// -----------------------------------------------------------------------------
// Copyright (c) 2026 Centre for Development of Advanced Computing (C-DAC)
//
// This file is part of Torch_ParaS, a component of the ParaS Ecosystem
//
// This library is free software: you can redistribute it and/or modify
// it under the terms of the GNU Lesser General Public License (LGPL)
// version 3 as published by the Free Software Foundation.
//
// This library is distributed in the hope that it will be useful,
// but WITHOUT ANY WARRANTY; without even the implied warranty of
// MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.
// See the GNU Lesser General Public License for more details.
//
// You should have received a copy of the GNU Lesser General Public License
// along with this library. If not, see <https://www.gnu.org/licenses/>.
// -----------------------------------------------------------------------------


#include <ATen/InferSize.h>
#include <ATen/native/CPUFallback.h>
#include <c10/core/DeviceGuard.h>

#include "core/allocator.h"
#include "core/kernels.h"

namespace ptsycl {
namespace {

using at::Tensor;

// -----------------------------------------------------------------------------
// Tensor construction
// -----------------------------------------------------------------------------
Tensor new_paras_tensor(c10::IntArrayRef sizes, c10::Device device,
                        c10::ScalarType dtype) {
    c10::DeviceGuard guard(device); // allocator reads the current device
    auto& alloc = allocator_instance();

    int64_t numel = 1;
    for (auto s : sizes) numel *= s;
    const std::size_t nbytes =
        static_cast<std::size_t>(numel) * c10::elementSize(dtype);

    c10::Storage storage(c10::Storage::use_byte_size_t(), nbytes,
                         alloc.allocate(nbytes), &alloc,
                         /*resizable=*/true);
    auto impl = c10::make_intrusive<c10::TensorImpl>(
        std::move(storage),
        c10::DispatchKeySet{c10::DispatchKey::PrivateUse1},
        caffe2::TypeMeta::fromScalarType(dtype));
    impl->set_sizes_contiguous(sizes);
    return Tensor(std::move(impl));
}

Tensor allocate_empty(c10::IntArrayRef size,
                      std::optional<c10::ScalarType> dtype,
                      std::optional<c10::Layout> layout,
                      std::optional<c10::Device> device,
                      std::optional<bool> /*pin_memory*/,
                      std::optional<c10::MemoryFormat> /*memory_format*/) {
    PTSYCL_TRACE_OP("empty.memory_format");
    TORCH_CHECK(!layout || *layout == c10::kStrided,
                "paras backend supports only strided layout");
    const c10::Device dev = device ? *device : c10::Device(kParasDevice, 0);
    const c10::ScalarType st = dtype ? *dtype : c10::kFloat;
    return new_paras_tensor(size, dev, st);
}

Tensor _reshape_alias(const Tensor& self, c10::IntArrayRef size,
                      c10::IntArrayRef stride) {
    PTSYCL_TRACE_OP("_reshape_alias");
    Tensor alias = at::alias(self);
    alias.getIntrusivePtr()->set_sizes_and_strides(size, stride);
    return alias;
}

Tensor empty_strided(c10::IntArrayRef size, c10::IntArrayRef stride,
                     std::optional<c10::ScalarType> dtype,
                     std::optional<c10::Layout> layout,
                     std::optional<c10::Device> device,
                     std::optional<bool> pin_memory) {
    PTSYCL_TRACE_OP("empty_strided");
    Tensor r = allocate_empty(size, dtype, layout, device, pin_memory,
                              std::nullopt);
    return ptsycl::_reshape_alias(r, size, stride);
}

Tensor view(const Tensor& self, c10::IntArrayRef size) {
    PTSYCL_TRACE_OP("view");
    auto inferred = at::infer_size_dv(size, self.numel());
    auto stride =
        at::detail::computeStride(self.sizes(), self.strides(), inferred);
    TORCH_CHECK(stride.has_value(),
                "view size is not compatible with input tensor's size and "
                "stride; use .reshape(...) instead");
    Tensor alias = at::alias(self);
    alias.getIntrusivePtr()->set_sizes_and_strides(inferred, *stride);
    return alias;
}

Tensor as_strided(const Tensor& self, c10::IntArrayRef size,
                  c10::IntArrayRef stride,
                  std::optional<int64_t> storage_offset) {
    PTSYCL_TRACE_OP("as_strided");
    Tensor alias = at::alias(self);
    alias.getIntrusivePtr()->set_sizes_and_strides(size, stride);
    if (storage_offset)
        alias.getIntrusivePtr()->set_storage_offset(*storage_offset);
    return alias;
}

// -----------------------------------------------------------------------------
// Copy machinery
// -----------------------------------------------------------------------------

#define PTSYCL_COPY_TYPES(_)                                                  \
    _(uint8_t, c10::kByte)                                                    \
    _(int8_t, c10::kChar)                                                     \
    _(int16_t, c10::kShort)                                                   \
    _(int32_t, c10::kInt)                                                     \
    _(int64_t, c10::kLong)                                                    \
    _(float, c10::kFloat)                                                     \
    _(double, c10::kDouble)                                                   \
    _(bool, c10::kBool)                                                       \
    _(c10::Half, c10::kHalf)                                                  \
    _(c10::BFloat16, c10::kBFloat16)

// Gathers a strided tensor into a contiguous buffer (same dtype).
void device_gather_contiguous(compat::Queue& q, const Tensor& src, void* dst) {
    const auto spec = make_spec(src);
    const int64_t n = src.numel();
    switch (src.scalar_type()) {
#define CASE(T, ST)                                                           \
    case ST: {                                                                \
        const T* s = data_ptr<T>(src);                                        \
        T*       d = static_cast<T*>(dst);                                    \
        launch_flat(q, n, [=](std::size_t i) {                                \
            d[i] = s[spec.index(static_cast<int64_t>(i))];                    \
        });                                                                   \
        break;                                                                \
    }
        PTSYCL_COPY_TYPES(CASE)
#undef CASE
        default:
            TORCH_CHECK(false, "paras copy: unsupported dtype ",
                        src.scalar_type());
    }
}

// Scatters a contiguous buffer into a strided tensor (same dtype).
void device_scatter_strided(compat::Queue& q, const void* src, Tensor& dst) {
    const auto spec = make_spec(dst);
    const int64_t n = dst.numel();
    switch (dst.scalar_type()) {
#define CASE(T, ST)                                                           \
    case ST: {                                                                \
        const T* s = static_cast<const T*>(src);                              \
        T*       d = data_ptr<T>(dst);                                        \
        launch_flat(q, n, [=](std::size_t i) {                                \
            d[spec.index(static_cast<int64_t>(i))] = s[i];                    \
        });                                                                   \
        break;                                                                \
    }
        PTSYCL_COPY_TYPES(CASE)
#undef CASE
        default:
            TORCH_CHECK(false, "paras copy: unsupported dtype ",
                        dst.scalar_type());
    }
}


template <typename SrcT>
void device_convert_from(compat::Queue& q, const SrcT* s, Tensor& dst,
                         int64_t n) {
    switch (dst.scalar_type()) {
#define CASE(T, ST)                                                           \
    case ST: {                                                                \
        T* d = data_ptr<T>(dst);                                              \
        launch_flat(q, n, [=](std::size_t i) {                                \
            d[i] = static_cast<T>(s[i]);                                      \
        });                                                                   \
        break;                                                                \
    }
        PTSYCL_COPY_TYPES(CASE)
#undef CASE
        default:
            TORCH_CHECK(false, "paras convert: unsupported dst dtype ",
                        dst.scalar_type());
    }
}

void device_convert(compat::Queue& q, const Tensor& src_contig, Tensor& dst_contig) {
    const int64_t n = src_contig.numel();
    switch (src_contig.scalar_type()) {
#define CASE(T, ST)                                                           \
    case ST:                                                                  \
        device_convert_from<T>(q, data_ptr<T>(src_contig), dst_contig, n);    \
        break;
        PTSYCL_COPY_TYPES(CASE)
#undef CASE
        default:
            TORCH_CHECK(false, "paras convert: unsupported src dtype ",
                        src_contig.scalar_type());
    }
}


void device_to_device_copy(const Tensor& src, Tensor& dst) {
    auto& q = queue_for(dst);
    const int64_t n = dst.numel();
    TORCH_CHECK(src.numel() == n, "paras copy: element count mismatch");
    if (n == 0) return;

    const bool same_type = src.scalar_type() == dst.scalar_type();
    const bool src_contig = src.is_contiguous();
    const bool dst_contig = dst.is_contiguous();

    if (same_type && src_contig && dst_contig) {
        q.copy(dst.data_ptr(), src.data_ptr(),
               static_cast<std::size_t>(n) * src.element_size(),
               /*blocking=*/false);
        return;
    }

  
    Tensor src_c = src_contig
                       ? src
                       : new_paras_tensor(src.sizes(), src.device(),
                                          src.scalar_type());
    if (!src_contig) device_gather_contiguous(q, src, src_c.data_ptr());

   
    Tensor conv = src_c;
    if (!same_type) {
        conv = new_paras_tensor(dst.sizes(), dst.device(), dst.scalar_type());
        device_convert(q, src_c, conv);
    }

    
    if (dst_contig) {
        q.copy(dst.data_ptr(), conv.data_ptr(),
               static_cast<std::size_t>(n) * dst.element_size(),
               /*blocking=*/false);
    } else {
        device_scatter_strided(q, conv.data_ptr(), dst);
    }
}

Tensor _copy_from(const Tensor& self, const Tensor& dst, bool non_blocking) {
    PTSYCL_TRACE_OP("_copy_from");
    if (self.numel() == 0) return self;

    const bool src_paras = is_paras_tensor(self);
    const bool dst_paras = is_paras_tensor(dst);

    // paras -> CPU
    if (src_paras && dst.device().is_cpu()) {
        auto& q = queue_for(self);
       
        Tensor staged = self;
        if (!self.is_contiguous() || self.scalar_type() != dst.scalar_type()) {
            staged = new_paras_tensor(self.sizes(), self.device(),
                                      dst.scalar_type());
            device_to_device_copy(self, staged);
        }
        if (dst.is_contiguous()) {
            q.copy(dst.data_ptr(), staged.data_ptr(),
                   static_cast<std::size_t>(staged.numel()) *
                       staged.element_size(),
                   /*blocking=*/true);
        } else {
            Tensor host_c = at::empty(staged.sizes(),
                                      dst.options().device(c10::kCPU));
            q.copy(host_c.data_ptr(), staged.data_ptr(),
                   static_cast<std::size_t>(staged.numel()) *
                       staged.element_size(),
                   /*blocking=*/true);
            const_cast<Tensor&>(dst).copy_(host_c);
        }
        return self;
    }

    // CPU -> paras
    if (self.device().is_cpu() && dst_paras) {
        auto& q = queue_for(dst);
        Tensor src_c = self;
        if (!self.is_contiguous() || self.scalar_type() != dst.scalar_type()) {
            src_c = self.to(dst.scalar_type()).contiguous();
        }
        if (dst.is_contiguous()) {
            q.copy(dst.data_ptr(), src_c.data_ptr(),
                   static_cast<std::size_t>(src_c.numel()) *
                       src_c.element_size(),
                   /*blocking=*/true);
        } else {
            Tensor staged = new_paras_tensor(dst.sizes(), dst.device(),
                                             dst.scalar_type());
            q.copy(staged.data_ptr(), src_c.data_ptr(),
                   static_cast<std::size_t>(src_c.numel()) *
                       src_c.element_size(),
                   /*blocking=*/true);
            device_scatter_strided(q, staged.data_ptr(),
                                   const_cast<Tensor&>(dst));
            q.synchronize();
        }
        return self;
    }

    // paras -> paras
    if (src_paras && dst_paras) {
        if (self.device().index() != dst.device().index()) {

            Tensor host = at::empty(
                self.sizes(),
                self.options().device(c10::kCPU).dtype(dst.scalar_type()));
            ptsycl::_copy_from(self, host, /*non_blocking=*/false);
            ptsycl::_copy_from(host, dst, /*non_blocking=*/false);
            return self;
        }
        device_to_device_copy(self, const_cast<Tensor&>(dst));
        if (!non_blocking) queue_for(dst).synchronize();
        return self;
    }

    TORCH_CHECK(false, "paras _copy_from: unsupported device pair ",
                self.device(), " -> ", dst.device());
}

Tensor _copy_from_and_resize(const Tensor& self, const Tensor& dst) {
    PTSYCL_TRACE_OP("_copy_from_and_resize");
    if (dst.sizes() != self.sizes())
        const_cast<Tensor&>(dst).resize_(self.sizes());
    return ptsycl::_copy_from(self, dst, /*non_blocking=*/false);
}

// -----------------------------------------------------------------------------
// fill_ / zero_
// -----------------------------------------------------------------------------
Tensor& fill_(Tensor& self, const c10::Scalar& value) {
    PTSYCL_TRACE_OP("fill_");
    const int64_t n = self.numel();
    if (n == 0) return self;
    auto& q = queue_for(self);

    AT_DISPATCH_ALL_TYPES_AND3(
        c10::kHalf, c10::kBFloat16, c10::kBool, self.scalar_type(), "fill_",
        [&] {
            const scalar_t val = value.to<scalar_t>();
            scalar_t* d = data_ptr<scalar_t>(self);
            if (self.is_contiguous()) {
                launch_flat(q, n, [=](std::size_t i) { d[i] = val; });
            } else {
                const auto spec = make_spec(self);
                launch_flat(q, n, [=](std::size_t i) {
                    d[spec.index(static_cast<int64_t>(i))] = val;
                });
            }
        });
    return self;
}

Tensor& zero_(Tensor& self) {
    PTSYCL_TRACE_OP("zero_");
    if (self.numel() == 0) return self;
    if (self.is_contiguous()) {
        queue_for(self).memset(self.data_ptr(), 0,
                               static_cast<std::size_t>(self.nbytes()));
        return self;
    }
    return ptsycl::fill_(self, 0);
}

// -----------------------------------------------------------------------------
// Scalar extraction
// -----------------------------------------------------------------------------
c10::Scalar _local_scalar_dense(const Tensor& self) {
    PTSYCL_TRACE_OP("_local_scalar_dense");
    TORCH_CHECK(self.numel() == 1,
                "_local_scalar_dense expects a single-element tensor");
    auto& q = queue_for(self);
    q.synchronize(); // order after pending kernels before host read

    switch (self.scalar_type()) {
#define CASE(T, ST)                                                           \
    case ST:                                                                  \
        return c10::Scalar(*data_ptr<T>(self));
        PTSYCL_COPY_TYPES(CASE)
#undef CASE
        default:
            TORCH_CHECK(false, "_local_scalar_dense: unsupported dtype ",
                        self.scalar_type());
    }
}

// -----------------------------------------------------------------------------
// Storage manipulation
// -----------------------------------------------------------------------------
Tensor& set_source_storage_offset(Tensor& self, c10::Storage source,
                                  c10::SymInt storage_offset,
                                  c10::SymIntArrayRef size,
                                  c10::SymIntArrayRef stride) {
    PTSYCL_TRACE_OP("set_.source_Storage_storage_offset");
    auto impl = self.getIntrusivePtr();
    impl->set_storage_keep_dtype(std::move(source));
    impl->set_sizes_and_strides(size, stride, storage_offset);
    return self;
}

Tensor& set_source_storage(Tensor& self, c10::Storage source) {
    PTSYCL_TRACE_OP("set_.source_Storage");
    auto impl = self.getIntrusivePtr();
    const auto nbytes = source.nbytes();
    impl->set_storage_keep_dtype(std::move(source));
    const int64_t elem = static_cast<int64_t>(self.element_size());
    const int64_t numel = elem > 0 ? static_cast<int64_t>(nbytes) / elem : 0;
    impl->set_sizes_contiguous({numel});
    return self;
}

const Tensor& resize_(const Tensor& self, c10::SymIntArrayRef size,
                      std::optional<c10::MemoryFormat> memory_format) {
    PTSYCL_TRACE_OP("resize_");
    if (memory_format)
        TORCH_CHECK(*memory_format == c10::MemoryFormat::Contiguous,
                    "paras resize_ supports only contiguous memory format");

    auto impl = self.getIntrusivePtr();
    const c10::Storage& storage = impl->storage();
    const int64_t old_bytes = static_cast<int64_t>(storage.nbytes());

    std::vector<int64_t> sizes;
    int64_t numel = 1;
    for (const auto& s : size) {
        const int64_t v = s.expect_int();
        sizes.push_back(v);
        numel *= v;
    }
    const int64_t new_bytes =
        numel * static_cast<int64_t>(self.element_size());

    if (new_bytes > old_bytes) {
        c10::DeviceGuard guard(self.device());
        auto& q = queue_for(self);
        at::DataPtr fresh =
            allocator_instance().allocate(static_cast<std::size_t>(new_bytes));
        if (old_bytes > 0) {
            q.copy(fresh.get(), storage.data(),
                   static_cast<std::size_t>(old_bytes), /*blocking=*/true);
        }
        storage.mutable_data_ptr() = std::move(fresh);
        storage.set_nbytes(static_cast<std::size_t>(new_bytes));
    }
    impl->set_sizes_contiguous(sizes);
    return self;
}

// -----------------------------------------------------------------------------
// Embedding
// -----------------------------------------------------------------------------
Tensor embedding(const Tensor& weight, const Tensor& indices,
                 c10::SymInt padding_idx, bool /*scale_grad_by_freq*/,
                 bool sparse) {
    PTSYCL_TRACE_OP("embedding");
    TORCH_CHECK(!sparse,
                "paras: sparse gradients for embedding are not supported");
    TORCH_CHECK(weight.dim() == 2, "paras embedding: weight must be 2-D");
    (void)padding_idx; // padding_idx only affects embedding_dense_backward

    auto& q = queue_for(weight);
    Tensor w   = weight.contiguous();
    Tensor idx = indices.contiguous();
    if (idx.scalar_type() != c10::kLong) idx = idx.to(c10::kLong);

    const int64_t dim = w.size(1);
    const int64_t n   = idx.numel();

    std::vector<int64_t> out_sizes(idx.sizes().begin(), idx.sizes().end());
    out_sizes.push_back(dim);
    Tensor out = at::empty(out_sizes, w.options());
    if (n == 0 || dim == 0) return out;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, w.scalar_type(), "ptsycl_embedding", [&] {
            const scalar_t* pw = data_ptr<scalar_t>(w);
            const int64_t*  pi = data_ptr<int64_t>(idx);
            scalar_t*       po = data_ptr<scalar_t>(out);
            launch_flat(q, n * dim, [=](std::size_t flat) {
                const int64_t row   = static_cast<int64_t>(flat) / dim;
                const int64_t col   = static_cast<int64_t>(flat) % dim;
                const int64_t w_row = pi[row];
                po[row * dim + col] = pw[w_row * dim + col];
            });
        });
    return out;
}

// grad_output has shape indices.sizes() + [dim]; grad_weight has shape
// [num_weights, dim]. Parallelizing over weight rows (rather than over
// indices) means every thread owns a disjoint output row, so contributions
// can be accumulated with a plain read-loop and no atomics are needed. This
// is O(num_weights * n) and favors correctness over throughput; a
// sorted-scatter kernel would be the natural follow-up for large
// vocabularies.
Tensor embedding_dense_backward(const Tensor& grad_output,
                                const Tensor& indices,
                                c10::SymInt num_weights_sym,
                                c10::SymInt padding_idx_sym,
                                bool scale_grad_by_freq) {
    PTSYCL_TRACE_OP("embedding_dense_backward");
    const int64_t num_weights = num_weights_sym.guard_int(__FILE__, __LINE__);
    const int64_t padding_idx = padding_idx_sym.guard_int(__FILE__, __LINE__);

    auto& q    = queue_for(grad_output);
    Tensor go  = grad_output.contiguous();
    Tensor idx = indices.contiguous();
    if (idx.scalar_type() != c10::kLong) idx = idx.to(c10::kLong);

    const int64_t dim = go.size(-1);
    const int64_t n   = idx.numel();

    Tensor grad_weight = at::zeros({num_weights, dim}, go.options());
    if (n == 0 || dim == 0 || num_weights == 0) return grad_weight;

    AT_DISPATCH_FLOATING_TYPES_AND2(
        c10::kHalf, c10::kBFloat16, go.scalar_type(),
        "ptsycl_embedding_backward", [&] {
            const scalar_t* pg = data_ptr<scalar_t>(go);
            const int64_t*  pi = data_ptr<int64_t>(idx);
            scalar_t*       pw = data_ptr<scalar_t>(grad_weight);

            launch_flat(q, num_weights, [=](std::size_t flat) {
                const int64_t w = static_cast<int64_t>(flat);
                if (w == padding_idx) return;

                int64_t count = 0;
                for (int64_t i = 0; i < n; ++i)
                    if (pi[i] == w) ++count;
                if (count == 0) return;

                const double scale =
                    scale_grad_by_freq ? 1.0 / static_cast<double>(count)
                                       : 1.0;
                for (int64_t col = 0; col < dim; ++col) {
                    double acc = 0.0;
                    for (int64_t i = 0; i < n; ++i)
                        if (pi[i] == w)
                            acc += static_cast<double>(pg[i * dim + col]);
                    pw[w * dim + col] = static_cast<scalar_t>(acc * scale);
                }
            });
        });
    return grad_weight;
}

bool _has_compatible_shallow_copy_type(const Tensor& self,
                                       const Tensor& from) {
    auto dense = [](c10::DispatchKeySet ks) {
        return ks.has(c10::DispatchKey::CPU) ||
               ks.has(c10::DispatchKey::PrivateUse1);
    };
    return self.key_set() == from.key_set() ||
           (dense(self.key_set()) && dense(from.key_set()));
}

// -----------------------------------------------------------------------------
// Boxed CPU fallback for everything not implemented natively.
// -----------------------------------------------------------------------------
void fallback(const c10::OperatorHandle& op, torch::jit::Stack* stack) {
    if (log::level() >= log::kInfo) {
        PTSYCL_INFO("falling back to CPU for %s",
                    op.schema().operator_name().name.c_str());
    }
    at::native::cpu_fallback(op, stack);
}

} // namespace
} // namespace ptsycl

TORCH_LIBRARY_IMPL(aten, PrivateUse1, m) {
    m.impl("aten::empty.memory_format", &ptsycl::allocate_empty);
    m.impl("aten::empty_strided", &ptsycl::empty_strided);
    m.impl("aten::_reshape_alias", &ptsycl::_reshape_alias);
    m.impl("aten::view", &ptsycl::view);
    m.impl("aten::as_strided", &ptsycl::as_strided);
    m.impl("aten::_copy_from", &ptsycl::_copy_from);
    m.impl("aten::_copy_from_and_resize", &ptsycl::_copy_from_and_resize);
    m.impl("aten::fill_.Scalar", &ptsycl::fill_);
    m.impl("aten::zero_", &ptsycl::zero_);
    m.impl("aten::_local_scalar_dense", &ptsycl::_local_scalar_dense);
    m.impl("aten::set_.source_Storage", &ptsycl::set_source_storage);
    m.impl("aten::set_.source_Storage_storage_offset",
           &ptsycl::set_source_storage_offset);
    m.impl("aten::resize_", &ptsycl::resize_);
    m.impl("aten::_has_compatible_shallow_copy_type",
           &ptsycl::_has_compatible_shallow_copy_type);
    m.impl("aten::embedding", &ptsycl::embedding);
    m.impl("aten::embedding_dense_backward",
           &ptsycl::embedding_dense_backward);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(
        torch::CppFunction::makeFromBoxedFunction<&ptsycl::fallback>());
}

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}






