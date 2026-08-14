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


#include <ATen/ExpandUtils.h>
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
// Shape / Movement Operations
// -----------------------------------------------------------------------------
Tensor transpose(const Tensor& self, int64_t dim0, int64_t dim1) {
    PTSYCL_TRACE_OP("transpose");
    const int64_t ndim = self.dim();
    const int64_t d0 = c10::maybe_wrap_dim(dim0, ndim);
    const int64_t d1 = c10::maybe_wrap_dim(dim1, ndim);
    if (d0 == d1) return self;
    auto sizes = self.sizes().vec();
    auto strides = self.strides().vec();
    std::swap(sizes[d0], sizes[d1]);
    std::swap(strides[d0], strides[d1]);
    return ptsycl::as_strided(self, sizes, strides, self.storage_offset());
}

Tensor permute(const Tensor& self, c10::IntArrayRef dims) {
    PTSYCL_TRACE_OP("permute");
    const int64_t ndim = self.dim();
    TORCH_CHECK(dims.size() == static_cast<size_t>(ndim),
                "permute: number of dimensions in dims must match tensor dimensions");
    std::vector<int64_t> new_sizes(ndim);
    std::vector<int64_t> new_strides(ndim);
    std::vector<bool> seen(ndim, false);
    for (int64_t i = 0; i < ndim; ++i) {
        const int64_t d = c10::maybe_wrap_dim(dims[i], ndim);
        TORCH_CHECK(!seen[d], "permute: duplicate dimension in dims");
        seen[d] = true;
        new_sizes[i] = self.size(d);
        new_strides[i] = self.stride(d);
    }
    return ptsycl::as_strided(self, new_sizes, new_strides, self.storage_offset());
}

Tensor expand(const Tensor& self, c10::SymIntArrayRef size, bool /*implicit*/) {
    PTSYCL_TRACE_OP("expand");
    auto target_sizes = c10::asIntArrayRefUnchecked(size);
    const auto result =
        at::inferExpandGeometry(self.sizes(), self.strides(), target_sizes);
    return ptsycl::as_strided(self, std::get<0>(result), std::get<1>(result),
                              self.storage_offset());
}

Tensor repeat(const Tensor& self, c10::SymIntArrayRef repeats_sym) {
    PTSYCL_TRACE_OP("repeat");
    auto repeats = c10::asIntArrayRefUnchecked(repeats_sym);
    const int64_t self_dim = self.dim();
    const int64_t repeats_dim = repeats.size();
    TORCH_CHECK(repeats_dim >= self_dim,
                "Number of dimensions of repeat dims can not be smaller than number of dimensions of tensor");

    Tensor src = self;
    if (repeats_dim > self_dim) {
        std::vector<int64_t> new_sizes(repeats_dim - self_dim, 1);
        new_sizes.insert(new_sizes.end(), self.sizes().begin(), self.sizes().end());
        src = ptsycl::view(self, new_sizes);
    }

    std::vector<int64_t> target_sizes(repeats_dim);
    std::vector<int64_t> expanded_sizes;
    std::vector<int64_t> expanded_strides;
    expanded_sizes.reserve(repeats_dim * 2);
    expanded_strides.reserve(repeats_dim * 2);

    for (int64_t i = 0; i < repeats_dim; ++i) {
        const int64_t r = repeats[i];
        const int64_t s = src.size(i);
        const int64_t st = src.stride(i);
        target_sizes[i] = r * s;
        expanded_sizes.push_back(r);
        expanded_sizes.push_back(s);
        expanded_strides.push_back(0);
        expanded_strides.push_back(st);
    }

    Tensor expanded = ptsycl::as_strided(src, expanded_sizes, expanded_strides, src.storage_offset());
    Tensor out = ptsycl::allocate_empty(target_sizes, src.scalar_type(), c10::kStrided, src.device(), std::nullopt, std::nullopt);
    ptsycl::_copy_from(expanded, out, /*non_blocking=*/false);
    return out;
}

Tensor narrow(const Tensor& self, int64_t dim, c10::SymInt start_sym, c10::SymInt length_sym) {
    PTSYCL_TRACE_OP("narrow.default");
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    const int64_t start = start_sym.expect_int();
    const int64_t length = length_sym.expect_int();
    const int64_t cur_size = self.size(d);
    TORCH_CHECK(start >= 0 && start <= cur_size, "narrow: start out of range");
    TORCH_CHECK(length >= 0 && start + length <= cur_size, "narrow: length out of range");

    auto new_sizes = self.sizes().vec();
    new_sizes[d] = length;
    const int64_t new_offset = self.storage_offset() + start * self.stride(d);
    return ptsycl::as_strided(self, new_sizes, self.strides(), new_offset);
}

Tensor contiguous(const Tensor& self, c10::MemoryFormat memory_format) {
    PTSYCL_TRACE_OP("contiguous");
    TORCH_CHECK(memory_format == c10::MemoryFormat::Contiguous,
                "paras contiguous supports only Contiguous memory format");
    if (self.is_contiguous(memory_format)) return self;
    Tensor result = ptsycl::allocate_empty(self.sizes(), self.scalar_type(),
                                           c10::kStrided, self.device(),
                                           std::nullopt, std::nullopt);
    ptsycl::_copy_from(self, result, /*non_blocking=*/false);
    return result;
}

Tensor reshape(const Tensor& self, c10::SymIntArrayRef shape_sym) {
    PTSYCL_TRACE_OP("reshape");
    const auto shape = c10::asIntArrayRefUnchecked(shape_sym);
    auto inferred = at::infer_size_dv(shape, self.numel());
    auto stride = at::detail::computeStride(self.sizes(), self.strides(), inferred);
    if (stride.has_value()) {
        return ptsycl::view(self, inferred);
    }
    return ptsycl::view(ptsycl::contiguous(self, c10::MemoryFormat::Contiguous), inferred);
}

Tensor repeat_interleave_int(const Tensor& self, c10::SymInt repeats_sym,
                             std::optional<int64_t> dim,
                             std::optional<c10::SymInt> /*output_size*/) {
    PTSYCL_TRACE_OP("repeat_interleave.self_int");
    const int64_t repeats = repeats_sym.expect_int();
    TORCH_CHECK(repeats >= 0, "repeat_interleave: repeats must be non-negative");

    Tensor src = self;
    int64_t d = 0;
    if (!dim.has_value()) {
        src = ptsycl::reshape(self, {self.numel()});
        d = 0;
    } else {
        d = c10::maybe_wrap_dim(*dim, self.dim());
    }

    const int64_t size_at_d = src.size(d);
    std::vector<int64_t> target_sizes = src.sizes().vec();
    target_sizes[d] = size_at_d * repeats;

    if (repeats == 0 || src.numel() == 0) {
        return ptsycl::allocate_empty(target_sizes, src.scalar_type(), c10::kStrided, src.device(), std::nullopt, std::nullopt);
    }

    std::vector<int64_t> exp_sizes;
    std::vector<int64_t> exp_strides;
    for (int64_t i = 0; i < src.dim(); ++i) {
        if (i == d) {
            exp_sizes.push_back(size_at_d);
            exp_sizes.push_back(repeats);
            exp_strides.push_back(src.stride(i));
            exp_strides.push_back(0);
        } else {
            exp_sizes.push_back(src.size(i));
            exp_strides.push_back(src.stride(i));
        }
    }

    Tensor expanded = ptsycl::as_strided(src, exp_sizes, exp_strides, src.storage_offset());
    Tensor out = ptsycl::allocate_empty(target_sizes, src.scalar_type(), c10::kStrided, src.device(), std::nullopt, std::nullopt);
    ptsycl::_copy_from(expanded, out, /*non_blocking=*/false);
    return out;
}

Tensor repeat_interleave_tensor(const Tensor& self, const Tensor& repeats,
                                std::optional<int64_t> dim,
                                std::optional<c10::SymInt> /*output_size*/) {
    PTSYCL_TRACE_OP("repeat_interleave.self_Tensor");
    Tensor rep = repeats.to(c10::kCPU).to(c10::kLong).contiguous();
    const int64_t n_rep = rep.numel();

    Tensor src = self;
    int64_t d = 0;
    if (!dim.has_value()) {
        src = ptsycl::reshape(self, {self.numel()});
        d = 0;
    } else {
        d = c10::maybe_wrap_dim(*dim, self.dim());
    }

    TORCH_CHECK(n_rep == src.size(d), "repeat_interleave: repeats length does not match dim size");
    const int64_t* p_rep = rep.data_ptr<int64_t>();

    int64_t total_out_dim = 0;
    std::vector<int64_t> offsets(n_rep + 1, 0);
    for (int64_t i = 0; i < n_rep; ++i) {
        TORCH_CHECK(p_rep[i] >= 0, "repeat_interleave: repeats must be non-negative");
        offsets[i] = total_out_dim;
        total_out_dim += p_rep[i];
    }
    offsets[n_rep] = total_out_dim;

    std::vector<int64_t> target_sizes = src.sizes().vec();
    target_sizes[d] = total_out_dim;

    Tensor out = ptsycl::allocate_empty(target_sizes, src.scalar_type(), c10::kStrided, src.device(), std::nullopt, std::nullopt);
    if (total_out_dim == 0 || src.numel() == 0) return out;

    for (int64_t i = 0; i < n_rep; ++i) {
        const int64_t count = p_rep[i];
        if (count > 0) {
            Tensor src_slice = ptsycl::narrow(src, d, i, 1);
            Tensor dst_slice = ptsycl::narrow(out, d, offsets[i], count);
            Tensor rep_src = ptsycl::repeat_interleave_int(src_slice, c10::SymInt(count), d, std::nullopt);
            ptsycl::_copy_from(rep_src, dst_slice, /*non_blocking=*/false);
        }
    }
    return out;
}

std::vector<Tensor> split_with_sizes(const Tensor& self, c10::SymIntArrayRef split_sizes_sym, int64_t dim) {
    PTSYCL_TRACE_OP("split_with_sizes");
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    auto split_sizes = c10::asIntArrayRefUnchecked(split_sizes_sym);
    const int64_t dim_size = self.size(d);

    int64_t total_size = 0;
    for (auto size : split_sizes) {
        TORCH_CHECK(size >= 0, "split_with_sizes: split size must be non-negative");
        total_size += size;
    }
    TORCH_CHECK(total_size == dim_size,
                "split_with_sizes: sum of split sizes must equal dim size");

    std::vector<Tensor> splits;
    splits.reserve(split_sizes.size());
    int64_t start = 0;
    for (auto length : split_sizes) {
        splits.push_back(ptsycl::narrow(self, d, start, length));
        start += length;
    }
    return splits;
}

std::vector<Tensor> split(const Tensor& self, c10::SymInt split_size_sym, int64_t dim) {
    PTSYCL_TRACE_OP("split.Tensor");
    const int64_t split_size = split_size_sym.expect_int();
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    const int64_t dim_size = self.size(d);
    TORCH_CHECK(split_size > 0, "split: split_size must be positive");

    std::vector<c10::SymInt> split_sizes;
    int64_t remaining = dim_size;
    while (remaining > 0) {
        const int64_t current_size = std::min(split_size, remaining);
        split_sizes.emplace_back(current_size);
        remaining -= current_size;
    }
    if (split_sizes.empty()) split_sizes.emplace_back(0);
    return ptsycl::split_with_sizes(self, split_sizes, d);
}

std::vector<Tensor> chunk(const Tensor& self, int64_t chunks, int64_t dim) {
    PTSYCL_TRACE_OP("chunk");
    TORCH_CHECK(chunks > 0, "chunk: chunks must be positive");
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    const int64_t dim_size = self.size(d);
    const int64_t split_size = (dim_size + chunks - 1) / chunks;
    return ptsycl::split(self, c10::SymInt(split_size > 0 ? split_size : 1), d);
}

Tensor squeeze_all(const Tensor& self) {
    PTSYCL_TRACE_OP("squeeze");
    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;
    for (int64_t d = 0; d < self.dim(); ++d) {
        if (self.size(d) != 1) {
            new_sizes.push_back(self.size(d));
            new_strides.push_back(self.stride(d));
        }
    }
    return ptsycl::as_strided(self, new_sizes, new_strides, self.storage_offset());
}

Tensor squeeze_dim(const Tensor& self, int64_t dim) {
    PTSYCL_TRACE_OP("squeeze.dim");
    const int64_t d = c10::maybe_wrap_dim(dim, self.dim());
    if (self.dim() > 0 && self.size(d) == 1) {
        std::vector<int64_t> new_sizes;
        std::vector<int64_t> new_strides;
        for (int64_t i = 0; i < self.dim(); ++i) {
            if (i != d) {
                new_sizes.push_back(self.size(i));
                new_strides.push_back(self.stride(i));
            }
        }
        return ptsycl::as_strided(self, new_sizes, new_strides, self.storage_offset());
    }
    return self;
}

Tensor squeeze_dims(const Tensor& self, c10::IntArrayRef dims) {
    PTSYCL_TRACE_OP("squeeze.dims");
    const int64_t ndim = self.dim();
    std::vector<bool> to_squeeze(ndim, false);
    for (auto d : dims) {
        const int64_t wrapped = c10::maybe_wrap_dim(d, ndim);
        if (self.size(wrapped) == 1) {
            to_squeeze[wrapped] = true;
        }
    }
    std::vector<int64_t> new_sizes;
    std::vector<int64_t> new_strides;
    for (int64_t i = 0; i < ndim; ++i) {
        if (!to_squeeze[i]) {
            new_sizes.push_back(self.size(i));
            new_strides.push_back(self.stride(i));
        }
    }
    return ptsycl::as_strided(self, new_sizes, new_strides, self.storage_offset());
}

Tensor unsqueeze(const Tensor& self, int64_t dim) {
    PTSYCL_TRACE_OP("unsqueeze");
    const int64_t new_dim = self.dim() + 1;
    const int64_t d = c10::maybe_wrap_dim(dim, new_dim);
    std::vector<int64_t> new_sizes(new_dim);
    std::vector<int64_t> new_strides(new_dim);
    for (int64_t i = 0, j = 0; i < new_dim; ++i) {
        if (i == d) {
            new_sizes[i] = 1;
            new_strides[i] = (i < new_dim - 1 && j < self.dim()) ? self.size(j) * self.stride(j) : 1;
        } else {
            new_sizes[i] = self.size(j);
            new_strides[i] = self.stride(j);
            ++j;
        }
    }
    return ptsycl::as_strided(self, new_sizes, new_strides, self.storage_offset());
}

Tensor slice(const Tensor& self, int64_t dim,
             std::optional<c10::SymInt> start_sym,
             std::optional<c10::SymInt> end_sym,
             c10::SymInt step_sym) {
    PTSYCL_TRACE_OP("slice.Tensor");
    const int64_t ndim = self.dim();
    if (ndim == 0) {
        TORCH_CHECK(dim == 0 || dim == -1, "slice: dim ", dim, " on 0-d tensor");
        return self.alias();
    }
    const int64_t d    = c10::maybe_wrap_dim(dim, ndim);
    const int64_t len  = self.size(d);
    const int64_t step = step_sym.expect_int();
    TORCH_CHECK(step > 0, "slice: step must be > 0, got ", step);

    int64_t start = start_sym.has_value() ? start_sym->expect_int() : 0;
    int64_t end   = end_sym.has_value() ? end_sym->expect_int() : len;
    if (start < 0) start += len;
    if (end < 0)   end   += len;
    start = std::clamp(start, int64_t(0), len);
    end   = std::clamp(end,   start,      len);

    auto new_sizes   = self.sizes().vec();
    auto new_strides = self.strides().vec();
    new_sizes[d]   = (end - start + step - 1) / step;
    new_strides[d] = self.stride(d) * step;
    const int64_t new_offset = self.storage_offset() + start * self.stride(d);

    return ptsycl::as_strided(self, new_sizes, new_strides, new_offset);
}

Tensor _to_copy(
    const Tensor& self,
    std::optional<c10::ScalarType> dtype,
    std::optional<c10::Layout> layout,
    std::optional<c10::Device> device,
    std::optional<bool> pin_memory,
    bool non_blocking,
    std::optional<c10::MemoryFormat> memory_format) {
    PTSYCL_TRACE_OP("_to_copy");

    auto target_dtype  = dtype.value_or(self.scalar_type());
    auto target_device = device.value_or(self.device());
    auto target_format = memory_format.value_or(at::MemoryFormat::Preserve);
    const bool preserve_strides = (target_format == at::MemoryFormat::Preserve) && self.is_non_overlapping_and_dense();
    const auto alloc_format = (target_format == at::MemoryFormat::Preserve) ? at::MemoryFormat::Contiguous : target_format;

    Tensor dst;
    if (target_device.is_cpu()) {
        if (preserve_strides) {
            dst = at::empty_strided(self.sizes(), self.strides().vec(),
                                    at::TensorOptions().dtype(target_dtype).device(c10::kCPU).pinned_memory(pin_memory.value_or(false)));
        } else {
            dst = at::empty(self.sizes(),
                            at::TensorOptions().dtype(target_dtype).device(c10::kCPU).pinned_memory(pin_memory.value_or(false)),
                            alloc_format);
        }
    } else {
        if (preserve_strides) {
            auto strides = self.strides().vec();
            dst = ptsycl::empty_strided(self.sizes(), strides, target_dtype, layout, target_device, pin_memory);
        } else {
            dst = ptsycl::allocate_empty(self.sizes(), target_dtype, layout.value_or(c10::kStrided), target_device, pin_memory, alloc_format);
        }
    }

    ptsycl::_copy_from(self, dst, non_blocking);
    return dst;
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
    m.impl("aten::_to_copy", &ptsycl::_to_copy);
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
    m.impl("aten::transpose.int", &ptsycl::transpose);
    m.impl("aten::permute", &ptsycl::permute);
    m.impl("aten::expand", &ptsycl::expand);
    m.impl("aten::repeat", &ptsycl::repeat);
    m.impl("aten::repeat_interleave.self_int", &ptsycl::repeat_interleave_int);
    m.impl("aten::repeat_interleave.self_Tensor", &ptsycl::repeat_interleave_tensor);
    m.impl("aten::split.Tensor", &ptsycl::split);
    m.impl("aten::split_with_sizes", &ptsycl::split_with_sizes);
    m.impl("aten::chunk", &ptsycl::chunk);
    m.impl("aten::contiguous", &ptsycl::contiguous);
    m.impl("aten::reshape", &ptsycl::reshape);
    m.impl("aten::squeeze", &ptsycl::squeeze_all);
    m.impl("aten::squeeze.dim", &ptsycl::squeeze_dim);
    m.impl("aten::squeeze.dims", &ptsycl::squeeze_dims);
    m.impl("aten::unsqueeze", &ptsycl::unsqueeze);
    m.impl("aten::narrow", &ptsycl::narrow);
    m.impl("aten::slice.Tensor", &ptsycl::slice);
}

TORCH_LIBRARY_IMPL(_, PrivateUse1, m) {
    m.fallback(
        torch::CppFunction::makeFromBoxedFunction<&ptsycl::fallback>());
}

TORCH_LIBRARY_IMPL(_, AutogradPrivateUse1, m) {
    m.fallback(torch::CppFunction::makeFallthrough());
}



