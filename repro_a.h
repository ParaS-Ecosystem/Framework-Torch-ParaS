// // repro_a.h
// #pragma once

// #if !defined(PTSYCL_BACKEND_SYCL) && !defined(PTSYCL_BACKEND_CPU)
// #error "Define exactly one of PTSYCL_BACKEND_SYCL / PTSYCL_BACKEND_CPU"
// #endif

// #if defined(PTSYCL_BACKEND_SYCL)
// #include <sycl/sycl.hpp>
// #endif

// namespace repro {

// template <typename F>
// class ParasKernelKey;

// class Queue {
// public:
//     void init();
//     void copy(void* dst, const void* src, std::size_t nbytes);

// #if defined(PTSYCL_BACKEND_SYCL)
//     sycl::queue& sycl_queue() { return *sycl_queue_; }
// #endif

//     template <typename F>
//     void parallel_for(std::size_t n, F f) {
// #if defined(PTSYCL_BACKEND_SYCL)
//         sycl_queue().template parallel_for<ParasKernelKey<F>>(
//             sycl::range<1>(n),
//             [f](sycl::id<1> i) { f(static_cast<std::size_t>(i[0])); });
// #endif
//     }

//     void* alloc(std::size_t nbytes);

// private:
// #if defined(PTSYCL_BACKEND_SYCL)
//     sycl::queue* sycl_queue_ = nullptr;
// #endif
// };

// } // namespace repro


// repro_a.h — add std::optional-based member alongside the existing pointer version
#pragma once
#include <sycl/sycl.hpp>
#include <optional>

namespace repro {

class QueueOpt {
public:
    void init();
    void copy(void* dst, const void* src, std::size_t nbytes);
    sycl::queue& sycl_queue() { return *sycl_queue_; }

private:
    std::optional<sycl::queue> sycl_queue_;
};

} // namespace repro