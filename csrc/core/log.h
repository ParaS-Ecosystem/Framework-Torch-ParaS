#pragma once


#include <atomic>
#include <cstdio>
#include <cstdlib>

namespace ptsycl {
namespace log {

enum Level : int { kWarn = 0, kInfo = 1, kTrace = 2 };

inline int level() {
    static const int lvl = [] {
        const char* env = std::getenv("PTSYCL_LOG");
        return env ? std::atoi(env) : 0;
    }();
    return lvl;
}

#define PTSYCL_WARN(...)                                            \
    do {                                                            \
        std::fprintf(stderr, "[ptsycl warn] " __VA_ARGS__);         \
        std::fprintf(stderr, "\n");                                 \
    } while (0)

#define PTSYCL_INFO(...)                                            \
    do {                                                            \
        if (::ptsycl::log::level() >= ::ptsycl::log::kInfo) {       \
            std::fprintf(stderr, "[ptsycl info] " __VA_ARGS__);     \
            std::fprintf(stderr, "\n");                             \
        }                                                           \
    } while (0)

#define PTSYCL_TRACE_OP(name)                                       \
    do {                                                            \
        if (::ptsycl::log::level() >= ::ptsycl::log::kTrace) {      \
            std::fprintf(stderr, "[ptsycl op] %s\n", (name));       \
        }                                                           \
    } while (0)

} // namespace log
} // namespace ptsycl
