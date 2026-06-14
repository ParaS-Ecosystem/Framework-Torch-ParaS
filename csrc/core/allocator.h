#pragma once



#include <cstddef>
#include <list>
#include <map>
#include <mutex>

#include <c10/core/Allocator.h>
#include <c10/core/Device.h>

namespace ptsycl {

namespace compat { class Queue; }

class MemoryPool {
public:
    MemoryPool(compat::Queue& queue, int device_index);
    ~MemoryPool();

    void* acquire(std::size_t nbytes);
    void  release(void* ptr);
    void  clear();

    std::size_t allocated_bytes() const { return allocated_; }
    std::size_t cached_bytes()    const { return cached_; }

private:
    struct Block {
        void*       ptr;
        std::size_t rounded;
    };

    static std::size_t round_size(std::size_t v);

    compat::Queue& queue_;
    int            device_index_;
    bool           caching_enabled_;
    bool           debug_;

    std::mutex                              mutex_;
    std::map<std::size_t, std::list<void*>> free_lists_;
    std::map<void*, std::size_t>            live_; // ptr -> rounded size
    std::size_t                             allocated_ = 0;
    std::size_t                             cached_    = 0;
};


struct ParasAllocator final : public c10::Allocator {
    c10::DataPtr allocate(size_t nbytes) override;
    c10::DeleterFnPtr raw_deleter() const override;
    void copy_data(void* dest, const void* src, std::size_t count) const override;
};

ParasAllocator& allocator_instance();

} // namespace ptsycl
