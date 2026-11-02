#pragma once

#include <cstddef>
#include <cstdint>
#include <stdexcept>

namespace qcs
{
// Non-owning, monotonic slices. The owner must keep the backing allocation alive
// until all streams using any slice have finished. Reset only between lifetimes.
class buffer_arena
{
    unsigned char* data_ = nullptr;
    size_t size_ = 0;
    size_t used_ = 0;

public:
    buffer_arena() = default;
    buffer_arena(void* data, size_t size) : data_(static_cast<unsigned char*>(data)), size_(size) {}

    static size_t budget(size_t free_bytes)
    {
        // floor(free_bytes * 95 / 100), without overflowing size_t.
        return free_bytes / 100 * 95 + free_bytes % 100 * 95 / 100;
    }

    size_t remaining() const { return size_ - used_; }
    size_t available(size_t alignment = 256) const
    {
        if (!alignment || (alignment & (alignment - 1)))
            throw std::invalid_argument("work buffer: alignment must be a power of two");
        size_t offset = (reinterpret_cast<uintptr_t>(data_) % alignment + used_ % alignment) % alignment;
        size_t padding = (alignment - offset) % alignment;
        return padding <= remaining() ? remaining() - padding : 0;
    }
    void* take(size_t bytes, size_t alignment = 256)
    {
        size_t aligned_bytes = available(alignment);
        size_t padding = remaining() - aligned_bytes;
        if (!data_ || bytes > aligned_bytes || (bytes == 0 && padding == remaining()))
            throw std::runtime_error("work buffer: insufficient capacity");
        used_ += padding;
        void* result = data_ + used_;
        used_ += bytes;
        return result;
    }
};
} // namespace qcs
