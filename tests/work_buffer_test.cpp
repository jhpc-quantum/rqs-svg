#include "../src/work_buffer.hpp"
#include <cassert>
#include <iostream>
#include <limits>

template<class F> void rejects(F operation)
{
    bool rejected = false;
    try { operation(); }
    catch (const std::exception&) { rejected = true; }
    assert(rejected);
}

int main()
{
    using qcs::buffer_arena;
    for (size_t n = 0; n < 10000; ++n)
        assert(buffer_arena::budget(n) == n * 95 / 100);
    auto max = std::numeric_limits<size_t>::max();
    assert(buffer_arena::budget(max) == max - max / 20 - (max % 20 != 0));

    alignas(256) unsigned char storage[1024]{};
    buffer_arena arena(storage, sizeof(storage));
    assert(arena.take(1) == storage);
    assert(arena.remaining() == 1023);
    assert(arena.available() == 768);
    assert(arena.take(257) == storage + 256);
    assert(arena.available() == 256);
    size_t before = arena.remaining();
    rejects([&] { arena.take(257); });
    rejects([&] { arena.take(max); });
    rejects([&] { arena.take(1, 0); });
    rejects([&] { arena.take(1, 3); });
    assert(arena.remaining() == before); // failed requests never consume storage
    assert(arena.take(256) == storage + 768);
    assert(arena.remaining() == 0);
    rejects([&] { arena.take(1); });

    // A non-aligned base and a tiny tail must not produce an invalid slice.
    buffer_arena shifted(storage + 1, 511);
    assert(shifted.available() == 256);
    assert(shifted.take(256) == storage + 256);
    buffer_arena tail(storage + 1, 254);
    assert(tail.available() == 0);
    rejects([&] { tail.take(1); });
    buffer_arena empty;
    rejects([&] { empty.take(1); });
    std::cout << "work buffer: budget, alignment, disjoint slices and exhaustion passed\n";
}
