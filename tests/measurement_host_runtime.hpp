// Synchronous, single-rank stand-ins used ONLY by measurement_host.py.
// These exercise the production orchestration and buffer lifecycle, not CUB
// algorithms or asynchronous/device/MPI behavior.
using cudaStream_t = int;
using cudaEvent_t = int;
using ncclUniqueId = int;
using ncclComm_t = int;
using bit_t = uint8_t;
using bit_num_t = int16_t;
constexpr int cudaMemcpyDeviceToHost = 0, cudaMemcpyHostToDevice = 1;
constexpr int MPI_DOUBLE = 0, MPI_SUM = 0, MPI_COMM_WORLD = 0, MPI_UINT64_T = 1;
static int reductions = 0, allocations = 0, allreduces = 0, broadcasts = 0;
#define ATLC_CHECK_CUDA(function, ...) function(__VA_ARGS__)
#define ATLC_CHECK_MPI(function, ...) function(__VA_ARGS__)
static void cudaMallocAsync(void **pointer, size_t size, int)
{
    *pointer = std::malloc(size);
    if (!*pointer) throw std::bad_alloc();
    ++allocations;
}
static void cudaFreeAsync(void *pointer, int) { std::free(pointer); }
static void cudaMemcpyAsync(void *dest, const void *src, size_t size, int, int) { std::memcpy(dest, src, size); }
static void cudaStreamSynchronize(int) {}
static void MPI_Allreduce(const double *src, double *dest, int count, int, int, int)
{
    ++allreduces;
    std::copy_n(src, count, dest);
}
static void MPI_Bcast(uint64_t *, int, int, int, int) { ++broadcasts; }
namespace atlc {
    template <typename... Args>
    std::string format(const char *message, Args...) { return message; }
}
namespace thrust {
    template <typename T> struct counting_iterator {
        T first;
        explicit counting_iterator(T value) : first(value) {}
        T operator[](uint64_t i) const { return first + i; }
    };
    template <typename F, typename I> struct transform_iterator {
        I input;
        F function;
        transform_iterator(I i, F f) : input(i), function(f) {}
        auto operator[](uint64_t i) const { return function(input[i]); }
    };
}
namespace cub {
    struct DeviceReduce {
        template <typename I, typename T, typename Add>
        static void Reduce(void *temp, size_t &size, I input, T *output,
                           uint64_t count, Add add, T zero, int)
        {
            // Deliberately vary requirements by value size and input count.
            size_t required = 64 + sizeof(T) * (count + 1);
            if (!temp) { size = required; return; }
            if (size < required) throw std::runtime_error("undersized scratch buffer");
            ++reductions;
            for (uint64_t i = 0; i < count; ++i) zero = add(zero, input[i]);
            *output = zero;
        }
    };
}
