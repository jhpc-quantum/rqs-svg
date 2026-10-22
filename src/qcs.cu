#include <cstdlib>
#include <cstdio>
#include <cstdarg>
#include <cstring>
#include <cmath>
#include <cstdint>
#include <cinttypes>
#include <unistd.h>
#include <dlfcn.h>

#include <stdexcept>
#include <string>
#include <sstream>
#include <algorithm>
#include <chrono>
#include <random>
#include <utility>
#include <unordered_set>
#include <string_view>
#include <algorithm>
#include <string>
#include <unordered_set>
#include <tuple>
#include <type_traits>

#include <mpi.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda/std/complex>
#include <cuda/std/array>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <nccl.h>

#include <atlc/format.hpp>
#include <atlc/log2_int.hpp>
#include <atlc/mpi.hpp>
#include <atlc/cuda.hpp>
#include <atlc/check_mpi.hpp>
#include <atlc/check_cuda.hpp>
#include <atlc/check_curand.hpp>
#include <atlc/check_nccl.hpp>

#include <qcs.h>

namespace qcs
{

    struct simulator_core;

    typedef double float_t;
    typedef cuda::std::complex<qcs::float_t> complex_t;
    typedef cuda::std::array<qcs::float_t, 2> float2_t;

    __device__ __host__ complex_t multiply_i(complex_t input)
    {
        return complex_t(-input.imag(), input.real());
    }

    __device__ __host__ complex_t multiply_i_m(complex_t input)
    {
        return complex_t(input.imag(), -input.real());
    }

    __device__ __host__ complex_t multiply_i_real(complex_t input, float_t multiplier)
    {
        return complex_t(-multiplier * input.imag(), +multiplier * input.real());
    }

    __global__ void initstate_sequential_kernel(qcs::complex_t *const data_global, int proc_num)
    {
        uint64_t const num_threads = (uint64_t)gridDim.x * (uint64_t)blockDim.x;
        uint64_t const idx = (uint64_t)blockDim.x * (uint64_t)blockIdx.x + (uint64_t)threadIdx.x;
        data_global[idx] = idx + num_threads * proc_num;
    }

    __global__ void initstate_flat_kernel(qcs::complex_t *const data_global)
    {
        uint64_t const idx = (uint64_t)blockDim.x * (uint64_t)blockIdx.x + (uint64_t)threadIdx.x;
        data_global[idx] = 1;
    }

    struct kernel_common_struct
    {
        int num_qubits;
        qcs::complex_t *state_data_device;
    };

    __constant__ qcs::complex_t zero_constant;

    __constant__ qcs::kernel_common_struct kernel_common_constant;

    struct kernel_input_qnlist_struct
    {
        int num_target_qubits;
        int num_negative_control_qubits;
        int num_positive_control_qubits;
        uint64_t is_measured_bits;
        uint64_t measured_value_bits;
        int qubit_num_list[1];

        static constexpr __host__ __device__ uint64_t needed_size(
            int const num_target_qubits,
            int const num_negative_control_qubits,
            int const num_positive_control_qubits)
        {
            return sizeof(qcs::kernel_input_qnlist_struct) - sizeof(qubit_num_list) + sizeof(int) * (2 * num_positive_control_qubits + num_negative_control_qubits + 2 * num_target_qubits);
        }

        __host__ __device__ uint64_t byte_size() const
        {
            return needed_size(this->num_target_qubits, this->num_negative_control_qubits, this->num_positive_control_qubits);
        }

        __host__ __device__ int get_num_operand_qubits() const
        {
            return this->num_positive_control_qubits + this->num_negative_control_qubits + this->num_target_qubits;
        }

        __host__ __device__ int const *get_operand_qubit_num_list_sorted() const
        {
            return this->qubit_num_list;
        }

        __host__ __device__ int *get_operand_qubit_num_list_sorted()
        {
            return this->qubit_num_list;
        }

        __host__ __device__ int const *get_positive_control_qubit_num_list() const
        {
            return this->qubit_num_list + this->get_num_operand_qubits();
        }

        __host__ __device__ int *get_positive_control_qubit_num_list()
        {
            return this->qubit_num_list + this->get_num_operand_qubits();
        }

        __host__ __device__ int const *get_target_qubit_num_list() const
        {
            return this->qubit_num_list + 2 * this->num_positive_control_qubits + this->num_negative_control_qubits + this->num_target_qubits;
        }

        __host__ __device__ int *get_target_qubit_num_list()
        {
            return this->qubit_num_list + 2 * this->num_positive_control_qubits + this->num_negative_control_qubits + this->num_target_qubits;
        }
    }; /* kernel_input_qnlist_struct */

    constexpr int max_num_qubits_local = 34;
    constexpr uint64_t kernel_input_max_size = qcs::kernel_input_qnlist_struct::needed_size(qcs::max_num_qubits_local, 0, 0);
    __constant__ unsigned char kernel_input_constant[qcs::kernel_input_max_size];

    static __device__ void thread_num_to_state_index_q0(uint64_t thread_num, uint64_t &index_state)
    {
        auto args = (qcs::kernel_input_qnlist_struct const *)(void *)qcs::kernel_input_constant;

        index_state = 0;

        int const num_operand_qubits = args->get_num_operand_qubits();
        int const *const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

        // generate index_state
        // ignoring positive control qubits
        uint64_t lower_mask = 0;
        for (int i = 0; i < num_operand_qubits; i++)
        {
            uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
            uint64_t const upper_mask = mask & ~lower_mask;
            lower_mask = mask;
            index_state |= (thread_num & upper_mask) << i;
        }
        index_state |= (thread_num & ~lower_mask) << num_operand_qubits;

        // update index_state
        // considering positive control qubits
        int const *const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
        for (int i = 0; i < args->num_positive_control_qubits; i++)
        {
            index_state |= UINT64_C(1) << positive_control_qubit_num_list[i];
        }

    } /* thread_num_to_state_index_q0 */

    static __device__ void thread_num_to_state_index_q1(
        uint64_t thread_num,
        uint64_t &index_state_0, uint64_t &index_state_1,
        int &is_measured_bits, int &measured_value_bits)
    {
        auto args = (qcs::kernel_input_qnlist_struct const *)(void *)qcs::kernel_input_constant;

        index_state_0 = 0;

        int const num_operand_qubits = args->get_num_operand_qubits();
        int const *const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

        // generate index_state_0
        // ignoring positive control qubits
        uint64_t lower_mask = 0;
        for (int i = 0; i < num_operand_qubits; i++)
        {
            uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
            uint64_t const upper_mask = mask & ~lower_mask;
            lower_mask = mask;
            index_state_0 |= (thread_num & upper_mask) << i;
        }
        index_state_0 |= (thread_num & ~lower_mask) << num_operand_qubits;

        // update index_state_0
        // considering positive control qubits
        int const *const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
        for (int i = 0; i < args->num_positive_control_qubits; i++)
        {
            index_state_0 |= UINT64_C(1) << positive_control_qubit_num_list[i];
        }

        // generate other indices
        auto target_qubit_num_list = args->get_target_qubit_num_list();

        index_state_1 = index_state_0 | (UINT64_C(1) << target_qubit_num_list[0]);

        is_measured_bits = args->is_measured_bits;
        measured_value_bits = args->measured_value_bits;

    } /* thread_num_to_state_index_q1 */

    static __device__ void thread_num_to_state_index_q2(
        uint64_t thread_num,
        uint64_t &index_state_00, uint64_t &index_state_01, uint64_t &index_state_10, uint64_t &index_state_11,
        int &is_measured_bits, int &measured_value_bits)
    {
        auto args = (qcs::kernel_input_qnlist_struct const *)(void *)qcs::kernel_input_constant;

        index_state_00 = 0;

        int const num_operand_qubits = args->get_num_operand_qubits();
        int const *const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

        // generate index_state_00
        // ignoring positive control qubits
        uint64_t lower_mask = 0;
        for (int i = 0; i < num_operand_qubits; i++)
        {
            uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
            uint64_t const upper_mask = mask & ~lower_mask;
            lower_mask = mask;
            index_state_00 |= (thread_num & upper_mask) << i;
        }
        index_state_00 |= (thread_num & ~lower_mask) << num_operand_qubits;

        // update index_state_00
        // considering positive control qubits
        int const *const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
        for (int i = 0; i < args->num_positive_control_qubits; i++)
        {
            index_state_00 |= UINT64_C(1) << positive_control_qubit_num_list[i];
        }

        // generate other indices
        auto target_qubit_num_list = args->get_target_qubit_num_list();

        index_state_01 = index_state_00 | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10 = index_state_00 | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11 = index_state_00 | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        is_measured_bits = args->is_measured_bits;
        measured_value_bits = args->measured_value_bits;

    } /* thread_num_to_state_index_q2 */

    static __device__ void thread_num_to_state_index_q3(
        uint64_t thread_num,
        uint64_t &index_state_000, uint64_t &index_state_001, uint64_t &index_state_010, uint64_t &index_state_011,
        uint64_t &index_state_100, uint64_t &index_state_101, uint64_t &index_state_110, uint64_t &index_state_111,
        int &is_measured_bits, int &measured_value_bits)
    {
        auto args = (qcs::kernel_input_qnlist_struct const *)(void *)qcs::kernel_input_constant;

        index_state_000 = 0;

        int const num_operand_qubits = args->get_num_operand_qubits();
        int const *const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

        // generate index_state_000
        // ignoring positive control qubits
        uint64_t lower_mask = 0;
        for (int i = 0; i < num_operand_qubits; i++)
        {
            uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
            uint64_t const upper_mask = mask & ~lower_mask;
            lower_mask = mask;
            index_state_000 |= (thread_num & upper_mask) << i;
        }
        index_state_000 |= (thread_num & ~lower_mask) << num_operand_qubits;

        // update index_state_000
        // considering positive control qubits
        int const *const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
        for (int i = 0; i < args->num_positive_control_qubits; i++)
        {
            index_state_000 |= UINT64_C(1) << positive_control_qubit_num_list[i];
        }

        // generate other indices
        auto target_qubit_num_list = args->get_target_qubit_num_list();

        index_state_001 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_010 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_011 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_100 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_101 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_110 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_111 = index_state_000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        is_measured_bits = args->is_measured_bits;
        measured_value_bits = args->measured_value_bits;

    } /* thread_num_to_state_index_q3 */

    static __device__ void thread_num_to_state_index_q4(
        uint64_t thread_num,
        uint64_t &index_state_0000, uint64_t &index_state_0001, uint64_t &index_state_0010, uint64_t &index_state_0011,
        uint64_t &index_state_0100, uint64_t &index_state_0101, uint64_t &index_state_0110, uint64_t &index_state_0111,
        uint64_t &index_state_1000, uint64_t &index_state_1001, uint64_t &index_state_1010, uint64_t &index_state_1011,
        uint64_t &index_state_1100, uint64_t &index_state_1101, uint64_t &index_state_1110, uint64_t &index_state_1111,
        int &is_measured_bits, int &measured_value_bits)
    {
        auto args = (qcs::kernel_input_qnlist_struct const *)(void *)qcs::kernel_input_constant;

        index_state_0000 = 0;

        int const num_operand_qubits = args->get_num_operand_qubits();
        int const *const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

        // generate index_state_0000
        // ignoring positive control qubits
        uint64_t lower_mask = 0;
        for (int i = 0; i < num_operand_qubits; i++)
        {
            uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
            uint64_t const upper_mask = mask & ~lower_mask;
            lower_mask = mask;
            index_state_0000 |= (thread_num & upper_mask) << i;
        }
        index_state_0000 |= (thread_num & ~lower_mask) << num_operand_qubits;

        // update index_state_0000
        // considering positive control qubits
        int const *const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
        for (int i = 0; i < args->num_positive_control_qubits; i++)
        {
            index_state_0000 |= UINT64_C(1) << positive_control_qubit_num_list[i];
        }

        // generate other indices
        auto target_qubit_num_list = args->get_target_qubit_num_list();

        index_state_0001 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0010 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0011 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0100 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_0101 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0110 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0111 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1000 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_1001 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1010 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1011 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1100 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_1101 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1110 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1111 = index_state_0000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        is_measured_bits = args->is_measured_bits;
        measured_value_bits = args->measured_value_bits;

    } /* thread_num_to_state_index_q4 */

    static __device__ void thread_num_to_state_index_q5(
        uint64_t thread_num,
        uint64_t &index_state_00000, uint64_t &index_state_00001, uint64_t &index_state_00010, uint64_t &index_state_00011,
        uint64_t &index_state_00100, uint64_t &index_state_00101, uint64_t &index_state_00110, uint64_t &index_state_00111,
        uint64_t &index_state_01000, uint64_t &index_state_01001, uint64_t &index_state_01010, uint64_t &index_state_01011,
        uint64_t &index_state_01100, uint64_t &index_state_01101, uint64_t &index_state_01110, uint64_t &index_state_01111,
        uint64_t &index_state_10000, uint64_t &index_state_10001, uint64_t &index_state_10010, uint64_t &index_state_10011,
        uint64_t &index_state_10100, uint64_t &index_state_10101, uint64_t &index_state_10110, uint64_t &index_state_10111,
        uint64_t &index_state_11000, uint64_t &index_state_11001, uint64_t &index_state_11010, uint64_t &index_state_11011,
        uint64_t &index_state_11100, uint64_t &index_state_11101, uint64_t &index_state_11110, uint64_t &index_state_11111,
        int &is_measured_bits, int &measured_value_bits)
    {
        auto args = (qcs::kernel_input_qnlist_struct const *)(void *)qcs::kernel_input_constant;

        index_state_00000 = 0;

        int const num_operand_qubits = args->get_num_operand_qubits();
        int const *const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

        // generate index_state_00000
        // ignoring positive control qubits
        uint64_t lower_mask = 0;
        for (int i = 0; i < num_operand_qubits; i++)
        {
            uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
            uint64_t const upper_mask = mask & ~lower_mask;
            lower_mask = mask;
            index_state_00000 |= (thread_num & upper_mask) << i;
        }
        index_state_00000 |= (thread_num & ~lower_mask) << num_operand_qubits;

        // update index_state_00000
        // considering positive control qubits
        int const *const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
        for (int i = 0; i < args->num_positive_control_qubits; i++)
        {
            index_state_00000 |= UINT64_C(1) << positive_control_qubit_num_list[i];
        }

        // generate other indices
        auto target_qubit_num_list = args->get_target_qubit_num_list();

        index_state_00001 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00010 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00011 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00100 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_00101 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00110 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00111 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01000 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_01001 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01010 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01011 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01100 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_01101 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01110 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01111 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10000 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_10001 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10010 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10011 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10100 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_10101 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10110 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10111 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11000 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_11001 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11010 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11011 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11100 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_11101 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11110 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11111 = index_state_00000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        is_measured_bits = args->is_measured_bits;
        measured_value_bits = args->measured_value_bits;

    } /* thread_num_to_state_index_q5 */

    static __device__ void thread_num_to_state_index_q6(
        uint64_t thread_num,
        uint64_t &index_state_000000, uint64_t &index_state_000001, uint64_t &index_state_000010, uint64_t &index_state_000011,
        uint64_t &index_state_000100, uint64_t &index_state_000101, uint64_t &index_state_000110, uint64_t &index_state_000111,
        uint64_t &index_state_001000, uint64_t &index_state_001001, uint64_t &index_state_001010, uint64_t &index_state_001011,
        uint64_t &index_state_001100, uint64_t &index_state_001101, uint64_t &index_state_001110, uint64_t &index_state_001111,
        uint64_t &index_state_010000, uint64_t &index_state_010001, uint64_t &index_state_010010, uint64_t &index_state_010011,
        uint64_t &index_state_010100, uint64_t &index_state_010101, uint64_t &index_state_010110, uint64_t &index_state_010111,
        uint64_t &index_state_011000, uint64_t &index_state_011001, uint64_t &index_state_011010, uint64_t &index_state_011011,
        uint64_t &index_state_011100, uint64_t &index_state_011101, uint64_t &index_state_011110, uint64_t &index_state_011111,
        uint64_t &index_state_100000, uint64_t &index_state_100001, uint64_t &index_state_100010, uint64_t &index_state_100011,
        uint64_t &index_state_100100, uint64_t &index_state_100101, uint64_t &index_state_100110, uint64_t &index_state_100111,
        uint64_t &index_state_101000, uint64_t &index_state_101001, uint64_t &index_state_101010, uint64_t &index_state_101011,
        uint64_t &index_state_101100, uint64_t &index_state_101101, uint64_t &index_state_101110, uint64_t &index_state_101111,
        uint64_t &index_state_110000, uint64_t &index_state_110001, uint64_t &index_state_110010, uint64_t &index_state_110011,
        uint64_t &index_state_110100, uint64_t &index_state_110101, uint64_t &index_state_110110, uint64_t &index_state_110111,
        uint64_t &index_state_111000, uint64_t &index_state_111001, uint64_t &index_state_111010, uint64_t &index_state_111011,
        uint64_t &index_state_111100, uint64_t &index_state_111101, uint64_t &index_state_111110, uint64_t &index_state_111111,
        int &is_measured_bits, int &measured_value_bits)
    {
        auto args = (qcs::kernel_input_qnlist_struct const *)(void *)qcs::kernel_input_constant;

        index_state_000000 = 0;

        int const num_operand_qubits = args->get_num_operand_qubits();
        int const *const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

        // generate index_state_000000
        // ignoring positive control qubits
        uint64_t lower_mask = 0;
        for (int i = 0; i < num_operand_qubits; i++)
        {
            uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
            uint64_t const upper_mask = mask & ~lower_mask;
            lower_mask = mask;
            index_state_000000 |= (thread_num & upper_mask) << i;
        }
        index_state_000000 |= (thread_num & ~lower_mask) << num_operand_qubits;

        // update index_state_000000
        // considering positive control qubits
        int const *const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
        for (int i = 0; i < args->num_positive_control_qubits; i++)
        {
            index_state_000000 |= UINT64_C(1) << positive_control_qubit_num_list[i];
        }

        // generate other indices
        auto target_qubit_num_list = args->get_target_qubit_num_list();

        index_state_000001 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_000010 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_000011 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_000100 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_000101 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_000110 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_000111 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_001000 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_001001 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_001010 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_001011 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_001100 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_001101 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_001110 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_001111 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_010000 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_010001 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_010010 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_010011 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_010100 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_010101 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_010110 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_010111 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_011000 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_011001 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_011010 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_011011 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_011100 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_011101 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_011110 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_011111 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_100000 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]);
        index_state_100001 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_100010 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_100011 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_100100 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_100101 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_100110 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_100111 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_101000 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_101001 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_101010 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_101011 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_101100 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_101101 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_101110 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_101111 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_110000 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_110001 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_110010 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_110011 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_110100 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_110101 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_110110 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_110111 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_111000 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_111001 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_111010 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_111011 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_111100 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_111101 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_111110 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_111111 = index_state_000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        is_measured_bits = args->is_measured_bits;
        measured_value_bits = args->measured_value_bits;

    } /* thread_num_to_state_index_q6 */

    static __device__ void thread_num_to_state_index_q7(
        uint64_t thread_num,
        uint64_t &index_state_0000000, uint64_t &index_state_0000001, uint64_t &index_state_0000010, uint64_t &index_state_0000011,
        uint64_t &index_state_0000100, uint64_t &index_state_0000101, uint64_t &index_state_0000110, uint64_t &index_state_0000111,
        uint64_t &index_state_0001000, uint64_t &index_state_0001001, uint64_t &index_state_0001010, uint64_t &index_state_0001011,
        uint64_t &index_state_0001100, uint64_t &index_state_0001101, uint64_t &index_state_0001110, uint64_t &index_state_0001111,
        uint64_t &index_state_0010000, uint64_t &index_state_0010001, uint64_t &index_state_0010010, uint64_t &index_state_0010011,
        uint64_t &index_state_0010100, uint64_t &index_state_0010101, uint64_t &index_state_0010110, uint64_t &index_state_0010111,
        uint64_t &index_state_0011000, uint64_t &index_state_0011001, uint64_t &index_state_0011010, uint64_t &index_state_0011011,
        uint64_t &index_state_0011100, uint64_t &index_state_0011101, uint64_t &index_state_0011110, uint64_t &index_state_0011111,
        uint64_t &index_state_0100000, uint64_t &index_state_0100001, uint64_t &index_state_0100010, uint64_t &index_state_0100011,
        uint64_t &index_state_0100100, uint64_t &index_state_0100101, uint64_t &index_state_0100110, uint64_t &index_state_0100111,
        uint64_t &index_state_0101000, uint64_t &index_state_0101001, uint64_t &index_state_0101010, uint64_t &index_state_0101011,
        uint64_t &index_state_0101100, uint64_t &index_state_0101101, uint64_t &index_state_0101110, uint64_t &index_state_0101111,
        uint64_t &index_state_0110000, uint64_t &index_state_0110001, uint64_t &index_state_0110010, uint64_t &index_state_0110011,
        uint64_t &index_state_0110100, uint64_t &index_state_0110101, uint64_t &index_state_0110110, uint64_t &index_state_0110111,
        uint64_t &index_state_0111000, uint64_t &index_state_0111001, uint64_t &index_state_0111010, uint64_t &index_state_0111011,
        uint64_t &index_state_0111100, uint64_t &index_state_0111101, uint64_t &index_state_0111110, uint64_t &index_state_0111111,
        uint64_t &index_state_1000000, uint64_t &index_state_1000001, uint64_t &index_state_1000010, uint64_t &index_state_1000011,
        uint64_t &index_state_1000100, uint64_t &index_state_1000101, uint64_t &index_state_1000110, uint64_t &index_state_1000111,
        uint64_t &index_state_1001000, uint64_t &index_state_1001001, uint64_t &index_state_1001010, uint64_t &index_state_1001011,
        uint64_t &index_state_1001100, uint64_t &index_state_1001101, uint64_t &index_state_1001110, uint64_t &index_state_1001111,
        uint64_t &index_state_1010000, uint64_t &index_state_1010001, uint64_t &index_state_1010010, uint64_t &index_state_1010011,
        uint64_t &index_state_1010100, uint64_t &index_state_1010101, uint64_t &index_state_1010110, uint64_t &index_state_1010111,
        uint64_t &index_state_1011000, uint64_t &index_state_1011001, uint64_t &index_state_1011010, uint64_t &index_state_1011011,
        uint64_t &index_state_1011100, uint64_t &index_state_1011101, uint64_t &index_state_1011110, uint64_t &index_state_1011111,
        uint64_t &index_state_1100000, uint64_t &index_state_1100001, uint64_t &index_state_1100010, uint64_t &index_state_1100011,
        uint64_t &index_state_1100100, uint64_t &index_state_1100101, uint64_t &index_state_1100110, uint64_t &index_state_1100111,
        uint64_t &index_state_1101000, uint64_t &index_state_1101001, uint64_t &index_state_1101010, uint64_t &index_state_1101011,
        uint64_t &index_state_1101100, uint64_t &index_state_1101101, uint64_t &index_state_1101110, uint64_t &index_state_1101111,
        uint64_t &index_state_1110000, uint64_t &index_state_1110001, uint64_t &index_state_1110010, uint64_t &index_state_1110011,
        uint64_t &index_state_1110100, uint64_t &index_state_1110101, uint64_t &index_state_1110110, uint64_t &index_state_1110111,
        uint64_t &index_state_1111000, uint64_t &index_state_1111001, uint64_t &index_state_1111010, uint64_t &index_state_1111011,
        uint64_t &index_state_1111100, uint64_t &index_state_1111101, uint64_t &index_state_1111110, uint64_t &index_state_1111111,
        int &is_measured_bits, int &measured_value_bits)
    {
        auto args = (qcs::kernel_input_qnlist_struct const *)(void *)qcs::kernel_input_constant;

        index_state_0000000 = 0;

        int const num_operand_qubits = args->get_num_operand_qubits();
        int const *const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

        // generate index_state_0000000
        // ignoring positive control qubits
        uint64_t lower_mask = 0;
        for (int i = 0; i < num_operand_qubits; i++)
        {
            uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
            uint64_t const upper_mask = mask & ~lower_mask;
            lower_mask = mask;
            index_state_0000000 |= (thread_num & upper_mask) << i;
        }
        index_state_0000000 |= (thread_num & ~lower_mask) << num_operand_qubits;

        // update index_state_0000000
        // considering positive control qubits
        int const *const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
        for (int i = 0; i < args->num_positive_control_qubits; i++)
        {
            index_state_0000000 |= UINT64_C(1) << positive_control_qubit_num_list[i];
        }

        // generate other indices
        auto target_qubit_num_list = args->get_target_qubit_num_list();

        index_state_0000001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0000010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0000011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0000100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_0000101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0000110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0000111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0001000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_0001001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0001010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0001011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0001100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_0001101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0001110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0001111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0010000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_0010001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0010010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0010011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0010100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_0010101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0010110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0010111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0011000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_0011001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0011010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0011011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0011100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_0011101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0011110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0011111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0100000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]);
        index_state_0100001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0100010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0100011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0100100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_0100101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0100110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0100111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0101000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_0101001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0101010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0101011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0101100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_0101101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0101110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0101111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0110000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_0110001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0110010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0110011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0110100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_0110101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0110110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0110111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0111000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_0111001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0111010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0111011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_0111100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_0111101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_0111110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_0111111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1000000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]);
        index_state_1000001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1000010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1000011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1000100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_1000101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1000110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1000111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1001000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_1001001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1001010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1001011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1001100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_1001101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1001110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1001111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1010000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_1010001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1010010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1010011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1010100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_1010101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1010110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1010111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1011000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_1011001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1011010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1011011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1011100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_1011101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1011110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1011111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1100000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]);
        index_state_1100001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1100010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1100011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1100100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_1100101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1100110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1100111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1101000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_1101001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1101010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1101011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1101100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_1101101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1101110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1101111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1110000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_1110001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1110010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1110011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1110100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_1110101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1110110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1110111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1111000 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_1111001 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1111010 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1111011 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_1111100 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_1111101 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_1111110 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_1111111 = index_state_0000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        is_measured_bits = args->is_measured_bits;
        measured_value_bits = args->measured_value_bits;

    } /* thread_num_to_state_index_q7 */

    static __device__ void thread_num_to_state_index_q8(
        uint64_t thread_num,
        uint64_t &index_state_00000000, uint64_t &index_state_00000001, uint64_t &index_state_00000010, uint64_t &index_state_00000011,
        uint64_t &index_state_00000100, uint64_t &index_state_00000101, uint64_t &index_state_00000110, uint64_t &index_state_00000111,
        uint64_t &index_state_00001000, uint64_t &index_state_00001001, uint64_t &index_state_00001010, uint64_t &index_state_00001011,
        uint64_t &index_state_00001100, uint64_t &index_state_00001101, uint64_t &index_state_00001110, uint64_t &index_state_00001111,
        uint64_t &index_state_00010000, uint64_t &index_state_00010001, uint64_t &index_state_00010010, uint64_t &index_state_00010011,
        uint64_t &index_state_00010100, uint64_t &index_state_00010101, uint64_t &index_state_00010110, uint64_t &index_state_00010111,
        uint64_t &index_state_00011000, uint64_t &index_state_00011001, uint64_t &index_state_00011010, uint64_t &index_state_00011011,
        uint64_t &index_state_00011100, uint64_t &index_state_00011101, uint64_t &index_state_00011110, uint64_t &index_state_00011111,
        uint64_t &index_state_00100000, uint64_t &index_state_00100001, uint64_t &index_state_00100010, uint64_t &index_state_00100011,
        uint64_t &index_state_00100100, uint64_t &index_state_00100101, uint64_t &index_state_00100110, uint64_t &index_state_00100111,
        uint64_t &index_state_00101000, uint64_t &index_state_00101001, uint64_t &index_state_00101010, uint64_t &index_state_00101011,
        uint64_t &index_state_00101100, uint64_t &index_state_00101101, uint64_t &index_state_00101110, uint64_t &index_state_00101111,
        uint64_t &index_state_00110000, uint64_t &index_state_00110001, uint64_t &index_state_00110010, uint64_t &index_state_00110011,
        uint64_t &index_state_00110100, uint64_t &index_state_00110101, uint64_t &index_state_00110110, uint64_t &index_state_00110111,
        uint64_t &index_state_00111000, uint64_t &index_state_00111001, uint64_t &index_state_00111010, uint64_t &index_state_00111011,
        uint64_t &index_state_00111100, uint64_t &index_state_00111101, uint64_t &index_state_00111110, uint64_t &index_state_00111111,
        uint64_t &index_state_01000000, uint64_t &index_state_01000001, uint64_t &index_state_01000010, uint64_t &index_state_01000011,
        uint64_t &index_state_01000100, uint64_t &index_state_01000101, uint64_t &index_state_01000110, uint64_t &index_state_01000111,
        uint64_t &index_state_01001000, uint64_t &index_state_01001001, uint64_t &index_state_01001010, uint64_t &index_state_01001011,
        uint64_t &index_state_01001100, uint64_t &index_state_01001101, uint64_t &index_state_01001110, uint64_t &index_state_01001111,
        uint64_t &index_state_01010000, uint64_t &index_state_01010001, uint64_t &index_state_01010010, uint64_t &index_state_01010011,
        uint64_t &index_state_01010100, uint64_t &index_state_01010101, uint64_t &index_state_01010110, uint64_t &index_state_01010111,
        uint64_t &index_state_01011000, uint64_t &index_state_01011001, uint64_t &index_state_01011010, uint64_t &index_state_01011011,
        uint64_t &index_state_01011100, uint64_t &index_state_01011101, uint64_t &index_state_01011110, uint64_t &index_state_01011111,
        uint64_t &index_state_01100000, uint64_t &index_state_01100001, uint64_t &index_state_01100010, uint64_t &index_state_01100011,
        uint64_t &index_state_01100100, uint64_t &index_state_01100101, uint64_t &index_state_01100110, uint64_t &index_state_01100111,
        uint64_t &index_state_01101000, uint64_t &index_state_01101001, uint64_t &index_state_01101010, uint64_t &index_state_01101011,
        uint64_t &index_state_01101100, uint64_t &index_state_01101101, uint64_t &index_state_01101110, uint64_t &index_state_01101111,
        uint64_t &index_state_01110000, uint64_t &index_state_01110001, uint64_t &index_state_01110010, uint64_t &index_state_01110011,
        uint64_t &index_state_01110100, uint64_t &index_state_01110101, uint64_t &index_state_01110110, uint64_t &index_state_01110111,
        uint64_t &index_state_01111000, uint64_t &index_state_01111001, uint64_t &index_state_01111010, uint64_t &index_state_01111011,
        uint64_t &index_state_01111100, uint64_t &index_state_01111101, uint64_t &index_state_01111110, uint64_t &index_state_01111111,
        uint64_t &index_state_10000000, uint64_t &index_state_10000001, uint64_t &index_state_10000010, uint64_t &index_state_10000011,
        uint64_t &index_state_10000100, uint64_t &index_state_10000101, uint64_t &index_state_10000110, uint64_t &index_state_10000111,
        uint64_t &index_state_10001000, uint64_t &index_state_10001001, uint64_t &index_state_10001010, uint64_t &index_state_10001011,
        uint64_t &index_state_10001100, uint64_t &index_state_10001101, uint64_t &index_state_10001110, uint64_t &index_state_10001111,
        uint64_t &index_state_10010000, uint64_t &index_state_10010001, uint64_t &index_state_10010010, uint64_t &index_state_10010011,
        uint64_t &index_state_10010100, uint64_t &index_state_10010101, uint64_t &index_state_10010110, uint64_t &index_state_10010111,
        uint64_t &index_state_10011000, uint64_t &index_state_10011001, uint64_t &index_state_10011010, uint64_t &index_state_10011011,
        uint64_t &index_state_10011100, uint64_t &index_state_10011101, uint64_t &index_state_10011110, uint64_t &index_state_10011111,
        uint64_t &index_state_10100000, uint64_t &index_state_10100001, uint64_t &index_state_10100010, uint64_t &index_state_10100011,
        uint64_t &index_state_10100100, uint64_t &index_state_10100101, uint64_t &index_state_10100110, uint64_t &index_state_10100111,
        uint64_t &index_state_10101000, uint64_t &index_state_10101001, uint64_t &index_state_10101010, uint64_t &index_state_10101011,
        uint64_t &index_state_10101100, uint64_t &index_state_10101101, uint64_t &index_state_10101110, uint64_t &index_state_10101111,
        uint64_t &index_state_10110000, uint64_t &index_state_10110001, uint64_t &index_state_10110010, uint64_t &index_state_10110011,
        uint64_t &index_state_10110100, uint64_t &index_state_10110101, uint64_t &index_state_10110110, uint64_t &index_state_10110111,
        uint64_t &index_state_10111000, uint64_t &index_state_10111001, uint64_t &index_state_10111010, uint64_t &index_state_10111011,
        uint64_t &index_state_10111100, uint64_t &index_state_10111101, uint64_t &index_state_10111110, uint64_t &index_state_10111111,
        uint64_t &index_state_11000000, uint64_t &index_state_11000001, uint64_t &index_state_11000010, uint64_t &index_state_11000011,
        uint64_t &index_state_11000100, uint64_t &index_state_11000101, uint64_t &index_state_11000110, uint64_t &index_state_11000111,
        uint64_t &index_state_11001000, uint64_t &index_state_11001001, uint64_t &index_state_11001010, uint64_t &index_state_11001011,
        uint64_t &index_state_11001100, uint64_t &index_state_11001101, uint64_t &index_state_11001110, uint64_t &index_state_11001111,
        uint64_t &index_state_11010000, uint64_t &index_state_11010001, uint64_t &index_state_11010010, uint64_t &index_state_11010011,
        uint64_t &index_state_11010100, uint64_t &index_state_11010101, uint64_t &index_state_11010110, uint64_t &index_state_11010111,
        uint64_t &index_state_11011000, uint64_t &index_state_11011001, uint64_t &index_state_11011010, uint64_t &index_state_11011011,
        uint64_t &index_state_11011100, uint64_t &index_state_11011101, uint64_t &index_state_11011110, uint64_t &index_state_11011111,
        uint64_t &index_state_11100000, uint64_t &index_state_11100001, uint64_t &index_state_11100010, uint64_t &index_state_11100011,
        uint64_t &index_state_11100100, uint64_t &index_state_11100101, uint64_t &index_state_11100110, uint64_t &index_state_11100111,
        uint64_t &index_state_11101000, uint64_t &index_state_11101001, uint64_t &index_state_11101010, uint64_t &index_state_11101011,
        uint64_t &index_state_11101100, uint64_t &index_state_11101101, uint64_t &index_state_11101110, uint64_t &index_state_11101111,
        uint64_t &index_state_11110000, uint64_t &index_state_11110001, uint64_t &index_state_11110010, uint64_t &index_state_11110011,
        uint64_t &index_state_11110100, uint64_t &index_state_11110101, uint64_t &index_state_11110110, uint64_t &index_state_11110111,
        uint64_t &index_state_11111000, uint64_t &index_state_11111001, uint64_t &index_state_11111010, uint64_t &index_state_11111011,
        uint64_t &index_state_11111100, uint64_t &index_state_11111101, uint64_t &index_state_11111110, uint64_t &index_state_11111111,
        int &is_measured_bits, int &measured_value_bits)
    {
        auto args = (qcs::kernel_input_qnlist_struct const *)(void *)qcs::kernel_input_constant;

        index_state_00000000 = 0;

        int const num_operand_qubits = args->get_num_operand_qubits();
        int const *const qubit_num_list_sorted = args->get_operand_qubit_num_list_sorted();

        // generate index_state_00000000
        // ignoring positive control qubits
        uint64_t lower_mask = 0;
        for (int i = 0; i < num_operand_qubits; i++)
        {
            uint64_t const mask = (UINT64_C(1) << (qubit_num_list_sorted[i] - i)) - 1;
            uint64_t const upper_mask = mask & ~lower_mask;
            lower_mask = mask;
            index_state_00000000 |= (thread_num & upper_mask) << i;
        }
        index_state_00000000 |= (thread_num & ~lower_mask) << num_operand_qubits;

        // update index_state_00000000
        // considering positive control qubits
        int const *const positive_control_qubit_num_list = args->get_positive_control_qubit_num_list();
        for (int i = 0; i < args->num_positive_control_qubits; i++)
        {
            index_state_00000000 |= UINT64_C(1) << positive_control_qubit_num_list[i];
        }

        // generate other indices
        auto target_qubit_num_list = args->get_target_qubit_num_list();

        index_state_00000001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00000010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00000011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00000100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_00000101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00000110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00000111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00001000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_00001001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00001010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00001011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00001100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_00001101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00001110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00001111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00010000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_00010001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00010010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00010011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00010100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_00010101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00010110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00010111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00011000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_00011001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00011010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00011011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00011100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_00011101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00011110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00011111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00100000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]);
        index_state_00100001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00100010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00100011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00100100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_00100101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00100110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00100111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00101000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_00101001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00101010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00101011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00101100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_00101101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00101110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00101111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00110000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_00110001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00110010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00110011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00110100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_00110101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00110110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00110111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00111000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_00111001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00111010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00111011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_00111100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_00111101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_00111110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_00111111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01000000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]);
        index_state_01000001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01000010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01000011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01000100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_01000101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01000110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01000111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01001000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_01001001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01001010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01001011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01001100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_01001101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01001110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01001111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01010000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_01010001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01010010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01010011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01010100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_01010101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01010110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01010111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01011000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_01011001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01011010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01011011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01011100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_01011101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01011110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01011111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01100000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]);
        index_state_01100001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01100010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01100011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01100100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_01100101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01100110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01100111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01101000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_01101001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01101010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01101011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01101100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_01101101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01101110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01101111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01110000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_01110001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01110010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01110011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01110100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_01110101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01110110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01110111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01111000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_01111001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01111010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01111011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_01111100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_01111101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_01111110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_01111111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10000000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]);
        index_state_10000001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10000010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10000011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10000100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_10000101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10000110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10000111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10001000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_10001001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10001010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10001011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10001100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_10001101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10001110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10001111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10010000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_10010001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10010010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10010011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10010100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_10010101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10010110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10010111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10011000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_10011001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10011010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10011011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10011100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_10011101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10011110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10011111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10100000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]);
        index_state_10100001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10100010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10100011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10100100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_10100101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10100110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10100111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10101000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_10101001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10101010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10101011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10101100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_10101101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10101110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10101111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10110000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_10110001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10110010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10110011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10110100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_10110101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10110110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10110111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10111000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_10111001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10111010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10111011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_10111100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_10111101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_10111110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_10111111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11000000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]);
        index_state_11000001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11000010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11000011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11000100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_11000101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11000110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11000111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11001000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_11001001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11001010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11001011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11001100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_11001101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11001110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11001111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11010000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_11010001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11010010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11010011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11010100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_11010101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11010110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11010111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11011000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_11011001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11011010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11011011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11011100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_11011101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11011110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11011111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11100000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]);
        index_state_11100001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11100010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11100011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11100100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_11100101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11100110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11100111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11101000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_11101001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11101010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11101011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11101100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_11101101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11101110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11101111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11110000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]);
        index_state_11110001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11110010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11110011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11110100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_11110101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11110110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11110111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11111000 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]);
        index_state_11111001 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11111010 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11111011 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        index_state_11111100 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]);
        index_state_11111101 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[0]);
        index_state_11111110 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]);
        index_state_11111111 = index_state_00000000 | (UINT64_C(1) << target_qubit_num_list[7]) | (UINT64_C(1) << target_qubit_num_list[6]) | (UINT64_C(1) << target_qubit_num_list[5]) | (UINT64_C(1) << target_qubit_num_list[4]) | (UINT64_C(1) << target_qubit_num_list[3]) | (UINT64_C(1) << target_qubit_num_list[2]) | (UINT64_C(1) << target_qubit_num_list[1]) | (UINT64_C(1) << target_qubit_num_list[0]);

        is_measured_bits = args->is_measured_bits;
        measured_value_bits = args->measured_value_bits;

    } /* thread_num_to_state_index_q8 */

    namespace gate
    {

        struct u4
        {
            static constexpr unsigned int num_target_qubits = 1;
            qcs::float_t cos_theta_2;
            qcs::float_t sin_theta_2;
            qcs::complex_t exp_i_phi;
            qcs::complex_t exp_i_lambda;
            qcs::complex_t exp_i_gamma;
            u4(double theta, double phi, double lambda, double gamma = 0.0)
            {
                qcs::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
                exp_i_phi = qcs::complex_t(cos(phi), sin(phi));
                exp_i_lambda = qcs::complex_t(cos(lambda), sin(lambda));
                exp_i_gamma = qcs::complex_t(cos(gamma), sin(gamma));
            }
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                qcs::complex_t const s0_in_copy = s0_in;
                qcs::complex_t const s1_in_copy = s1_in;
                s0_out = exp_i_gamma * (cos_theta_2 * s0_in_copy - sin_theta_2 * exp_i_lambda * s1_in_copy);
                s1_out = exp_i_gamma * (exp_i_phi * (sin_theta_2 * s0_in_copy + exp_i_lambda * cos_theta_2 * s1_in_copy));
            }
        };

        struct u3
        {
            static constexpr unsigned int num_target_qubits = 1;
            qcs::float_t cos_theta_2;
            qcs::float_t sin_theta_2;
            qcs::complex_t exp_i_phi;
            qcs::complex_t exp_i_lambda;
            qcs::complex_t exp_i_phi_plus_lambda;
            u3(double theta, double phi, double lambda)
            {
                qcs::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
                exp_i_phi = qcs::complex_t(cos(phi), sin(phi));
                exp_i_lambda = qcs::complex_t(cos(lambda), sin(lambda));
                qcs::float_t const phi_plus_lambda = phi + lambda;
                exp_i_phi_plus_lambda = qcs::complex_t(cos(phi_plus_lambda), sin(phi_plus_lambda));
            }
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                qcs::complex_t const s0_in_copy = s0_in;
                qcs::complex_t const s1_in_copy = s1_in;
                s0_out = cos_theta_2 * s0_in_copy - sin_theta_2 * exp_i_lambda * s1_in_copy;
                s1_out = exp_i_phi * sin_theta_2 * s0_in_copy + exp_i_phi_plus_lambda * cos_theta_2 * s1_in_copy;
            }
        };

        struct u2
        {
            static constexpr unsigned int num_target_qubits = 1;
            qcs::complex_t exp_i_phi;
            qcs::complex_t exp_i_lambda;
            qcs::complex_t exp_i_phi_plus_lambda;
            u2(double phi, double lambda)
            {
                exp_i_phi = qcs::complex_t(cos(phi), sin(phi));
                exp_i_lambda = qcs::complex_t(cos(lambda), sin(lambda));
                qcs::float_t const phi_plus_lambda = phi + lambda;
                exp_i_phi_plus_lambda = qcs::complex_t(cos(phi_plus_lambda), sin(phi_plus_lambda));
            }
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                qcs::complex_t const s0_in_copy = s0_in;
                qcs::complex_t const s1_in_copy = s1_in;
                s0_out = M_SQRT1_2 * (s0_in_copy - exp_i_lambda * s1_in_copy);
                s1_out = M_SQRT1_2 * (exp_i_phi * s0_in_copy + exp_i_phi_plus_lambda * s1_in_copy);
            }
        };

        struct u1
        {
            static constexpr unsigned int num_target_qubits = 1;
            qcs::complex_t exp_i_lambda;
            u1(double lambda)
            {
                exp_i_lambda = qcs::complex_t(cos(lambda), sin(lambda));
            }
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                s0_out = s0_in;
                s1_out = exp_i_lambda * s1_in;
            }
        };

        typedef u3 u;
        typedef u1 p;

        struct global_phase
        {
            static constexpr unsigned int num_target_qubits = 0;
            qcs::complex_t exp_i_theta;
            global_phase(double theta)
            {
                exp_i_theta.real(cos(theta));
                exp_i_theta.imag(sin(theta));
            }
            __device__ void apply(qcs::complex_t const &s_in, qcs::complex_t &s_out) const
            {
                s_out = exp_i_theta * s_in;
            }
        };

        struct hadamard
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                auto const s0_in_copy = s0_in;
                auto const s1_in_copy = s1_in;
                s0_out = M_SQRT1_2 * (s0_in_copy + s1_in_copy);
                s1_out = M_SQRT1_2 * (s0_in_copy - s1_in_copy);
            }
        };

        struct identity
        {
            static constexpr unsigned int num_target_qubits = 0;
            __device__ void apply() const
            {
            }
        };

        struct x
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                auto const s0_in_copy = s0_in;
                auto const s1_in_copy = s1_in;
                s0_out = s1_in_copy;
                s1_out = s0_in_copy;
            }
        };

        struct y
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                auto const s0_in_copy = s0_in;
                auto const s1_in_copy = s1_in;
                s0_out.real(s1_in_copy.imag());
                s0_out.imag(-s1_in_copy.real());
                s1_out.real(-s0_in_copy.imag());
                s1_out.imag(s0_in_copy.real());
            }
        };

        struct z
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                // auto const s0_in_copy = s0_in;
                auto const s1_in_copy = s1_in;
                // s0_out = s0_in_copy;
                s1_out = -s1_in_copy;
            }
        };

        struct s
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                auto const s1_in_copy = s1_in;
                s1_out.real(-s1_in_copy.imag());
                s1_out.imag(s1_in_copy.real());
            }
        };

        struct sdg
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                auto const s1_in_copy = s1_in;
                s1_out.real(s1_in_copy.imag());
                s1_out.imag(-s1_in_copy.real());
            }
        };

        struct t
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                auto const s1_in_copy = s1_in;
                s1_out = qcs::complex_t(M_SQRT1_2, M_SQRT1_2) * s1_in_copy;
            }
        };

        struct tdg
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                auto const s1_in_copy = s1_in;
                s1_out = qcs::complex_t(M_SQRT1_2, -M_SQRT1_2) * s1_in_copy;
            }
        };

        struct sx
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                // auto const s0_in_copy = s0_in;
                // auto const s1_in_copy = s1_in;
                // s0_out = qcs::complex_t(0.5, 0.5) * s0_in_copy + qcs::complex_t(0.5, - 0.5) * s1_in_copy;
                // s1_out = qcs::complex_t(0.5, - 0.5) * s0_in_copy + qcs::complex_t(0.5, 0.5) * s1_in_copy;

                auto const a = 0.5 * (s0_in + s1_in);
                auto const b = 0.5 * multiply_i(s0_in - s1_in);
                s0_out = a + b;
                s1_out = a - b;
            }
        };

        struct sxdg
        {
            static constexpr unsigned int num_target_qubits = 1;

            __device__ void apply(qcs::complex_t const &s0_in,
                                  qcs::complex_t const &s1_in,
                                  qcs::complex_t &s0_out,
                                  qcs::complex_t &s1_out) const
            {
                auto const a = 0.5 * (s0_in + s1_in);
                auto const b = -0.5 * multiply_i(s0_in - s1_in);
                s0_out = a + b;
                s1_out = a - b;
            }
        };

        struct rx
        {
            static constexpr unsigned int num_target_qubits = 1;
            qcs::float_t cos_theta_2;
            qcs::float_t sin_theta_2;
            rx(double theta)
            {
                cos_theta_2 = cos(0.5 * theta);
                sin_theta_2 = sin(0.5 * theta);
            }
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                auto const s0_in_copy = s0_in;
                auto const s1_in_copy = s1_in;
                s0_out = cos_theta_2 * s0_in_copy + multiply_i_real(s1_in_copy, -sin_theta_2);
                s1_out = multiply_i_real(s0_in_copy, -sin_theta_2) + cos_theta_2 * s1_in_copy;
            }
        };

        struct ry
        {
            static constexpr unsigned int num_target_qubits = 1;
            qcs::float_t cos_theta_2;
            qcs::float_t sin_theta_2;
            ry(double theta)
            {
                cos_theta_2 = cos(0.5 * theta);
                sin_theta_2 = sin(0.5 * theta);
            }
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                auto const s0_in_copy = s0_in;
                auto const s1_in_copy = s1_in;
                s0_out = cos_theta_2 * s0_in_copy - sin_theta_2 * s1_in_copy;
                s1_out = sin_theta_2 * s0_in_copy + cos_theta_2 * s1_in_copy;
            }
        };

        struct rz
        {
            static constexpr unsigned int num_target_qubits = 1;
            qcs::float_t cos_theta_2;
            qcs::float_t sin_theta_2;
            rz(double theta)
            {
                cos_theta_2 = cos(0.5 * theta);
                sin_theta_2 = sin(0.5 * theta);
            }
            __device__ void apply(qcs::complex_t const &s0_in, qcs::complex_t const &s1_in, qcs::complex_t &s0_out, qcs::complex_t &s1_out) const
            {
                s0_out = qcs::complex_t(cos_theta_2, -sin_theta_2) * s0_in;
                s1_out = qcs::complex_t(cos_theta_2, sin_theta_2) * s1_in;
            }
        };

        struct swap
        {
            static constexpr unsigned int num_target_qubits = 2;
            __device__ void apply(qcs::complex_t const &s00_in, qcs::complex_t const &s01_in, qcs::complex_t const &s10_in, qcs::complex_t const &s11_in, qcs::complex_t &s00_out, qcs::complex_t &s01_out, qcs::complex_t &s10_out, qcs::complex_t &s11_out) const
            {
                auto const s01_in_copy = s01_in;
                auto const s10_in_copy = s10_in;
                // s00_out = s00_in;
                s01_out = s10_in_copy;
                s10_out = s01_in_copy;
                // s11_out = s11_in;
            }
        };

        struct swap2
        {
            static constexpr unsigned int num_target_qubits = 4;

            __device__ void apply(qcs::complex_t const &s0000_in,
                                  qcs::complex_t const &s0001_in,
                                  qcs::complex_t const &s0010_in,
                                  qcs::complex_t const &s0011_in,
                                  qcs::complex_t const &s0100_in,
                                  qcs::complex_t const &s0101_in,
                                  qcs::complex_t const &s0110_in,
                                  qcs::complex_t const &s0111_in,
                                  qcs::complex_t const &s1000_in,
                                  qcs::complex_t const &s1001_in,
                                  qcs::complex_t const &s1010_in,
                                  qcs::complex_t const &s1011_in,
                                  qcs::complex_t const &s1100_in,
                                  qcs::complex_t const &s1101_in,
                                  qcs::complex_t const &s1110_in,
                                  qcs::complex_t const &s1111_in,
                                  qcs::complex_t &s0000_out,
                                  qcs::complex_t &s0001_out,
                                  qcs::complex_t &s0010_out,
                                  qcs::complex_t &s0011_out,
                                  qcs::complex_t &s0100_out,
                                  qcs::complex_t &s0101_out,
                                  qcs::complex_t &s0110_out,
                                  qcs::complex_t &s0111_out,
                                  qcs::complex_t &s1000_out,
                                  qcs::complex_t &s1001_out,
                                  qcs::complex_t &s1010_out,
                                  qcs::complex_t &s1011_out,
                                  qcs::complex_t &s1100_out,
                                  qcs::complex_t &s1101_out,
                                  qcs::complex_t &s1110_out,
                                  qcs::complex_t &s1111_out) const
            {
                // Swap target bits 0/1 and 2/3; inputs and outputs may alias.
                auto const s0001_in_copy = s0001_in;
                auto const s0010_in_copy = s0010_in;
                auto const s0100_in_copy = s0100_in;
                auto const s0101_in_copy = s0101_in;
                auto const s0110_in_copy = s0110_in;
                auto const s0111_in_copy = s0111_in;
                auto const s1000_in_copy = s1000_in;
                auto const s1001_in_copy = s1001_in;
                auto const s1010_in_copy = s1010_in;
                auto const s1011_in_copy = s1011_in;
                auto const s1101_in_copy = s1101_in;
                auto const s1110_in_copy = s1110_in;
                // s0000_out = s0000_in;
                s0001_out = s0010_in_copy;
                s0010_out = s0001_in_copy;
                // s0011_out = s0011_in;
                s0100_out = s1000_in_copy;
                s0101_out = s1010_in_copy;
                s0110_out = s1001_in_copy;
                s0111_out = s1011_in_copy;
                s1000_out = s0100_in_copy;
                s1001_out = s0110_in_copy;
                s1010_out = s0101_in_copy;
                s1011_out = s0111_in_copy;
                // s1100_out = s1100_in;
                s1101_out = s1110_in_copy;
                s1110_out = s1101_in_copy;
                // s1111_out = s1111_in;
            }
        };

        struct iswap
        {
            static constexpr unsigned int num_target_qubits = 2;
            __device__ void apply(qcs::complex_t const &s00_in, qcs::complex_t const &s01_in, qcs::complex_t const &s10_in, qcs::complex_t const &s11_in, qcs::complex_t &s00_out, qcs::complex_t &s01_out, qcs::complex_t &s10_out, qcs::complex_t &s11_out) const
            {
                auto const s01_in_copy = s01_in;
                auto const s10_in_copy = s10_in;
                // s00_out = s00_in;
                s01_out.real(-s10_in_copy.imag());
                s01_out.imag(s10_in_copy.real());
                s10_out.real(-s01_in_copy.imag());
                s10_out.imag(s01_in_copy.real());
                // s11_out = s11_in;
            }
        };

        struct id
        {
            static constexpr unsigned int num_target_qubits = 1;

            __device__ void apply(qcs::complex_t const &s0_in,
                                  qcs::complex_t const &s1_in,
                                  qcs::complex_t &s0_out,
                                  qcs::complex_t &s1_out) const
            {
                // s0_out = s0_in;
                // s1_out = s1_in;
            }
        };

        struct r
        {
            static constexpr unsigned int num_target_qubits = 1;

            qcs::float_t cos_theta_2;
            qcs::float_t sin_theta_2;
            qcs::complex_t exp_i_phi;
            qcs::complex_t exp_minus_i_phi;

            r(double theta, double phi)
            {
                qcs::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
                qcs::float_t cos_phi = cos(phi);
                qcs::float_t sin_phi = sin(phi);
                exp_i_phi = qcs::complex_t(cos_phi, sin_phi);
                exp_minus_i_phi = qcs::complex_t(cos_phi, -sin_phi);
            }

            __device__ void apply(qcs::complex_t const &s0_in,
                                  qcs::complex_t const &s1_in,
                                  qcs::complex_t &s0_out,
                                  qcs::complex_t &s1_out) const
            {
                auto const s0_in_copy = s0_in;
                auto const s1_in_copy = s1_in;
                s0_out = cos_theta_2 * s0_in_copy + multiply_i_real(exp_minus_i_phi * s1_in_copy, -sin_theta_2);
                s1_out = multiply_i_real(exp_i_phi * s0_in_copy, -sin_theta_2) + cos_theta_2 * s1_in_copy;
            }
        };

        struct rzz
        {
            static constexpr unsigned int num_target_qubits = 2;

            qcs::complex_t phase_minus;
            qcs::complex_t phase_plus;

            rzz(double theta)
            {
                qcs::float_t const theta_2 = 0.5 * theta;
                phase_minus = qcs::complex_t(cos(theta_2), -sin(theta_2));
                phase_plus = qcs::complex_t(cos(theta_2), sin(theta_2));
            }

            __device__ void apply(qcs::complex_t const &s00_in,
                                  qcs::complex_t const &s01_in,
                                  qcs::complex_t const &s10_in,
                                  qcs::complex_t const &s11_in,
                                  qcs::complex_t &s00_out,
                                  qcs::complex_t &s01_out,
                                  qcs::complex_t &s10_out,
                                  qcs::complex_t &s11_out) const
            {
                s00_out = phase_minus * s00_in;
                s01_out = phase_plus * s01_in;
                s10_out = phase_plus * s10_in;
                s11_out = phase_minus * s11_in;
            }
        };

        struct rxx
        {
            static constexpr unsigned int num_target_qubits = 2;

            qcs::float_t cos_theta_2;
            qcs::float_t m_sin_theta_2;

            rxx(double theta)
            {
                qcs::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                m_sin_theta_2 = -sin(theta_2);
            }

            __device__ void apply(qcs::complex_t const &s00_in,
                                  qcs::complex_t const &s01_in,
                                  qcs::complex_t const &s10_in,
                                  qcs::complex_t const &s11_in,
                                  qcs::complex_t &s00_out,
                                  qcs::complex_t &s01_out,
                                  qcs::complex_t &s10_out,
                                  qcs::complex_t &s11_out) const
            {
                auto const s00_in_copy = s00_in;
                auto const s01_in_copy = s01_in;
                auto const s10_in_copy = s10_in;
                auto const s11_in_copy = s11_in;

                s00_out = cos_theta_2 * s00_in_copy + multiply_i_real(s11_in_copy, m_sin_theta_2);
                s01_out = cos_theta_2 * s01_in_copy + multiply_i_real(s10_in_copy, m_sin_theta_2);
                s10_out = cos_theta_2 * s10_in_copy + multiply_i_real(s01_in_copy, m_sin_theta_2);
                s11_out = cos_theta_2 * s11_in_copy + multiply_i_real(s00_in_copy, m_sin_theta_2);
            }
        };

        struct ryy
        {
            static constexpr unsigned int num_target_qubits = 2;

            qcs::float_t cos_theta_2;
            qcs::float_t sin_theta_2;

            ryy(double theta)
            {
                qcs::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
            }

            __device__ void apply(qcs::complex_t const &s00_in,
                                  qcs::complex_t const &s01_in,
                                  qcs::complex_t const &s10_in,
                                  qcs::complex_t const &s11_in,
                                  qcs::complex_t &s00_out,
                                  qcs::complex_t &s01_out,
                                  qcs::complex_t &s10_out,
                                  qcs::complex_t &s11_out) const
            {
                auto const s00_in_copy = s00_in;
                auto const s01_in_copy = s01_in;
                auto const s10_in_copy = s10_in;
                auto const s11_in_copy = s11_in;

                s00_out = cos_theta_2 * s00_in_copy + multiply_i_real(s11_in_copy, sin_theta_2);
                s01_out = cos_theta_2 * s01_in_copy + multiply_i_real(s10_in_copy, -sin_theta_2);
                s10_out = cos_theta_2 * s10_in_copy + multiply_i_real(s01_in_copy, -sin_theta_2);
                s11_out = cos_theta_2 * s11_in_copy + multiply_i_real(s00_in_copy, sin_theta_2);
            }
        };

        struct rzx
        {
            static constexpr unsigned int num_target_qubits = 2;

            qcs::float_t cos_theta_2;
            qcs::float_t sin_theta_2;

            rzx(double theta)
            {
                qcs::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
            }

            __device__ void apply(qcs::complex_t const &s00_in,
                                  qcs::complex_t const &s01_in,
                                  qcs::complex_t const &s10_in,
                                  qcs::complex_t const &s11_in,
                                  qcs::complex_t &s00_out,
                                  qcs::complex_t &s01_out,
                                  qcs::complex_t &s10_out,
                                  qcs::complex_t &s11_out) const
            {
                auto const s00_in_copy = s00_in;
                auto const s01_in_copy = s01_in;
                auto const s10_in_copy = s10_in;
                auto const s11_in_copy = s11_in;

                s00_out = cos_theta_2 * s00_in_copy + multiply_i_real(s10_in_copy, -sin_theta_2);
                s01_out = cos_theta_2 * s01_in_copy + multiply_i(s11_in_copy) * sin_theta_2;
                s10_out = cos_theta_2 * s10_in_copy + multiply_i_real(s00_in_copy, -sin_theta_2);
                s11_out = cos_theta_2 * s11_in_copy + multiply_i(s01_in_copy) * sin_theta_2;
            }
        };

        struct xx_plus_yy
        {
            static constexpr unsigned int num_target_qubits = 2;

            qcs::float_t cos_theta_2;
            qcs::float_t sin_theta_2;
            qcs::complex_t exp_i_beta;
            qcs::complex_t exp_minus_i_beta;

            xx_plus_yy(double theta, double beta = 0.0)
            {
                qcs::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
                exp_i_beta = qcs::complex_t(cos(beta), sin(beta));
                exp_minus_i_beta = qcs::complex_t(cos(beta), -sin(beta));
            }

            __device__ void apply(qcs::complex_t const &s00_in,
                                  qcs::complex_t const &s01_in,
                                  qcs::complex_t const &s10_in,
                                  qcs::complex_t const &s11_in,
                                  qcs::complex_t &s00_out,
                                  qcs::complex_t &s01_out,
                                  qcs::complex_t &s10_out,
                                  qcs::complex_t &s11_out) const
            {
                auto const s01_in_copy = s01_in;
                auto const s10_in_copy = s10_in;

                // s00_out = s00_in;
                s01_out = cos_theta_2 * s01_in_copy + multiply_i_real(exp_minus_i_beta * s10_in_copy, -sin_theta_2);
                s10_out = multiply_i_real(exp_i_beta * s01_in_copy, -sin_theta_2) + cos_theta_2 * s10_in_copy;
                // s11_out = s11_in;
            }
        };

        struct xx_minus_yy
        {
            static constexpr unsigned int num_target_qubits = 2;

            qcs::float_t cos_theta_2;
            qcs::float_t sin_theta_2;
            qcs::complex_t exp_i_beta;
            qcs::complex_t exp_minus_i_beta;

            xx_minus_yy(double theta, double beta = 0.0)
            {
                qcs::float_t const theta_2 = 0.5 * theta;
                cos_theta_2 = cos(theta_2);
                sin_theta_2 = sin(theta_2);
                exp_i_beta = qcs::complex_t(cos(beta), sin(beta));
                exp_minus_i_beta = qcs::complex_t(cos(beta), -sin(beta));
            }

            __device__ void apply(qcs::complex_t const &s00_in,
                                  qcs::complex_t const &s01_in,
                                  qcs::complex_t const &s10_in,
                                  qcs::complex_t const &s11_in,
                                  qcs::complex_t &s00_out,
                                  qcs::complex_t &s01_out,
                                  qcs::complex_t &s10_out,
                                  qcs::complex_t &s11_out) const
            {
                auto const s00_in_copy = s00_in;
                auto const s11_in_copy = s11_in;

                s00_out = cos_theta_2 * s00_in_copy + multiply_i_real(exp_minus_i_beta * s11_in_copy, -sin_theta_2);
                // s01_out = s01_in;
                // s10_out = s10_in;
                s11_out = multiply_i_real(exp_i_beta * s00_in_copy, -sin_theta_2) + cos_theta_2 * s11_in_copy;
            }
        };

        struct dcx
        {
            static constexpr unsigned int num_target_qubits = 2;

            __device__ void apply(qcs::complex_t const &s00_in,
                                  qcs::complex_t const &s01_in,
                                  qcs::complex_t const &s10_in,
                                  qcs::complex_t const &s11_in,
                                  qcs::complex_t &s00_out,
                                  qcs::complex_t &s01_out,
                                  qcs::complex_t &s10_out,
                                  qcs::complex_t &s11_out) const
            {
                auto const s10_in_copy = s10_in;
                auto const s11_in_copy = s11_in;
                auto const s01_in_copy = s01_in;
                // s00_out = s00_in;
                s01_out = s11_in_copy;
                s10_out = s01_in_copy;
                s11_out = s10_in_copy;
            }
        };

        struct ecr
        {
            static constexpr unsigned int num_target_qubits = 2;

            __device__ void apply(qcs::complex_t const &s00_in,
                                  qcs::complex_t const &s01_in,
                                  qcs::complex_t const &s10_in,
                                  qcs::complex_t const &s11_in,
                                  qcs::complex_t &s00_out,
                                  qcs::complex_t &s01_out,
                                  qcs::complex_t &s10_out,
                                  qcs::complex_t &s11_out) const
            {
                auto const s00_in_copy = s00_in;
                auto const s01_in_copy = s01_in;
                auto const s10_in_copy = s10_in;
                auto const s11_in_copy = s11_in;
                s00_out = M_SQRT1_2 * (s01_in_copy + multiply_i(s11_in_copy));
                s01_out = M_SQRT1_2 * (s00_in_copy + multiply_i_m(s10_in_copy));
                s10_out = M_SQRT1_2 * (multiply_i(s01_in_copy) + s11_in_copy);
                s11_out = M_SQRT1_2 * (multiply_i_m(s00_in_copy) + s10_in_copy);
            }
        };

        struct rccx
        {
            static constexpr unsigned int num_target_qubits = 3;

            __device__ void apply(qcs::complex_t const &s000_in,
                                  qcs::complex_t const &s001_in,
                                  qcs::complex_t const &s010_in,
                                  qcs::complex_t const &s011_in,
                                  qcs::complex_t const &s100_in,
                                  qcs::complex_t const &s101_in,
                                  qcs::complex_t const &s110_in,
                                  qcs::complex_t const &s111_in,
                                  qcs::complex_t &s000_out,
                                  qcs::complex_t &s001_out,
                                  qcs::complex_t &s010_out,
                                  qcs::complex_t &s011_out,
                                  qcs::complex_t &s100_out,
                                  qcs::complex_t &s101_out,
                                  qcs::complex_t &s110_out,
                                  qcs::complex_t &s111_out) const
            {
                auto const s111_in_copy = s111_in;
                auto const s101_in_copy = s101_in;
                auto const s011_in_copy = s011_in;
                // s000_out = s000_in;
                // s001_out = s001_in;
                // s010_out = s010_in;
                s011_out = multiply_i_m(s111_in_copy);
                // s100_out = s100_in;
                s101_out = -s101_in_copy;
                // s110_out = s110_in;
                s111_out = multiply_i(s011_in_copy);
            }
        };

        struct rcccx
        {
            static constexpr unsigned int num_target_qubits = 4;

            __device__ void apply(qcs::complex_t const &s0000_in,
                                  qcs::complex_t const &s0001_in,
                                  qcs::complex_t const &s0010_in,
                                  qcs::complex_t const &s0011_in,
                                  qcs::complex_t const &s0100_in,
                                  qcs::complex_t const &s0101_in,
                                  qcs::complex_t const &s0110_in,
                                  qcs::complex_t const &s0111_in,
                                  qcs::complex_t const &s1000_in,
                                  qcs::complex_t const &s1001_in,
                                  qcs::complex_t const &s1010_in,
                                  qcs::complex_t const &s1011_in,
                                  qcs::complex_t const &s1100_in,
                                  qcs::complex_t const &s1101_in,
                                  qcs::complex_t const &s1110_in,
                                  qcs::complex_t const &s1111_in,
                                  qcs::complex_t &s0000_out,
                                  qcs::complex_t &s0001_out,
                                  qcs::complex_t &s0010_out,
                                  qcs::complex_t &s0011_out,
                                  qcs::complex_t &s0100_out,
                                  qcs::complex_t &s0101_out,
                                  qcs::complex_t &s0110_out,
                                  qcs::complex_t &s0111_out,
                                  qcs::complex_t &s1000_out,
                                  qcs::complex_t &s1001_out,
                                  qcs::complex_t &s1010_out,
                                  qcs::complex_t &s1011_out,
                                  qcs::complex_t &s1100_out,
                                  qcs::complex_t &s1101_out,
                                  qcs::complex_t &s1110_out,
                                  qcs::complex_t &s1111_out) const
            {
                auto const s0011_in_copy = s0011_in;
                auto const s1111_in_copy = s1111_in;
                auto const s1011_in_copy = s1011_in;
                auto const s0111_in_copy = s0111_in;
                // s0000_out = s0000_in;
                // s0001_out = s0001_in;
                // s0010_out = s0010_in;
                s0011_out = multiply_i(s0011_in_copy);
                // s0100_out = s0100_in;
                // s0101_out = s0101_in;
                // s0110_out = s0110_in;
                s0111_out = s1111_in_copy;
                // s1000_out = s1000_in;
                // s1001_out = s1001_in;
                // s1010_out = s1010_in;
                s1011_out = multiply_i_m(s1011_in_copy);
                // s1100_out = s1100_in;
                // s1101_out = s1101_in;
                // s1110_out = s1110_in;
                s1111_out = -s0111_in_copy;
            }
        };

    }

    __global__ void materialize_projection_kernel(
        qcs::complex_t *state,
        uint64_t num_states_local,
        uint64_t local_mask,
        uint64_t local_value)
    {
        uint64_t const first = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        uint64_t const stride = (uint64_t)gridDim.x * blockDim.x;
        for (uint64_t i = first; i < num_states_local; i += stride)
        {
            if (((i ^ local_value) & local_mask) != 0)
            {
                state[i] = qcs::complex_t(0.0, 0.0);
            }
        }
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 0>::type
        cuda_gate(GateType const gateobj)
    {
        int64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;

        uint64_t index_state;
        thread_num_to_state_index_q0(thread_num, index_state);

        qcs::complex_t const *ps_in = &qcs::kernel_common_constant.state_data_device[index_state];
        gateobj.apply(*ps_in, qcs::kernel_common_constant.state_data_device[index_state]);
    }

    template <typename GateType, std::size_t... I>
    __device__ void apply_gate_q1(GateType const &gateobj, uint64_t thread_num, std::index_sequence<I...>)
    {
        uint64_t index_state[2];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index_q1(
            thread_num, index_state[I]..., is_measured_bits, measured_value_bits);

        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;

        // Basis indices run from 0 to 1, with target 0
        // as the least significant bit. Pass all inputs before all outputs.
        gateobj.apply(
            ((((I ^ measured_value_bits) & is_measured_bits) == 0)
                 ? state_in[index_state[I]]
                 : zero_constant)...,
            state[index_state[I]]...);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 1>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        apply_gate_q1(gateobj, thread_num, std::make_index_sequence<2>{});
    }

    template <typename GateType, std::size_t... I>
    __device__ void apply_gate_q2(GateType const &gateobj, uint64_t thread_num, std::index_sequence<I...>)
    {
        uint64_t index_state[4];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index_q2(
            thread_num, index_state[I]..., is_measured_bits, measured_value_bits);

        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;

        // Basis indices run from 00 to 11, with target 0
        // as the least significant bit. Pass all inputs before all outputs.
        gateobj.apply(
            ((((I ^ measured_value_bits) & is_measured_bits) == 0)
                 ? state_in[index_state[I]]
                 : zero_constant)...,
            state[index_state[I]]...);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 2>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        apply_gate_q2(gateobj, thread_num, std::make_index_sequence<4>{});
    }

    template <typename GateType, std::size_t... I>
    __device__ void apply_gate_q3(GateType const &gateobj, uint64_t thread_num, std::index_sequence<I...>)
    {
        uint64_t index_state[8];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index_q3(
            thread_num, index_state[I]..., is_measured_bits, measured_value_bits);

        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;

        // Basis indices run from 000 to 111, with target 0
        // as the least significant bit. Pass all inputs before all outputs.
        gateobj.apply(
            ((((I ^ measured_value_bits) & is_measured_bits) == 0)
                 ? state_in[index_state[I]]
                 : zero_constant)...,
            state[index_state[I]]...);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 3>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        apply_gate_q3(gateobj, thread_num, std::make_index_sequence<8>{});
    }

    template <typename GateType, std::size_t... I>
    __device__ void apply_gate_q4(GateType const &gateobj, uint64_t thread_num, std::index_sequence<I...>)
    {
        uint64_t index_state[16];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index_q4(
            thread_num, index_state[I]..., is_measured_bits, measured_value_bits);

        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;

        // Basis indices run from 0000 to 1111, with target 0
        // as the least significant bit. Pass all inputs before all outputs.
        gateobj.apply(
            ((((I ^ measured_value_bits) & is_measured_bits) == 0)
                 ? state_in[index_state[I]]
                 : zero_constant)...,
            state[index_state[I]]...);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 4>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        apply_gate_q4(gateobj, thread_num, std::make_index_sequence<16>{});
    }

    template <typename GateType, std::size_t... I>
    __device__ void apply_gate_q5(GateType const &gateobj, uint64_t thread_num, std::index_sequence<I...>)
    {
        uint64_t index_state[32];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index_q5(
            thread_num, index_state[I]..., is_measured_bits, measured_value_bits);

        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;

        // Basis indices run from 00000 to 11111, with target 0
        // as the least significant bit. Pass all inputs before all outputs.
        gateobj.apply(
            ((((I ^ measured_value_bits) & is_measured_bits) == 0)
                 ? state_in[index_state[I]]
                 : zero_constant)...,
            state[index_state[I]]...);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 5>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        apply_gate_q5(gateobj, thread_num, std::make_index_sequence<32>{});
    }

    template <typename GateType, std::size_t... I>
    __device__ void apply_gate_q6(GateType const &gateobj, uint64_t thread_num, std::index_sequence<I...>)
    {
        uint64_t index_state[64];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index_q6(
            thread_num, index_state[I]..., is_measured_bits, measured_value_bits);

        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;

        // Basis indices run from 000000 to 111111, with target 0
        // as the least significant bit. Pass all inputs before all outputs.
        gateobj.apply(
            ((((I ^ measured_value_bits) & is_measured_bits) == 0)
                 ? state_in[index_state[I]]
                 : zero_constant)...,
            state[index_state[I]]...);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 6>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        apply_gate_q6(gateobj, thread_num, std::make_index_sequence<64>{});
    }

    template <typename GateType, std::size_t... I>
    __device__ void apply_gate_q7(GateType const &gateobj, uint64_t thread_num, std::index_sequence<I...>)
    {
        uint64_t index_state[128];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index_q7(
            thread_num, index_state[I]..., is_measured_bits, measured_value_bits);

        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;

        // Basis indices run from 0000000 to 1111111, with target 0
        // as the least significant bit. Pass all inputs before all outputs.
        gateobj.apply(
            ((((I ^ measured_value_bits) & is_measured_bits) == 0)
                 ? state_in[index_state[I]]
                 : zero_constant)...,
            state[index_state[I]]...);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 7>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        apply_gate_q7(gateobj, thread_num, std::make_index_sequence<128>{});
    }

    template <typename GateType, std::size_t... I>
    __device__ void apply_gate_q8(GateType const &gateobj, uint64_t thread_num, std::index_sequence<I...>)
    {
        uint64_t index_state[256];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index_q8(
            thread_num, index_state[I]..., is_measured_bits, measured_value_bits);

        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;

        // Basis indices run from 00000000 to 11111111, with target 0
        // as the least significant bit. Pass all inputs before all outputs.
        gateobj.apply(
            ((((I ^ measured_value_bits) & is_measured_bits) == 0)
                 ? state_in[index_state[I]]
                 : zero_constant)...,
            state[index_state[I]]...);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 8>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        apply_gate_q8(gateobj, thread_num, std::make_index_sequence<256>{});
    }

    namespace cubUtility
    {

        struct float2Add
        {
            __device__ qcs::float2_t operator()(const qcs::float2_t &a, const qcs::float2_t &b) const
            {
                return {a[0] + b[0], a[1] + b[1]};
            }
        };

        struct IndirectLoad
        {
            bool is_global;
            int rank_value;

            __device__ qcs::float2_t operator()(uint64_t thread_num) const
            {
                if (is_global)
                {
                    uint64_t index_state;
                    thread_num_to_state_index_q0(thread_num, index_state);

                    qcs::float_t const norm = cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state]);
                    return rank_value
                               ? qcs::float2_t{0, norm}
                               : qcs::float2_t{norm, 0};
                }

                uint64_t index_state_0, index_state_1;
                int is_measured_bits, measured_value_bits;
                thread_num_to_state_index_q1(thread_num, index_state_0, index_state_1, is_measured_bits, measured_value_bits);

                // since target_qubit must be unmeasured, branching is not necessary.
                return qcs::float2_t{
                    cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_0]),
                    cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_1])};

                // return qcs::float2_t{
                //     (measured_state != 1)? cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_0]): 0,
                //     (measured_state != 0)? cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state_1]): 0
                // };
            }
        };

    } /* cubUtility */

    enum class initstate_enum
    {
        sequential,
        flat,
        zero,
        entangled,
        use_curand,
        laod_statevector,
    };

    struct simulator_core
    {

        /* begin simulator variables */

        int num_qubits;

        bool use_unified_memory;

        initstate_enum initstate_choice;

        float elapsed_ms, elapsed_ms_2;

        int num_procs, proc_num;

        int num_rand_areas;

        std::string my_hostname;
        int my_node_number;
        int my_node_local_rank;
        int node_count;

        int gpu_id;

        ncclUniqueId nccl_id;
        ncclComm_t nccl_comm;
        int nccl_rank;

        std::vector<int> perm_p2l;
        std::vector<int> perm_l2p;
        std::vector<int> initial_perm_p2l;

        int num_samples;
        unsigned int rng_seed;
        std::mt19937_64 engine;

        int log_num_procs;
        int max_num_qubits_local_device;
        uint64_t initial_free_memory_bytes;
        int log_block_size_max;
        int block_size_max;
        int target_qubit_num_begin;
        int target_qubit_num_end;

        cudaStream_t stream;
        std::vector<cudaEvent_t> event_list;

        uint64_t num_states;
        int num_qubits_local;
        uint64_t num_states_local;

        uint64_t num_blocks_gateop;
        uint64_t block_size_gateop;
        uint64_t num_operand_qubits;

        qcs::complex_t *state_data_device;

        qcs::complex_t *zero_constant_addr;
        qcs::kernel_common_struct *qcs_kernel_common_constant_addr;
        qcs::kernel_common_struct qcs_kernel_common_host;
        qcs::kernel_input_qnlist_struct *qcs_kernel_input_constant_addr;
        std::vector<char> qcs_kernel_input_host_buffer;
        int log_swap_buffer_total_length;
        uint64_t swap_buffer_total_length;
        qcs::complex_t *swap_buffer;
        qcs::float2_t *measure_norm_device;

        void *cub_temp_buffer_device;
        uint64_t cub_temp_buffer_device_size;

        std::vector<int> operand_qubit_num_list;
        std::vector<int> target_qubit_num_physical_list;
        std::vector<int> swap_target_global_list;
        std::vector<int> swap_target_local_list;
        std::vector<int> swap_target_local_logical_list;
        std::vector<int> swap_target_global_logical_list;
        std::vector<int> positive_control_qubit_num_physical_list;
        std::vector<int> positive_control_qubit_num_physical_global_list;
        std::vector<int> positive_control_qubit_num_physical_local_list;
        std::vector<int> negative_control_qubit_num_physical_list;
        std::vector<int> negative_control_qubit_num_physical_global_list;
        std::vector<int> negative_control_qubit_num_physical_local_list;

        std::vector<int> measured_1_qubit_num_logical_list;
        std::vector<int> measured_0_qubit_num_logical_list;
        std::vector<int> measured_1_qubit_num_logical_list_copy;
        std::vector<int> measured_0_qubit_num_logical_list_copy;

        uint64_t measured_mask_logical = 0;
        uint64_t measured_value_logical = 0;
        uint64_t pending_projection_mask_logical = 0;

        std::vector<int> target_qubit_num_logical_list;
        std::vector<int> positive_control_qubit_num_logical_list;
        std::vector<int> negative_control_qubit_num_logical_list;

        bool measured_control_condition;
        bool proc_num_control_condition;

        /* end simulator variables */

        simulator_core() : num_qubits(0),
                           use_unified_memory(false),
                           num_rand_areas(1),
                           initial_free_memory_bytes(0)
        {
        }

        void setup()
        {

            MPI_Init(NULL, NULL);
            MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
            MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);

            atlc::group_by_hostname(proc_num, num_procs, my_hostname, my_node_number, my_node_local_rank, node_count);
            // fprintf(stderr, "[debug] Rank %d on host %s -> assigned node number: %d, local node rank: %d (total nodes: %d)\n", proc_num, my_hostname.c_str(), my_node_number, my_node_local_rank, node_count);

            gpu_id = my_node_local_rank;
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);

            size_t total_memory_bytes = 0;
            size_t free_memory_bytes = 0;
            ATLC_CHECK_CUDA(cudaMemGetInfo, &free_memory_bytes, &total_memory_bytes);

            uint64_t const used_memory_bytes = (uint64_t)total_memory_bytes - (uint64_t)free_memory_bytes;
            if (used_memory_bytes > (UINT64_C(1) << 27))
            {
                fprintf(
                    stderr,
                    "[warn] GPU memory already in use before allocations: %.2f GiB (used=%" PRIu64 " bytes, total=%zu bytes).\n",
                    (double)used_memory_bytes / (double)(UINT64_C(1) << 30),
                    used_memory_bytes,
                    total_memory_bytes);
            }

            if (proc_num == 0)
            {
                ATLC_CHECK_NCCL(ncclGetUniqueId, &nccl_id);
            }

            MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
            nccl_rank = proc_num;
            ATLC_CHECK_NCCL(ncclCommInitRank, &nccl_comm, num_procs, nccl_id, nccl_rank);

            log_num_procs = atlc::log2_int(num_procs);

            log_block_size_max = 9;

            ATLC_CHECK_CUDA(cudaStreamCreate, &stream);
            // ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event_1, cudaEventDefault);
            // ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event_2, cudaEventDefault);

            block_size_max = 1 << log_block_size_max;

            ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void **)&qcs_kernel_common_constant_addr, qcs::kernel_common_constant);

            ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void **)&qcs_kernel_input_constant_addr, qcs::kernel_input_constant);

            ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void **)&zero_constant_addr, qcs::zero_constant);

            qcs::complex_t const complex_zero = 0;
            ATLC_CHECK_CUDA(cudaMemcpyAsync, zero_constant_addr, &complex_zero, sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);

            if (proc_num == 0)
            {
                std::random_device rng;
                this->rng_seed = rng(); // & ((UINT64_C(1)<<12)-1);
            }
            ATLC_CHECK_MPI(MPI_Bcast, &this->rng_seed, 1, MPI_UNSIGNED, 0, MPI_COMM_WORLD);

            engine = std::mt19937_64(rng_seed);

            cub_temp_buffer_device = NULL;
            state_data_device = NULL;
            swap_buffer = NULL;

        } /* setup */

        void reinitialize_mapping()
        {

            if (initial_perm_p2l.empty())
            {
                for (int qubit_num = 0; qubit_num < num_qubits; qubit_num++)
                {
                    perm_p2l[qubit_num] = qubit_num;
                    perm_l2p[qubit_num] = qubit_num;
                }
            }
            else
            {
                if ((int)initial_perm_p2l.size() != num_qubits)
                {
                    throw std::runtime_error(atlc::format("mapping size %d does not match num_qubits %d", (int)initial_perm_p2l.size(), num_qubits));
                }

                perm_p2l = initial_perm_p2l;
                for (int physical_qubit_num = 0; physical_qubit_num < num_qubits; physical_qubit_num++)
                {
                    int const logical_qubit_num = perm_p2l[physical_qubit_num];
                    perm_l2p[logical_qubit_num] = physical_qubit_num;
                }
            }
        }

        void allocate_memory(bit_num_t num_qubits)
        {

            size_t total_memory_bytes = 0;
            size_t free_memory_bytes = 0;
            ATLC_CHECK_CUDA(cudaMemGetInfo, &free_memory_bytes, &total_memory_bytes);
            initial_free_memory_bytes = free_memory_bytes;

            max_num_qubits_local_device = atlc::log2_int((uint64_t)initial_free_memory_bytes >> 4);
            if (proc_num == 0 && max_num_qubits_local_device > qcs::max_num_qubits_local)
            {
                fprintf(
                    stderr,
                    "[warn] recommendation: increase qcs::max_num_qubits_local (current=%d, device-based=%d).\n",
                    qcs::max_num_qubits_local,
                    max_num_qubits_local_device);
            }

            if (num_qubits > max_num_qubits_local_device + log_num_procs)
            {
                throw std::runtime_error(atlc::format("num_qubits(%d) > max_num_qubits_local_device(%d) + log_num_procs(%d)", num_qubits, max_num_qubits_local_device, log_num_procs));
            }

            std::vector<std::string> exmes_list;
            if (this->num_qubits > 0)
            {
                exmes_list.push_back(atlc::format("num_qubits is already set %d > 0", this->num_qubits));
            }
            if (cub_temp_buffer_device)
            {
                exmes_list.push_back(atlc::format("cub_temp_buffer_device is %p not NULL", cub_temp_buffer_device));
            }
            if (state_data_device)
            {
                exmes_list.push_back(atlc::format("state_data_device is %p not NULL", state_data_device));
            }
            if (swap_buffer)
            {
                exmes_list.push_back(atlc::format("swap_buffer is %p not NULL", swap_buffer));
            }
            if (exmes_list.size() > 0)
            {
                std::string exmes_all = std::move(exmes_list[0]);
                for (int i = 1; i < exmes_list.size(); i++)
                {
                    exmes_all += "\n" + exmes_list[i];
                }
                throw std::runtime_error(exmes_all);
            }

            this->num_qubits = num_qubits;

            perm_p2l.resize(num_qubits);
            perm_l2p.resize(num_qubits);

            reinitialize_mapping();

            num_states = UINT64_C(1) << num_qubits;

            num_qubits_local = num_qubits - log_num_procs;

            num_states_local = UINT64_C(1) << num_qubits_local;

            if (use_unified_memory)
            {
                ATLC_CHECK_CUDA(cudaMallocManaged, &state_data_device, num_states_local * sizeof(*state_data_device));
            }
            else
            {
                ATLC_CHECK_CUDA(cudaMallocAsync, &state_data_device, num_states_local * sizeof(*state_data_device), stream);
            }
            initialize_zero();

            qcs_kernel_common_host.num_qubits = num_qubits;
            qcs_kernel_common_host.state_data_device = state_data_device;
            ATLC_CHECK_CUDA(cudaMemcpyAsync, qcs_kernel_common_constant_addr, &qcs_kernel_common_host, sizeof(qcs::kernel_common_struct), cudaMemcpyHostToDevice, stream);

            uint64_t const allocatable_states = (uint64_t)initial_free_memory_bytes >> 4;
            if (allocatable_states > num_states_local)
            {
                uint64_t const swap_buffer_total_length_avail = allocatable_states - num_states_local;
                int const log_swap_buffer_total_length_avail = atlc::log2_int(swap_buffer_total_length_avail);
                if (log_swap_buffer_total_length_avail >= num_qubits_local - 1)
                {
                    log_swap_buffer_total_length = num_qubits_local - 1;
                }
                else
                {
                    log_swap_buffer_total_length = log_swap_buffer_total_length_avail;
                    fprintf(
                        stderr,
                        "[warn] swap buffer length reduced due to free memory limit (log=%d, target=%d).\n",
                        log_swap_buffer_total_length,
                        num_qubits_local - 1);
                }
            }
            else
            {
                log_swap_buffer_total_length = 0;
                fprintf(
                    stderr,
                    "[warn] insufficient free memory for additional swap buffer allocation (bytes=%zu).\n",
                    initial_free_memory_bytes);
            }
            swap_buffer_total_length = UINT64_C(1) << log_swap_buffer_total_length;
            ATLC_CHECK_CUDA(cudaMallocAsync, &swap_buffer, swap_buffer_total_length * sizeof(qcs::complex_t), stream);

            ATLC_CHECK_CUDA(cudaMallocAsync, &cub_temp_buffer_device, 1, stream);
            cub_temp_buffer_device_size = 1;

            ATLC_CHECK_CUDA(cudaMallocAsync, &measure_norm_device, sizeof(qcs::complex_t), stream);
        }

        void free_memory()
        {
            if (cub_temp_buffer_device)
            {
                ATLC_CHECK_CUDA(cudaFreeAsync, cub_temp_buffer_device, stream);
                cub_temp_buffer_device = NULL;
            }
            if (state_data_device)
            {
                ATLC_CHECK_CUDA(cudaFreeAsync, state_data_device, stream);
                state_data_device = NULL;
            }
            if (swap_buffer)
            {
                ATLC_CHECK_CUDA(cudaFreeAsync, swap_buffer, stream);
                swap_buffer = NULL;
            }
            if (measure_norm_device)
            {
                ATLC_CHECK_CUDA(cudaFreeAsync, measure_norm_device, stream);
                measure_norm_device = NULL;
            }
            this->num_qubits = 0;
        }

        void initialize_sequential()
        {
            discard_measurement_state();
            uint64_t num_blocks_init;
            uint64_t block_size_init;
            if (num_qubits_local >= log_block_size_max)
            {
                num_blocks_init = num_states_local >> log_block_size_max;
                block_size_init = block_size_max;
            }
            else
            {
                num_blocks_init = 1;
                block_size_init = num_states_local;
            }

            ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, qcs::initstate_sequential_kernel, num_blocks_init, block_size_init, 0, stream, state_data_device, proc_num);
        } /* initialize_sequential */

        void initialize_flat()
        {
            discard_measurement_state();
            uint64_t num_blocks_init;
            uint64_t block_size_init;
            if (num_qubits_local >= log_block_size_max)
            {
                num_blocks_init = num_states_local >> log_block_size_max;
                block_size_init = block_size_max;
            }
            else
            {
                num_blocks_init = 1;
                block_size_init = num_states_local;
            }

            ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, qcs::initstate_flat_kernel, num_blocks_init, block_size_init, 0, stream, state_data_device);
        } /* initialize_flat */

        void initialize_zero()
        {
            discard_measurement_state();
            if (proc_num == 0)
            {
                qcs::complex_t const one = 1;
                ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device, &one, sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);
                ATLC_CHECK_CUDA(cudaMemset, state_data_device + 1, 0, sizeof(qcs::complex_t) * (num_states_local - 1));
            }
            else
            {
                ATLC_CHECK_CUDA(cudaMemset, state_data_device, 0, sizeof(qcs::complex_t) * num_states_local);
            }
        }

        void initialize_entangled()
        {
            discard_measurement_state();
            if (proc_num == 0)
            {
                qcs::complex_t const one = 1;
                ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device, &one, sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);
            }
            else if (proc_num == num_procs - 1)
            {
                qcs::complex_t const one = 1;
                ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device + num_states_local - 1, &one, sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);
            }
        }

        void initialize_use_curand()
        {
            discard_measurement_state();

            MPI_Barrier(MPI_COMM_WORLD);
            if (proc_num == 0)
            {
                fprintf(stderr, "[info] generating random state\n");
            }
            curandGenerator_t rng_device;

            {
                int const log_num_rand_areas = atlc::log2_int(num_rand_areas);
                uint64_t const num_states_rand_area = num_states_local >> log_num_rand_areas;
                for (int rand_area_num = 0; rand_area_num < num_rand_areas; rand_area_num++)
                {
                    ATLC_CHECK_CURAND(curandCreateGenerator, &rng_device, CURAND_RNG_PSEUDO_DEFAULT);
                    ATLC_CHECK_CURAND(curandSetStream, rng_device, stream);
                    ATLC_CHECK_CURAND(curandSetPseudoRandomGeneratorSeed, rng_device, rng_seed + proc_num * num_rand_areas + rand_area_num);
                    ATLC_CHECK_CURAND(curandGenerateNormalDouble, rng_device, (qcs::float_t *)(void *)(state_data_device + num_states_rand_area * ((uint64_t)rand_area_num)), num_states_rand_area * 2 /* complex */, 0.0, 1.0);
                    ATLC_CHECK_CURAND(curandDestroyGenerator, rng_device);
                }
            }
        } /* initialize_use_curand */

        void initialize_laod_statevector()
        {
            discard_measurement_state();

            MPI_Barrier(MPI_COMM_WORLD);

            if (proc_num == 0)
            {
                fprintf(stderr, "[info] load statevector\n");
            }

            qcs::complex_t *state_data_host = (qcs::complex_t *)malloc(num_states_local * sizeof(qcs::complex_t));
            ATLC_DEFER_FUNC(free, state_data_host);

            for (int proc_num_active = 0; proc_num_active < num_procs; proc_num_active++)
            {
                if (proc_num_active == proc_num)
                {
                    FILE *const fp = fopen("statevector_input.bin", "rb");
                    if (fp == NULL)
                    {
                        throw std::runtime_error("open failed");
                    }
                    fseek(fp, proc_num * num_states_local * sizeof(qcs::complex_t), SEEK_SET);
                    size_t const ret = fread(state_data_host, sizeof(qcs::complex_t), num_states_local, fp);
                    if (ret != num_states_local)
                    {
                        throw std::runtime_error("fread failed");
                    }
                    fclose(fp);

                    ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_device, state_data_host, num_states_local * sizeof(qcs::complex_t), cudaMemcpyHostToDevice, stream);
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }

            ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
        } /* initialize_laod_statevector */

        void prepare_control_qubit_num_list()
        {

            std::vector<int> *const measured_X_qubit_num_logical_list_list[] = {
                &measured_0_qubit_num_logical_list,
                &measured_1_qubit_num_logical_list};

            std::vector<int> *const measured_X_qubit_num_logical_list_copy_list[] = {
                &measured_0_qubit_num_logical_list_copy,
                &measured_1_qubit_num_logical_list_copy};

            std::vector<int> *const X_control_qubit_num_logical_list_list[] = {
                &negative_control_qubit_num_logical_list,
                &positive_control_qubit_num_logical_list};

            measured_control_condition = true;

#pragma unroll
            for (int measured_value = 0; measured_value < 2; measured_value++)
            {
                measured_X_qubit_num_logical_list_copy_list[measured_value]->clear();
                for (int mXqnl_idx = 0; mXqnl_idx < measured_X_qubit_num_logical_list_list[measured_value]->size(); mXqnl_idx++)
                {
                    auto const mXqn = measured_X_qubit_num_logical_list_list[measured_value]->operator[](mXqnl_idx);
                    bool const is_target = std::find(target_qubit_num_logical_list.begin(), target_qubit_num_logical_list.end(), mXqn) != target_qubit_num_logical_list.end();
                    if (!is_target)
                    {
                        // the qubit is kept measured
                        measured_X_qubit_num_logical_list_copy_list[measured_value]->push_back(mXqn);
                        bool const is_control = std::find(X_control_qubit_num_logical_list_list[measured_value]->begin(), X_control_qubit_num_logical_list_list[measured_value]->end(), mXqn) != X_control_qubit_num_logical_list_list[measured_value]->end();
                        bool const is_control_other = std::find(X_control_qubit_num_logical_list_list[1 - measured_value]->begin(), X_control_qubit_num_logical_list_list[1 - measured_value]->end(), mXqn) != X_control_qubit_num_logical_list_list[1 - measured_value]->end();
                        if ((!is_control) && (!is_control_other))
                        {
                            X_control_qubit_num_logical_list_list[measured_value]->push_back(mXqn);
                        }
                        else if (is_control_other)
                        {
                            measured_control_condition = false;
                        }
                    }
                }
            }

        } /* prepare_control_qubit_num_list */

        void ensure_local_qubits()
        {
            target_qubit_num_physical_list.resize(target_qubit_num_logical_list.size());
            for (int tqni = 0; tqni < target_qubit_num_logical_list.size(); tqni++)
            {
                target_qubit_num_physical_list[tqni] = perm_l2p[target_qubit_num_logical_list[tqni]];
            }

            swap_target_global_list.resize(0);
            swap_target_local_list.resize(0);
            for (int tqni = 0; tqni < target_qubit_num_physical_list.size(); tqni++)
            {
                auto const tqn_i = target_qubit_num_physical_list[tqni];
                if (tqn_i >= num_qubits_local)
                {
                    swap_target_global_list.push_back(tqn_i);
                    int const swap_target_local = num_qubits_local - swap_target_global_list.size();
                    swap_target_local_list.push_back(swap_target_local);
                    target_qubit_num_physical_list[tqni] = swap_target_local;
                }
            }
            int const num_swap_qubits = swap_target_global_list.size();

            /* target qubits is global */
            if (swap_target_global_list.size() > 0)
            {

                // b_min
                int const swap_target_local_min = *std::min_element(swap_target_local_list.data(), swap_target_local_list.data() + num_swap_qubits);

                uint64_t const local_buf_length = UINT64_C(1) << swap_target_local_min;
                uint64_t swap_buffer_length = swap_buffer_total_length;
                if (swap_buffer_length > local_buf_length)
                {
                    swap_buffer_length = local_buf_length;
                }

                // generate a mask for generating global_nonswap_self
                uint64_t global_swap_self_mask = 0;
                for (int target_num = 0; target_num < num_swap_qubits; target_num++)
                {
                    // a_delta = a – n_local
                    int const swap_target_global_delta = swap_target_global_list[target_num] - num_qubits_local;
                    global_swap_self_mask |= (UINT64_C(1) << swap_target_global_delta);
                }

                // global_nonswap_self = make proc_num_self's a_delta_i-th digit zero
                uint64_t const global_nonswap_self = proc_num & ~global_swap_self_mask;

                // 1<<(num_local_qubits - b_min)
                uint64_t const num_local_areas = UINT64_C(1) << (num_qubits_local - swap_target_local_min);
                for (uint64_t local_num_self = 0; local_num_self < num_local_areas; local_num_self++)
                {

                    // global_swap_peer = OR_i (local_num_selfのb_delta_i桁目)をa_delta_i桁目にする
                    uint64_t global_swap_peer = 0;
                    for (int target_num = 0; target_num < num_swap_qubits; target_num++)
                    {
                        // a_delta_i
                        int const swap_target_global_delta = swap_target_global_list[target_num] - num_qubits_local;
                        // b_delta_i
                        int const swap_target_local_delta = swap_target_local_list[target_num] - swap_target_local_min;
                        global_swap_peer |=
                            // local_num_selfのb_delta_i桁目
                            ((local_num_self >> swap_target_local_delta) & 1)
                            // をa_delta_i桁目にする
                            << swap_target_global_delta;
                    }

                    uint64_t const proc_num_peer = global_swap_peer | global_nonswap_self;

                    // send & recv
                    if (proc_num_peer == proc_num)
                    {
                        continue;
                    }

                    bool is_peer_greater = proc_num_peer > proc_num;
                    for (uint64_t buffer_pos = 0; buffer_pos < local_buf_length; buffer_pos += swap_buffer_length)
                    {
                        ATLC_CHECK_NCCL(ncclGroupStart);
                        for (int send_recv = 0; send_recv < 2; send_recv++)
                        {
                            if (send_recv ^ is_peer_greater)
                            {
                                ATLC_CHECK_NCCL(ncclSend, &state_data_device[local_num_self * local_buf_length + buffer_pos], swap_buffer_length * 2 /* complex */, ncclDouble, proc_num_peer, nccl_comm, stream);
                            }
                            else
                            {
                                ATLC_CHECK_NCCL(ncclRecv, swap_buffer, swap_buffer_length * 2 /* complex */, ncclDouble, proc_num_peer, nccl_comm, stream);
                            }
                        }
                        ATLC_CHECK_NCCL(ncclGroupEnd);
                        ATLC_CHECK_CUDA(cudaMemcpyAsync, &state_data_device[local_num_self * local_buf_length + buffer_pos], swap_buffer, swap_buffer_length * sizeof(qcs::complex_t), cudaMemcpyDeviceToDevice, stream);
                    }
                }

                // swap_target_global_logical_list[:] = perm_p2l[swap_target_global_list[:]]
                // swap_target_local_logical_list[:] = perm_p2l[swap_target_local_list[:]]
                swap_target_local_logical_list.resize(num_swap_qubits);
                swap_target_global_logical_list.resize(num_swap_qubits);
                for (int target_num = 0; target_num < num_swap_qubits; target_num++)
                {
                    swap_target_local_logical_list[target_num] = perm_p2l[swap_target_local_list[target_num]];
                    swap_target_global_logical_list[target_num] = perm_p2l[swap_target_global_list[target_num]];
                }

                // update p2l & l2p
                // perm_p2l[swap_target_global_list[:]] = swap_target_local_logical_list[:]
                // perm_p2l[swap_target_local_list[:]] = swap_target_global_logical_list[:]
                // perm_l2p[swap_target_global_logical_list[:]] = swap_target_local_list[:]
                // perm_l2p[swap_target_local_logical_list[:]] = swap_target_global_list[:]

                for (int target_num = 0; target_num < num_swap_qubits; target_num++)
                {
                    perm_p2l[swap_target_global_list[target_num]] = swap_target_local_logical_list[target_num];
                    perm_p2l[swap_target_local_list[target_num]] = swap_target_global_logical_list[target_num];
                    perm_l2p[swap_target_global_logical_list[target_num]] = swap_target_local_list[target_num];
                    perm_l2p[swap_target_local_logical_list[target_num]] = swap_target_global_list[target_num];
                }

                // target_qubit_num_physical = swap_target_local;
            }
        }; /* ensure_local_qubits */

        void check_control_qubit_num_physical()
        {

            /* check whether proc_num is under control condition */
            proc_num_control_condition = true;

            std::vector<int> *x_control_qubit_num_logical_list_list[] = {&negative_control_qubit_num_logical_list, &positive_control_qubit_num_logical_list};
            std::vector<int> *x_control_qubit_num_physical_list_list[] = {&negative_control_qubit_num_physical_list, &positive_control_qubit_num_physical_list};
            std::vector<int> *x_control_qubit_num_physical_global_list_list[] = {&negative_control_qubit_num_physical_global_list, &positive_control_qubit_num_physical_global_list};
            std::vector<int> *x_control_qubit_num_physical_local_list_list[] = {&negative_control_qubit_num_physical_local_list, &positive_control_qubit_num_physical_local_list};

#pragma unroll
            for (int control_np = 0; control_np < 2; control_np++)
            {
                x_control_qubit_num_physical_list_list[control_np]->resize(x_control_qubit_num_logical_list_list[control_np]->size());
                x_control_qubit_num_physical_global_list_list[control_np]->resize(0);
                x_control_qubit_num_physical_local_list_list[control_np]->resize(0);

                for (int cqni = 0; cqni < x_control_qubit_num_logical_list_list[control_np]->size(); cqni++)
                {
                    auto const x_control_qubit_num_physical = perm_l2p[x_control_qubit_num_logical_list_list[control_np]->operator[](cqni)];
                    x_control_qubit_num_physical_list_list[control_np]->operator[](cqni) = x_control_qubit_num_physical;
                    if (x_control_qubit_num_physical >= num_qubits_local)
                    {
                        x_control_qubit_num_physical_global_list_list[control_np]->push_back(x_control_qubit_num_physical);
                        if ((1 & (proc_num >> (x_control_qubit_num_physical - num_qubits_local))) != control_np)
                        {
                            proc_num_control_condition = false;
                        }
                    }
                    else
                    {
                        x_control_qubit_num_physical_local_list_list[control_np]->push_back(x_control_qubit_num_physical);
                    }
                }
            }

        }; /* check_control_qubit_num_physical */

        void prepare_operating_gate()
        {

            if (!proc_num_control_condition)
            {
                return;
            }

            uint64_t const qkiqn_size = qcs::kernel_input_qnlist_struct::needed_size(
                target_qubit_num_physical_list.size(),
                negative_control_qubit_num_physical_local_list.size(),
                positive_control_qubit_num_physical_local_list.size());
            qcs_kernel_input_host_buffer.resize(qkiqn_size);
            qcs::kernel_input_qnlist_struct *const qcs_kernel_input_host = (qcs::kernel_input_qnlist_struct *)qcs_kernel_input_host_buffer.data();

            qcs_kernel_input_host->num_positive_control_qubits = positive_control_qubit_num_physical_local_list.size();
            qcs_kernel_input_host->num_negative_control_qubits = negative_control_qubit_num_physical_local_list.size();
            qcs_kernel_input_host->num_target_qubits = target_qubit_num_physical_list.size();

            auto positive_control_qubit_num_list_kernel_arg = qcs_kernel_input_host->get_positive_control_qubit_num_list();
            for (int pcqi = 0; pcqi < positive_control_qubit_num_physical_local_list.size(); pcqi++)
            {
                positive_control_qubit_num_list_kernel_arg[pcqi] = positive_control_qubit_num_physical_local_list[pcqi];
            }

            auto target_qubit_num_list_kernel_arg = qcs_kernel_input_host->get_target_qubit_num_list();
            qcs_kernel_input_host->is_measured_bits = 0;
            qcs_kernel_input_host->measured_value_bits = 0;

            for (int tqi = 0; tqi < target_qubit_num_physical_list.size(); tqi++)
            {
                auto const tqn_phys = target_qubit_num_physical_list[tqi];
                target_qubit_num_list_kernel_arg[tqi] = tqn_phys;

            }

            num_operand_qubits =
                positive_control_qubit_num_physical_local_list.size() + negative_control_qubit_num_physical_local_list.size() + target_qubit_num_physical_list.size();

            /* get sorted operand qubits */
            operand_qubit_num_list.clear();
            operand_qubit_num_list.insert(operand_qubit_num_list.end(), positive_control_qubit_num_physical_local_list.begin(), positive_control_qubit_num_physical_local_list.end());
            operand_qubit_num_list.insert(operand_qubit_num_list.end(), negative_control_qubit_num_physical_local_list.begin(), negative_control_qubit_num_physical_local_list.end());
            operand_qubit_num_list.insert(operand_qubit_num_list.end(), target_qubit_num_physical_list.begin(), target_qubit_num_physical_list.end());

            std::sort(operand_qubit_num_list.begin(), operand_qubit_num_list.end()); /* ascending order */

            auto qubit_num_list_sorted_kernel_arg = qcs_kernel_input_host->get_operand_qubit_num_list_sorted();
            for (int qni = 0; qni < operand_qubit_num_list.size(); qni++)
            {
                qubit_num_list_sorted_kernel_arg[qni] = operand_qubit_num_list[qni];
            }

            ATLC_CHECK_CUDA(cudaMemcpyAsync, qcs_kernel_input_constant_addr, qcs_kernel_input_host, qkiqn_size, cudaMemcpyHostToDevice, stream);

            uint64_t const log_num_threads = num_qubits_local - num_operand_qubits;

            uint64_t log_block_size_gateop;

            if (log_block_size_max > log_num_threads)
            {
                log_block_size_gateop = log_num_threads;
                num_blocks_gateop = 1;
            }
            else
            {
                log_block_size_gateop = log_block_size_max;
                num_blocks_gateop = UINT64_C(1) << (log_num_threads - log_block_size_max);
            }

            block_size_gateop = UINT64_C(1) << log_block_size_gateop;

        } /* prepare_operating_gate */

        void update_measured_list()
        {
            measured_0_qubit_num_logical_list = measured_0_qubit_num_logical_list_copy;
            measured_1_qubit_num_logical_list = measured_1_qubit_num_logical_list_copy;
        }

        void discard_measurement_state()
        {
            measured_mask_logical = 0;
            measured_value_logical = 0;
            pending_projection_mask_logical = 0;
            measured_0_qubit_num_logical_list.clear();
            measured_1_qubit_num_logical_list.clear();
            measured_0_qubit_num_logical_list_copy.clear();
            measured_1_qubit_num_logical_list_copy.clear();
        }

        void materialize_pending_projection(uint64_t requested_mask)
        {
            uint64_t const todo_mask = requested_mask & pending_projection_mask_logical;
            if (todo_mask == 0)
                return;

            uint64_t local_mask = 0;
            uint64_t local_value = 0;
            uint64_t rank_mask = 0;
            uint64_t rank_value = 0;
            for (int logical_q = 0; logical_q < num_qubits; ++logical_q)
            {
                uint64_t const logical_bit = UINT64_C(1) << logical_q;
                if ((todo_mask & logical_bit) == 0)
                    continue;
                int const physical_q = perm_l2p[logical_q];
                uint64_t const value = (measured_value_logical & logical_bit) != 0;
                if (physical_q < num_qubits_local)
                {
                    local_mask |= UINT64_C(1) << physical_q;
                    local_value |= value << physical_q;
                }
                else
                {
                    int const rank_q = physical_q - num_qubits_local;
                    rank_mask |= UINT64_C(1) << rank_q;
                    rank_value |= value << rank_q;
                }
            }

            if ((((uint64_t)proc_num ^ rank_value) & rank_mask) != 0)
            {
                ATLC_CHECK_CUDA(cudaMemsetAsync, state_data_device, 0, num_states_local * sizeof(qcs::complex_t), stream);
            }
            else if (local_mask != 0)
            {
                uint64_t const block_size = std::min<uint64_t>(block_size_max, num_states_local);
                uint64_t const num_blocks = (num_states_local + block_size - 1) / block_size;
                ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, qcs::materialize_projection_kernel, num_blocks, block_size, 0, stream,
                                state_data_device, num_states_local, local_mask, local_value);
            }
            pending_projection_mask_logical &= ~todo_mask;
        }

        void clear_measurement_state()
        {
            materialize_pending_projection(pending_projection_mask_logical);
            discard_measurement_state();
        }

        void save_statevector(char const *const outfn)
        {

            materialize_pending_projection(pending_projection_mask_logical);

            MPI_Barrier(MPI_COMM_WORLD);
            // if (proc_num == 0) { fprintf(stderr, "[info] dump statevector\n"); }

            qcs::complex_t *state_data_host = (qcs::complex_t *)malloc(num_states_local * sizeof(qcs::complex_t));
            ATLC_DEFER_FUNC(free, state_data_host);

            ATLC_CHECK_CUDA(cudaMemcpyAsync, state_data_host, state_data_device, num_states_local * sizeof(qcs::complex_t), cudaMemcpyDeviceToHost, stream);

            for (int proc_num_active = 0; proc_num_active < num_procs; proc_num_active++)
            {
                if (proc_num_active == proc_num)
                {
                    FILE *const fp = fopen(outfn, (proc_num == 0) ? "wb" : "rb+");
                    if (fp == NULL)
                    {
                        throw std::runtime_error("open failed");
                    }

                    for (uint64_t state_num_physical_local = 0; state_num_physical_local < num_states_local; state_num_physical_local++)
                    {
                        uint64_t const state_num_physical = state_num_physical_local | (((uint64_t)proc_num) << num_qubits_local);
                        uint64_t state_num_logical = 0;
                        for (int qubit_num_physical = 0; qubit_num_physical < num_qubits; qubit_num_physical++)
                        {
                            int qubit_num_logical = perm_p2l[qubit_num_physical];
                            state_num_logical = state_num_logical | (((state_num_physical >> qubit_num_physical) & 1) << qubit_num_logical);
                        }
                        int const ret_fseek = fseek(fp, state_num_logical * sizeof(qcs::complex_t), SEEK_SET);
                        if (ret_fseek != 0)
                        {
                            auto const errno_saved = errno;
                            throw std::runtime_error(atlc::format("errorno=%d ret_fseek=%d", errno_saved, ret_fseek));
                        }
                        size_t const ret = fwrite(&state_data_host[state_num_physical_local], sizeof(qcs::complex_t), 1, fp);
                        if (ret != 1)
                        {
                            auto const errno_saved = errno;
                            throw std::runtime_error(atlc::format("fwrite failed ret=%zu errno=%d", ret, errno_saved));
                        }
                    }
                    fflush(fp);
                    fclose(fp);
                    fsync(fileno(fp));
                }
                MPI_Barrier(MPI_COMM_WORLD);
            }
        } /* save_statevector */

        int measure_qubit(int const measure_qubit_num_logical)
        {

            for (auto const m0qn : measured_0_qubit_num_logical_list)
            {
                if (measure_qubit_num_logical == m0qn)
                {
                    return 0;
                }
            }

            for (auto const m1qn : measured_1_qubit_num_logical_list)
            {
                if (measure_qubit_num_logical == m1qn)
                {
                    return 1;
                }
            }

            int const measure_qubit_num_physical = perm_l2p[measure_qubit_num_logical];
            bool const is_global = measure_qubit_num_physical >= num_qubits_local;
            int const rank_value = is_global
                                       ? (uint64_t(proc_num) >> (measure_qubit_num_physical - num_qubits_local)) & 1
                                       : 0;

            target_qubit_num_logical_list.clear();
            if (!is_global)
            {
                target_qubit_num_logical_list.push_back(measure_qubit_num_logical);
            }
            positive_control_qubit_num_logical_list = measured_1_qubit_num_logical_list;
            negative_control_qubit_num_logical_list = measured_0_qubit_num_logical_list;

            // A global measurement does not need a statevector exchange.  Keeping
            // ensure_local_qubits() here clears its working lists for both paths.
            ensure_local_qubits();
            check_control_qubit_num_physical();
            prepare_operating_gate();

            float2_t measure_norm_host;

            if (proc_num_control_condition)
            {

                cubUtility::IndirectLoad loader{is_global, rank_value};

                using CountingIter = thrust::counting_iterator<uint64_t>;
                using TransformIter = thrust::transform_iterator<decltype(loader), CountingIter>;

                CountingIter counting(0);
                TransformIter in_it(counting, loader);

                uint64_t temp_sz_required;

                cubUtility::float2Add float2AddObj;
                float2_t zero{};
                ATLC_CHECK_CUDA(cub::DeviceReduce::Reduce, NULL, temp_sz_required, in_it, measure_norm_device, num_states_local >> num_operand_qubits, float2AddObj, zero, stream);

                if (cub_temp_buffer_device_size < temp_sz_required)
                {
                    ATLC_CHECK_CUDA(cudaFreeAsync, cub_temp_buffer_device, stream);
                    ATLC_CHECK_CUDA(cudaMallocAsync, &cub_temp_buffer_device, temp_sz_required, stream);
                    cub_temp_buffer_device_size = temp_sz_required;
                }

                ATLC_CHECK_CUDA(cub::DeviceReduce::Reduce, cub_temp_buffer_device, cub_temp_buffer_device_size, in_it, measure_norm_device, num_states_local >> num_operand_qubits, float2AddObj, zero, stream);

                ATLC_CHECK_CUDA(cudaMemcpyAsync, &measure_norm_host, measure_norm_device, sizeof(float2_t), cudaMemcpyDeviceToHost, stream);

                ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
            }
            else
            {
                measure_norm_host = {0, 0};
            }

#if 1 /* parallel measurement */
            qcs::float_t measure_norm_global[2];
            ATLC_CHECK_MPI(MPI_Allreduce, measure_norm_host.data(), measure_norm_global, 2, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);

            qcs::float_t const measure_norm_sum = measure_norm_global[0] + measure_norm_global[1];
            if (!(measure_norm_sum > 0) || !std::isfinite(measure_norm_sum))
            {
                throw std::runtime_error(atlc::format(
                    "invalid measurement norm: norm0=%g norm1=%g",
                    measure_norm_global[0], measure_norm_global[1]));
            }

            int measure_result = 0;
            if (proc_num == 0)
            {
                std::uniform_real_distribution<qcs::float_t> dist1(0, measure_norm_sum);
                qcs::float_t const random_value = dist1(engine);
                if (!(measure_norm_global[0] > 0))
                    measure_result = 1;
                else if (measure_norm_global[1] > 0)
                    measure_result = measure_norm_global[0] < random_value;
            }
            ATLC_CHECK_MPI(MPI_Bcast, &measure_result, 1, MPI_INT, 0, MPI_COMM_WORLD);

            if (measure_result)
            { /* 1 */
                measured_1_qubit_num_logical_list.push_back(measure_qubit_num_logical);
            }
            else
            { /* 0 */
                measured_0_qubit_num_logical_list.push_back(measure_qubit_num_logical);
            }
#else /* measurement by master */
            qcs::float_t measure_norm_global[2];
            MPI_Reduce(measure_norm_host.data(), measure_norm_global, 2, MPI_DOUBLE, MPI_SUM, 0, MPI_COMM_WORLD);

            bool measure_result_0;
            if (proc_num == 0)
            {
                qcs::float_t const measure_norm_sum = measure_norm_global[0] + measure_norm_global[1];

                std::uniform_real_distribution<qcs::float_t> dist1(0, measure_norm_sum);
                qcs::float_t const random_value = dist1(engine);
                measure_result_0 = measure_norm_global[0] < random_value;
            }
            bool measure_result;
            MPI_Scatter(&measure_result_0, 1, MPI_C_BOOL, &measure_result, 1, MPI_C_BOOL, 0, MPI_COMM_WORLD);

            if (measure_result)
            { /* 1 */
                measured_1_qubit_num_logical_list.push_back(measure_qubit_num_logical);
                // measured_bit |= UINT64_C(1) << measure_qubit_num_logical;
            }
            else
            { /* 0 */
                measured_0_qubit_num_logical_list.push_back(measure_qubit_num_logical);
            }
#endif

            uint64_t const measured_bit = UINT64_C(1) << measure_qubit_num_logical;
            measured_mask_logical |= measured_bit;
            measured_value_logical = (measured_value_logical & ~measured_bit) |
                                     ((uint64_t)measure_result << measure_qubit_num_logical);
            pending_projection_mask_logical |= measured_bit;

            return measure_result;
        }

        template <typename GateType>
        void operate_gate(GateType gateobj, std::vector<int> target_qubit_num_logical_list_input, std::vector<int> negative_control_qubit_num_logical_list_input, std::vector<int> positive_control_qubit_num_logical_list_input)
        {

            assert(target_qubit_num_logical_list_input.size() == GateType::num_target_qubits);

            target_qubit_num_logical_list = std::move(target_qubit_num_logical_list_input);
            negative_control_qubit_num_logical_list = std::move(negative_control_qubit_num_logical_list_input);
            positive_control_qubit_num_logical_list = std::move(positive_control_qubit_num_logical_list_input);

            uint64_t target_mask = 0;
            for (int const target : target_qubit_num_logical_list)
                target_mask |= UINT64_C(1) << target;
            uint64_t const release_mask = measured_mask_logical & target_mask;

            prepare_control_qubit_num_list();
            if (!measured_control_condition)
                return;

            ensure_local_qubits();
            materialize_pending_projection(release_mask);
            assert((pending_projection_mask_logical & target_mask) == 0);
            check_control_qubit_num_physical();
            prepare_operating_gate();

            if (proc_num_control_condition)
            {
                ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, cuda_gate<GateType>, num_blocks_gateop, block_size_gateop, 0, stream, gateobj);
            }

            update_measured_list();
            measured_mask_logical &= ~release_mask;
            measured_value_logical &= measured_mask_logical;
            pending_projection_mask_logical &= measured_mask_logical;
        }

#if 0
void GHZ_circuit_sample() {

    allocate_memory(14);

    uint64_t measured_bit = 0;

    for (int measure_qubit_num_logical = 0; measure_qubit_num_logical < num_qubits; measure_qubit_num_logical++) {
        int const measured_value = measure_qubit(measure_qubit_num_logical);
        if (measured_value) {
            measured_bit |= UINT64_C(1) << measure_qubit_num_logical;
        }
    }

    if (proc_num == 0) {
        fprintf(stdout, "%" PRIu64 "\n", measured_bit);
    }

    uint64_t const num_samples = UINT64_C(1) << num_qubits;

    for(int sample_num = 0; sample_num < num_samples; ++sample_num) {

        /* begin gate operation */
        operate_gate(qcs::gate::hadamard(), {0}, {}, {});

        for(int target_qubit_num_logical = 1; target_qubit_num_logical < num_qubits; target_qubit_num_logical++)
        {
            if ((measured_bit>>target_qubit_num_logical)&1) {
                operate_gate(qcs::gate::x(), {target_qubit_num_logical}, {0}, {});
            } else {
                operate_gate(qcs::gate::x(), {target_qubit_num_logical}, {}, {0});
            }

        } /* target_qubit_num_logical loop */

        /* end gate operation */

        /* begin measurement */
        measured_bit = 0;

        for (int measure_qubit_num_logical = 0; measure_qubit_num_logical < num_qubits; measure_qubit_num_logical++) {
            int const measured_value = measure_qubit(measure_qubit_num_logical);
            if (measured_value) {
                measured_bit |= UINT64_C(1) << measure_qubit_num_logical;
            }
        }

        if (proc_num == 0) {
            fprintf(stdout, "%" PRIu64 "\n", measured_bit);
        }
        /* end measurement */

    }

}; /* GHZ_circuit_sample */

void measurement_sample() {

    allocate_memory(14);

    constexpr initstate_enum initstate_choice = initstate_enum::flat;
    switch (initstate_choice) {
        case initstate_enum::sequential:
            initialize_sequential();
            break;
        case initstate_enum::flat:
            initialize_flat();
            break;
        case initstate_enum::zero:
            initialize_zero();
            break;
        case initstate_enum::entangled:
            initialize_entangled();
            break;
        case initstate_enum::use_curand:
            initialize_use_curand();
            break;
        case initstate_enum::laod_statevector:
            initialize_laod_statevector();
            break;
        default:
            throw initstate_choice;
    }

    uint64_t measured_bit = 0;
    uint64_t const num_samples = UINT64_C(1) << num_qubits;

    for(int sample_num = 0; sample_num < num_samples; ++sample_num) {

        // forget measurement
        // warn: do not use it, unless you fully understand the behavior of lazy view.
        measured_0_qubit_num_logical_list.clear();
        measured_1_qubit_num_logical_list.clear();

        /* begin measurement */
        measured_bit = 0;

        for (int measure_qubit_num_logical = 0; measure_qubit_num_logical < num_qubits; measure_qubit_num_logical++) {
            int const measured_value = measure_qubit(measure_qubit_num_logical);
            if (measured_value) {
                measured_bit |= UINT64_C(1) << measure_qubit_num_logical;
            }
        }

        if (proc_num == 0) {
            fprintf(stdout, "%" PRIu64 "\n", measured_bit);
        }
        /* end measurement */

    }

} /* measurement_sample */

int main() {

    constexpr bool flag_save_statevector = false;

    setup();
    ATLC_DEFER_FUNC(dispose);

    ATLC_DEFER_FUNC(free_memory);

    // measurement_sample();
    GHZ_circuit_sample();

    if (flag_save_statevector) { save_statevector("statevector_output.bin"); }
    return 0;

}; /* main */
#endif

        void dispose()
        {
            free_memory();
            for (cudaEvent_t event : event_list)
            {
                ATLC_CHECK_CUDA(cudaEventDestroy, event);
            }
            ATLC_CHECK_CUDA(cudaStreamDestroy, stream);
            MPI_Finalize();
        };

        int event_create()
        {
            int const event_num = event_list.size();
            cudaEvent_t event;
            ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event, cudaEventDefault);
            this->event_list.push_back(event);
            return event_num;
        }

        void event_record(int const event_num)
        {
            cudaEvent_t const event = event_list[event_num];
            ATLC_CHECK_CUDA(cudaEventRecordWithFlags, event, stream, cudaEventDefault);
        }

        double event_get_elapsed_time(int const start_event_num, int const stop_event_num)
        {
            cudaEvent_t const start = event_list[start_event_num];
            cudaEvent_t const stop = event_list[stop_event_num];

            cudaError_t q;

            q = cudaEventQuery(start);
            if (q == cudaErrorNotReady)
            {
                ATLC_CHECK_CUDA(cudaEventSynchronize, start);
            }
            else if (q != cudaSuccess)
            {
                throw std::runtime_error(cudaGetErrorString(q));
            }

            q = cudaEventQuery(stop);
            if (q == cudaErrorNotReady)
            {
                ATLC_CHECK_CUDA(cudaEventSynchronize, stop);
            }
            else if (q != cudaSuccess)
            {
                throw std::runtime_error(cudaGetErrorString(q));
            }

            float elapsed_ms = 0.0f;
            ATLC_CHECK_CUDA(cudaEventElapsedTime, &elapsed_ms, start, stop);

            float elapsed_ms_global;

            MPI_Allreduce(&elapsed_ms, &elapsed_ms_global, 1, MPI_FLOAT, MPI_MAX, MPI_COMM_WORLD);

            return elapsed_ms_global * 1e-3;
        }

    }; /* simulator_core */

} /* qcs */

struct qcs_simulator
{
    qcs::simulator_core *core;
    int num_qubits;
    int num_clbits;
    std::vector<bool> clbits;
};

static void qcs_simulator_init(qcs_simulator *sim);
static void qcs_simulator_setup(qcs_simulator *sim);

static std::vector<int> qcs_vector_from_array(bit_num_t const *values, bit_num_t count)
{
    if (count < 0)
    {
        throw std::runtime_error("array element count must be non-negative");
    }
    if (count > 0 && values == NULL)
    {
        throw std::runtime_error("array pointer must not be NULL when element count is positive");
    }
    return std::vector<int>(values, values + count);
}

static void qcs_simulator_init(qcs_simulator *sim)
{
    sim->num_qubits = 0;
    sim->num_clbits = 0;
    sim->core = NULL;
    sim->clbits.clear();
}

namespace
{
    qcs_exception_callback qcs_current_exception_callback = nullptr;

    void qcs_report_exception(char const *message)
    {
        if (qcs_current_exception_callback == nullptr)
            return;
        qcs_current_exception_callback(message, std::strlen(message));
    }

    void qcs_report_exception(std::exception const &exception)
    {
        qcs_report_exception(exception.what());
    }

    void qcs_report_unknown_exception()
    {
        qcs_report_exception("Unknown C++ exception");
    }

    template <typename Result, typename Value>
    void qcs_write_result(Result *result, Value value)
    {
        if (result == NULL)
            throw std::runtime_error("result pointer must not be NULL");
        *result = static_cast<Result>(value);
    }

    template <typename Function>
    auto qcs_try_cxx(Function &&function, std::invoke_result_t<Function> fallback) -> std::invoke_result_t<Function>
    {
        try
        {
            return function();
        }
        catch (std::exception const &exception)
        {
            qcs_report_exception(exception);
        }
        catch (...)
        {
            qcs_report_unknown_exception();
        }
        return fallback;
    }

    template <typename Function>
    int qcs_try_cxx(Function &&function)
    {
        try
        {
            function();
            return 1;
        }
        catch (std::exception const &exception)
        {
            qcs_report_exception(exception);
        }
        catch (...)
        {
            qcs_report_unknown_exception();
        }
        return 0;
    }
}

extern "C" int qcs_set_exception_callback(qcs_exception_callback callback)
{
    qcs_current_exception_callback = callback;
    return 1;
}

qcs_simulator *qcs_simulator_create_cxx()
{
    qcs_simulator *sim = new qcs_simulator();
    qcs_simulator_setup(sim);
    return sim;
}

void qcs_simulator_destroy_cxx(qcs_simulator *sim)
{
    if (sim == NULL)
    {
        return;
    }
    qcs_simulator_dispose(sim);
    delete sim;
}

static void qcs_simulator_setup(qcs_simulator *sim)
{
    qcs_simulator_init(sim);
    sim->core = new qcs::simulator_core();
    sim->core->setup();
}

void qcs_simulator_allocate_memory_cxx(qcs_simulator *sim) { sim->core->allocate_memory(sim->num_qubits); }
void qcs_simulator_dispose_cxx(qcs_simulator *sim)
{
    sim->core->dispose();
    delete sim->core;
    sim->core = NULL;
}
int qcs_simulator_get_proc_num_cxx(qcs_simulator *sim) { return sim->core->proc_num; }
int qcs_simulator_get_num_procs_cxx(qcs_simulator *sim) { return sim->core->num_procs; }
int qcs_simulator_get_num_qubits_cxx(qcs_simulator const *sim) { return sim->num_qubits; }
int qcs_simulator_get_num_clbits_cxx(qcs_simulator const *sim) { return sim->num_clbits; }
void qcs_simulator_set_num_qubits_cxx(qcs_simulator *sim, bit_num_t num_qubits) { sim->num_qubits = num_qubits; }

void qcs_simulator_set_mapping_cxx(qcs_simulator *sim, bit_num_t const *perm_p2l_array, bit_num_t perm_p2l_count)
{
    std::vector<int> const perm_p2l = qcs_vector_from_array(perm_p2l_array, perm_p2l_count);
    if (sim->num_qubits <= 0)
        throw std::runtime_error("set_num_qubits must be called before set_mapping");
    if ((int)perm_p2l.size() != sim->num_qubits)
        throw std::runtime_error(atlc::format("mapping size %d does not match num_qubits %d", (int)perm_p2l.size(), sim->num_qubits));
    std::vector<bool> used(sim->num_qubits, false);
    for (int logical_qubit_num : perm_p2l)
    {
        if (logical_qubit_num < 0 || logical_qubit_num >= sim->num_qubits)
            throw std::runtime_error(atlc::format("mapping value %d is out of range [0, %d)", logical_qubit_num, sim->num_qubits));
        if (used[logical_qubit_num])
            throw std::runtime_error(atlc::format("mapping value %d appears multiple times", logical_qubit_num));
        used[logical_qubit_num] = true;
    }
    sim->core->initial_perm_p2l = perm_p2l;
}

void qcs_simulator_set_num_clbits_cxx(qcs_simulator *sim, bit_num_t num_clbits)
{
    sim->num_clbits = num_clbits;
    sim->clbits.resize(num_clbits);
}
int qcs_simulator_measure_cxx(qcs_simulator *sim, bit_num_t qubit_num) { return sim->core->measure_qubit(qubit_num); }
int qcs_simulator_measure_to_clbit_cxx(qcs_simulator *sim, bit_num_t qubit_num, bit_num_t clbit_num)
{
    int const result = qcs_simulator_measure_cxx(sim, qubit_num);
    sim->clbits[clbit_num] = result;
    return result;
}

static std::vector<int> qcs_validate_measure_many(
    qcs_simulator *sim, bit_num_t const *qubit_num_list,
    bit_num_t qubit_num_count, bit_t *results)
{
    if (sim == NULL)
        throw std::runtime_error("simulator pointer must not be NULL");
    if (qubit_num_count < 0)
        throw std::runtime_error("qubit count must be non-negative");
    if (qubit_num_count > 0 && qubit_num_list == NULL)
        throw std::runtime_error("qubit list must not be NULL when qubit count is positive");
    if (qubit_num_count > 0 && results == NULL)
        throw std::runtime_error("results pointer must not be NULL when qubit count is positive");

    std::vector<int> qubits;
    qubits.reserve(qubit_num_count);
    std::vector<bool> used(sim->num_qubits, false);
    for (int i = 0; i < qubit_num_count; ++i)
    {
        int const qubit_num = qubit_num_list[i];
        if (qubit_num < 0 || qubit_num >= sim->num_qubits)
            throw std::runtime_error(atlc::format("qubit %d is out of range [0, %d)", qubit_num, sim->num_qubits));
        if (used[qubit_num])
            throw std::runtime_error(atlc::format("qubit %d appears multiple times", qubit_num));
        used[qubit_num] = true;
        qubits.push_back(qubit_num);
    }
    return qubits;
}

void qcs_simulator_measure_many_cxx(qcs_simulator *sim, bit_num_t const *qubit_num_list,
                                    bit_num_t qubit_num_count, bit_t *results)
{
    std::vector<int> const qubits = qcs_validate_measure_many(sim, qubit_num_list, qubit_num_count, results);
    std::vector<bit_t> temporary_results(qubits.size());
    for (size_t i = 0; i < qubits.size(); ++i)
        temporary_results[i] = static_cast<bit_t>(sim->core->measure_qubit(qubits[i]));
    if (!temporary_results.empty())
        std::copy(temporary_results.begin(), temporary_results.end(), results);
}

void qcs_simulator_measure_many_to_clbits_cxx(
    qcs_simulator *sim, bit_num_t const *qubit_num_list, bit_num_t qubit_num_count,
    bit_num_t const *clbit_num_list, bit_num_t clbit_num_count, bit_t *results)
{
    std::vector<int> const qubits = qcs_validate_measure_many(sim, qubit_num_list, qubit_num_count, results);
    if (clbit_num_count < 0)
        throw std::runtime_error("classical bit count must be non-negative");
    if (clbit_num_count != qubit_num_count)
        throw std::runtime_error("qubit and classical bit counts must match");
    if (clbit_num_count > 0 && clbit_num_list == NULL)
        throw std::runtime_error("classical bit list must not be NULL when its count is positive");

    std::vector<bool> used(sim->num_clbits, false);
    for (int i = 0; i < clbit_num_count; ++i)
    {
        int const clbit_num = clbit_num_list[i];
        if (clbit_num < 0 || clbit_num >= sim->num_clbits)
            throw std::runtime_error(atlc::format("classical bit %d is out of range [0, %d)", clbit_num, sim->num_clbits));
        if (used[clbit_num])
            throw std::runtime_error(atlc::format("classical bit %d appears multiple times", clbit_num));
        used[clbit_num] = true;
    }

    std::vector<bit_t> temporary_results(qubits.size());
    for (size_t i = 0; i < qubits.size(); ++i)
        temporary_results[i] = static_cast<bit_t>(sim->core->measure_qubit(qubits[i]));
    for (size_t i = 0; i < qubits.size(); ++i)
        sim->clbits[clbit_num_list[i]] = temporary_results[i];
    if (!temporary_results.empty())
        std::copy(temporary_results.begin(), temporary_results.end(), results);
}
int qcs_simulator_read_cxx(qcs_simulator *sim, bit_num_t clbit_num) { return sim->clbits[clbit_num]; }
void qcs_simulator_reset_cxx(qcs_simulator *sim, bit_num_t qubit_num)
{
    if (qcs_simulator_measure_cxx(sim, qubit_num))
        qcs_simulator_gate_x_cxx(sim, &qubit_num, 1, NULL, 0, NULL, 0);
}
void qcs_simulator_set_zero_state_cxx(qcs_simulator *sim) { sim->core->initialize_zero(); }
void qcs_simulator_set_sequential_state_cxx(qcs_simulator *sim) { sim->core->initialize_sequential(); }
void qcs_simulator_set_flat_state_cxx(qcs_simulator *sim) { sim->core->initialize_flat(); }
void qcs_simulator_set_entangled_state_cxx(qcs_simulator *sim) { sim->core->initialize_entangled(); }
void qcs_simulator_set_random_state_cxx(qcs_simulator *sim) { sim->core->initialize_use_curand(); }

void qcs_simulator_gate_global_phase_cxx(qcs_simulator *sim, double theta, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::global_phase(theta), {}, negctrls, ctrls);
}

void qcs_simulator_gate_swap_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::swap(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_swap2_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::swap2(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_iswap_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::iswap(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_h_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::hadamard(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_x_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::x(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_y_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::y(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_z_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::z(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_s_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::s(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_sdg_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::sdg(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_t_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::t(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_tdg_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::tdg(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_sx_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::sx(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_sxdg_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::sxdg(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_rx_cxx(qcs_simulator *sim, double theta, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::rx(theta), targets, negctrls, ctrls);
}

void qcs_simulator_gate_ry_cxx(qcs_simulator *sim, double theta, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::ry(theta), targets, negctrls, ctrls);
}

void qcs_simulator_gate_rz_cxx(qcs_simulator *sim, double theta, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::rz(theta), targets, negctrls, ctrls);
}

void qcs_simulator_gate_u4_cxx(qcs_simulator *sim, double theta, double phi, double lambda, double gamma, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::u4(theta, phi, lambda, gamma), targets, negctrls, ctrls);
}

void qcs_simulator_gate_u3_cxx(qcs_simulator *sim, double theta, double phi, double lambda, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::u3(theta, phi, lambda), targets, negctrls, ctrls);
}

void qcs_simulator_gate_u2_cxx(qcs_simulator *sim, double phi, double lambda, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::u2(phi, lambda), targets, negctrls, ctrls);
}

void qcs_simulator_gate_u1_cxx(qcs_simulator *sim, double lambda, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::u1(lambda), targets, negctrls, ctrls);
}

void qcs_simulator_gate_u_cxx(qcs_simulator *sim, double theta, double phi, double lambda, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::u(theta, phi, lambda), targets, negctrls, ctrls);
}

void qcs_simulator_gate_p_cxx(qcs_simulator *sim, double theta, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::p(theta), targets, negctrls, ctrls);
}

void qcs_simulator_gate_id_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::id(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_r_cxx(qcs_simulator *sim, double theta, double phi, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::r(theta, phi), targets, negctrls, ctrls);
}

void qcs_simulator_gate_rzz_cxx(qcs_simulator *sim, double theta, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::rzz(theta), targets, negctrls, ctrls);
}

void qcs_simulator_gate_rxx_cxx(qcs_simulator *sim, double theta, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::rxx(theta), targets, negctrls, ctrls);
}

void qcs_simulator_gate_ryy_cxx(qcs_simulator *sim, double theta, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::ryy(theta), targets, negctrls, ctrls);
}

void qcs_simulator_gate_rzx_cxx(qcs_simulator *sim, double theta, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::rzx(theta), targets, negctrls, ctrls);
}

void qcs_simulator_gate_dcx_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::dcx(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_ecr_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::ecr(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_xx_plus_yy_cxx(qcs_simulator *sim, double theta, double beta, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::xx_plus_yy(theta, beta), targets, negctrls, ctrls);
}

void qcs_simulator_gate_xx_minus_yy_cxx(qcs_simulator *sim, double theta, double beta, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::xx_minus_yy(theta, beta), targets, negctrls, ctrls);
}

void qcs_simulator_gate_rccx_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::rccx(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_rcccx_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::rcccx(), targets, negctrls, ctrls);
}

void qcs_simulator_get_clbits_cxx(qcs_simulator const *sim, bit_t *clbits)
{
    for (int i = 0; i < sim->num_clbits; ++i)
        clbits[i] = sim->clbits[i] ? 1 : 0;
}
void qcs_simulator_reset_clbits_cxx(qcs_simulator *sim)
{
    for (int i = 0; i < (int)sim->clbits.size(); ++i)
        sim->clbits[i] = 0;
}
void qcs_simulator_reset_measurement_state_cxx(qcs_simulator *sim) { sim->core->clear_measurement_state(); }
void qcs_simulator_reinitialize_mapping_cxx(qcs_simulator *sim) { sim->core->reinitialize_mapping(); }
void qcs_simulator_get_clbits_string_cxx(qcs_simulator const *sim, char *clbits_string)
{
    int out = 0;
    for (auto it = sim->clbits.rbegin(); it != sim->clbits.rend(); ++it)
        clbits_string[out++] = *it ? '1' : '0';
    clbits_string[out] = '\0';
}
void qcs_simulator_save_statevector_cxx(qcs_simulator *sim, char const *outfn) { sim->core->save_statevector(outfn); }
int qcs_simulator_event_create_cxx(qcs_simulator *sim) { return sim->core->event_create(); }
void qcs_simulator_event_record_cxx(qcs_simulator *sim, int event_num) { sim->core->event_record(event_num); }
double qcs_simulator_event_get_elapsed_time_cxx(qcs_simulator *sim, event_num_t start_event_num, event_num_t stop_event_num) { return sim->core->event_get_elapsed_time(start_event_num, stop_event_num); }

extern "C" int qcs_simulator_create(qcs_simulator **result)
{
    return qcs_try_cxx([&]()
                       { qcs_write_result(result, qcs_simulator_create_cxx()); });
}

extern "C" int qcs_simulator_destroy(qcs_simulator *sim)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_destroy_cxx(sim); });
}

extern "C" int qcs_simulator_allocate_memory(qcs_simulator *sim)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_allocate_memory_cxx(sim); });
}

extern "C" int qcs_simulator_dispose(qcs_simulator *sim)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_dispose_cxx(sim); });
}

extern "C" int qcs_simulator_get_num_procs(qcs_simulator *sim, bit_t *result)
{
    return qcs_try_cxx([&]()
                       { qcs_write_result(result, qcs_simulator_get_num_procs_cxx(sim)); });
}

extern "C" int qcs_simulator_get_proc_num(qcs_simulator *sim, bit_t *result)
{
    return qcs_try_cxx([&]()
                       { qcs_write_result(result, qcs_simulator_get_proc_num_cxx(sim)); });
}

extern "C" int qcs_simulator_get_num_qubits(const qcs_simulator *sim, bit_t *result)
{
    return qcs_try_cxx([&]()
                       { qcs_write_result(result, qcs_simulator_get_num_qubits_cxx(sim)); });
}

extern "C" int qcs_simulator_get_num_clbits(const qcs_simulator *sim, bit_t *result)
{
    return qcs_try_cxx([&]()
                       { qcs_write_result(result, qcs_simulator_get_num_clbits_cxx(sim)); });
}

extern "C" int qcs_simulator_set_num_qubits(qcs_simulator *sim, bit_num_t num_qubits)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_set_num_qubits_cxx(sim, num_qubits); });
}

extern "C" int qcs_simulator_set_mapping(qcs_simulator *sim, const bit_num_t *perm_p2l, bit_num_t perm_p2l_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_set_mapping_cxx(sim, perm_p2l, perm_p2l_count); });
}

extern "C" int qcs_simulator_set_num_clbits(qcs_simulator *sim, bit_num_t num_clbits)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_set_num_clbits_cxx(sim, num_clbits); });
}

extern "C" int qcs_simulator_get_clbits(const qcs_simulator *sim, bit_t *clbits)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_get_clbits_cxx(sim, clbits); });
}

extern "C" int qcs_simulator_get_clbits_string(const qcs_simulator *sim, char *clbits_string)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_get_clbits_string_cxx(sim, clbits_string); });
}

extern "C" int qcs_simulator_reset(qcs_simulator *sim, bit_num_t qubit_num)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_reset_cxx(sim, qubit_num); });
}

extern "C" int qcs_simulator_set_zero_state(qcs_simulator *sim)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_set_zero_state_cxx(sim); });
}

extern "C" int qcs_simulator_set_sequential_state(qcs_simulator *sim)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_set_sequential_state_cxx(sim); });
}

extern "C" int qcs_simulator_set_flat_state(qcs_simulator *sim)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_set_flat_state_cxx(sim); });
}

extern "C" int qcs_simulator_set_entangled_state(qcs_simulator *sim)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_set_entangled_state_cxx(sim); });
}

extern "C" int qcs_simulator_set_random_state(qcs_simulator *sim)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_set_random_state_cxx(sim); });
}

extern "C" int qcs_simulator_reset_clbits(qcs_simulator *sim)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_reset_clbits_cxx(sim); });
}

extern "C" int qcs_simulator_reset_measurement_state(qcs_simulator *sim)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_reset_measurement_state_cxx(sim); });
}

extern "C" int qcs_simulator_reinitialize_mapping(qcs_simulator *sim)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_reinitialize_mapping_cxx(sim); });
}

extern "C" int qcs_simulator_gate_global_phase(qcs_simulator *sim, double theta, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_global_phase_cxx(sim, theta, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_h(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_h_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_x(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_x_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_y(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_y_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_z(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_z_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_s(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_s_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_sdg(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_sdg_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_t(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_t_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_tdg(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_tdg_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_sx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_sx_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_rx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_rx_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_ry(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_ry_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_rz(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_rz_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_u4(qcs_simulator *sim, double theta, double phi, double lambda, double gamma, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_u4_cxx(sim, theta, phi, lambda, gamma, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_u3(qcs_simulator *sim, double theta, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_u3_cxx(sim, theta, phi, lambda, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_u2(qcs_simulator *sim, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_u2_cxx(sim, phi, lambda, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_u1(qcs_simulator *sim, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_u1_cxx(sim, lambda, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_u(qcs_simulator *sim, double theta, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_u_cxx(sim, theta, phi, lambda, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_p(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_p_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_swap(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_swap_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_swap2(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_swap2_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_iswap(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_iswap_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_id(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_id_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_sxdg(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_sxdg_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_r(qcs_simulator *sim, double theta, double phi, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_r_cxx(sim, theta, phi, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_rxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_rxx_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_ryy(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_ryy_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_rzz(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_rzz_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_rzx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_rzx_cxx(sim, theta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_dcx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_dcx_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_ecr(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_ecr_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_xx_plus_yy(qcs_simulator *sim, double theta, double beta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_xx_plus_yy_cxx(sim, theta, beta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_xx_minus_yy(qcs_simulator *sim, double theta, double beta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_xx_minus_yy_cxx(sim, theta, beta, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_rccx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_rccx_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_rcccx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_rcccx_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_measure(qcs_simulator *sim, bit_num_t qubit_num, bit_t *result)
{
    return qcs_try_cxx([&]()
                       { qcs_write_result(result, qcs_simulator_measure_cxx(sim, qubit_num)); });
}

extern "C" int qcs_simulator_measure_to_clbit(qcs_simulator *sim, bit_num_t qubit_num, bit_num_t clbit_num, bit_t *result)
{
    return qcs_try_cxx([&]()
                       { qcs_write_result(result, qcs_simulator_measure_to_clbit_cxx(sim, qubit_num, clbit_num)); });
}

extern "C" int qcs_simulator_measure_many(qcs_simulator *sim, bit_num_t const *qubit_num_list, bit_num_t qubit_num_count, bit_t *results)
{
    return qcs_try_cxx([&]()
                       { qcs_simulator_measure_many_cxx(sim, qubit_num_list, qubit_num_count, results); });
}

extern "C" int qcs_simulator_measure_many_to_clbits(qcs_simulator *sim, bit_num_t const *qubit_num_list, bit_num_t qubit_num_count, bit_num_t const *clbit_num_list, bit_num_t clbit_num_count, bit_t *results)
{
    return qcs_try_cxx([&]()
                       { qcs_simulator_measure_many_to_clbits_cxx(sim, qubit_num_list, qubit_num_count, clbit_num_list, clbit_num_count, results); });
}

extern "C" int qcs_simulator_read(qcs_simulator *sim, bit_num_t clbit_num, bit_t *result)
{
    return qcs_try_cxx([&]()
                       { qcs_write_result(result, qcs_simulator_read_cxx(sim, clbit_num)); });
}

extern "C" int qcs_simulator_save_statevector(qcs_simulator *sim, const char *outfn)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_save_statevector_cxx(sim, outfn); });
}

extern "C" int qcs_simulator_event_create(qcs_simulator *sim, bit_t *result)
{
    return qcs_try_cxx([&]()
                       { qcs_write_result(result, qcs_simulator_event_create_cxx(sim)); });
}

extern "C" int qcs_simulator_event_record(qcs_simulator *sim, int event_num)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_event_record_cxx(sim, event_num); });
}

extern "C" int qcs_simulator_event_get_elapsed_time(qcs_simulator *sim, event_num_t start_event_num, event_num_t stop_event_num, double *result)
{
    return qcs_try_cxx([&]()
                       { qcs_write_result(result, qcs_simulator_event_get_elapsed_time_cxx(sim, start_event_num, stop_event_num)); });
}
