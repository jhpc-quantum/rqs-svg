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

    template <int num_target_qubits>
    static __device__ void add_target_qubit_indices(
        uint64_t *index_state,
        int const *target_qubit_num_list)
    {
        if constexpr (num_target_qubits > 0)
        {
            add_target_qubit_indices<num_target_qubits - 1>(index_state, target_qubit_num_list);

            uint64_t const target_mask = UINT64_C(1) << target_qubit_num_list[num_target_qubits - 1];
            constexpr int num_basis_indices = 1 << (num_target_qubits - 1);
            #pragma unroll num_basis_indices
            for (int basis_index = 0; basis_index < num_basis_indices; ++basis_index)
            {
                index_state[num_basis_indices + basis_index] =
                    index_state[basis_index] | target_mask;
            }
        }
    }

    template <int num_target_qubits>
    static __device__ void thread_num_to_state_index(
        uint64_t thread_num,
        uint64_t *index_state,
        int &is_measured_bits, int &measured_value_bits)
    {
        static_assert(num_target_qubits >= 1,
                      "num_target_qubits must be at least 1");

        thread_num_to_state_index_q0(thread_num, index_state[0]);

        auto args = (qcs::kernel_input_qnlist_struct const *)(void *)qcs::kernel_input_constant;
        int const *const target_qubit_num_list = args->get_target_qubit_num_list();

        // Add target bits in ascending basis-index order, with target 0 as LSB.
        add_target_qubit_indices<num_target_qubits>(index_state, target_qubit_num_list);

        is_measured_bits = args->is_measured_bits;
        measured_value_bits = args->measured_value_bits;

    } /* thread_num_to_state_index */

    namespace gate
    {
        // Pointer arrays use ascending basis indices, with target 0 as the least
        // significant bit. Inputs and outputs may alias; unchanged outputs are not written.

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
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                qcs::complex_t const s0_in_copy = *s_in[0];
                qcs::complex_t const s1_in_copy = *s_in[1];
                *s_out[0] = exp_i_gamma * (cos_theta_2 * s0_in_copy - sin_theta_2 * exp_i_lambda * s1_in_copy);
                *s_out[1] = exp_i_gamma * (exp_i_phi * (sin_theta_2 * s0_in_copy + exp_i_lambda * cos_theta_2 * s1_in_copy));
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
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                qcs::complex_t const s0_in_copy = *s_in[0];
                qcs::complex_t const s1_in_copy = *s_in[1];
                *s_out[0] = cos_theta_2 * s0_in_copy - sin_theta_2 * exp_i_lambda * s1_in_copy;
                *s_out[1] = exp_i_phi * sin_theta_2 * s0_in_copy + exp_i_phi_plus_lambda * cos_theta_2 * s1_in_copy;
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
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                qcs::complex_t const s0_in_copy = *s_in[0];
                qcs::complex_t const s1_in_copy = *s_in[1];
                *s_out[0] = M_SQRT1_2 * (s0_in_copy - exp_i_lambda * s1_in_copy);
                *s_out[1] = M_SQRT1_2 * (exp_i_phi * s0_in_copy + exp_i_phi_plus_lambda * s1_in_copy);
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
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                *s_out[0] = *s_in[0];
                *s_out[1] = exp_i_lambda * *s_in[1];
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
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                *s_out[0] = exp_i_theta * *s_in[0];
            }
        };

        struct hadamard
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                *s_out[0] = M_SQRT1_2 * (s0_in_copy + s1_in_copy);
                *s_out[1] = M_SQRT1_2 * (s0_in_copy - s1_in_copy);
            }
        };

        struct identity
        {
            static constexpr unsigned int num_target_qubits = 0;
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
            }
        };

        struct x
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                *s_out[0] = s1_in_copy;
                *s_out[1] = s0_in_copy;
            }
        };

        struct y
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                s_out[0]->real(s1_in_copy.imag());
                s_out[0]->imag(-s1_in_copy.real());
                s_out[1]->real(-s0_in_copy.imag());
                s_out[1]->imag(s0_in_copy.real());
            }
        };

        struct z
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                // auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                // *s_out[0] = s0_in_copy;
                *s_out[1] = -s1_in_copy;
            }
        };

        struct s
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s1_in_copy = *s_in[1];
                s_out[1]->real(-s1_in_copy.imag());
                s_out[1]->imag(s1_in_copy.real());
            }
        };

        struct sdg
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s1_in_copy = *s_in[1];
                s_out[1]->real(s1_in_copy.imag());
                s_out[1]->imag(-s1_in_copy.real());
            }
        };

        struct t
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s1_in_copy = *s_in[1];
                *s_out[1] = qcs::complex_t(M_SQRT1_2, M_SQRT1_2) * s1_in_copy;
            }
        };

        struct tdg
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s1_in_copy = *s_in[1];
                *s_out[1] = qcs::complex_t(M_SQRT1_2, -M_SQRT1_2) * s1_in_copy;
            }
        };

        struct sx
        {
            static constexpr unsigned int num_target_qubits = 1;
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                // auto const s0_in_copy = *s_in[0];
                // auto const s1_in_copy = *s_in[1];
                // *s_out[0] = qcs::complex_t(0.5, 0.5) * s0_in_copy + qcs::complex_t(0.5, - 0.5) * s1_in_copy;
                // *s_out[1] = qcs::complex_t(0.5, - 0.5) * s0_in_copy + qcs::complex_t(0.5, 0.5) * s1_in_copy;

                auto const a = 0.5 * (*s_in[0] + *s_in[1]);
                auto const b = 0.5 * multiply_i(*s_in[0] - *s_in[1]);
                *s_out[0] = a + b;
                *s_out[1] = a - b;
            }
        };

        struct sxdg
        {
            static constexpr unsigned int num_target_qubits = 1;

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const a = 0.5 * (*s_in[0] + *s_in[1]);
                auto const b = -0.5 * multiply_i(*s_in[0] - *s_in[1]);
                *s_out[0] = a + b;
                *s_out[1] = a - b;
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
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                *s_out[0] = cos_theta_2 * s0_in_copy + multiply_i_real(s1_in_copy, -sin_theta_2);
                *s_out[1] = multiply_i_real(s0_in_copy, -sin_theta_2) + cos_theta_2 * s1_in_copy;
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
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                *s_out[0] = cos_theta_2 * s0_in_copy - sin_theta_2 * s1_in_copy;
                *s_out[1] = sin_theta_2 * s0_in_copy + cos_theta_2 * s1_in_copy;
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
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                *s_out[0] = qcs::complex_t(cos_theta_2, -sin_theta_2) * *s_in[0];
                *s_out[1] = qcs::complex_t(cos_theta_2, sin_theta_2) * *s_in[1];
            }
        };

        struct swap
        {
            static constexpr unsigned int num_target_qubits = 2;
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];
                // *s_out[0] = *s_in[0];
                *s_out[1] = s10_in_copy;
                *s_out[2] = s01_in_copy;
                // *s_out[3] = *s_in[3];
            }
        };

        using swap1 = swap;

        struct swap2
        {
            static constexpr unsigned int num_target_qubits = 4;

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                // Swap target bits 0/1 and 2/3; inputs and outputs may alias.
                auto const s0001_in_copy = *s_in[1];
                auto const s0010_in_copy = *s_in[2];
                auto const s0100_in_copy = *s_in[4];
                auto const s0101_in_copy = *s_in[5];
                auto const s0110_in_copy = *s_in[6];
                auto const s0111_in_copy = *s_in[7];
                auto const s1000_in_copy = *s_in[8];
                auto const s1001_in_copy = *s_in[9];
                auto const s1010_in_copy = *s_in[10];
                auto const s1011_in_copy = *s_in[11];
                auto const s1101_in_copy = *s_in[13];
                auto const s1110_in_copy = *s_in[14];
                // *s_out[0] = *s_in[0];
                *s_out[1] = s0010_in_copy;
                *s_out[2] = s0001_in_copy;
                // *s_out[3] = *s_in[3];
                *s_out[4] = s1000_in_copy;
                *s_out[5] = s1010_in_copy;
                *s_out[6] = s1001_in_copy;
                *s_out[7] = s1011_in_copy;
                *s_out[8] = s0100_in_copy;
                *s_out[9] = s0110_in_copy;
                *s_out[10] = s0101_in_copy;
                *s_out[11] = s0111_in_copy;
                // *s_out[12] = *s_in[12];
                *s_out[13] = s1110_in_copy;
                *s_out[14] = s1101_in_copy;
                // *s_out[15] = *s_in[15];
            }
        };

        struct swap3
        {
            static constexpr unsigned int num_target_qubits = 6;

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                // Swap target bits 0/1, 2/3 and 4/5; inputs and outputs may alias.
                auto const s000001_in_copy = *s_in[1];
                auto const s000010_in_copy = *s_in[2];
                auto const s000100_in_copy = *s_in[4];
                auto const s000101_in_copy = *s_in[5];
                auto const s000110_in_copy = *s_in[6];
                auto const s000111_in_copy = *s_in[7];
                auto const s001000_in_copy = *s_in[8];
                auto const s001001_in_copy = *s_in[9];
                auto const s001010_in_copy = *s_in[10];
                auto const s001011_in_copy = *s_in[11];
                auto const s001101_in_copy = *s_in[13];
                auto const s001110_in_copy = *s_in[14];
                auto const s010000_in_copy = *s_in[16];
                auto const s010001_in_copy = *s_in[17];
                auto const s010010_in_copy = *s_in[18];
                auto const s010011_in_copy = *s_in[19];
                auto const s010100_in_copy = *s_in[20];
                auto const s010101_in_copy = *s_in[21];
                auto const s010110_in_copy = *s_in[22];
                auto const s010111_in_copy = *s_in[23];
                auto const s011000_in_copy = *s_in[24];
                auto const s011001_in_copy = *s_in[25];
                auto const s011010_in_copy = *s_in[26];
                auto const s011011_in_copy = *s_in[27];
                auto const s011100_in_copy = *s_in[28];
                auto const s011101_in_copy = *s_in[29];
                auto const s011110_in_copy = *s_in[30];
                auto const s011111_in_copy = *s_in[31];
                auto const s100000_in_copy = *s_in[32];
                auto const s100001_in_copy = *s_in[33];
                auto const s100010_in_copy = *s_in[34];
                auto const s100011_in_copy = *s_in[35];
                auto const s100100_in_copy = *s_in[36];
                auto const s100101_in_copy = *s_in[37];
                auto const s100110_in_copy = *s_in[38];
                auto const s100111_in_copy = *s_in[39];
                auto const s101000_in_copy = *s_in[40];
                auto const s101001_in_copy = *s_in[41];
                auto const s101010_in_copy = *s_in[42];
                auto const s101011_in_copy = *s_in[43];
                auto const s101100_in_copy = *s_in[44];
                auto const s101101_in_copy = *s_in[45];
                auto const s101110_in_copy = *s_in[46];
                auto const s101111_in_copy = *s_in[47];
                auto const s110001_in_copy = *s_in[49];
                auto const s110010_in_copy = *s_in[50];
                auto const s110100_in_copy = *s_in[52];
                auto const s110101_in_copy = *s_in[53];
                auto const s110110_in_copy = *s_in[54];
                auto const s110111_in_copy = *s_in[55];
                auto const s111000_in_copy = *s_in[56];
                auto const s111001_in_copy = *s_in[57];
                auto const s111010_in_copy = *s_in[58];
                auto const s111011_in_copy = *s_in[59];
                auto const s111101_in_copy = *s_in[61];
                auto const s111110_in_copy = *s_in[62];
                // *s_out[0] = *s_in[0];
                *s_out[1] = s000010_in_copy;
                *s_out[2] = s000001_in_copy;
                // *s_out[3] = *s_in[3];
                *s_out[4] = s001000_in_copy;
                *s_out[5] = s001010_in_copy;
                *s_out[6] = s001001_in_copy;
                *s_out[7] = s001011_in_copy;
                *s_out[8] = s000100_in_copy;
                *s_out[9] = s000110_in_copy;
                *s_out[10] = s000101_in_copy;
                *s_out[11] = s000111_in_copy;
                // *s_out[12] = *s_in[12];
                *s_out[13] = s001110_in_copy;
                *s_out[14] = s001101_in_copy;
                // *s_out[15] = *s_in[15];
                *s_out[16] = s100000_in_copy;
                *s_out[17] = s100010_in_copy;
                *s_out[18] = s100001_in_copy;
                *s_out[19] = s100011_in_copy;
                *s_out[20] = s101000_in_copy;
                *s_out[21] = s101010_in_copy;
                *s_out[22] = s101001_in_copy;
                *s_out[23] = s101011_in_copy;
                *s_out[24] = s100100_in_copy;
                *s_out[25] = s100110_in_copy;
                *s_out[26] = s100101_in_copy;
                *s_out[27] = s100111_in_copy;
                *s_out[28] = s101100_in_copy;
                *s_out[29] = s101110_in_copy;
                *s_out[30] = s101101_in_copy;
                *s_out[31] = s101111_in_copy;
                *s_out[32] = s010000_in_copy;
                *s_out[33] = s010010_in_copy;
                *s_out[34] = s010001_in_copy;
                *s_out[35] = s010011_in_copy;
                *s_out[36] = s011000_in_copy;
                *s_out[37] = s011010_in_copy;
                *s_out[38] = s011001_in_copy;
                *s_out[39] = s011011_in_copy;
                *s_out[40] = s010100_in_copy;
                *s_out[41] = s010110_in_copy;
                *s_out[42] = s010101_in_copy;
                *s_out[43] = s010111_in_copy;
                *s_out[44] = s011100_in_copy;
                *s_out[45] = s011110_in_copy;
                *s_out[46] = s011101_in_copy;
                *s_out[47] = s011111_in_copy;
                // *s_out[48] = *s_in[48];
                *s_out[49] = s110010_in_copy;
                *s_out[50] = s110001_in_copy;
                // *s_out[51] = *s_in[51];
                *s_out[52] = s111000_in_copy;
                *s_out[53] = s111010_in_copy;
                *s_out[54] = s111001_in_copy;
                *s_out[55] = s111011_in_copy;
                *s_out[56] = s110100_in_copy;
                *s_out[57] = s110110_in_copy;
                *s_out[58] = s110101_in_copy;
                *s_out[59] = s110111_in_copy;
                // *s_out[60] = *s_in[60];
                *s_out[61] = s111110_in_copy;
                *s_out[62] = s111101_in_copy;
                // *s_out[63] = *s_in[63];
            }
        };

        struct swap4
        {
            static constexpr unsigned int num_target_qubits = 8;

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                // Swap target bits 0/1, 2/3, 4/5 and 6/7; inputs and outputs may alias.
                auto const s00000001_in_copy = *s_in[1];
                auto const s00000010_in_copy = *s_in[2];
                auto const s00000100_in_copy = *s_in[4];
                auto const s00000101_in_copy = *s_in[5];
                auto const s00000110_in_copy = *s_in[6];
                auto const s00000111_in_copy = *s_in[7];
                auto const s00001000_in_copy = *s_in[8];
                auto const s00001001_in_copy = *s_in[9];
                auto const s00001010_in_copy = *s_in[10];
                auto const s00001011_in_copy = *s_in[11];
                auto const s00001101_in_copy = *s_in[13];
                auto const s00001110_in_copy = *s_in[14];
                auto const s00010000_in_copy = *s_in[16];
                auto const s00010001_in_copy = *s_in[17];
                auto const s00010010_in_copy = *s_in[18];
                auto const s00010011_in_copy = *s_in[19];
                auto const s00010100_in_copy = *s_in[20];
                auto const s00010101_in_copy = *s_in[21];
                auto const s00010110_in_copy = *s_in[22];
                auto const s00010111_in_copy = *s_in[23];
                auto const s00011000_in_copy = *s_in[24];
                auto const s00011001_in_copy = *s_in[25];
                auto const s00011010_in_copy = *s_in[26];
                auto const s00011011_in_copy = *s_in[27];
                auto const s00011100_in_copy = *s_in[28];
                auto const s00011101_in_copy = *s_in[29];
                auto const s00011110_in_copy = *s_in[30];
                auto const s00011111_in_copy = *s_in[31];
                auto const s00100000_in_copy = *s_in[32];
                auto const s00100001_in_copy = *s_in[33];
                auto const s00100010_in_copy = *s_in[34];
                auto const s00100011_in_copy = *s_in[35];
                auto const s00100100_in_copy = *s_in[36];
                auto const s00100101_in_copy = *s_in[37];
                auto const s00100110_in_copy = *s_in[38];
                auto const s00100111_in_copy = *s_in[39];
                auto const s00101000_in_copy = *s_in[40];
                auto const s00101001_in_copy = *s_in[41];
                auto const s00101010_in_copy = *s_in[42];
                auto const s00101011_in_copy = *s_in[43];
                auto const s00101100_in_copy = *s_in[44];
                auto const s00101101_in_copy = *s_in[45];
                auto const s00101110_in_copy = *s_in[46];
                auto const s00101111_in_copy = *s_in[47];
                auto const s00110001_in_copy = *s_in[49];
                auto const s00110010_in_copy = *s_in[50];
                auto const s00110100_in_copy = *s_in[52];
                auto const s00110101_in_copy = *s_in[53];
                auto const s00110110_in_copy = *s_in[54];
                auto const s00110111_in_copy = *s_in[55];
                auto const s00111000_in_copy = *s_in[56];
                auto const s00111001_in_copy = *s_in[57];
                auto const s00111010_in_copy = *s_in[58];
                auto const s00111011_in_copy = *s_in[59];
                auto const s00111101_in_copy = *s_in[61];
                auto const s00111110_in_copy = *s_in[62];
                auto const s01000000_in_copy = *s_in[64];
                auto const s01000001_in_copy = *s_in[65];
                auto const s01000010_in_copy = *s_in[66];
                auto const s01000011_in_copy = *s_in[67];
                auto const s01000100_in_copy = *s_in[68];
                auto const s01000101_in_copy = *s_in[69];
                auto const s01000110_in_copy = *s_in[70];
                auto const s01000111_in_copy = *s_in[71];
                auto const s01001000_in_copy = *s_in[72];
                auto const s01001001_in_copy = *s_in[73];
                auto const s01001010_in_copy = *s_in[74];
                auto const s01001011_in_copy = *s_in[75];
                auto const s01001100_in_copy = *s_in[76];
                auto const s01001101_in_copy = *s_in[77];
                auto const s01001110_in_copy = *s_in[78];
                auto const s01001111_in_copy = *s_in[79];
                auto const s01010000_in_copy = *s_in[80];
                auto const s01010001_in_copy = *s_in[81];
                auto const s01010010_in_copy = *s_in[82];
                auto const s01010011_in_copy = *s_in[83];
                auto const s01010100_in_copy = *s_in[84];
                auto const s01010101_in_copy = *s_in[85];
                auto const s01010110_in_copy = *s_in[86];
                auto const s01010111_in_copy = *s_in[87];
                auto const s01011000_in_copy = *s_in[88];
                auto const s01011001_in_copy = *s_in[89];
                auto const s01011010_in_copy = *s_in[90];
                auto const s01011011_in_copy = *s_in[91];
                auto const s01011100_in_copy = *s_in[92];
                auto const s01011101_in_copy = *s_in[93];
                auto const s01011110_in_copy = *s_in[94];
                auto const s01011111_in_copy = *s_in[95];
                auto const s01100000_in_copy = *s_in[96];
                auto const s01100001_in_copy = *s_in[97];
                auto const s01100010_in_copy = *s_in[98];
                auto const s01100011_in_copy = *s_in[99];
                auto const s01100100_in_copy = *s_in[100];
                auto const s01100101_in_copy = *s_in[101];
                auto const s01100110_in_copy = *s_in[102];
                auto const s01100111_in_copy = *s_in[103];
                auto const s01101000_in_copy = *s_in[104];
                auto const s01101001_in_copy = *s_in[105];
                auto const s01101010_in_copy = *s_in[106];
                auto const s01101011_in_copy = *s_in[107];
                auto const s01101100_in_copy = *s_in[108];
                auto const s01101101_in_copy = *s_in[109];
                auto const s01101110_in_copy = *s_in[110];
                auto const s01101111_in_copy = *s_in[111];
                auto const s01110000_in_copy = *s_in[112];
                auto const s01110001_in_copy = *s_in[113];
                auto const s01110010_in_copy = *s_in[114];
                auto const s01110011_in_copy = *s_in[115];
                auto const s01110100_in_copy = *s_in[116];
                auto const s01110101_in_copy = *s_in[117];
                auto const s01110110_in_copy = *s_in[118];
                auto const s01110111_in_copy = *s_in[119];
                auto const s01111000_in_copy = *s_in[120];
                auto const s01111001_in_copy = *s_in[121];
                auto const s01111010_in_copy = *s_in[122];
                auto const s01111011_in_copy = *s_in[123];
                auto const s01111100_in_copy = *s_in[124];
                auto const s01111101_in_copy = *s_in[125];
                auto const s01111110_in_copy = *s_in[126];
                auto const s01111111_in_copy = *s_in[127];
                auto const s10000000_in_copy = *s_in[128];
                auto const s10000001_in_copy = *s_in[129];
                auto const s10000010_in_copy = *s_in[130];
                auto const s10000011_in_copy = *s_in[131];
                auto const s10000100_in_copy = *s_in[132];
                auto const s10000101_in_copy = *s_in[133];
                auto const s10000110_in_copy = *s_in[134];
                auto const s10000111_in_copy = *s_in[135];
                auto const s10001000_in_copy = *s_in[136];
                auto const s10001001_in_copy = *s_in[137];
                auto const s10001010_in_copy = *s_in[138];
                auto const s10001011_in_copy = *s_in[139];
                auto const s10001100_in_copy = *s_in[140];
                auto const s10001101_in_copy = *s_in[141];
                auto const s10001110_in_copy = *s_in[142];
                auto const s10001111_in_copy = *s_in[143];
                auto const s10010000_in_copy = *s_in[144];
                auto const s10010001_in_copy = *s_in[145];
                auto const s10010010_in_copy = *s_in[146];
                auto const s10010011_in_copy = *s_in[147];
                auto const s10010100_in_copy = *s_in[148];
                auto const s10010101_in_copy = *s_in[149];
                auto const s10010110_in_copy = *s_in[150];
                auto const s10010111_in_copy = *s_in[151];
                auto const s10011000_in_copy = *s_in[152];
                auto const s10011001_in_copy = *s_in[153];
                auto const s10011010_in_copy = *s_in[154];
                auto const s10011011_in_copy = *s_in[155];
                auto const s10011100_in_copy = *s_in[156];
                auto const s10011101_in_copy = *s_in[157];
                auto const s10011110_in_copy = *s_in[158];
                auto const s10011111_in_copy = *s_in[159];
                auto const s10100000_in_copy = *s_in[160];
                auto const s10100001_in_copy = *s_in[161];
                auto const s10100010_in_copy = *s_in[162];
                auto const s10100011_in_copy = *s_in[163];
                auto const s10100100_in_copy = *s_in[164];
                auto const s10100101_in_copy = *s_in[165];
                auto const s10100110_in_copy = *s_in[166];
                auto const s10100111_in_copy = *s_in[167];
                auto const s10101000_in_copy = *s_in[168];
                auto const s10101001_in_copy = *s_in[169];
                auto const s10101010_in_copy = *s_in[170];
                auto const s10101011_in_copy = *s_in[171];
                auto const s10101100_in_copy = *s_in[172];
                auto const s10101101_in_copy = *s_in[173];
                auto const s10101110_in_copy = *s_in[174];
                auto const s10101111_in_copy = *s_in[175];
                auto const s10110000_in_copy = *s_in[176];
                auto const s10110001_in_copy = *s_in[177];
                auto const s10110010_in_copy = *s_in[178];
                auto const s10110011_in_copy = *s_in[179];
                auto const s10110100_in_copy = *s_in[180];
                auto const s10110101_in_copy = *s_in[181];
                auto const s10110110_in_copy = *s_in[182];
                auto const s10110111_in_copy = *s_in[183];
                auto const s10111000_in_copy = *s_in[184];
                auto const s10111001_in_copy = *s_in[185];
                auto const s10111010_in_copy = *s_in[186];
                auto const s10111011_in_copy = *s_in[187];
                auto const s10111100_in_copy = *s_in[188];
                auto const s10111101_in_copy = *s_in[189];
                auto const s10111110_in_copy = *s_in[190];
                auto const s10111111_in_copy = *s_in[191];
                auto const s11000001_in_copy = *s_in[193];
                auto const s11000010_in_copy = *s_in[194];
                auto const s11000100_in_copy = *s_in[196];
                auto const s11000101_in_copy = *s_in[197];
                auto const s11000110_in_copy = *s_in[198];
                auto const s11000111_in_copy = *s_in[199];
                auto const s11001000_in_copy = *s_in[200];
                auto const s11001001_in_copy = *s_in[201];
                auto const s11001010_in_copy = *s_in[202];
                auto const s11001011_in_copy = *s_in[203];
                auto const s11001101_in_copy = *s_in[205];
                auto const s11001110_in_copy = *s_in[206];
                auto const s11010000_in_copy = *s_in[208];
                auto const s11010001_in_copy = *s_in[209];
                auto const s11010010_in_copy = *s_in[210];
                auto const s11010011_in_copy = *s_in[211];
                auto const s11010100_in_copy = *s_in[212];
                auto const s11010101_in_copy = *s_in[213];
                auto const s11010110_in_copy = *s_in[214];
                auto const s11010111_in_copy = *s_in[215];
                auto const s11011000_in_copy = *s_in[216];
                auto const s11011001_in_copy = *s_in[217];
                auto const s11011010_in_copy = *s_in[218];
                auto const s11011011_in_copy = *s_in[219];
                auto const s11011100_in_copy = *s_in[220];
                auto const s11011101_in_copy = *s_in[221];
                auto const s11011110_in_copy = *s_in[222];
                auto const s11011111_in_copy = *s_in[223];
                auto const s11100000_in_copy = *s_in[224];
                auto const s11100001_in_copy = *s_in[225];
                auto const s11100010_in_copy = *s_in[226];
                auto const s11100011_in_copy = *s_in[227];
                auto const s11100100_in_copy = *s_in[228];
                auto const s11100101_in_copy = *s_in[229];
                auto const s11100110_in_copy = *s_in[230];
                auto const s11100111_in_copy = *s_in[231];
                auto const s11101000_in_copy = *s_in[232];
                auto const s11101001_in_copy = *s_in[233];
                auto const s11101010_in_copy = *s_in[234];
                auto const s11101011_in_copy = *s_in[235];
                auto const s11101100_in_copy = *s_in[236];
                auto const s11101101_in_copy = *s_in[237];
                auto const s11101110_in_copy = *s_in[238];
                auto const s11101111_in_copy = *s_in[239];
                auto const s11110001_in_copy = *s_in[241];
                auto const s11110010_in_copy = *s_in[242];
                auto const s11110100_in_copy = *s_in[244];
                auto const s11110101_in_copy = *s_in[245];
                auto const s11110110_in_copy = *s_in[246];
                auto const s11110111_in_copy = *s_in[247];
                auto const s11111000_in_copy = *s_in[248];
                auto const s11111001_in_copy = *s_in[249];
                auto const s11111010_in_copy = *s_in[250];
                auto const s11111011_in_copy = *s_in[251];
                auto const s11111101_in_copy = *s_in[253];
                auto const s11111110_in_copy = *s_in[254];
                // *s_out[0] = *s_in[0];
                *s_out[1] = s00000010_in_copy;
                *s_out[2] = s00000001_in_copy;
                // *s_out[3] = *s_in[3];
                *s_out[4] = s00001000_in_copy;
                *s_out[5] = s00001010_in_copy;
                *s_out[6] = s00001001_in_copy;
                *s_out[7] = s00001011_in_copy;
                *s_out[8] = s00000100_in_copy;
                *s_out[9] = s00000110_in_copy;
                *s_out[10] = s00000101_in_copy;
                *s_out[11] = s00000111_in_copy;
                // *s_out[12] = *s_in[12];
                *s_out[13] = s00001110_in_copy;
                *s_out[14] = s00001101_in_copy;
                // *s_out[15] = *s_in[15];
                *s_out[16] = s00100000_in_copy;
                *s_out[17] = s00100010_in_copy;
                *s_out[18] = s00100001_in_copy;
                *s_out[19] = s00100011_in_copy;
                *s_out[20] = s00101000_in_copy;
                *s_out[21] = s00101010_in_copy;
                *s_out[22] = s00101001_in_copy;
                *s_out[23] = s00101011_in_copy;
                *s_out[24] = s00100100_in_copy;
                *s_out[25] = s00100110_in_copy;
                *s_out[26] = s00100101_in_copy;
                *s_out[27] = s00100111_in_copy;
                *s_out[28] = s00101100_in_copy;
                *s_out[29] = s00101110_in_copy;
                *s_out[30] = s00101101_in_copy;
                *s_out[31] = s00101111_in_copy;
                *s_out[32] = s00010000_in_copy;
                *s_out[33] = s00010010_in_copy;
                *s_out[34] = s00010001_in_copy;
                *s_out[35] = s00010011_in_copy;
                *s_out[36] = s00011000_in_copy;
                *s_out[37] = s00011010_in_copy;
                *s_out[38] = s00011001_in_copy;
                *s_out[39] = s00011011_in_copy;
                *s_out[40] = s00010100_in_copy;
                *s_out[41] = s00010110_in_copy;
                *s_out[42] = s00010101_in_copy;
                *s_out[43] = s00010111_in_copy;
                *s_out[44] = s00011100_in_copy;
                *s_out[45] = s00011110_in_copy;
                *s_out[46] = s00011101_in_copy;
                *s_out[47] = s00011111_in_copy;
                // *s_out[48] = *s_in[48];
                *s_out[49] = s00110010_in_copy;
                *s_out[50] = s00110001_in_copy;
                // *s_out[51] = *s_in[51];
                *s_out[52] = s00111000_in_copy;
                *s_out[53] = s00111010_in_copy;
                *s_out[54] = s00111001_in_copy;
                *s_out[55] = s00111011_in_copy;
                *s_out[56] = s00110100_in_copy;
                *s_out[57] = s00110110_in_copy;
                *s_out[58] = s00110101_in_copy;
                *s_out[59] = s00110111_in_copy;
                // *s_out[60] = *s_in[60];
                *s_out[61] = s00111110_in_copy;
                *s_out[62] = s00111101_in_copy;
                // *s_out[63] = *s_in[63];
                *s_out[64] = s10000000_in_copy;
                *s_out[65] = s10000010_in_copy;
                *s_out[66] = s10000001_in_copy;
                *s_out[67] = s10000011_in_copy;
                *s_out[68] = s10001000_in_copy;
                *s_out[69] = s10001010_in_copy;
                *s_out[70] = s10001001_in_copy;
                *s_out[71] = s10001011_in_copy;
                *s_out[72] = s10000100_in_copy;
                *s_out[73] = s10000110_in_copy;
                *s_out[74] = s10000101_in_copy;
                *s_out[75] = s10000111_in_copy;
                *s_out[76] = s10001100_in_copy;
                *s_out[77] = s10001110_in_copy;
                *s_out[78] = s10001101_in_copy;
                *s_out[79] = s10001111_in_copy;
                *s_out[80] = s10100000_in_copy;
                *s_out[81] = s10100010_in_copy;
                *s_out[82] = s10100001_in_copy;
                *s_out[83] = s10100011_in_copy;
                *s_out[84] = s10101000_in_copy;
                *s_out[85] = s10101010_in_copy;
                *s_out[86] = s10101001_in_copy;
                *s_out[87] = s10101011_in_copy;
                *s_out[88] = s10100100_in_copy;
                *s_out[89] = s10100110_in_copy;
                *s_out[90] = s10100101_in_copy;
                *s_out[91] = s10100111_in_copy;
                *s_out[92] = s10101100_in_copy;
                *s_out[93] = s10101110_in_copy;
                *s_out[94] = s10101101_in_copy;
                *s_out[95] = s10101111_in_copy;
                *s_out[96] = s10010000_in_copy;
                *s_out[97] = s10010010_in_copy;
                *s_out[98] = s10010001_in_copy;
                *s_out[99] = s10010011_in_copy;
                *s_out[100] = s10011000_in_copy;
                *s_out[101] = s10011010_in_copy;
                *s_out[102] = s10011001_in_copy;
                *s_out[103] = s10011011_in_copy;
                *s_out[104] = s10010100_in_copy;
                *s_out[105] = s10010110_in_copy;
                *s_out[106] = s10010101_in_copy;
                *s_out[107] = s10010111_in_copy;
                *s_out[108] = s10011100_in_copy;
                *s_out[109] = s10011110_in_copy;
                *s_out[110] = s10011101_in_copy;
                *s_out[111] = s10011111_in_copy;
                *s_out[112] = s10110000_in_copy;
                *s_out[113] = s10110010_in_copy;
                *s_out[114] = s10110001_in_copy;
                *s_out[115] = s10110011_in_copy;
                *s_out[116] = s10111000_in_copy;
                *s_out[117] = s10111010_in_copy;
                *s_out[118] = s10111001_in_copy;
                *s_out[119] = s10111011_in_copy;
                *s_out[120] = s10110100_in_copy;
                *s_out[121] = s10110110_in_copy;
                *s_out[122] = s10110101_in_copy;
                *s_out[123] = s10110111_in_copy;
                *s_out[124] = s10111100_in_copy;
                *s_out[125] = s10111110_in_copy;
                *s_out[126] = s10111101_in_copy;
                *s_out[127] = s10111111_in_copy;
                *s_out[128] = s01000000_in_copy;
                *s_out[129] = s01000010_in_copy;
                *s_out[130] = s01000001_in_copy;
                *s_out[131] = s01000011_in_copy;
                *s_out[132] = s01001000_in_copy;
                *s_out[133] = s01001010_in_copy;
                *s_out[134] = s01001001_in_copy;
                *s_out[135] = s01001011_in_copy;
                *s_out[136] = s01000100_in_copy;
                *s_out[137] = s01000110_in_copy;
                *s_out[138] = s01000101_in_copy;
                *s_out[139] = s01000111_in_copy;
                *s_out[140] = s01001100_in_copy;
                *s_out[141] = s01001110_in_copy;
                *s_out[142] = s01001101_in_copy;
                *s_out[143] = s01001111_in_copy;
                *s_out[144] = s01100000_in_copy;
                *s_out[145] = s01100010_in_copy;
                *s_out[146] = s01100001_in_copy;
                *s_out[147] = s01100011_in_copy;
                *s_out[148] = s01101000_in_copy;
                *s_out[149] = s01101010_in_copy;
                *s_out[150] = s01101001_in_copy;
                *s_out[151] = s01101011_in_copy;
                *s_out[152] = s01100100_in_copy;
                *s_out[153] = s01100110_in_copy;
                *s_out[154] = s01100101_in_copy;
                *s_out[155] = s01100111_in_copy;
                *s_out[156] = s01101100_in_copy;
                *s_out[157] = s01101110_in_copy;
                *s_out[158] = s01101101_in_copy;
                *s_out[159] = s01101111_in_copy;
                *s_out[160] = s01010000_in_copy;
                *s_out[161] = s01010010_in_copy;
                *s_out[162] = s01010001_in_copy;
                *s_out[163] = s01010011_in_copy;
                *s_out[164] = s01011000_in_copy;
                *s_out[165] = s01011010_in_copy;
                *s_out[166] = s01011001_in_copy;
                *s_out[167] = s01011011_in_copy;
                *s_out[168] = s01010100_in_copy;
                *s_out[169] = s01010110_in_copy;
                *s_out[170] = s01010101_in_copy;
                *s_out[171] = s01010111_in_copy;
                *s_out[172] = s01011100_in_copy;
                *s_out[173] = s01011110_in_copy;
                *s_out[174] = s01011101_in_copy;
                *s_out[175] = s01011111_in_copy;
                *s_out[176] = s01110000_in_copy;
                *s_out[177] = s01110010_in_copy;
                *s_out[178] = s01110001_in_copy;
                *s_out[179] = s01110011_in_copy;
                *s_out[180] = s01111000_in_copy;
                *s_out[181] = s01111010_in_copy;
                *s_out[182] = s01111001_in_copy;
                *s_out[183] = s01111011_in_copy;
                *s_out[184] = s01110100_in_copy;
                *s_out[185] = s01110110_in_copy;
                *s_out[186] = s01110101_in_copy;
                *s_out[187] = s01110111_in_copy;
                *s_out[188] = s01111100_in_copy;
                *s_out[189] = s01111110_in_copy;
                *s_out[190] = s01111101_in_copy;
                *s_out[191] = s01111111_in_copy;
                // *s_out[192] = *s_in[192];
                *s_out[193] = s11000010_in_copy;
                *s_out[194] = s11000001_in_copy;
                // *s_out[195] = *s_in[195];
                *s_out[196] = s11001000_in_copy;
                *s_out[197] = s11001010_in_copy;
                *s_out[198] = s11001001_in_copy;
                *s_out[199] = s11001011_in_copy;
                *s_out[200] = s11000100_in_copy;
                *s_out[201] = s11000110_in_copy;
                *s_out[202] = s11000101_in_copy;
                *s_out[203] = s11000111_in_copy;
                // *s_out[204] = *s_in[204];
                *s_out[205] = s11001110_in_copy;
                *s_out[206] = s11001101_in_copy;
                // *s_out[207] = *s_in[207];
                *s_out[208] = s11100000_in_copy;
                *s_out[209] = s11100010_in_copy;
                *s_out[210] = s11100001_in_copy;
                *s_out[211] = s11100011_in_copy;
                *s_out[212] = s11101000_in_copy;
                *s_out[213] = s11101010_in_copy;
                *s_out[214] = s11101001_in_copy;
                *s_out[215] = s11101011_in_copy;
                *s_out[216] = s11100100_in_copy;
                *s_out[217] = s11100110_in_copy;
                *s_out[218] = s11100101_in_copy;
                *s_out[219] = s11100111_in_copy;
                *s_out[220] = s11101100_in_copy;
                *s_out[221] = s11101110_in_copy;
                *s_out[222] = s11101101_in_copy;
                *s_out[223] = s11101111_in_copy;
                *s_out[224] = s11010000_in_copy;
                *s_out[225] = s11010010_in_copy;
                *s_out[226] = s11010001_in_copy;
                *s_out[227] = s11010011_in_copy;
                *s_out[228] = s11011000_in_copy;
                *s_out[229] = s11011010_in_copy;
                *s_out[230] = s11011001_in_copy;
                *s_out[231] = s11011011_in_copy;
                *s_out[232] = s11010100_in_copy;
                *s_out[233] = s11010110_in_copy;
                *s_out[234] = s11010101_in_copy;
                *s_out[235] = s11010111_in_copy;
                *s_out[236] = s11011100_in_copy;
                *s_out[237] = s11011110_in_copy;
                *s_out[238] = s11011101_in_copy;
                *s_out[239] = s11011111_in_copy;
                // *s_out[240] = *s_in[240];
                *s_out[241] = s11110010_in_copy;
                *s_out[242] = s11110001_in_copy;
                // *s_out[243] = *s_in[243];
                *s_out[244] = s11111000_in_copy;
                *s_out[245] = s11111010_in_copy;
                *s_out[246] = s11111001_in_copy;
                *s_out[247] = s11111011_in_copy;
                *s_out[248] = s11110100_in_copy;
                *s_out[249] = s11110110_in_copy;
                *s_out[250] = s11110101_in_copy;
                *s_out[251] = s11110111_in_copy;
                // *s_out[252] = *s_in[252];
                *s_out[253] = s11111110_in_copy;
                *s_out[254] = s11111101_in_copy;
                // *s_out[255] = *s_in[255];
            }
        };

        struct iswap
        {
            static constexpr unsigned int num_target_qubits = 2;
            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];
                // *s_out[0] = *s_in[0];
                s_out[1]->real(-s10_in_copy.imag());
                s_out[1]->imag(s10_in_copy.real());
                s_out[2]->real(-s01_in_copy.imag());
                s_out[2]->imag(s01_in_copy.real());
                // *s_out[3] = *s_in[3];
            }
        };

        struct id
        {
            static constexpr unsigned int num_target_qubits = 1;

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                // *s_out[0] = *s_in[0];
                // *s_out[1] = *s_in[1];
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

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s0_in_copy = *s_in[0];
                auto const s1_in_copy = *s_in[1];
                *s_out[0] = cos_theta_2 * s0_in_copy + multiply_i_real(exp_minus_i_phi * s1_in_copy, -sin_theta_2);
                *s_out[1] = multiply_i_real(exp_i_phi * s0_in_copy, -sin_theta_2) + cos_theta_2 * s1_in_copy;
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

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                *s_out[0] = phase_minus * *s_in[0];
                *s_out[1] = phase_plus * *s_in[1];
                *s_out[2] = phase_plus * *s_in[2];
                *s_out[3] = phase_minus * *s_in[3];
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

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s00_in_copy = *s_in[0];
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];
                auto const s11_in_copy = *s_in[3];

                *s_out[0] = cos_theta_2 * s00_in_copy + multiply_i_real(s11_in_copy, m_sin_theta_2);
                *s_out[1] = cos_theta_2 * s01_in_copy + multiply_i_real(s10_in_copy, m_sin_theta_2);
                *s_out[2] = cos_theta_2 * s10_in_copy + multiply_i_real(s01_in_copy, m_sin_theta_2);
                *s_out[3] = cos_theta_2 * s11_in_copy + multiply_i_real(s00_in_copy, m_sin_theta_2);
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

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s00_in_copy = *s_in[0];
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];
                auto const s11_in_copy = *s_in[3];

                *s_out[0] = cos_theta_2 * s00_in_copy + multiply_i_real(s11_in_copy, sin_theta_2);
                *s_out[1] = cos_theta_2 * s01_in_copy + multiply_i_real(s10_in_copy, -sin_theta_2);
                *s_out[2] = cos_theta_2 * s10_in_copy + multiply_i_real(s01_in_copy, -sin_theta_2);
                *s_out[3] = cos_theta_2 * s11_in_copy + multiply_i_real(s00_in_copy, sin_theta_2);
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

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s00_in_copy = *s_in[0];
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];
                auto const s11_in_copy = *s_in[3];

                *s_out[0] = cos_theta_2 * s00_in_copy + multiply_i_real(s10_in_copy, -sin_theta_2);
                *s_out[1] = cos_theta_2 * s01_in_copy + multiply_i(s11_in_copy) * sin_theta_2;
                *s_out[2] = cos_theta_2 * s10_in_copy + multiply_i_real(s00_in_copy, -sin_theta_2);
                *s_out[3] = cos_theta_2 * s11_in_copy + multiply_i(s01_in_copy) * sin_theta_2;
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

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];

                // *s_out[0] = *s_in[0];
                *s_out[1] = cos_theta_2 * s01_in_copy + multiply_i_real(exp_minus_i_beta * s10_in_copy, -sin_theta_2);
                *s_out[2] = multiply_i_real(exp_i_beta * s01_in_copy, -sin_theta_2) + cos_theta_2 * s10_in_copy;
                // *s_out[3] = *s_in[3];
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

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s00_in_copy = *s_in[0];
                auto const s11_in_copy = *s_in[3];

                *s_out[0] = cos_theta_2 * s00_in_copy + multiply_i_real(exp_minus_i_beta * s11_in_copy, -sin_theta_2);
                // *s_out[1] = *s_in[1];
                // *s_out[2] = *s_in[2];
                *s_out[3] = multiply_i_real(exp_i_beta * s00_in_copy, -sin_theta_2) + cos_theta_2 * s11_in_copy;
            }
        };

        struct dcx
        {
            static constexpr unsigned int num_target_qubits = 2;

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s10_in_copy = *s_in[2];
                auto const s11_in_copy = *s_in[3];
                auto const s01_in_copy = *s_in[1];
                // *s_out[0] = *s_in[0];
                *s_out[1] = s11_in_copy;
                *s_out[2] = s01_in_copy;
                *s_out[3] = s10_in_copy;
            }
        };

        struct ecr
        {
            static constexpr unsigned int num_target_qubits = 2;

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s00_in_copy = *s_in[0];
                auto const s01_in_copy = *s_in[1];
                auto const s10_in_copy = *s_in[2];
                auto const s11_in_copy = *s_in[3];
                *s_out[0] = M_SQRT1_2 * (s01_in_copy + multiply_i(s11_in_copy));
                *s_out[1] = M_SQRT1_2 * (s00_in_copy + multiply_i_m(s10_in_copy));
                *s_out[2] = M_SQRT1_2 * (multiply_i(s01_in_copy) + s11_in_copy);
                *s_out[3] = M_SQRT1_2 * (multiply_i_m(s00_in_copy) + s10_in_copy);
            }
        };

        struct rccx
        {
            static constexpr unsigned int num_target_qubits = 3;

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s111_in_copy = *s_in[7];
                auto const s101_in_copy = *s_in[5];
                auto const s011_in_copy = *s_in[3];
                // *s_out[0] = *s_in[0];
                // *s_out[1] = *s_in[1];
                // *s_out[2] = *s_in[2];
                *s_out[3] = multiply_i_m(s111_in_copy);
                // *s_out[4] = *s_in[4];
                *s_out[5] = -s101_in_copy;
                // *s_out[6] = *s_in[6];
                *s_out[7] = multiply_i(s011_in_copy);
            }
        };

        struct rcccx
        {
            static constexpr unsigned int num_target_qubits = 4;

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                auto const s0011_in_copy = *s_in[3];
                auto const s1111_in_copy = *s_in[15];
                auto const s1011_in_copy = *s_in[11];
                auto const s0111_in_copy = *s_in[7];
                // *s_out[0] = *s_in[0];
                // *s_out[1] = *s_in[1];
                // *s_out[2] = *s_in[2];
                *s_out[3] = multiply_i(s0011_in_copy);
                // *s_out[4] = *s_in[4];
                // *s_out[5] = *s_in[5];
                // *s_out[6] = *s_in[6];
                *s_out[7] = s1111_in_copy;
                // *s_out[8] = *s_in[8];
                // *s_out[9] = *s_in[9];
                // *s_out[10] = *s_in[10];
                *s_out[11] = multiply_i_m(s1011_in_copy);
                // *s_out[12] = *s_in[12];
                // *s_out[13] = *s_in[13];
                // *s_out[14] = *s_in[14];
                *s_out[15] = -s0111_in_copy;
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

        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const s_in[] = {&state[index_state]};
        qcs::complex_t *const s_out[] = {&state[index_state]};
        gateobj.apply(s_in, s_out);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 1>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        uint64_t index_state[2];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index<1>(thread_num, index_state, is_measured_bits, measured_value_bits);
        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;
        qcs::complex_t const *s_in[2];
        qcs::complex_t *s_out[2];
        for (int i = 0; i < 2; ++i)
        {
            s_in[i] = (((i ^ measured_value_bits) & is_measured_bits) == 0) ? &state_in[index_state[i]] : &zero_constant;
            s_out[i] = &state[index_state[i]];
        }
        gateobj.apply(s_in, s_out);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 2>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        uint64_t index_state[4];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index<2>(thread_num, index_state, is_measured_bits, measured_value_bits);
        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;
        qcs::complex_t const *s_in[4];
        qcs::complex_t *s_out[4];
        for (int i = 0; i < 4; ++i)
        {
            s_in[i] = (((i ^ measured_value_bits) & is_measured_bits) == 0) ? &state_in[index_state[i]] : &zero_constant;
            s_out[i] = &state[index_state[i]];
        }
        gateobj.apply(s_in, s_out);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 3>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        uint64_t index_state[8];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index<3>(thread_num, index_state, is_measured_bits, measured_value_bits);
        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;
        qcs::complex_t const *s_in[8];
        qcs::complex_t *s_out[8];
        for (int i = 0; i < 8; ++i)
        {
            s_in[i] = (((i ^ measured_value_bits) & is_measured_bits) == 0) ? &state_in[index_state[i]] : &zero_constant;
            s_out[i] = &state[index_state[i]];
        }
        gateobj.apply(s_in, s_out);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 4>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        uint64_t index_state[16];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index<4>(thread_num, index_state, is_measured_bits, measured_value_bits);
        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;
        qcs::complex_t const *s_in[16];
        qcs::complex_t *s_out[16];
        for (int i = 0; i < 16; ++i)
        {
            s_in[i] = (((i ^ measured_value_bits) & is_measured_bits) == 0) ? &state_in[index_state[i]] : &zero_constant;
            s_out[i] = &state[index_state[i]];
        }
        gateobj.apply(s_in, s_out);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 5>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        uint64_t index_state[32];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index<5>(thread_num, index_state, is_measured_bits, measured_value_bits);
        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;
        qcs::complex_t const *s_in[32];
        qcs::complex_t *s_out[32];
        for (int i = 0; i < 32; ++i)
        {
            s_in[i] = (((i ^ measured_value_bits) & is_measured_bits) == 0) ? &state_in[index_state[i]] : &zero_constant;
            s_out[i] = &state[index_state[i]];
        }
        gateobj.apply(s_in, s_out);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 6>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        uint64_t index_state[64];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index<6>(thread_num, index_state, is_measured_bits, measured_value_bits);
        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;
        qcs::complex_t const *s_in[64];
        qcs::complex_t *s_out[64];
        for (int i = 0; i < 64; ++i)
        {
            s_in[i] = (((i ^ measured_value_bits) & is_measured_bits) == 0) ? &state_in[index_state[i]] : &zero_constant;
            s_out[i] = &state[index_state[i]];
        }
        gateobj.apply(s_in, s_out);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 7>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        uint64_t index_state[128];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index<7>(thread_num, index_state, is_measured_bits, measured_value_bits);
        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;
        qcs::complex_t const *s_in[128];
        qcs::complex_t *s_out[128];
        for (int i = 0; i < 128; ++i)
        {
            s_in[i] = (((i ^ measured_value_bits) & is_measured_bits) == 0) ? &state_in[index_state[i]] : &zero_constant;
            s_out[i] = &state[index_state[i]];
        }
        gateobj.apply(s_in, s_out);
    }

    template <typename GateType>
    __global__
        typename std::enable_if<GateType::num_target_qubits == 8>::type
        cuda_gate(GateType const gateobj)
    {
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        uint64_t index_state[256];
        int is_measured_bits, measured_value_bits;
        thread_num_to_state_index<8>(thread_num, index_state, is_measured_bits, measured_value_bits);
        qcs::complex_t *const state = qcs::kernel_common_constant.state_data_device;
        qcs::complex_t const *const state_in = state;
        qcs::complex_t const *s_in[256];
        qcs::complex_t *s_out[256];
        for (int i = 0; i < 256; ++i)
        {
            s_in[i] = (((i ^ measured_value_bits) & is_measured_bits) == 0) ? &state_in[index_state[i]] : &zero_constant;
            s_out[i] = &state[index_state[i]];
        }
        gateobj.apply(s_in, s_out);
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

                uint64_t index_state[2];
                int is_measured_bits, measured_value_bits;
                thread_num_to_state_index<1>(thread_num, index_state, is_measured_bits, measured_value_bits);

                // since target_qubit must be unmeasured, branching is not necessary.
                return qcs::float2_t{
                    cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state[0]]),
                    cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state[1]])};

                // return qcs::float2_t{
                //     (measured_state != 1)? cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state[0]]): 0,
                //     (measured_state != 0)? cuda::std::norm(qcs::kernel_common_constant.state_data_device[index_state[1]]): 0
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

void qcs_simulator_gate_swap1_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    qcs_simulator_gate_swap_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count);
}

void qcs_simulator_gate_swap2_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::swap2(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_swap3_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::swap3(), targets, negctrls, ctrls);
}

void qcs_simulator_gate_swap4_cxx(qcs_simulator *sim, bit_num_t const *target_qubit_num_list, bit_num_t target_qubit_num_count, bit_num_t const *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, bit_num_t const *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    auto const targets = qcs_vector_from_array(target_qubit_num_list, target_qubit_num_count);
    auto const negctrls = qcs_vector_from_array(negctrl_qubit_num_list, negctrl_qubit_num_count);
    auto const ctrls = qcs_vector_from_array(ctrl_qubit_num_list, ctrl_qubit_num_count);
    sim->core->operate_gate(qcs::gate::swap4(), targets, negctrls, ctrls);
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

extern "C" int qcs_simulator_gate_swap1(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_swap1_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_swap2(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_swap2_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_swap3(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_swap3_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
}

extern "C" int qcs_simulator_gate_swap4(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_gate_swap4_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count); });
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
