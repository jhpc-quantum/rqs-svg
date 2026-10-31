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
#include <array>
#include <thread>

#include <mpi.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda/std/complex>
#include <cuda/std/array>
#include <cub/cub.cuh>
#include <thrust/iterator/counting_iterator.h>
#include <thrust/iterator/transform_iterator.h>
#include <nccl.h>
#ifdef QCS_ENABLE_NVSHMEM
#include <nvshmem.h>
#include <nvshmemx.h>
#endif

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
    constexpr int max_measure_qubits = 4;
    constexpr int measure_group_size = max_measure_qubits;
    constexpr bool use_global_memory_for_constants = false;
    // Applies only to the NCCL implementation.
    constexpr bool exchange_pipeline_copy_back = true;

    constexpr unsigned int stream_sync_flags = cudaDeviceScheduleBlockingSync;
    static_assert(stream_sync_flags == cudaDeviceScheduleSpin ||
                  stream_sync_flags == cudaDeviceScheduleYield ||
                  stream_sync_flags == cudaDeviceScheduleAuto ||
                  stream_sync_flags == cudaDeviceScheduleBlockingSync,
                  "Unsupported CUDA stream synchronization setting");

    enum class event_sync_mode
    {
        default_sync,
        blocking_sync,
        busy_loop,
    };
    constexpr event_sync_mode event_sync = event_sync_mode::blocking_sync;
    constexpr bool event_busy_loop_yield = true;

    template <int N>
    using MeasureNorm = cuda::std::array<qcs::float_t, (1 << N)>;

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
    __device__ qcs::complex_t zero_global;

    __constant__ qcs::kernel_common_struct kernel_common_constant;
    __device__ qcs::kernel_common_struct kernel_common_global;

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
    alignas(kernel_input_qnlist_struct) __constant__ unsigned char kernel_input_constant[qcs::kernel_input_max_size];
    alignas(kernel_input_qnlist_struct) __device__ unsigned char kernel_input_global[qcs::kernel_input_max_size];

    // The host resolves the corresponding symbol with the same selector below.
    __device__ inline qcs::complex_t const *zero_data()
    {
        if constexpr (use_global_memory_for_constants)
            return &zero_global;
        else
            return &zero_constant;
    }

    __device__ inline qcs::kernel_common_struct const &kernel_common_data()
    {
        if constexpr (use_global_memory_for_constants)
            return kernel_common_global;
        else
            return kernel_common_constant;
    }

    __device__ inline qcs::kernel_input_qnlist_struct const *kernel_input_data()
    {
        if constexpr (use_global_memory_for_constants)
            return reinterpret_cast<qcs::kernel_input_qnlist_struct const *>(kernel_input_global);
        else
            return reinterpret_cast<qcs::kernel_input_qnlist_struct const *>(kernel_input_constant);
    }

    static __device__ void thread_num_to_state_index_q0(uint64_t thread_num, uint64_t &index_state)
    {
        auto args = kernel_input_data();

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

        auto args = kernel_input_data();
        int const *const target_qubit_num_list = args->get_target_qubit_num_list();

        // Add target bits in ascending basis-index order, with target 0 as LSB.
        add_target_qubit_indices<num_target_qubits>(index_state, target_qubit_num_list);

        is_measured_bits = args->is_measured_bits;
        measured_value_bits = args->measured_value_bits;

    } /* thread_num_to_state_index */

    namespace gate
    {
        constexpr unsigned int max_num_swaps = 4;

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

        template <unsigned int num_swaps>
        struct swap
        {
            static_assert(num_swaps > 0 && num_swaps <= max_num_swaps,
                          "num_swaps must be between 1 and max_num_swaps");
            static constexpr unsigned int num_target_qubits = 2 * num_swaps;

            __device__ void apply(qcs::complex_t const *const s_in[], qcs::complex_t *const s_out[]) const
            {
                constexpr uint64_t num_basis_indices = UINT64_C(1) << num_target_qubits;
                constexpr uint64_t even_bits = UINT64_C(0x5555555555555555);
                // Exchange target bits 0/1, 2/3, ... in each basis index.
                #pragma unroll
                for (uint64_t i = 0; i < num_basis_indices; ++i)
                {
                    uint64_t const j = ((i & even_bits) << 1) | ((i >> 1) & even_bits);
                    // Each nontrivial cycle has two entries. Read both before
                    // writing so inputs and outputs may alias; skip fixed points.
                    if (i < j)
                    {
                        auto const si_in_copy = *s_in[i];
                        auto const sj_in_copy = *s_in[j];
                        *s_out[i] = sj_in_copy;
                        *s_out[j] = si_in_copy;
                    }
                }
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

    template <typename GateType, int num_target_qubits = GateType::num_target_qubits>
    __global__ void cuda_gate(GateType const gateobj)
    {
        static_assert(num_target_qubits == GateType::num_target_qubits,
                      "num_target_qubits must match the gate");
        constexpr int num_basis_indices = 1 << num_target_qubits;
        uint64_t const thread_num = (uint64_t)threadIdx.x + (uint64_t)blockIdx.x * (uint64_t)blockDim.x;
        uint64_t index_state[num_basis_indices];
        int is_measured_bits = 0, measured_value_bits = 0;
        if constexpr (num_target_qubits == 0)
        {
            thread_num_to_state_index_q0(thread_num, index_state[0]);
        }
        else
        {
            thread_num_to_state_index<num_target_qubits>(thread_num, index_state, is_measured_bits, measured_value_bits);
        }

        qcs::complex_t *const state = kernel_common_data().state_data_device;
        qcs::complex_t const *const state_in = state;
        qcs::complex_t const *s_in[num_basis_indices];
        qcs::complex_t *s_out[num_basis_indices];
        #pragma unroll num_basis_indices
        for (int i = 0; i < num_basis_indices; ++i)
        {
            s_in[i] = (((i ^ measured_value_bits) & is_measured_bits) == 0) ? &state_in[index_state[i]] : zero_data();
            s_out[i] = &state[index_state[i]];
        }
        gateobj.apply(s_in, s_out);
    }

    namespace cubUtility
    {

        template <int N>
        struct MeasureNormAdd
        {
            __host__ __device__ MeasureNorm<N> operator()(const MeasureNorm<N> &a, const MeasureNorm<N> &b) const
            {
                MeasureNorm<N> result{};
                #pragma unroll
                for (int i = 0; i < (1 << N); ++i)
                    result[i] = a[i] + b[i];
                return result;
            }
        };

        // L counts only unmeasured local targets. All other output bits are
        // fixed by either the rank or a previously recorded measurement.
        template <int N, int L>
        struct IndirectLoad
        {
            static_assert(N >= 1 && N <= max_measure_qubits);
            static_assert(L >= 0 && L <= N);
            cuda::std::array<int, N> local_outcome_bits;
            uint64_t fixed_outcome;

            __device__ MeasureNorm<N> operator()(uint64_t thread_num) const
            {
                uint64_t index_state[1 << L];
                if constexpr (L == 0)
                {
                    thread_num_to_state_index_q0(thread_num, index_state[0]);
                }
                else
                {
                    int is_measured_bits, measured_value_bits;
                    thread_num_to_state_index<L>(thread_num, index_state, is_measured_bits, measured_value_bits);
                }

                MeasureNorm<N> result{};
                #pragma unroll
                for (int basis = 0; basis < (1 << L); ++basis)
                {
                    uint64_t outcome = fixed_outcome;
                    #pragma unroll
                    for (int bit = 0; bit < L; ++bit)
                        outcome |= uint64_t((basis >> bit) & 1) << local_outcome_bits[bit];
                    result[outcome] = cuda::std::norm(kernel_common_data().state_data_device[index_state[basis]]);
                }
                return result;
            }
        };

        template <int N>
        qcs::float_t measure_norm_sum(const MeasureNorm<N> &norm)
        {
            qcs::float_t total = 0;
            for (auto const value : norm)
            {
                if (!(value >= 0) || !std::isfinite(value))
                    throw std::runtime_error("invalid measurement probability component");
                total += value;
            }
            if (!(total > 0) || !std::isfinite(total))
                throw std::runtime_error("invalid measurement norm");
            return total;
        }

        // The caller validates the distribution and draws in [0, total).
        template <int N>
        uint64_t select_measure_outcome(const MeasureNorm<N> &norm, qcs::float_t draw)
        {
            qcs::float_t cumulative = 0;
            uint64_t last_positive = 0;
            for (int outcome = 0; outcome < (1 << N); ++outcome)
            {
                if (!(norm[outcome] > 0))
                    continue;
                last_positive = outcome;
                cumulative += norm[outcome];
                if (draw < cumulative)
                    return outcome;
            }
            // Rounding at the upper endpoint must never select a zero bin.
            return last_positive;
        }

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

#ifdef QCS_ENABLE_NVSHMEM
    static void check_nvshmem(int result, const char *operation)
    {
        if (result != 0) throw std::runtime_error(std::string(operation) + " failed: " + std::to_string(result));
    }

    // One resident block owns the staging slot and both signals.
    __global__ void exchange_registered_put(qcs::complex_t *state, qcs::complex_t *staging,
                                             uint64_t *signals, uint64_t offset,
                                             uint64_t length, uint64_t generation, int peer)
    {
        nvshmemx_putmem_signal_block(staging, state + offset, length * sizeof(qcs::complex_t),
                                     signals, generation, NVSHMEM_SIGNAL_SET, peer);
        __syncthreads();
        if (threadIdx.x == 0) nvshmem_signal_wait_until(signals, NVSHMEM_CMP_GE, generation);
        __syncthreads();
        for (uint64_t i = threadIdx.x; i < length; i += blockDim.x) state[offset + i] = staging[i];
        __syncthreads();
        if (threadIdx.x == 0)
        {
            nvshmemx_signal_op(signals + 1, generation, NVSHMEM_SIGNAL_SET, peer);
            nvshmem_quiet();
            nvshmem_signal_wait_until(signals + 1, NVSHMEM_CMP_GE, generation);
        }
        __syncthreads();
        __threadfence_system();
    }
#endif

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
        cudaStream_t exchange_copy_stream;
        cudaEvent_t exchange_transfer_done[2];
        cudaEvent_t exchange_copy_done[2];
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
#ifdef QCS_ENABLE_NVSHMEM
        uint64_t *exchange_signals = nullptr;
        uint64_t exchange_generation = 0;
#endif
        void *measure_norm_device = NULL;
        size_t measure_norm_device_size = 0;

        void *cub_temp_buffer_device;
        size_t cub_temp_buffer_device_size;

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
            ATLC_CHECK_CUDA(cudaSetDeviceFlags, stream_sync_flags);

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

#ifdef QCS_ENABLE_NVSHMEM
            {
                MPI_Comm comm = MPI_COMM_WORLD;
                nvshmemx_init_attr_t attr{};
                attr.mpi_comm = &comm;
                check_nvshmem(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr), "nvshmemx_init_attr");
                if (nvshmem_n_pes() != num_procs || nvshmem_my_pe() != proc_num)
                    throw std::runtime_error("NVSHMEM PE mapping differs from MPI ranks");
            }
#endif
            log_num_procs = atlc::log2_int(num_procs);

            log_block_size_max = 9;

            ATLC_CHECK_CUDA(cudaStreamCreate, &stream);
#ifndef QCS_ENABLE_NVSHMEM
            if constexpr (exchange_pipeline_copy_back)
            {
                ATLC_CHECK_CUDA(cudaStreamCreateWithFlags, &exchange_copy_stream, cudaStreamNonBlocking);
                for (int slot = 0; slot < 2; slot++)
                {
                    ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &exchange_transfer_done[slot], cudaEventDisableTiming);
                    ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &exchange_copy_done[slot], cudaEventDisableTiming);
                }
            }
#endif

            block_size_max = 1 << log_block_size_max;

            if constexpr (use_global_memory_for_constants)
            {
                ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void **)&qcs_kernel_common_constant_addr, qcs::kernel_common_global);
                ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void **)&qcs_kernel_input_constant_addr, qcs::kernel_input_global);
                ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void **)&zero_constant_addr, qcs::zero_global);
            }
            else
            {
                ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void **)&qcs_kernel_common_constant_addr, qcs::kernel_common_constant);
                ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void **)&qcs_kernel_input_constant_addr, qcs::kernel_input_constant);
                ATLC_CHECK_CUDA(cudaGetSymbolAddress, (void **)&zero_constant_addr, qcs::zero_constant);
            }

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

#ifdef QCS_ENABLE_NVSHMEM
            {
                if (use_unified_memory) throw std::runtime_error("NVSHMEM exchange does not support unified memory");
                ATLC_CHECK_CUDA(cudaMalloc, &state_data_device, num_states_local * sizeof(*state_data_device));
                check_nvshmem(nvshmemx_buffer_register(state_data_device, num_states_local * sizeof(*state_data_device)), "nvshmemx_buffer_register");
            }
#else
            if (use_unified_memory)
            {
                ATLC_CHECK_CUDA(cudaMallocManaged, &state_data_device, num_states_local * sizeof(*state_data_device));
            }
            else
            {
                ATLC_CHECK_CUDA(cudaMallocAsync, &state_data_device, num_states_local * sizeof(*state_data_device), stream);
            }
#endif
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
#ifdef QCS_ENABLE_NVSHMEM
            {
                // Bound the symmetric heap footprint and agree on the smallest rank's capacity.
                swap_buffer_total_length = std::min(swap_buffer_total_length, UINT64_C(262144));
                uint64_t common_length = 0;
                ATLC_CHECK_MPI(MPI_Allreduce, &swap_buffer_total_length, &common_length, 1, MPI_UINT64_T, MPI_MIN, MPI_COMM_WORLD);
                swap_buffer_total_length = common_length;
                swap_buffer = static_cast<qcs::complex_t *>(nvshmem_malloc(common_length * sizeof(qcs::complex_t)));
                exchange_signals = static_cast<uint64_t *>(nvshmem_malloc(2 * sizeof(uint64_t)));
                if (!swap_buffer || !exchange_signals) throw std::runtime_error("NVSHMEM symmetric allocation failed");
                ATLC_CHECK_CUDA(cudaMemsetAsync, exchange_signals, 0, 2 * sizeof(uint64_t), stream);
                exchange_generation = 0;
            }
#else
                ATLC_CHECK_CUDA(cudaMallocAsync, &swap_buffer, swap_buffer_total_length * sizeof(qcs::complex_t), stream);
#endif

            ATLC_CHECK_CUDA(cudaMallocAsync, &cub_temp_buffer_device, 1, stream);
            cub_temp_buffer_device_size = 1;
        }

        void free_memory()
        {
            if (cub_temp_buffer_device)
            {
                ATLC_CHECK_CUDA(cudaFreeAsync, cub_temp_buffer_device, stream);
                cub_temp_buffer_device = NULL;
            }
#ifdef QCS_ENABLE_NVSHMEM
            {
                if (state_data_device || swap_buffer || exchange_signals)
                {
                    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
                    ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
                    if (exchange_signals) { nvshmem_free(exchange_signals); exchange_signals = nullptr; }
                    if (swap_buffer) { nvshmem_free(swap_buffer); swap_buffer = nullptr; }
                    if (state_data_device)
                    {
                        check_nvshmem(nvshmemx_buffer_unregister(state_data_device), "nvshmemx_buffer_unregister");
                        ATLC_CHECK_CUDA(cudaFree, state_data_device);
                        state_data_device = nullptr;
                    }
                }
            }
#endif
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
                measure_norm_device_size = 0;
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

        void determine_qubit_exchange_mapping()
        {
            if (target_qubit_num_logical_list.size() > num_qubits_local)
            {
                throw std::runtime_error("number of target qubits exceeds number of local qubits");
            }

            target_qubit_num_physical_list.resize(target_qubit_num_logical_list.size());
            for (int tqni = 0; tqni < target_qubit_num_logical_list.size(); tqni++)
            {
                target_qubit_num_physical_list[tqni] = perm_l2p[target_qubit_num_logical_list[tqni]];
            }

            swap_target_global_list.resize(0);
            swap_target_local_list.resize(0);
            int swap_target_local = num_qubits_local - 1;
            for (int tqni = 0; tqni < target_qubit_num_physical_list.size(); tqni++)
            {
                auto const tqn_i = target_qubit_num_physical_list[tqni];
                if (tqn_i >= num_qubits_local)
                {
                    // Keep existing local targets local, including those later in the list.
                    while (swap_target_local >= 0 &&
                           std::find(target_qubit_num_physical_list.begin(), target_qubit_num_physical_list.end(), swap_target_local) != target_qubit_num_physical_list.end())
                    {
                        --swap_target_local;
                    }
                    if (swap_target_local < 0)
                    {
                        throw std::runtime_error("not enough non-target local qubits to localize targets");
                    }
                    swap_target_global_list.push_back(tqn_i);
                    swap_target_local_list.push_back(swap_target_local);
                    --swap_target_local;
                }
            }
            int const num_swap_qubits = swap_target_global_list.size();

            // Prepare all working storage before submitting any exchange.
            swap_target_local_logical_list.resize(num_swap_qubits);
            swap_target_global_logical_list.resize(num_swap_qubits);
            for (int target_num = 0; target_num < num_swap_qubits; target_num++)
            {
                swap_target_local_logical_list[target_num] = perm_p2l[swap_target_local_list[target_num]];
                swap_target_global_logical_list[target_num] = perm_p2l[swap_target_global_list[target_num]];
            }
        } /* determine_qubit_exchange_mapping */

#ifdef QCS_ENABLE_NVSHMEM
        void exchange_qubits_nvshmem()
        {
            int const count = swap_target_global_list.size();
            int const low = *std::min_element(swap_target_local_list.begin(), swap_target_local_list.end());
            uint64_t const area_length = UINT64_C(1) << low;
            uint64_t const areas = UINT64_C(1) << (num_qubits_local - low);
            uint64_t mask = 0;
            for (int i = 0; i < count; ++i) mask |= UINT64_C(1) << (swap_target_global_list[i] - num_qubits_local);
            uint64_t const chunk = std::min(area_length, swap_buffer_total_length);
            // Every PE visits the same XOR masks, including disjoint peer groups.
            for (uint64_t delta = 1; delta <= mask; ++delta)
            {
                if (delta & ~mask) continue;
                int peer = proc_num ^ delta;
                for (uint64_t area = 0; area < areas; ++area)
                {
                    uint64_t peer_bits = proc_num & ~mask;
                    for (int i = 0; i < count; ++i)
                        peer_bits |= ((area >> (swap_target_local_list[i] - low)) & 1)
                                     << (swap_target_global_list[i] - num_qubits_local);
                    if (peer_bits != static_cast<uint64_t>(peer)) continue;
                    for (uint64_t position = 0; position < area_length; position += chunk)
                    {
                        uint64_t offset = area * area_length + position;
                        uint64_t length = std::min(chunk, area_length - position);
                        if (exchange_generation == UINT64_MAX) throw std::runtime_error("NVSHMEM signal generation exhausted");
                        uint64_t generation = ++exchange_generation;
                        void *args[] = {&state_data_device, &swap_buffer, &exchange_signals, &offset, &length, &generation, &peer};
                        int capacity = 0;
                        check_nvshmem(nvshmemx_collective_launch_query_gridsize((const void *)exchange_registered_put, dim3(256), args, 0, &capacity), "NVSHMEM launch capacity query");
                        int common_capacity = 0;
                        ATLC_CHECK_MPI(MPI_Allreduce, &capacity, &common_capacity, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                        if (common_capacity < 1) throw std::runtime_error("NVSHMEM collective launch has no capacity on at least one PE");
                        check_nvshmem(nvshmemx_collective_launch((const void *)exchange_registered_put, dim3(1), dim3(256), args, 0, stream), "NVSHMEM collective launch");
                    }
                }
            }
        }
#endif

        void exchange_qubits()
        {
            int const num_swap_qubits = swap_target_global_list.size();
            if (num_swap_qubits == 0)
            {
                return;
            }

#ifdef QCS_ENABLE_NVSHMEM
            {
                exchange_qubits_nvshmem();
                return;
            }
#endif
            // b_min
            int const swap_target_local_min = *std::min_element(swap_target_local_list.data(), swap_target_local_list.data() + num_swap_qubits);

            uint64_t const local_buf_length = UINT64_C(1) << swap_target_local_min;
            uint64_t swap_buffer_length = swap_buffer_total_length;
            bool const use_pipeline = exchange_pipeline_copy_back && swap_buffer_total_length >= 2;
            uint64_t const slot_capacity = use_pipeline ? swap_buffer_total_length / 2 : swap_buffer_total_length;
            if (use_pipeline)
            {
                swap_buffer_length = slot_capacity;
            }
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
            bool slot_used[2] = {false, false};
            int next_slot = 0;
            int last_slot = -1;
            for (uint64_t local_num_self = 0; local_num_self < num_local_areas; local_num_self++)
            {

                // global_swap_peer = OR_i (the b_delta_i-th bit of local_num_self) placed at the a_delta_i-th bit
                uint64_t global_swap_peer = 0;
                for (int target_num = 0; target_num < num_swap_qubits; target_num++)
                {
                    // a_delta_i
                    int const swap_target_global_delta = swap_target_global_list[target_num] - num_qubits_local;
                    // b_delta_i
                    int const swap_target_local_delta = swap_target_local_list[target_num] - swap_target_local_min;
                    global_swap_peer |=
                        // The b_delta_i-th bit of local_num_self
                        ((local_num_self >> swap_target_local_delta) & 1)
                        // Place it at the a_delta_i-th bit
                        << swap_target_global_delta;
                }

                uint64_t const proc_num_peer = global_swap_peer | global_nonswap_self;

                // send & recv
                if (proc_num_peer == proc_num)
                {
                    continue;
                }

                bool is_peer_greater = proc_num_peer > proc_num;
                for (uint64_t buffer_pos = 0; buffer_pos < local_buf_length; )
                {
                    uint64_t const chunk_length = std::min(swap_buffer_length, local_buf_length - buffer_pos);
                    int const slot = use_pipeline ? next_slot : 0;
                    qcs::complex_t *const receive_buffer = swap_buffer + slot * slot_capacity;
                    qcs::complex_t *const state_region = &state_data_device[local_num_self * local_buf_length + buffer_pos];
                    if constexpr (exchange_pipeline_copy_back)
                    {
                        if (use_pipeline && slot_used[slot])
                        {
                            ATLC_CHECK_CUDA(cudaStreamWaitEvent, stream, exchange_copy_done[slot], 0);
                        }
                    }
                    ATLC_CHECK_NCCL(ncclGroupStart);
                    for (int send_recv = 0; send_recv < 2; send_recv++)
                    {
                        if (send_recv ^ is_peer_greater)
                        {
                            ATLC_CHECK_NCCL(ncclSend, state_region, chunk_length * 2 /* complex */, ncclDouble, proc_num_peer, nccl_comm, stream);
                        }
                        else
                        {
                            ATLC_CHECK_NCCL(ncclRecv, receive_buffer, chunk_length * 2 /* complex */, ncclDouble, proc_num_peer, nccl_comm, stream);
                        }
                    }
                    ATLC_CHECK_NCCL(ncclGroupEnd);
                    if constexpr (exchange_pipeline_copy_back)
                    {
                        if (use_pipeline)
                        {
                            // Both the send and receive must finish before overwriting the state region.
                            ATLC_CHECK_CUDA(cudaEventRecord, exchange_transfer_done[slot], stream);
                            ATLC_CHECK_CUDA(cudaStreamWaitEvent, exchange_copy_stream, exchange_transfer_done[slot], 0);
                            ATLC_CHECK_CUDA(cudaMemcpyAsync, state_region, receive_buffer, chunk_length * sizeof(qcs::complex_t), cudaMemcpyDeviceToDevice, exchange_copy_stream);
                            ATLC_CHECK_CUDA(cudaEventRecord, exchange_copy_done[slot], exchange_copy_stream);
                            slot_used[slot] = true;
                            last_slot = slot;
                            next_slot = 1 - slot;
                        }
                        else
                        {
                            ATLC_CHECK_CUDA(cudaMemcpyAsync, state_region, receive_buffer, chunk_length * sizeof(qcs::complex_t), cudaMemcpyDeviceToDevice, stream);
                        }
                    }
                    else
                    {
                        ATLC_CHECK_CUDA(cudaMemcpyAsync, state_region, receive_buffer, chunk_length * sizeof(qcs::complex_t), cudaMemcpyDeviceToDevice, stream);
                    }
                    buffer_pos += chunk_length;
                }
            }
#ifndef QCS_ENABLE_NVSHMEM
            if constexpr (exchange_pipeline_copy_back)
            {
                if (last_slot >= 0)
                {
                    // Copies share one stream, so the last event covers every copy in this exchange.
                    ATLC_CHECK_CUDA(cudaStreamWaitEvent, stream, exchange_copy_done[last_slot], 0);
                }
            }
#endif
        } /* exchange_qubits */

        void update_qubit_permutation_mapping()
        {
            int const num_swap_qubits = swap_target_global_list.size();
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

            for (int tqni = 0; tqni < target_qubit_num_logical_list.size(); tqni++)
            {
                target_qubit_num_physical_list[tqni] = perm_l2p[target_qubit_num_logical_list[tqni]];
            }
        } /* update_qubit_permutation_mapping */

        void localize_target_qubits()
        {
            determine_qubit_exchange_mapping();
            exchange_qubits();
            // exchange_qubits orders all copy-backs before subsequent work on stream.
            update_qubit_permutation_mapping();
        }; /* localize_target_qubits */

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

        template <int N, int L = N>
        MeasureNorm<N> reduce_measure_norm(
            int num_local_targets, const cuda::std::array<int, N> &local_outcome_bits,
            uint64_t fixed_outcome)
        {
            if (num_local_targets != L)
            {
                if constexpr (L > 0)
                    return reduce_measure_norm<N, L - 1>(num_local_targets, local_outcome_bits, fixed_outcome);
                throw std::runtime_error("invalid local measurement target count");
            }

            cubUtility::IndirectLoad<N, L> loader{local_outcome_bits, fixed_outcome};
            using CountingIter = thrust::counting_iterator<uint64_t>;
            using TransformIter = thrust::transform_iterator<decltype(loader), CountingIter>;
            TransformIter in_it(CountingIter(0), loader);

            if (measure_norm_device_size < sizeof(MeasureNorm<N>))
            {
                if (measure_norm_device)
                    ATLC_CHECK_CUDA(cudaFreeAsync, measure_norm_device, stream);
                measure_norm_device = NULL;
                measure_norm_device_size = 0;
                ATLC_CHECK_CUDA(cudaMallocAsync, &measure_norm_device, sizeof(MeasureNorm<N>), stream);
                measure_norm_device_size = sizeof(MeasureNorm<N>);
            }
            auto output = static_cast<MeasureNorm<N> *>(measure_norm_device);
            cubUtility::MeasureNormAdd<N> add;
            MeasureNorm<N> zero{};
            size_t temp_sz_required = 0;
            uint64_t const count = num_states_local >> num_operand_qubits;
            // CUB requirements depend on N, L and the number of active states.
            ATLC_CHECK_CUDA(cub::DeviceReduce::Reduce, NULL, temp_sz_required, in_it, output, count, add, zero, stream);
            if (cub_temp_buffer_device_size < temp_sz_required)
            {
                ATLC_CHECK_CUDA(cudaFreeAsync, cub_temp_buffer_device, stream);
                cub_temp_buffer_device = NULL;
                cub_temp_buffer_device_size = 0;
                ATLC_CHECK_CUDA(cudaMallocAsync, &cub_temp_buffer_device, temp_sz_required, stream);
                cub_temp_buffer_device_size = temp_sz_required;
            }
            ATLC_CHECK_CUDA(cub::DeviceReduce::Reduce, cub_temp_buffer_device, cub_temp_buffer_device_size, in_it, output, count, add, zero, stream);

            MeasureNorm<N> norm{};
            ATLC_CHECK_CUDA(cudaMemcpyAsync, &norm, output, sizeof(MeasureNorm<N>), cudaMemcpyDeviceToHost, stream);
            ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
            return norm;
        }

        template <int N>
        uint64_t measure_qubits(const std::array<int, N> &qubits)
        {
            static_assert(N >= 1 && N <= max_measure_qubits);
            uint64_t requested_mask = 0;
            for (int const qubit : qubits)
            {
                if (qubit < 0 || qubit >= num_qubits)
                    throw std::runtime_error(atlc::format("qubit %d is out of range [0, %d)", qubit, num_qubits));
                uint64_t const bit = UINT64_C(1) << qubit;
                if (requested_mask & bit)
                    throw std::runtime_error(atlc::format("qubit %d appears multiple times", qubit));
                requested_mask |= bit;
            }

            uint64_t fixed_outcome = 0;
            cuda::std::array<int, N> local_outcome_bits{};
            target_qubit_num_logical_list.clear();
            for (int i = 0; i < N; ++i)
            {
                int const qubit = qubits[i];
                int const physical = perm_l2p[qubit];
                if (measured_mask_logical & (UINT64_C(1) << qubit))
                {
                    fixed_outcome |= ((measured_value_logical >> qubit) & 1) << i;
                }
                else if (physical >= num_qubits_local)
                {
                    fixed_outcome |= ((uint64_t(proc_num) >> (physical - num_qubits_local)) & 1) << i;
                }
                else
                {
                    local_outcome_bits[target_qubit_num_logical_list.size()] = i;
                    target_qubit_num_logical_list.push_back(qubit);
                }
            }
            if ((requested_mask & ~measured_mask_logical) == 0)
                return fixed_outcome;

            // Already measured targets stay controls, so they are never counted
            // twice or loaded from the discarded branch of a pending projection.
            positive_control_qubit_num_logical_list = measured_1_qubit_num_logical_list;
            negative_control_qubit_num_logical_list = measured_0_qubit_num_logical_list;
            // Only local targets enter this path; it clears exchange work lists
            // without moving statevector data or changing the permutation.
            localize_target_qubits();
            check_control_qubit_num_physical();
            prepare_operating_gate();

            MeasureNorm<N> local_norm{};
            if (proc_num_control_condition)
                local_norm = reduce_measure_norm<N>(target_qubit_num_logical_list.size(), local_outcome_bits, fixed_outcome);

            MeasureNorm<N> global_norm{};
            static_assert(std::is_same<qcs::float_t, double>::value, "measurement MPI type must match float_t");
            ATLC_CHECK_MPI(MPI_Allreduce, local_norm.data(), global_norm.data(), 1 << N, MPI_DOUBLE, MPI_SUM, MPI_COMM_WORLD);
            // Validate on every rank before the broadcast, including each bin.
            qcs::float_t const total = cubUtility::measure_norm_sum<N>(global_norm);
            uint64_t outcome = 0;
            if (proc_num == 0)
            {
                std::uniform_real_distribution<qcs::float_t> distribution(0, total);
                outcome = cubUtility::select_measure_outcome<N>(global_norm, distribution(engine));
            }
            ATLC_CHECK_MPI(MPI_Bcast, &outcome, 1, MPI_UINT64_T, 0, MPI_COMM_WORLD);

            for (int i = 0; i < N; ++i)
            {
                int const qubit = qubits[i];
                uint64_t const bit = UINT64_C(1) << qubit;
                if (measured_mask_logical & bit)
                    continue;
                uint64_t const value = (outcome >> i) & 1;
                if (value)
                    measured_1_qubit_num_logical_list.push_back(qubit);
                else
                    measured_0_qubit_num_logical_list.push_back(qubit);
                measured_mask_logical |= bit;
                measured_value_logical = (measured_value_logical & ~bit) | (value << qubit);
                pending_projection_mask_logical |= bit;
            }
            return outcome;
        }

        int measure_qubit(int const qubit)
        {
            return static_cast<int>(measure_qubits<1>({qubit}));
        }

        template <size_t group_index>
        bool try_measure_qubit_group(const std::vector<int> &qubits, std::vector<bit_t> &results,
                                     size_t offset, size_t group_size)
        {
            constexpr int GroupSize = group_index + 1;
            if (group_size != GroupSize)
                return false;

            std::array<int, GroupSize> group;
            std::copy_n(qubits.begin() + offset, GroupSize, group.begin());
            uint64_t const outcome = measure_qubits<GroupSize>(group);
            for (int bit = 0; bit < GroupSize; ++bit)
                results[offset + bit] = static_cast<bit_t>((outcome >> bit) & 1);
            return true;
        }

        template <size_t... group_indices>
        bool try_measure_qubit_group(const std::vector<int> &qubits, std::vector<bit_t> &results,
                                     size_t offset, size_t group_size,
                                     std::index_sequence<group_indices...>)
        {
            return (try_measure_qubit_group<group_indices>(qubits, results, offset, group_size) || ...);
        }

        template <int GroupSize = measure_group_size>
        void measure_qubit_groups(const std::vector<int> &qubits, std::vector<bit_t> &results, size_t offset = 0)
        {
            static_assert(GroupSize >= 1 && GroupSize <= max_measure_qubits);
            for (size_t i = offset; i < qubits.size(); i += GroupSize)
            {
                size_t const group_size = std::min<size_t>(GroupSize, qubits.size() - i);
                if (!try_measure_qubit_group(qubits, results, i, group_size,
                                             std::make_index_sequence<GroupSize>{}))
                {
                    throw std::runtime_error("invalid measurement group size");
                }
            }
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

            localize_target_qubits();
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
#ifndef QCS_ENABLE_NVSHMEM
            if constexpr (exchange_pipeline_copy_back)
            {
                for (int slot = 0; slot < 2; slot++)
                {
                    ATLC_CHECK_CUDA(cudaEventDestroy, exchange_transfer_done[slot]);
                    ATLC_CHECK_CUDA(cudaEventDestroy, exchange_copy_done[slot]);
                }
                ATLC_CHECK_CUDA(cudaStreamDestroy, exchange_copy_stream);
            }
#endif
            ATLC_CHECK_CUDA(cudaStreamDestroy, stream);
#ifdef QCS_ENABLE_NVSHMEM
            nvshmem_finalize();
#endif
            MPI_Finalize();
        };

        int event_create()
        {
            int const event_num = event_list.size();
            cudaEvent_t event;
            constexpr unsigned int event_flags = event_sync == event_sync_mode::blocking_sync
                                                     ? cudaEventBlockingSync
                                                     : cudaEventDefault;
            ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &event, event_flags);
            this->event_list.push_back(event);
            return event_num;
        }

        void event_record(int const event_num)
        {
            cudaEvent_t const event = event_list[event_num];
            ATLC_CHECK_CUDA(cudaEventRecordWithFlags, event, stream, cudaEventDefault);
        }

        void event_wait(cudaEvent_t const event)
        {
            while (true)
            {
                cudaError_t const result = cudaEventQuery(event);
                if (result == cudaSuccess)
                    return;
                if (result != cudaErrorNotReady)
                    throw std::runtime_error(cudaGetErrorString(result));

                if constexpr (event_sync == event_sync_mode::busy_loop)
                {
                    if constexpr (event_busy_loop_yield)
                        std::this_thread::yield();
                }
                else
                {
                    ATLC_CHECK_CUDA(cudaEventSynchronize, event);
                    return;
                }
            }
        }

        double event_get_elapsed_time(int const start_event_num, int const stop_event_num)
        {
            cudaEvent_t const start = event_list[start_event_num];
            cudaEvent_t const stop = event_list[stop_event_num];

            event_wait(start);
            event_wait(stop);

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

template <size_t swap_index>
static bool qcs_simulator_try_operate_swap(
    qcs_simulator *sim, bit_num_t target_qubit_num_count,
    std::vector<int> const &targets,
    std::vector<int> const &negctrls,
    std::vector<int> const &ctrls)
{
    constexpr unsigned int num_swaps = swap_index + 1;
    if (target_qubit_num_count != 2 * num_swaps)
        return false;

    sim->core->operate_gate(qcs::gate::swap<num_swaps>(), targets, negctrls, ctrls);
    return true;
}

template <size_t... swap_indices>
static bool qcs_simulator_try_operate_swap(
    qcs_simulator *sim, bit_num_t target_qubit_num_count,
    std::vector<int> const &targets,
    std::vector<int> const &negctrls,
    std::vector<int> const &ctrls,
    std::index_sequence<swap_indices...>)
{
    return (qcs_simulator_try_operate_swap<swap_indices>(
                sim, target_qubit_num_count, targets, negctrls, ctrls) || ...);
}

static void qcs_simulator_operate_swap(qcs_simulator *sim, bit_num_t target_qubit_num_count,
                                       std::vector<int> const &targets,
                                       std::vector<int> const &negctrls,
                                       std::vector<int> const &ctrls)
{
    if (!qcs_simulator_try_operate_swap(
            sim, target_qubit_num_count, targets, negctrls, ctrls,
            std::make_index_sequence<qcs::gate::max_num_swaps>{}))
    {
        throw std::runtime_error("SWAP target count must be an even number from 2 through 2 * max_num_swaps");
    }
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
    sim->core->measure_qubit_groups(qubits, temporary_results);
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
    sim->core->measure_qubit_groups(qubits, temporary_results);
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
    qcs_simulator_operate_swap(sim, target_qubit_num_count, targets, negctrls, ctrls);
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
                {
                    if (target_qubit_num_count != 2)
                        throw std::runtime_error("SWAP1 target count must be 2");
                    qcs_simulator_gate_swap_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count);
                });
}

extern "C" int qcs_simulator_gate_swap2(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                {
                    if (target_qubit_num_count != 4)
                        throw std::runtime_error("SWAP2 target count must be 4");
                    qcs_simulator_gate_swap_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count);
                });
}

extern "C" int qcs_simulator_gate_swap3(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                {
                    if (target_qubit_num_count != 6)
                        throw std::runtime_error("SWAP3 target count must be 6");
                    qcs_simulator_gate_swap_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count);
                });
}

extern "C" int qcs_simulator_gate_swap4(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count)
{
    return qcs_try_cxx([&]()
                {
                    if (target_qubit_num_count != 8)
                        throw std::runtime_error("SWAP4 target count must be 8");
                    qcs_simulator_gate_swap_cxx(sim, target_qubit_num_list, target_qubit_num_count, negctrl_qubit_num_list, negctrl_qubit_num_count, ctrl_qubit_num_list, ctrl_qubit_num_count);
                });
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
