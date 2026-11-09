#include "measurement_profile.hpp"
#include "measurement_backend.hpp"
#include "measurement_tiled_cdf.cuh"
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
#include <memory>
#include <limits>

#include <mpi.h>
#include <cuda_runtime.h>
#include <curand.h>
#include <cuda/std/complex>
#include <cuda/std/array>
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
#include "measurement_reduce.cuh"
#include "pauli_reduce.cuh"
#include "work_buffer.cuh"
#include "swap_launch.hpp"

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
#ifdef QCS_ENABLE_NVSHMEM
    constexpr bool exchange_nvshmem_registered_put = true;
#else
    constexpr bool exchange_nvshmem_registered_put = false;
#endif

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

    static_assert(std::is_same<float_t, measurement_detail::real_type>::value);
    static_assert(std::is_same<complex_t, measurement_detail::state_type>::value);

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

    __global__ void cuda_swap(uint64_t num_threads, uint64_t ordinal_offset)
    {
        uint64_t ordinal = (uint64_t)blockIdx.x * blockDim.x + threadIdx.x;
        if (ordinal >= num_threads) return;
        ordinal += ordinal_offset;
        auto const *const args = kernel_input_data();
        int const num_targets = args->num_target_qubits;
        int const *const targets = args->get_target_qubit_num_list();
        qcs::complex_t *const state = kernel_common_data().state_data_device;
        // High ordinal bits enumerate non-operand qubits; controls are fixed
        // by the shared index helper. Low bits enumerate the target states.
        uint64_t index;
        thread_num_to_state_index_q0(ordinal >> num_targets, index);
        uint64_t partner = index;
        #pragma unroll 1
        for (int t = 0; t < num_targets; t += 2)
        {
            uint64_t const a = (ordinal >> t) & 1;
            uint64_t const b = (ordinal >> (t + 1)) & 1;
            index |= (a << targets[t]) | (b << targets[t + 1]);
            partner |= (b << targets[t]) | (a << targets[t + 1]);
        }
        // Disjoint SWAPs form an involution. Only the lower index owns each
        // two-element cycle; fixed points never access amplitude memory.
        if (index < partner)
        {
            auto const value = state[index];
            auto const partner_value = state[partner];
            state[index] = partner_value;
            state[partner] = value;
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

    static_assert(sizeof(qcs::complex_t) == 16, "NVSHMEM exchange requires 16-byte complex values");
    constexpr int exchange_threads = 32;
    constexpr int exchange_blocks_per_sm = 16;
    constexpr uint64_t exchange_chunk_bytes = UINT64_C(512) << 10;
    constexpr uint64_t exchange_chunk_elements = exchange_chunk_bytes / sizeof(qcs::complex_t);

    struct exchange_phase
    {
        uint64_t area_length, chunks_per_area, total_chunks;
        uint64_t fixed_mask, fixed_value, generation;
        int area_bits, peer;
    };

    __global__ void exchange_registered_put(qcs::complex_t *state, qcs::complex_t *staging,
                                            uint64_t *signals, exchange_phase phase)
    {
        qcs::complex_t *slot = staging + uint64_t(blockIdx.x) * exchange_chunk_elements;
        uint64_t *ready = signals + blockIdx.x;
        uint64_t *consumed = signals + gridDim.x + blockIdx.x;
        uint64_t round = 0;
        for (uint64_t task = blockIdx.x; task < phase.total_chunks; task += gridDim.x, ++round)
        {
            uint64_t ordinal = task / phase.chunks_per_area;
            uint64_t area = phase.fixed_value;
            // Deposit the common ordinal into the non-swapped local bits.
            for (int bit = 0; bit < phase.area_bits; ++bit)
            {
                uint64_t const flag = UINT64_C(1) << bit;
                if (!(phase.fixed_mask & flag))
                {
                    if (ordinal & 1) area |= flag;
                    ordinal >>= 1;
                }
            }
            uint64_t const position = (task % phase.chunks_per_area) * exchange_chunk_elements;
            uint64_t const remaining = phase.area_length - position;
            uint64_t const length = remaining < exchange_chunk_elements ? remaining : exchange_chunk_elements;
            uint64_t const offset = area * phase.area_length + position;
            uint64_t const value = phase.generation + round + 1;
            nvshmemx_putmem_signal_block(slot, state + offset, length * sizeof(qcs::complex_t),
                                         ready, value, NVSHMEM_SIGNAL_SET, phase.peer);
            __syncthreads();
            if (threadIdx.x == 0) nvshmem_signal_wait_until(ready, NVSHMEM_CMP_GE, value);
            __syncthreads();
            for (uint64_t i = threadIdx.x; i < length; i += blockDim.x) state[offset + i] = slot[i];
            // Reference notify_and_wait: drain the SET before waiting for its peer.
            __syncthreads();
            if (threadIdx.x == 0)
            {
                nvshmemx_signal_op(consumed, value, NVSHMEM_SIGNAL_SET, phase.peer);
                nvshmem_quiet();
                nvshmem_signal_wait_until(consumed, NVSHMEM_CMP_GE, value);
            }
            __syncthreads();
        }
        __threadfence_system();
    }

#endif

    namespace rank_sampling_detail
    {
        constexpr uint64_t chunk_size = 4096;

        // Lazy, simulator-owned storage; calls are serialized on the owner stream.
        struct workspace
        {
            double* device = nullptr;
            size_t capacity = 0;
            std::vector<double> weights;
            std::vector<double> exchange, masses;

            workspace() = default;
            workspace(const workspace&) = delete;
            workspace& operator=(const workspace&) = delete;

            void reserve(size_t chunks, int ranks)
            {
                weights.resize(chunks + 1);
                exchange.resize(size_t(ranks) * 2);
                masses.resize(ranks);
                if (capacity >= chunks + 2) return;
                double* replacement = nullptr;
                ATLC_CHECK_CUDA(cudaMalloc, &replacement, (chunks + 2) * sizeof(double));
                if (device) cudaFree(device);
                device = replacement;
                capacity = chunks + 2;
            }
            void close()
            {
                if (device) { ATLC_CHECK_CUDA(cudaFree, device); device = nullptr; }
                capacity = 0;
            }
            ~workspace() { if (device) cudaFree(device); }
        };

        __global__ void tiled_uniform(tiled_cdf_detail::rank_summary* summary,
                                      uint64_t seed, uint64_t counter)
        {
            auto bits = measurement_detail::measurement_random(seed, counter);
            summary->uniform = double((uint64_t(bits.x) << 21) |
                                      (uint64_t(bits.y) >> 11)) * 0x1.0p-53;
        }

        // One block per contiguous chunk. Storage scales with state size, never
        // with the number of requested outcomes. No state is projected here.
        __global__ void chunk_weights(const complex_t* state, uint64_t count,
                                     uint64_t mask, uint64_t value, bool active,
                                     double* weights, double* uniform,
                                     uint64_t seed, uint64_t counter)
        {
            __shared__ double partial[8];
            if (blockIdx.x == 0 && threadIdx.x == 0)
            {
                auto bits = measurement_detail::measurement_random(seed, counter);
                *uniform = double((uint64_t(bits.x) << 21) | (uint64_t(bits.y) >> 11)) * 0x1.0p-53;
            }
            uint64_t chunks = (count + chunk_size - 1) / chunk_size;
            for (uint64_t chunk = blockIdx.x; chunk < chunks; chunk += gridDim.x)
            {
                double sum = 0;
                uint64_t end = (chunk + 1) * chunk_size;
                if (end > count) end = count;
                if (active)
                    for (uint64_t i = chunk * chunk_size + threadIdx.x; i < end; i += blockDim.x)
                        if ((i & mask) == value)
                        {
                            auto a = state[i];
                            sum += a.real() * a.real() + a.imag() * a.imag();
                        }
                // One warp shuffle reduction and eight shared partials replace
                // the full-block shared-memory reduction tree.
                unsigned lane = threadIdx.x % 32, warp = threadIdx.x / 32;
                sum = measurement_detail::warp_sum(sum);
                if (lane == 0) partial[warp] = sum;
                __syncthreads();
                double total = 0;
                if (warp == 0) {
                    total = lane < 8 ? partial[lane] : 0.0;
                    total = measurement_detail::warp_sum(total);
                }
                if (threadIdx.x == 0) weights[chunk] = total;
                __syncthreads();
            }
        }

        __global__ void sample_chunk(const complex_t* state, uint64_t count,
                                     uint64_t chunk, uint64_t mask, uint64_t value,
                                     double fraction, uint64_t* result)
        {
            uint64_t begin = chunk * chunk_size, end = begin + chunk_size;
            if (end > count) end = count;
            // Use the same serial accumulation for total and CDF so rounding in
            // the parallel chunk reduction cannot select an impossible basis.
            double total = 0;
            for (uint64_t i = begin; i < end; ++i)
                if ((i & mask) == value) {
                    auto a = state[i];
                    total += a.real() * a.real() + a.imag() * a.imag();
                }
            *result = UINT64_MAX;
            if (!isfinite(total) || !(total > 0)) return;
            double threshold = fraction * total;
            if (!(threshold < total)) threshold = nextafter(total, 0.0);
            double cumulative = 0;
            for (uint64_t i = begin; i < end; ++i)
                if ((i & mask) == value) {
                    auto a = state[i];
                    double w = a.real() * a.real() + a.imag() * a.imag();
                    if (w > 0) {
                        *result = i;
                        cumulative += w;
                        if (threshold < cumulative) return;
                    }
                }
        }
    }

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
        ncclComm_t nccl_comm = nullptr;
        void* state_registration = nullptr;
        void* scratch_registration = nullptr;
        double allocation_seconds = 0;
        double registration_seconds = 0;
        int nccl_rank;

        std::vector<int> perm_p2l;
        std::vector<int> perm_l2p;
        std::vector<int> initial_perm_p2l;

        qcs_initial_state initial_state = QCS_INITIAL_STATE_ZERO;
        int num_samples;
        unsigned int rng_seed;
        // Simulator-lifetime RNG cursor; state/bookkeeping/shot resets leave it
        // untouched. Exhaustion is separate so UINT64_MAX can be sampled once.
        uint64_t measurement_counter = 0;
        bool measurement_counter_exhausted = false;
        bool rank_sampled_measurement = false;
        bool tiled_cdf_measurement = false;
        bool auto_measurement = false;
        rank_sampling_detail::workspace rank_sampling_workspace;
        std::vector<tiled_cdf_detail::rank_summary> tiled_rank_summaries;
        // Byte collectives preserve host long-double precision without requiring
        // an NCCL arithmetic datatype. Ranks must use the same host ABI.
        unsigned char* measurement_collective_send = nullptr;
        unsigned char* measurement_collective_receive = nullptr;
        static constexpr size_t measurement_collective_bytes =
            sizeof(double) * 2 > sizeof(tiled_cdf_detail::rank_summary)
            ? sizeof(double) * 2 : sizeof(tiled_cdf_detail::rank_summary);
        std::unique_ptr<tiled_cdf_detail::workspace> tiled_workspace;

        int log_num_procs;
        int max_num_qubits_local_device;
        uint64_t initial_free_memory_bytes;
        int log_block_size_max;
        int block_size_max;
        swap_detail::launch_limits swap_launch_config{1, 1, 1, 1, 1};
        int target_qubit_num_begin;
        int target_qubit_num_end;

        cudaStream_t stream{};
        cudaStream_t exchange_copy_stream{};
        cudaEvent_t exchange_transfer_done[2]{};
        cudaEvent_t exchange_copy_done[2]{};
        std::vector<cudaEvent_t> event_list;

        uint64_t num_states;
        int num_qubits_local;
        uint64_t num_states_local;

        uint64_t num_blocks_gateop;
        uint64_t block_size_gateop;
        uint64_t num_operand_qubits;

        qcs::complex_t *state_data_device = nullptr;

        qcs::complex_t *zero_constant_addr;
        qcs::kernel_common_struct *qcs_kernel_common_constant_addr;
        qcs::kernel_common_struct qcs_kernel_common_host;
        qcs::kernel_input_qnlist_struct *qcs_kernel_input_constant_addr;
        std::vector<char> qcs_kernel_input_host_buffer;
        qcs::work_buffer scratch;
        int log_swap_buffer_total_length;
        uint64_t swap_buffer_total_length;
        qcs::complex_t *swap_buffer = nullptr;
#ifdef QCS_ENABLE_NVSHMEM
        uint64_t *exchange_signals = nullptr;
        uint64_t exchange_generation = 0;
        int exchange_grid_blocks = 0;
#endif
        std::unique_ptr<measurement_detail::workspace> measurement_workspace = nullptr;
        std::unique_ptr<pauli_detail::workspace> pauli_workspace = nullptr;

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
            try { setup_after_mpi(); }
            catch (...)
            {
                if (nccl_comm) { ncclCommAbort(nccl_comm); nccl_comm = nullptr; }
                for (int slot = 0; slot < 2; ++slot)
                {
                    if (exchange_transfer_done[slot]) cudaEventDestroy(exchange_transfer_done[slot]);
                    if (exchange_copy_done[slot]) cudaEventDestroy(exchange_copy_done[slot]);
                }
                if (exchange_copy_stream) cudaStreamDestroy(exchange_copy_stream);
                if (stream) cudaStreamDestroy(stream);
                MPI_Finalize();
                throw;
            }
        }

        void setup_after_mpi()
        {
            MPI_Comm_size(MPI_COMM_WORLD, &num_procs);
            MPI_Comm_rank(MPI_COMM_WORLD, &proc_num);

            const char* backend = std::getenv("QCS_MEASUREMENT_BACKEND");
            int mode = measurement_detail::backend_mode(backend);
            int low_mode, high_mode;
            ATLC_CHECK_MPI(MPI_Allreduce, &mode, &low_mode, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            ATLC_CHECK_MPI(MPI_Allreduce, &mode, &high_mode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (low_mode < 0 || low_mode != high_mode)
                throw std::runtime_error("invalid or inconsistent QCS_MEASUREMENT_BACKEND across ranks");
            rank_sampled_measurement = mode == 1;
            auto_measurement = mode == 2;
            tiled_cdf_measurement = mode == 3;
            if (tiled_cdf_measurement || rank_sampled_measurement || auto_measurement) {
                int ready = 1, all_ready = 0;
                try {
                    tiled_rank_summaries.resize(num_procs);
                    rank_sampling_workspace.exchange.resize(size_t(num_procs) * 2);
                    rank_sampling_workspace.masses.resize(num_procs);
                }
                catch (...) { ready = 0; }
                ATLC_CHECK_MPI(MPI_Allreduce, &ready, &all_ready, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                if (!all_ready) throw std::runtime_error("measurement rank storage allocation failed");
            }

            // Check before creating the communicator, including mixed binaries.
            int local_mode = nccl_buffer_mode, min_mode = 0, max_mode = 0;
            ATLC_CHECK_MPI(MPI_Allreduce, &local_mode, &min_mode, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            ATLC_CHECK_MPI(MPI_Allreduce, &local_mode, &max_mode, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (min_mode != max_mode) throw std::runtime_error("NCCL buffer mode differs across ranks");
            int environment_ok = 1;
            const auto set_environment_default = [&environment_ok](const char* name, const char* value)
            {
                const char* current = std::getenv(name);
                if ((!current || !*current) && setenv(name, value, 1) != 0)
                    environment_ok = 0;
            };
            // Configurable performance defaults; preserve non-empty user overrides.
            set_environment_default("NCCL_NCHANNELS_PER_NET_PEER", "8");
            set_environment_default("NCCL_P2P_NET_CHUNKSIZE", "262144");
            if constexpr (nccl_buffer_mode == 2)
            {
                set_environment_default("NCCL_PXN_DISABLE", "1");
            }
            // A local setenv failure must not strand peers in the validation collective.
            int all_environment_ok = 0;
            ATLC_CHECK_MPI(MPI_Allreduce, &environment_ok, &all_environment_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (!all_environment_ok) throw std::runtime_error(
                "Failed to set NCCL environment defaults on one or more ranks");
            int version = 0;
            ATLC_CHECK_NCCL(ncclGetVersion, &version);
            const char* pxn = std::getenv("NCCL_PXN_DISABLE");
            int supported = nccl_buffer_mode == 0 ||
                (NCCL_VERSION_CODE >= 23102 && version >= 23102 && !exchange_nvshmem_registered_put &&
                 (nccl_buffer_mode != 2 || (pxn && std::string(pxn) == "1")));
            int all_supported = 0;
            ATLC_CHECK_MPI(MPI_Allreduce, &supported, &all_supported, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (!all_supported) throw std::runtime_error(
                "NCCL buffer modes 1/2 require NCCL >= 2.31.2 and NCCL exchange; mode 2 requires NCCL_PXN_DISABLE=1 on every rank");

            atlc::group_by_hostname(proc_num, num_procs, my_hostname, my_node_number, my_node_local_rank, node_count);
            // fprintf(stderr, "[debug] Rank %d on host %s -> assigned node number: %d, local node rank: %d (total nodes: %d)\n", proc_num, my_hostname.c_str(), my_node_number, my_node_local_rank, node_count);

            gpu_id = my_node_local_rank;
            ATLC_CHECK_CUDA(cudaSetDevice, gpu_id);
            ATLC_CHECK_CUDA(cudaSetDeviceFlags, stream_sync_flags);

            size_t total_memory_bytes = 0;
            size_t free_memory_bytes = 0;
            ATLC_CHECK_CUDA(cudaMemGetInfo, &free_memory_bytes, &total_memory_bytes);

            if (proc_num == 0)
            {
                ATLC_CHECK_NCCL(ncclGetUniqueId, &nccl_id);
            }

            MPI_Bcast(&nccl_id, sizeof(nccl_id), MPI_BYTE, 0, MPI_COMM_WORLD);
            nccl_rank = proc_num;
            ATLC_CHECK_NCCL(ncclCommInitRank, &nccl_comm, num_procs, nccl_id, nccl_rank);

#ifdef QCS_ENABLE_NVSHMEM
            if constexpr (exchange_nvshmem_registered_put)
            {
                MPI_Comm comm = MPI_COMM_WORLD;
                nvshmemx_init_attr_t attr{};
                attr.mpi_comm = &comm;
                check_nvshmem(nvshmemx_init_attr(NVSHMEMX_INIT_WITH_MPI_COMM, &attr),
                    "nvshmemx_init_attr (check NVSHMEM_SYMMETRIC_SIZE and available GPU memory)");
                if (nvshmem_n_pes() != num_procs || nvshmem_my_pe() != proc_num)
                    throw std::runtime_error("NVSHMEM PE mapping differs from MPI ranks");
            }
#endif
            log_num_procs = atlc::log2_int(num_procs);

            log_block_size_max = 9;

            ATLC_CHECK_CUDA(cudaStreamCreate, &stream);
            if constexpr (exchange_pipeline_copy_back && !exchange_nvshmem_registered_put)
            {
                ATLC_CHECK_CUDA(cudaStreamCreateWithFlags, &exchange_copy_stream, cudaStreamNonBlocking);
                for (int slot = 0; slot < 2; slot++)
                {
                    ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &exchange_transfer_done[slot], cudaEventDisableTiming);
                    ATLC_CHECK_CUDA(cudaEventCreateWithFlags, &exchange_copy_done[slot], cudaEventDisableTiming);
                }
            }

            block_size_max = 1 << log_block_size_max;

            cudaDeviceProp swap_device_properties{};
            ATLC_CHECK_CUDA(cudaGetDeviceProperties, &swap_device_properties, gpu_id);
            cudaFuncAttributes swap_kernel_properties{};
            ATLC_CHECK_CUDA(cudaFuncGetAttributes, &swap_kernel_properties, cuda_swap);
            swap_launch_config = swap_detail::launch_limits(block_size_max,
                swap_device_properties.maxThreadsPerBlock,
                swap_device_properties.maxThreadsDim[0],
                swap_kernel_properties.maxThreadsPerBlock,
                swap_device_properties.maxGridSize[0]);

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

        void allocate_memory(bit_num_t num_qubits, qcs_initial_state selected_state = QCS_INITIAL_STATE_ZERO)
        {
            if constexpr (exchange_nvshmem_registered_put)
            {
                allocate_memory_impl(num_qubits, selected_state);
                return;
            }
            // All ranks enter initialization together. Never roll back an existing lifetime.
            int active = this->num_qubits != 0 || state_data_device || scratch.data();
            int any_active = 0;
            ATLC_CHECK_MPI(MPI_Allreduce, &active, &any_active, 1, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            if (any_active) throw std::runtime_error("free existing memory on all ranks before reallocating");
            int dimensions[] = {int(num_qubits), int(selected_state), int(use_unified_memory)};
            int min_dimensions[3]{}, max_dimensions[3]{};
            ATLC_CHECK_MPI(MPI_Allreduce, dimensions, min_dimensions, 3, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            ATLC_CHECK_MPI(MPI_Allreduce, dimensions, max_dimensions, 3, MPI_INT, MPI_MAX, MPI_COMM_WORLD);
            for (int i = 0; i < 3; ++i)
                if (min_dimensions[i] != max_dimensions[i])
                    throw std::runtime_error("allocation configuration differs across ranks");
            std::string error;
            try
            {
                if (nccl_buffer_mode != 0 && use_unified_memory)
                    throw std::runtime_error("NCCL buffer modes do not support managed memory");
                double start = MPI_Wtime();
                allocate_memory_impl(num_qubits, selected_state);
                ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
                allocation_seconds = MPI_Wtime() - start;
                start = MPI_Wtime();
                if (nccl_buffer_mode == 2)
                {
                    register_nccl_buffer(nccl_comm, state_data_device,
                        num_states_local * sizeof(*state_data_device), &state_registration);
                    // Includes every swap slot and measurement AllReduce region.
                    register_nccl_buffer(nccl_comm, scratch.data(), scratch.size(), &scratch_registration);
                }
                registration_seconds = nccl_buffer_mode == 2 ? MPI_Wtime() - start : 0;
            }
            catch (const std::exception& e) { error = e.what(); }
            int ok = error.empty(), all_ok = 0;
            ATLC_CHECK_MPI(MPI_Allreduce, &ok, &all_ok, 1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
            if (!all_ok)
            {
                if (!error.empty()) fprintf(stderr, "[error] rank %d NCCL buffer initialization: %s\n", proc_num, error.c_str());
                free_memory();
                throw std::runtime_error("NCCL buffer initialization failed on at least one rank; see rank diagnostics");
            }
            if (nccl_buffer_mode != 0)
            {
                // Free-memory differences must not produce different chunk boundaries.
                int common_log = 0;
                ATLC_CHECK_MPI(MPI_Allreduce, &log_swap_buffer_total_length, &common_log,
                               1, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                log_swap_buffer_total_length = common_log;
                swap_buffer_total_length = UINT64_C(1) << common_log;
            }
        }

        void allocate_memory_impl(bit_num_t num_qubits, qcs_initial_state selected_state)
        {
            // Reject invalid API values before allocating device resources.
            switch (selected_state)
            {
            case QCS_INITIAL_STATE_ZERO:
            case QCS_INITIAL_STATE_FLAT:
            case QCS_INITIAL_STATE_ENTANGLED:
            case QCS_INITIAL_STATE_SEQUENTIAL:
            case QCS_INITIAL_STATE_USE_CURAND:
            case QCS_INITIAL_STATE_LOAD_STATEVECTOR:
                break;
            default:
                throw std::invalid_argument("invalid initial state");
            }
            initial_state = selected_state;

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
            if (measurement_workspace)
            {
                exmes_list.push_back("measurement workspace is already allocated");
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
            if constexpr (exchange_nvshmem_registered_put)
            {
                if (use_unified_memory) throw std::runtime_error("NVSHMEM exchange does not support unified memory");
                ATLC_CHECK_CUDA(cudaMalloc, &state_data_device, num_states_local * sizeof(*state_data_device));
                check_nvshmem(nvshmemx_buffer_register(state_data_device, num_states_local * sizeof(*state_data_device)), "nvshmemx_buffer_register");
            }
            else
#endif
            if (nccl_buffer_mode != 0)
            {
                state_data_device = static_cast<qcs::complex_t *>(allocate_nccl_buffer(
                    num_states_local * sizeof(*state_data_device)));
            }
            else if (use_unified_memory)
            {
                ATLC_CHECK_CUDA(cudaMallocManaged, &state_data_device, num_states_local * sizeof(*state_data_device));
            }
            else
            {
                ATLC_CHECK_CUDA(cudaMallocAsync, &state_data_device, num_states_local * sizeof(*state_data_device), stream);
            }
            // State-vector contents remain uninitialized until an explicit setter.

            qcs_kernel_common_host.num_qubits = num_qubits;
            qcs_kernel_common_host.state_data_device = state_data_device;
            ATLC_CHECK_CUDA(cudaMemcpyAsync, qcs_kernel_common_constant_addr, &qcs_kernel_common_host, sizeof(qcs::kernel_common_struct), cudaMemcpyHostToDevice, stream);

#ifdef QCS_ENABLE_NVSHMEM
            if constexpr (exchange_nvshmem_registered_put)
            {
                cudaDeviceProp properties{};
                ATLC_CHECK_CUDA(cudaGetDeviceProperties, &properties, gpu_id);
                exchange_phase phase{};
                void *args[] = {&state_data_device, &swap_buffer, &exchange_signals, &phase};
                int capacity = 0;
                check_nvshmem(nvshmemx_collective_launch_query_gridsize((const void *)exchange_registered_put,
                    dim3(exchange_threads), args, 0, &capacity), "NVSHMEM launch capacity query");
                int const requested_blocks = properties.multiProcessorCount * exchange_blocks_per_sm;
                bool const supported = properties.multiProcessorCount > 0 &&
                    exchange_threads <= properties.maxThreadsPerBlock &&
                    exchange_threads <= properties.maxThreadsDim[0] &&
                    requested_blocks <= properties.maxGridSize[0] && requested_blocks <= capacity;
                if (!supported)
                    fprintf(stderr, "[error] rank %d: NVSHMEM exchange requests %d blocks (%d SMs x %d), "
                        "%d threads/block; supported collective blocks=%d, grid blocks=%d, "
                        "threads/block=%d, thread dimension=%d\n", proc_num, requested_blocks,
                        properties.multiProcessorCount, exchange_blocks_per_sm, exchange_threads,
                        capacity, properties.maxGridSize[0], properties.maxThreadsPerBlock, properties.maxThreadsDim[0]);
                // The same grid and allocation sizes are required on every PE.
                int local_limits[] = {properties.multiProcessorCount, -properties.multiProcessorCount, supported ? 1 : 0};
                int global_limits[3]{};
                ATLC_CHECK_MPI(MPI_Allreduce, local_limits, global_limits, 3, MPI_INT, MPI_MIN, MPI_COMM_WORLD);
                if (global_limits[0] != -global_limits[1])
                    throw std::runtime_error("NVSHMEM exchange requires identical SM counts on all PEs; observed range " +
                        std::to_string(global_limits[0]) + " to " + std::to_string(-global_limits[1]));
                if (!global_limits[2])
                    throw std::runtime_error("NVSHMEM exchange requested grid is unsupported on at least one PE; see rank diagnostics");
                exchange_grid_blocks = requested_blocks;
                swap_buffer_total_length = uint64_t(exchange_grid_blocks) * exchange_chunk_elements;
                size_t const staging_bytes = swap_buffer_total_length * sizeof(qcs::complex_t);
                size_t const signal_offset = (staging_bytes + 255) / 256 * 256;
                size_t const signal_bytes = size_t(2) * exchange_grid_blocks * sizeof(uint64_t);
                scratch.allocate_symmetric(signal_offset + signal_bytes);
                swap_buffer = static_cast<qcs::complex_t *>(scratch.symmetric.take(staging_bytes));
                exchange_signals = static_cast<uint64_t *>(scratch.symmetric.take(signal_bytes));
                ATLC_CHECK_CUDA(cudaMemsetAsync, exchange_signals, 0, 2 * exchange_grid_blocks * sizeof(uint64_t), stream);
                // No peer may publish a signal until every PE has initialized
                // its local counters, including after a later reallocation.
                nvshmemx_barrier_all_on_stream(stream);
                exchange_generation = 0;
                if (proc_num == 0) fprintf(stderr, "[info] NVSHMEM exchange: %d SMs, %d blocks, %d threads/block, "
                    "%" PRIu64 " bytes/block, %" PRIu64 " bytes symmetric staging\n",
                    properties.multiProcessorCount, exchange_grid_blocks, exchange_threads,
                    exchange_chunk_bytes, uint64_t(exchange_grid_blocks) * exchange_chunk_bytes);
            }
#endif
            scratch.allocate_remaining(stream, nccl_buffer_mode != 0);
            measurement_workspace = std::make_unique<measurement_detail::workspace>(stream, scratch.device);
            pauli_workspace = std::make_unique<pauli_detail::workspace>(*measurement_workspace, stream);
            if (tiled_cdf_measurement || rank_sampled_measurement || auto_measurement) {
                measurement_collective_send = static_cast<unsigned char*>(
                    scratch.device.take(measurement_collective_bytes));
                measurement_collective_receive = static_cast<unsigned char*>(
                    scratch.device.take(size_t(num_procs) * measurement_collective_bytes));
            }
            if constexpr (!exchange_nvshmem_registered_put)
            {
                uint64_t const available = scratch.device.available() / sizeof(qcs::complex_t);
                if (!available) throw std::runtime_error("work buffer: no room for swap buffer");
                log_swap_buffer_total_length = std::min(std::max(0, num_qubits_local - 1),
                                                        static_cast<int>(atlc::log2_int(available)));
                swap_buffer_total_length = UINT64_C(1) << log_swap_buffer_total_length;
                if (log_swap_buffer_total_length < num_qubits_local - 1)
                    fprintf(stderr, "[warn] swap buffer length reduced by work buffer capacity (log=%d, target=%d).\n",
                            log_swap_buffer_total_length, num_qubits_local - 1);
                // take() aligns slices to 256 bytes; include padding in capacity.
                swap_buffer = static_cast<qcs::complex_t *>(scratch.device.take(
                    swap_buffer_total_length * sizeof(qcs::complex_t)));
            }
        }

        void free_memory()
        {
            // Join both streams even after a partially enqueued exchange failed.
            ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
            if constexpr (exchange_pipeline_copy_back && !exchange_nvshmem_registered_put)
                ATLC_CHECK_CUDA(cudaStreamSynchronize, exchange_copy_stream);
            deregister_nccl_buffer(nccl_comm, scratch_registration);
            deregister_nccl_buffer(nccl_comm, state_registration);
            tiled_workspace.reset();
            measurement_collective_send = measurement_collective_receive = nullptr;
            rank_sampling_workspace.close();
            pauli_workspace.reset();
            if (measurement_workspace)
            {
                // Queue release after every use, before releasing the state or stream.
                measurement_workspace->close();
                measurement_workspace.reset();
            }
#ifdef QCS_ENABLE_NVSHMEM
            if constexpr (exchange_nvshmem_registered_put)
            {
                if (state_data_device || swap_buffer || exchange_signals)
                {
                    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
                    ATLC_CHECK_MPI(MPI_Barrier, MPI_COMM_WORLD);
                    scratch.close();
                    exchange_signals = nullptr;
                    swap_buffer = nullptr;
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
                if (nccl_buffer_mode != 0) free_nccl_buffer(state_data_device);
                else ATLC_CHECK_CUDA(cudaFreeAsync, state_data_device, stream);
                state_data_device = NULL;
            }
            scratch.close();
            swap_buffer = nullptr;
            swap_buffer_total_length = 0;
            this->num_qubits = 0;
        }

        void initialize_selected_state()
        {
            switch (initial_state)
            {
            case QCS_INITIAL_STATE_ZERO: initialize_zero(); break;
            case QCS_INITIAL_STATE_FLAT: initialize_flat(); break;
            case QCS_INITIAL_STATE_ENTANGLED: initialize_entangled(); break;
            case QCS_INITIAL_STATE_SEQUENTIAL: initialize_sequential(); break;
            case QCS_INITIAL_STATE_USE_CURAND: initialize_use_curand(); break;
            case QCS_INITIAL_STATE_LOAD_STATEVECTOR: initialize_laod_statevector(); break;
            default: throw std::invalid_argument("invalid initial state");
            }
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
                ATLC_CHECK_CUDA(cudaMemsetAsync, state_data_device + 1, 0, sizeof(qcs::complex_t) * (num_states_local - 1), stream);
            }
            else
            {
                ATLC_CHECK_CUDA(cudaMemsetAsync, state_data_device, 0, sizeof(qcs::complex_t) * num_states_local, stream);
            }
        }

        void initialize_entangled()
        {
            discard_measurement_state();
            ATLC_CHECK_CUDA(cudaMemsetAsync, state_data_device, 0, sizeof(qcs::complex_t) * num_states_local, stream);
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
            exchange_phase phase{};
            phase.area_length = UINT64_C(1) << low;
            phase.area_bits = num_qubits_local - low;
            phase.chunks_per_area = (phase.area_length - 1) / exchange_chunk_elements + 1;
            uint64_t const regions = UINT64_C(1) << (phase.area_bits - count);
            phase.total_chunks = regions * phase.chunks_per_area;
            uint64_t mask = 0;
            for (int i = 0; i < count; ++i)
            {
                mask |= UINT64_C(1) << (swap_target_global_list[i] - num_qubits_local);
                phase.fixed_mask |= UINT64_C(1) << (swap_target_local_list[i] - low);
            }
            uint64_t const rounds = (phase.total_chunks - 1) / exchange_grid_blocks + 1;
            // XOR phases pair every PE exactly once, including disjoint peer groups.
            for (uint64_t delta = 1; delta <= mask; ++delta)
            {
                if (delta & ~mask) continue;
                phase.peer = proc_num ^ delta;
                phase.fixed_value = 0;
                for (int i = 0; i < count; ++i)
                    phase.fixed_value |= ((uint64_t(phase.peer) >> (swap_target_global_list[i] - num_qubits_local)) & 1)
                                         << (swap_target_local_list[i] - low);
                if (exchange_generation > UINT64_MAX - rounds)
                    throw std::runtime_error("NVSHMEM signal generation exhausted");
                phase.generation = exchange_generation;
                exchange_generation += rounds;
                void *args[] = {&state_data_device, &swap_buffer, &exchange_signals, &phase};
                check_nvshmem(nvshmemx_collective_launch((const void *)exchange_registered_put,
                    dim3(exchange_grid_blocks), dim3(exchange_threads), args, 0, stream), "NVSHMEM collective launch");
                // No PE may signal a new peer until every PE has finished this phase.
                nvshmemx_barrier_all_on_stream(stream);
            }
        }
#endif

        // Enqueues device work only. All host mapping updates remain in the caller.
        void exchange_qubits()
        {
            int const num_swap_qubits = swap_target_global_list.size();
            if (num_swap_qubits == 0)
            {
                return;
            }

#ifdef QCS_ENABLE_NVSHMEM
            if constexpr (exchange_nvshmem_registered_put)
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
            if constexpr (exchange_pipeline_copy_back && !exchange_nvshmem_registered_put)
            {
                if (last_slot >= 0)
                {
                    // Copies share one stream, so the last event covers every copy in this exchange.
                    ATLC_CHECK_CUDA(cudaStreamWaitEvent, stream, exchange_copy_done[last_slot], 0);
                }
            }
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

        // Collective across all simulator ranks with identical arguments/order.
        // This observes the effective state without releasing measured qubits or
        // consuming a measurement draw. It may retain a new physical mapping.
        double expectation_pauli(const bit_num_t* qubits, const qcs_pauli_operator* operators,
                                 bit_num_t count, double coefficient)
        {
            if (!state_data_device || !measurement_workspace || !pauli_workspace)
                throw std::runtime_error("Pauli expectation: simulator storage is not allocated");
            pauli_detail::pauli_string input(num_qubits, num_qubits_local,
                                             qubits, operators, count, coefficient);
            std::vector<int> targets;
            targets.reserve(input.terms().size());
            for (const auto& term : input.terms()) targets.push_back(term.qubit);

            // Project every measured bit, including bits outside this string,
            // using the old mapping before any exchange. Projection preserves
            // the current (possibly unnormalized) amplitudes of the kept branch.
            materialize_pending_projection(pending_projection_mask_logical);
            target_qubit_num_logical_list = std::move(targets);
            localize_target_qubits();
            auto descriptor = input.descriptor(perm_l2p);
            double* result_device = pauli_workspace->enqueue(
                state_data_device, num_states_local, descriptor, stream);
            // Zero-contribution ranks participate too. Same-stream ordering
            // covers exchange copy-back, both reductions, NCCL and readback.
            ATLC_CHECK_NCCL(ncclAllReduce, result_device, result_device, 1,
                            ncclDouble, ncclSum, nccl_comm, stream);
            double result;
            ATLC_CHECK_CUDA(cudaMemcpyAsync, &result, result_device, sizeof(result), cudaMemcpyDeviceToHost, stream);
            ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
            return coefficient * result;
        }

        template <int N>
        qcs::float_t* reduce_measure_norm(
            int num_local_targets, const cuda::std::array<int, N> &local_outcome_bits,
            uint64_t fixed_outcome)
        {
            measurement_detail::descriptor<N> descriptor{};
            descriptor.local_targets = num_local_targets;
            descriptor.fixed_outcome = fixed_outcome;
            // Physical targets retain caller order; only previously measured
            // qubits belong in the control domain.
            for (int i = 0; i < num_local_targets; ++i)
            {
                descriptor.local_target_physical[i] = target_qubit_num_physical_list[i];
                descriptor.local_outcome_bits[i] = local_outcome_bits[i];
            }
            for (int physical : positive_control_qubit_num_physical_local_list)
            {
                descriptor.control_mask |= UINT64_C(1) << physical;
                descriptor.control_value |= UINT64_C(1) << physical;
            }
            for (int physical : negative_control_qubit_num_physical_local_list)
                descriptor.control_mask |= UINT64_C(1) << physical;

            return measurement_workspace->enqueue<N>(
                state_data_device, num_states_local, descriptor, stream);
        }

        template <int N>
        uint64_t measure_qubits(const std::array<int, N> &qubits)
        {
            QCS_MEASURE_RANGE("grouped.metadata_and_inter_group_host");
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

            if (measurement_counter_exhausted)
                throw std::runtime_error("measurement RNG counter exhausted");

            // Already measured targets stay controls, so they are never counted
            // twice or loaded from the discarded branch of a pending projection.
            positive_control_qubit_num_logical_list = measured_1_qubit_num_logical_list;
            negative_control_qubit_num_logical_list = measured_0_qubit_num_logical_list;
            // Only local targets enter this path; it clears exchange work lists
            // without moving statevector data or changing the permutation.
            localize_target_qubits();
            check_control_qubit_num_physical();
            // The reducer passes its own descriptor by value. Gate metadata,
            // sorted operands and gate launch dimensions are not consumed here.

            // Every rank supplies all outcome bins, including ranks excluded by
            // earlier measurements. Clear their output rather than reuse weights.
            QCS_MEASURE_STAGE("grouped.local_probability_reduction");
            qcs::float_t* const norm_device = proc_num_control_condition
                ? reduce_measure_norm<N>(target_qubit_num_logical_list.size(), local_outcome_bits, fixed_outcome)
                : measurement_workspace->enqueue_zero<N>(stream);

            QCS_MEASURE_STAGE("grouped.nccl_allreduce");
            static_assert(std::is_same<qcs::float_t, double>::value, "measurement NCCL type must match float_t");
            ATLC_CHECK_NCCL(ncclAllReduce, norm_device, norm_device, 1 << N, ncclDouble, ncclSum, nccl_comm, stream);

            // Same-stream ordering: local reduction -> NCCL -> sampler ->
            // compact readback. No host boundary or RNG/outcome collective.
            QCS_MEASURE_STAGE("grouped.gpu_sampling");
            auto* const result_device = measurement_workspace->enqueue_sample<N>(
                norm_device, rng_seed, measurement_counter, stream);
            QCS_MEASURE_STAGE("grouped.device_to_host");
            measurement_detail::sample_result result{};
            ATLC_CHECK_CUDA(cudaMemcpyAsync, &result, result_device, sizeof(result), cudaMemcpyDeviceToHost, stream);
            QCS_MEASURE_STAGE("grouped.host_synchronization");
            ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
            if (result.status == measurement_detail::sample_status::invalid_weight)
                throw std::runtime_error("invalid measurement probability component");
            if (result.status != measurement_detail::sample_status::success)
                throw std::runtime_error("invalid measurement norm");
            QCS_MEASURE_STAGE("grouped.metadata_update");
            uint64_t const outcome = result.outcome;
            if (measurement_counter == std::numeric_limits<uint64_t>::max())
                measurement_counter_exhausted = true;
            else
                ++measurement_counter;

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

        // A broken CUDA context or transport cannot safely carry a failure
        // sentinel. Abort the job rather than leave peers waiting in NCCL.
        template<class F> void measurement_transport(F operation)
        {
            try { operation(); }
            catch (...) {
                fprintf(stderr, "[error] rank %d: measurement CUDA/NCCL transport failed; aborting job\n", proc_num);
                ncclCommAbort(nccl_comm);
                nccl_comm = nullptr;
                MPI_Abort(MPI_COMM_WORLD, 1);
                throw;
            }
        }

        void measurement_allgather(const void* source, void* output,
                                   size_t bytes, bool device_source = false)
        {
            measurement_transport([&] {
                const void* input = source;
                if (!device_source) {
                    ATLC_CHECK_CUDA(cudaMemcpyAsync, measurement_collective_send,
                        source, bytes, cudaMemcpyHostToDevice, stream);
                    input = measurement_collective_send;
                }
                ATLC_CHECK_NCCL(ncclAllGather, input, measurement_collective_receive,
                    bytes, ncclUint8, nccl_comm, stream);
                ATLC_CHECK_CUDA(cudaMemcpyAsync, output, measurement_collective_receive,
                    bytes * size_t(num_procs), cudaMemcpyDeviceToHost, stream);
                ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
            });
        }

        void measurement_broadcast(void* output, size_t bytes, int owner,
                                   const void* device_source = nullptr)
        {
            measurement_transport([&] {
                if (proc_num == owner && !device_source)
                    ATLC_CHECK_CUDA(cudaMemcpyAsync, measurement_collective_send,
                        output, bytes, cudaMemcpyHostToDevice, stream);
                const void* input = device_source ? device_source : measurement_collective_send;
                ATLC_CHECK_NCCL(ncclBroadcast, input, measurement_collective_receive,
                    bytes, ncclUint8, owner, nccl_comm, stream);
                ATLC_CHECK_CUDA(cudaMemcpyAsync, output, measurement_collective_receive,
                    bytes, cudaMemcpyDeviceToHost, stream);
                ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
            });
        }

        void measure_tiled_cdf(const std::vector<int>& qubits, std::vector<bit_t>& results)
        {
            QCS_MEASURE_RANGE("tiled_cdf.metadata_and_allocation");
            uint64_t requested = 0;
            for (int q : qubits) requested |= UINT64_C(1) << q;
            uint64_t fresh = requested & ~measured_mask_logical;
            uint64_t sampled = measured_value_logical;
            if (fresh)
            {
                uint64_t mask = 0, value = 0;
                bool active = true;
                for (int q = 0; q < num_qubits; ++q)
                    if (measured_mask_logical & (UINT64_C(1) << q)) {
                        int p = perm_l2p[q];
                        uint64_t b = (measured_value_logical >> q) & 1;
                        if (p < num_qubits_local) {
                            mask |= UINT64_C(1) << p;
                            value |= b << p;
                        } else if (((uint64_t(proc_num) >> (p - num_qubits_local)) & 1) != b)
                            active = false;
                    }
                // Allocate all commit storage before entering the collective protocol.
                // Any recoverable local build failure is carried by the summary.
                tiled_cdf_detail::rank_summary summary{};
                auto& ranks = tiled_rank_summaries;
                const tiled_cdf_detail::rank_summary* device_summary = nullptr;
                try {
                    measured_1_qubit_num_logical_list.reserve(num_qubits);
                    measured_0_qubit_num_logical_list.reserve(num_qubits);
                    if (measurement_counter_exhausted)
                        throw std::runtime_error("measurement RNG counter exhausted");
                    if (!tiled_workspace)
                        tiled_workspace = std::make_unique<tiled_cdf_detail::workspace>(stream);
                    device_summary = tiled_workspace->build(state_data_device, num_states_local,
                                                        mask, value, active);
                    rank_sampling_detail::tiled_uniform<<<1, 1, 0, stream>>>(
                        const_cast<tiled_cdf_detail::rank_summary*>(device_summary),
                        rng_seed, measurement_counter);
                    tiled_cdf_detail::check(cudaGetLastError());
                } catch (...) {
                    summary.status = tiled_cdf_detail::invalid_distribution;
                    device_summary = nullptr;
                }
                QCS_MEASURE_STAGE("tiled_cdf.rank_exchange");
                measurement_allgather(device_summary ? static_cast<const void*>(device_summary) : &summary,
                                      ranks.data(), sizeof(summary), device_summary != nullptr);
                QCS_MEASURE_STAGE("tiled_cdf.rank_selection");
                long double total = 0;
                for (const auto& r : ranks) {
                    if ((r.status & ~unsigned(tiled_cdf_detail::zero_mass)) ||
                        !std::isfinite(r.mass) || r.mass < 0)
                        throw std::runtime_error("invalid tiled CDF rank summary");
                    total += r.mass;
                }
                if (!std::isfinite(total) || !(total > 0))
                    throw std::runtime_error("invalid measurement norm");
                double uniform = ranks[0].uniform;
                if (!std::isfinite(uniform) || uniform < 0 || uniform > 1)
                    throw std::runtime_error("invalid measurement uniform");
                double threshold = std::min(uniform * total, std::nextafter(total, 0.0));
                int selected = -1;
                for (int r = 0; r < num_procs; ++r) {
                    if (ranks[r].mass <= 0) continue;
                    selected = r;
                    if (threshold < ranks[r].mass) break;
                    threshold -= ranks[r].mass;
                }
                double fraction = std::min(double(threshold / ranks[selected].mass),
                                           std::nextafter(1.0, 0.0));
                tiled_cdf_detail::sample_result result{UINT64_MAX,
                                                       tiled_cdf_detail::invalid_distribution};
                QCS_MEASURE_STAGE("tiled_cdf.gpu_tile_lookup_and_amplitude_sampling");
                const tiled_cdf_detail::sample_result* device_result = nullptr;
                if (proc_num == selected) {
                    try {
                        device_result = tiled_workspace->draw(fraction);
                    } catch (...) {
                        device_result = nullptr;
                        result = {UINT64_MAX, tiled_cdf_detail::invalid_distribution};
                    }
                }
                QCS_MEASURE_STAGE("tiled_cdf.result_broadcast");
                measurement_broadcast(&result, sizeof(result), selected, device_result);
                if (result.status || result.index >= num_states_local)
                    throw std::runtime_error("invalid tiled CDF owner sample");
                QCS_MEASURE_STAGE("tiled_cdf.metadata_update");
                uint64_t index = result.index;
                uint64_t physical = index | (uint64_t(selected) << num_qubits_local);
                for (int q : qubits) if (fresh & (UINT64_C(1) << q)) {
                    uint64_t b = (physical >> perm_l2p[q]) & 1;
                    sampled = (sampled & ~(UINT64_C(1) << q)) | (b << q);
                    (b ? measured_1_qubit_num_logical_list : measured_0_qubit_num_logical_list).push_back(q);
                }
                measured_mask_logical |= fresh;
                measured_value_logical = sampled;
                pending_projection_mask_logical |= fresh;
                if (measurement_counter == UINT64_MAX) measurement_counter_exhausted = true;
                else ++measurement_counter;
            }
            for (size_t i = 0; i < qubits.size(); ++i)
                results[i] = static_cast<bit_t>((sampled >> qubits[i]) & 1);
        }

        void measure_rank_sampled(const std::vector<int>& qubits, std::vector<bit_t>& results)
        {
            QCS_MEASURE_RANGE("rank_sampled.metadata_and_allocation");
            uint64_t requested = 0;
            for (int q : qubits) requested |= UINT64_C(1) << q;
            uint64_t fresh = requested & ~measured_mask_logical;
            uint64_t sampled = measured_value_logical;
            if (fresh)
            {
                uint64_t mask = 0, value = 0;
                bool active = true;
                for (int q = 0; q < num_qubits; ++q)
                    if (measured_mask_logical & (UINT64_C(1) << q)) {
                        int p = perm_l2p[q];
                        uint64_t b = (measured_value_logical >> q) & 1;
                        if (p < num_qubits_local) {
                            mask |= UINT64_C(1) << p;
                            value |= b << p;
                        } else if (((uint64_t(proc_num) >> (p - num_qubits_local)) & 1) != b)
                            active = false;
                    }
                size_t chunks = (num_states_local + rank_sampling_detail::chunk_size - 1) /
                                rank_sampling_detail::chunk_size;
                double* device = nullptr;
                auto& weights = rank_sampling_workspace.weights;
                auto& masses = rank_sampling_workspace.masses;
                auto& exchange = rank_sampling_workspace.exchange;
                double mass = -1;
                double uniform_local = 0;
                try {
                    measured_1_qubit_num_logical_list.reserve(num_qubits);
                    measured_0_qubit_num_logical_list.reserve(num_qubits);
                    if (measurement_counter_exhausted)
                        throw std::runtime_error("measurement RNG counter exhausted");
                    rank_sampling_workspace.reserve(chunks, num_procs);
                    device = rank_sampling_workspace.device;
                    QCS_MEASURE_STAGE("rank_sampled.local_mass");
                    rank_sampling_detail::chunk_weights<<<std::min<size_t>(chunks, 65535), 256, 0, stream>>>(
                        state_data_device, num_states_local, mask, value, active,
                        device, device + chunks, rng_seed, measurement_counter);
                    if (cudaGetLastError() != cudaSuccess)
                        throw std::runtime_error("rank sampling CUDA launch failed");
                    QCS_MEASURE_STAGE("rank_sampled.device_to_host_and_wait");
                    ATLC_CHECK_CUDA(cudaMemcpyAsync, weights.data(), device,
                                    weights.size() * sizeof(double), cudaMemcpyDeviceToHost, stream);
                    ATLC_CHECK_CUDA(cudaStreamSynchronize, stream);
                    QCS_MEASURE_STAGE("rank_sampled.host_mass_sum");
                    mass = 0;
                    for (size_t i = 0; i < chunks; ++i) {
                        if (!std::isfinite(weights[i]) || weights[i] < 0) {
                            mass = -1;
                            break;
                        }
                        mass += weights[i];
                    }
                    uniform_local = weights[chunks];
                } catch (...) { mass = -1; }
                // Include the authoritative rank-zero draw in the mass exchange:
                // no separate broadcast or rank-dependent dispatch collective.
                double payload[2] = {mass, uniform_local};
                QCS_MEASURE_STAGE("rank_sampled.rank_mass_collection");
                measurement_allgather(payload, exchange.data(), sizeof(payload));
                for (int rank = 0; rank < num_procs; ++rank) masses[rank] = exchange[2 * rank];
                QCS_MEASURE_STAGE("rank_sampled.rank_selection");
                double total = 0;
                for (auto w : masses) {
                    if (!std::isfinite(w) || w < 0)
                        throw std::runtime_error("invalid measurement probability component");
                    total += w;
                }
                if (!std::isfinite(total) || !(total > 0))
                    throw std::runtime_error("invalid measurement norm");
                // Rank zero supplies the authoritative draw, even if seeds differ.
                double uniform = static_cast<double>(exchange[1]);
                if (!std::isfinite(uniform) || uniform < 0 || uniform >= 1)
                    throw std::runtime_error("invalid measurement uniform");
                auto select = [](const auto& bins, size_t count, double& threshold) {
                    size_t last = 0;
                    for (size_t i = 0; i < count; ++i) if (bins[i] > 0) {
                        last = i;
                        if (threshold < bins[i]) return i;
                        threshold -= bins[i];
                    }
                    threshold = std::nextafter(bins[last], 0.0L);
                    return last;
                };
                double threshold = std::min(uniform * total, std::nextafter(total, 0.0));
                int selected = static_cast<int>(select(masses, masses.size(), threshold));
                uint64_t index = UINT64_MAX;
                const uint64_t* device_index = nullptr;
                QCS_MEASURE_STAGE("rank_sampled.owner_local_sampling");
                if (proc_num == selected) {
                    // Rank CDF residual is already a conditional local draw.
                    size_t chunk = select(weights, chunks, threshold);
                    double fraction = static_cast<double>(threshold / weights[chunk]);
                    fraction = std::min(fraction, std::nextafter(1.0, 0.0));
                    auto* index_device = reinterpret_cast<uint64_t*>(device + chunks + 1);
                    try {
                        rank_sampling_detail::sample_chunk<<<1, 1, 0, stream>>>(
                            state_data_device, num_states_local, chunk, mask, value, fraction, index_device);
                        if (cudaGetLastError() != cudaSuccess)
                            throw std::runtime_error("rank sampling CUDA launch failed");
                        device_index = index_device;
                    } catch (...) {
                        index = UINT64_MAX;
                    }
                }
                QCS_MEASURE_STAGE("rank_sampled.result_broadcast");
                measurement_broadcast(&index, sizeof(index), selected, device_index);
                if (index >= num_states_local) throw std::runtime_error("invalid local measurement norm");
                QCS_MEASURE_STAGE("rank_sampled.metadata_update");
                uint64_t physical = index | (uint64_t(selected) << num_qubits_local);
                for (int q : qubits) if (fresh & (UINT64_C(1) << q)) {
                    uint64_t b = (physical >> perm_l2p[q]) & 1;
                    sampled = (sampled & ~(UINT64_C(1) << q)) | (b << q);
                    (b ? measured_1_qubit_num_logical_list : measured_0_qubit_num_logical_list).push_back(q);
                }
                measured_mask_logical |= fresh;
                measured_value_logical = sampled;
                pending_projection_mask_logical |= fresh;
                if (measurement_counter == UINT64_MAX) measurement_counter_exhausted = true;
                else ++measurement_counter;
            }
            for (size_t i = 0; i < qubits.size(); ++i)
                results[i] = static_cast<bit_t>((sampled >> qubits[i]) & 1);
        }

        bool use_rank_sampling(const std::vector<int>& qubits) const
        {
            if (!auto_measurement) return rank_sampled_measurement;
            measurement_detail::backend_workload workload{};
            workload.ranks = num_procs;
            workload.local_states = num_states_local;
            for (int q = 0; q < num_qubits; ++q) {
                if (!(measured_mask_logical & (UINT64_C(1) << q))) continue;
                if (perm_l2p[q] < num_qubits_local) ++workload.measured_local;
                else ++workload.measured_global;
            }
            for (int q : qubits) {
                if (measured_mask_logical & (UINT64_C(1) << q)) continue;
                if (perm_l2p[q] < num_qubits_local) ++workload.fresh_local;
                else ++workload.fresh_global;
            }
            // Only replicated metadata enters dispatch; never rank-local mass,
            // control compatibility, RNG state, or observed runtime timings.
            return measurement_detail::prefer_rank_sampling(workload);
        }

        void measure_many(const std::vector<int>& qubits, std::vector<bit_t>& results)
        {
            if (tiled_cdf_measurement) measure_tiled_cdf(qubits, results);
            else if (use_rank_sampling(qubits)) measure_rank_sampled(qubits, results);
            else measure_qubit_groups(qubits, results);
        }

        int measure_qubit(int const qubit)
        {
            if (qubit < 0 || qubit >= num_qubits) throw std::runtime_error("qubit is out of range");
            if (!tiled_cdf_measurement && !use_rank_sampling({qubit}))
                return static_cast<int>(measure_qubits<1>({qubit}));
            std::vector<bit_t> result(1);
            if (tiled_cdf_measurement) measure_tiled_cdf({qubit}, result);
            else measure_rank_sampled({qubit}, result);
            return result[0];
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

        bool prepare_gate_operation(std::vector<int> target_qubit_num_logical_list_input,
                                    std::vector<int> negative_control_qubit_num_logical_list_input,
                                    std::vector<int> positive_control_qubit_num_logical_list_input,
                                    uint64_t &release_mask)
        {
            target_qubit_num_logical_list = std::move(target_qubit_num_logical_list_input);
            negative_control_qubit_num_logical_list = std::move(negative_control_qubit_num_logical_list_input);
            positive_control_qubit_num_logical_list = std::move(positive_control_qubit_num_logical_list_input);

            uint64_t target_mask = 0;
            for (int const target : target_qubit_num_logical_list)
                target_mask |= UINT64_C(1) << target;
            release_mask = measured_mask_logical & target_mask;

            prepare_control_qubit_num_list();
            if (!measured_control_condition)
                return false;

            localize_target_qubits();
            materialize_pending_projection(release_mask);
            assert((pending_projection_mask_logical & target_mask) == 0);
            check_control_qubit_num_physical();
            prepare_operating_gate();
            return true;
        }

        void finish_gate_operation(uint64_t release_mask)
        {
            update_measured_list();
            measured_mask_logical &= ~release_mask;
            measured_value_logical &= measured_mask_logical;
            pending_projection_mask_logical &= measured_mask_logical;
        }

        template <typename GateType>
        void operate_gate(GateType gateobj, std::vector<int> targets, std::vector<int> negctrls, std::vector<int> ctrls)
        {
            assert(targets.size() == GateType::num_target_qubits);
            uint64_t release_mask;
            if (!prepare_gate_operation(std::move(targets), std::move(negctrls), std::move(ctrls), release_mask))
                return;

            if (proc_num_control_condition)
            {
                ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, cuda_gate<GateType>, num_blocks_gateop, block_size_gateop, 0, stream, gateobj);
            }
            finish_gate_operation(release_mask);
        }

        void operate_swap(std::vector<int> targets, std::vector<int> negctrls, std::vector<int> ctrls)
        {
            if (targets.empty() || targets.size() > 2 * qcs::gate::max_num_swaps || targets.size() % 2 != 0)
                throw std::runtime_error("SWAP target count must be an even number from 2 through 2 * max_num_swaps");

            uint64_t release_mask;
            if (!prepare_gate_operation(std::move(targets), std::move(negctrls), std::move(ctrls), release_mask))
                return;

            if (proc_num_control_condition)
            {
                // Unlike cuda_gate, cuda_swap enumerates every target state as
                // well as every non-operand state. Exclude only local controls.
                int const num_controls = negative_control_qubit_num_physical_local_list.size() +
                                         positive_control_qubit_num_physical_local_list.size();
                uint64_t const num_threads = num_states_local >> num_controls;
                for (uint64_t offset = 0; offset < num_threads; )
                {
                    auto const launch = swap_launch_config.next(num_threads, offset);
                    ATLC_CHECK_CUDA(atlc::cudaLaunchKernel, cuda_swap, launch.blocks, launch.threads,
                        0, stream, launch.count, launch.offset);
                    offset += launch.count;
                }
            }
            finish_gate_operation(release_mask);
        }

#if 0
void GHZ_circuit_sample() {

    allocate_memory(14);
    initialize_selected_state();

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
            if constexpr (exchange_pipeline_copy_back && !exchange_nvshmem_registered_put)
            {
                for (int slot = 0; slot < 2; slot++)
                {
                    ATLC_CHECK_CUDA(cudaEventDestroy, exchange_transfer_done[slot]);
                    ATLC_CHECK_CUDA(cudaEventDestroy, exchange_copy_done[slot]);
                }
                ATLC_CHECK_CUDA(cudaStreamDestroy, exchange_copy_stream);
            }
            ATLC_CHECK_CUDA(cudaStreamDestroy, stream);
#ifdef QCS_ENABLE_NVSHMEM
            if constexpr (exchange_nvshmem_registered_put) nvshmem_finalize();
#endif
            if (nccl_comm)
            {
                ATLC_CHECK_NCCL(ncclCommDestroy, nccl_comm);
                nccl_comm = nullptr;
            }
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
    try { qcs_simulator_setup(sim); }
    catch (...) { delete sim; throw; }
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
    try { sim->core->setup(); }
    catch (...) { delete sim->core; sim->core = nullptr; throw; }
}

void qcs_simulator_allocate_memory_cxx(qcs_simulator *sim) { qcs_simulator_allocate_memory_cxx(sim, QCS_INITIAL_STATE_ZERO); }
void qcs_simulator_allocate_memory_cxx(qcs_simulator *sim, qcs_initial_state initial_state)
{
    sim->core->allocate_memory(sim->num_qubits, initial_state);
}
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
double qcs_simulator_expectation_pauli_cxx(qcs_simulator *sim, const bit_num_t *qubit_num_list,
                                          const qcs_pauli_operator *operators,
                                          bit_num_t qubit_num_count, double coefficient)
{
    if (!sim || !sim->core)
        throw std::runtime_error("Pauli expectation: null simulator/core");
    qcs::pauli_detail::validate_simulator_state(sim, sim->core->state_data_device);
    return sim->core->expectation_pauli(qubit_num_list, operators, qubit_num_count, coefficient);
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
    sim->core->measure_many(qubits, temporary_results);
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

    std::vector<size_t> measurement_order(qubits.size());
    for (size_t i = 0; i < qubits.size(); ++i)
        measurement_order[i] = i;
    std::sort(measurement_order.begin(), measurement_order.end(),
              [&](size_t a, size_t b) {
                  int const physical_a = sim->core->perm_l2p[qubits[a]];
                  int const physical_b = sim->core->perm_l2p[qubits[b]];
                  bool const local_a = physical_a < sim->core->num_qubits_local;
                  bool const local_b = physical_b < sim->core->num_qubits_local;
                  if (local_a != local_b)
                      return local_a;
                  return physical_a > physical_b;
              });
    std::vector<int> sorted_qubits;
    sorted_qubits.reserve(qubits.size());
    for (size_t const i : measurement_order)
        sorted_qubits.push_back(qubits[i]);

    std::vector<bit_t> sorted_results(qubits.size());
    sim->core->measure_many(sorted_qubits, sorted_results);
    std::vector<bit_t> temporary_results(qubits.size());
    for (size_t i = 0; i < qubits.size(); ++i)
        temporary_results[measurement_order[i]] = sorted_results[i];
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
void qcs_simulator_set_zero_state_cxx(qcs_simulator *sim) { sim->core->initialize_selected_state(); }
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
    sim->core->operate_swap(targets, negctrls, ctrls);
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

extern "C" int qcs_simulator_allocate_memory_with_initial_state(qcs_simulator *sim, qcs_initial_state initial_state)
{
    return qcs_try_cxx([&]()
                { qcs_simulator_allocate_memory_cxx(sim, initial_state); });
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

extern "C" int qcs_simulator_expectation_pauli(qcs_simulator *sim, const bit_num_t *qubit_num_list,
                                              const qcs_pauli_operator *operators,
                                              bit_num_t qubit_num_count, double coefficient, double *result)
{
    return qcs_try_cxx([&]()
    {
        // Check before entering the collective or materializing projections.
        if (!result) throw std::runtime_error("Pauli expectation: null result pointer");
        double value = qcs_simulator_expectation_pauli_cxx(
            sim, qubit_num_list, operators, qubit_num_count, coefficient);
        qcs_write_result(result, value);
    });
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
