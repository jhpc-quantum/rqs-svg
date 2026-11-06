#pragma once
#ifdef __cplusplus
#include <cstdio>
#include <cstdint>
#else
#include <stdio.h>
#include <stdint.h>
#endif

/* Initializers retain their existing amplitudes (no implicit normalization). */
typedef enum qcs_initial_state
{
    QCS_INITIAL_STATE_ZERO = 0,
    QCS_INITIAL_STATE_FLAT,
    QCS_INITIAL_STATE_ENTANGLED,
    QCS_INITIAL_STATE_SEQUENTIAL,
    QCS_INITIAL_STATE_USE_CURAND,
    QCS_INITIAL_STATE_LOAD_STATEVECTOR
} qcs_initial_state;

/* Operators for a single Pauli string; qubit indices are logical indices. */
typedef enum qcs_pauli_operator
{
    QCS_PAULI_I = 0,
    QCS_PAULI_X,
    QCS_PAULI_Y,
    QCS_PAULI_Z
} qcs_pauli_operator;

typedef struct qcs_simulator qcs_simulator;
typedef void (*qcs_exception_callback)(char const *message, size_t message_length);
typedef int16_t bit_num_t;
typedef int32_t proc_num_t;
typedef uint8_t bit_t;
typedef int16_t event_num_t;

#ifdef __cplusplus
qcs_simulator *qcs_simulator_create_cxx(void);
void qcs_simulator_destroy_cxx(qcs_simulator *sim);
/* Allocate simulator storage without initializing state-vector contents.
 * Call qcs_simulator_set_zero_state_cxx (or another state setter) before use.
 * The overload only selects the method for subsequent set_zero_state calls. */
void qcs_simulator_allocate_memory_cxx(qcs_simulator *sim);
void qcs_simulator_allocate_memory_cxx(qcs_simulator *sim, qcs_initial_state initial_state);
void qcs_simulator_dispose_cxx(qcs_simulator *sim);
int qcs_simulator_get_num_procs_cxx(qcs_simulator *sim);
int qcs_simulator_get_proc_num_cxx(qcs_simulator *sim);
int qcs_simulator_get_num_qubits_cxx(const qcs_simulator *sim);
int qcs_simulator_get_num_clbits_cxx(const qcs_simulator *sim);
void qcs_simulator_set_num_qubits_cxx(qcs_simulator *sim, bit_num_t num_qubits);
void qcs_simulator_set_mapping_cxx(qcs_simulator *sim, const bit_num_t *perm_p2l, bit_num_t perm_p2l_count);
void qcs_simulator_set_num_clbits_cxx(qcs_simulator *sim, bit_num_t num_clbits);
void qcs_simulator_get_clbits_cxx(const qcs_simulator *sim, bit_t *clbits);
void qcs_simulator_get_clbits_string_cxx(const qcs_simulator *sim, char *clbits_string);
void qcs_simulator_reset_cxx(qcs_simulator *sim, bit_num_t qubit_num);
/* Initializes/restores the state selected at allocation (zero by default). */
void qcs_simulator_set_zero_state_cxx(qcs_simulator *sim);
void qcs_simulator_set_sequential_state_cxx(qcs_simulator *sim);
void qcs_simulator_set_flat_state_cxx(qcs_simulator *sim);
void qcs_simulator_set_entangled_state_cxx(qcs_simulator *sim);
void qcs_simulator_set_random_state_cxx(qcs_simulator *sim);
void qcs_simulator_reset_clbits_cxx(qcs_simulator *sim);
void qcs_simulator_reset_measurement_state_cxx(qcs_simulator *sim);
void qcs_simulator_reinitialize_mapping_cxx(qcs_simulator *sim);
void qcs_simulator_gate_global_phase_cxx(qcs_simulator *sim, double theta, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_h_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_x_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_y_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_z_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_s_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_sdg_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_t_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_tdg_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_sx_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rx_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_ry_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rz_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_u4_cxx(qcs_simulator *sim, double theta, double phi, double lambda, double gamma, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_u3_cxx(qcs_simulator *sim, double theta, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_u2_cxx(qcs_simulator *sim, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_u1_cxx(qcs_simulator *sim, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_u_cxx(qcs_simulator *sim, double theta, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_p_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
// Apply 1-4 disjoint SWAPs to 2, 4, 6, or 8 targets, paired in list order.
void qcs_simulator_gate_swap_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_iswap_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_id_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_sxdg_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_r_cxx(qcs_simulator *sim, double theta, double phi, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rxx_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_ryy_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rzz_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rzx_cxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_dcx_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_ecr_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_xx_plus_yy_cxx(qcs_simulator *sim, double theta, double beta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_xx_minus_yy_cxx(qcs_simulator *sim, double theta, double beta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rccx_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
void qcs_simulator_gate_rcccx_cxx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
/* Same expectation contract as the C entry point below; throws on error. */
double qcs_simulator_expectation_pauli_cxx(qcs_simulator *sim, const bit_num_t *qubit_num_list, const qcs_pauli_operator *operators, bit_num_t qubit_num_count, double coefficient);
int qcs_simulator_measure_cxx(qcs_simulator *sim, bit_num_t qubit_num);
int qcs_simulator_measure_to_clbit_cxx(qcs_simulator *sim, bit_num_t qubit_num, bit_num_t clbit_num);
void qcs_simulator_measure_many_cxx(qcs_simulator *sim, const bit_num_t *qubit_num_list, bit_num_t qubit_num_count, bit_t *results);
void qcs_simulator_measure_many_to_clbits_cxx(qcs_simulator *sim, const bit_num_t *qubit_num_list, bit_num_t qubit_num_count, const bit_num_t *clbit_num_list, bit_num_t clbit_num_count, bit_t *results);
int qcs_simulator_read_cxx(qcs_simulator *sim, bit_num_t clbit_num);
void qcs_simulator_save_statevector_cxx(qcs_simulator *sim, const char *outfn);
int qcs_simulator_event_create_cxx(qcs_simulator *sim);
void qcs_simulator_event_record_cxx(qcs_simulator *sim, int event_num);
double qcs_simulator_event_get_elapsed_time_cxx(qcs_simulator *sim, event_num_t start_event_num, event_num_t stop_event_num);
extern "C"
{
#endif

int qcs_set_exception_callback(qcs_exception_callback callback);
int qcs_simulator_create(qcs_simulator **result);
int qcs_simulator_destroy(qcs_simulator *sim);
/* Allocate simulator storage without initializing or zero-filling the state vector.
 * Call qcs_simulator_set_zero_state (or another state setter) before use.
 * The default method is zero; with_initial_state only selects the method for
 * subsequent set_zero_state calls and does not execute it during allocation. */
int qcs_simulator_allocate_memory(qcs_simulator *sim);
int qcs_simulator_allocate_memory_with_initial_state(qcs_simulator *sim, qcs_initial_state initial_state);
int qcs_simulator_dispose(qcs_simulator *sim);
int qcs_simulator_get_num_procs(qcs_simulator *sim, bit_t *result);
int qcs_simulator_get_proc_num(qcs_simulator *sim, bit_t *result);
int qcs_simulator_get_num_qubits(const qcs_simulator *sim, bit_t *result);
int qcs_simulator_get_num_clbits(const qcs_simulator *sim, bit_t *result);
int qcs_simulator_set_num_qubits(qcs_simulator *sim, bit_num_t num_qubits);
int qcs_simulator_set_mapping(qcs_simulator *sim, const bit_num_t *perm_p2l, bit_num_t perm_p2l_count);
int qcs_simulator_set_num_clbits(qcs_simulator *sim, bit_num_t num_clbits);
int qcs_simulator_get_clbits(const qcs_simulator *sim, bit_t *clbits);
int qcs_simulator_get_clbits_string(const qcs_simulator *sim, char *clbits_string);
int qcs_simulator_reset(qcs_simulator *sim, bit_num_t qubit_num);
/* Initializes/restores the method selected at allocation (zero by default)
 * and clears measurement metadata.
 * Mapping and classical bits are reset separately by the caller. */
int qcs_simulator_set_zero_state(qcs_simulator *sim);
int qcs_simulator_set_sequential_state(qcs_simulator *sim);
int qcs_simulator_set_flat_state(qcs_simulator *sim);
int qcs_simulator_set_entangled_state(qcs_simulator *sim);
int qcs_simulator_set_random_state(qcs_simulator *sim);
int qcs_simulator_reset_clbits(qcs_simulator *sim);
int qcs_simulator_reset_measurement_state(qcs_simulator *sim);
int qcs_simulator_reinitialize_mapping(qcs_simulator *sim);
int qcs_simulator_gate_global_phase(qcs_simulator *sim, double theta, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_h(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_x(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_y(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_z(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_s(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_sdg(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_t(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_tdg(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_sx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_ry(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rz(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_u4(qcs_simulator *sim, double theta, double phi, double lambda, double gamma, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_u3(qcs_simulator *sim, double theta, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_u2(qcs_simulator *sim, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_u1(qcs_simulator *sim, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_u(qcs_simulator *sim, double theta, double phi, double lambda, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_p(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_swap(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_swap1(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_swap2(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_swap3(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_swap4(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_iswap(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_id(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_sxdg(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_r(qcs_simulator *sim, double theta, double phi, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rxx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_ryy(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rzz(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rzx(qcs_simulator *sim, double theta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_dcx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_ecr(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_xx_plus_yy(qcs_simulator *sim, double theta, double beta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_xx_minus_yy(qcs_simulator *sim, double theta, double beta, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rccx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
int qcs_simulator_gate_rcccx(qcs_simulator *sim, const bit_num_t *target_qubit_num_list, bit_num_t target_qubit_num_count, const bit_num_t *negctrl_qubit_num_list, bit_num_t negctrl_qubit_num_count, const bit_num_t *ctrl_qubit_num_list, bit_num_t ctrl_qubit_num_count);
/*
 * Compute coefficient * <psi|P|psi> without implicit normalization.
 * Parallel arrays specify distinct logical qubits and I/X/Y/Z operators.
 * A zero count is identity and permits NULL arrays; result must not be NULL.
 * The coefficient must be finite. Non-identity support must fit local qubits.
 * Requires allocated, explicitly initialized state storage. Pending measurement
 * projections are applied; recorded outcomes, classical bits, and RNG are kept.
 * Global targets are localized and the updated mapping is retained.
 * Collective: all ranks must supply identical arguments in the same call order.
 * Calls on a simulator must be serialized. Returns 1 on success, 0 on error
 * through the exception callback. The result is written only on success.
 */
int qcs_simulator_expectation_pauli(qcs_simulator *sim, const bit_num_t *qubit_num_list, const qcs_pauli_operator *operators, bit_num_t qubit_num_count, double coefficient, double *result);
int qcs_simulator_measure(qcs_simulator *sim, bit_num_t qubit_num, bit_t *result);
int qcs_simulator_measure_to_clbit(qcs_simulator *sim, bit_num_t qubit_num, bit_num_t clbit_num, bit_t *result);
/*
 * Measure distinct logical qubits in Z basis.  Results preserve input order:
 * results[i] belongs to qubit_num_list[i].  A zero-count request is a
 * successful no-op and permits NULL list/result pointers.
 */
int qcs_simulator_measure_many(qcs_simulator *sim, const bit_num_t *qubit_num_list, bit_num_t qubit_num_count, bit_t *results);
/* Measures local qubits in descending physical order, then global qubits in
 * ascending physical order, using the current mapping. Assigns results[i] to
 * clbit_num_list[i]. Results retain input order. Counts must match.
 */
int qcs_simulator_measure_many_to_clbits(qcs_simulator *sim, const bit_num_t *qubit_num_list, bit_num_t qubit_num_count, const bit_num_t *clbit_num_list, bit_num_t clbit_num_count, bit_t *results);
int qcs_simulator_read(qcs_simulator *sim, bit_num_t clbit_num, bit_t *result);
int qcs_simulator_save_statevector(qcs_simulator *sim, const char *outfn);
int qcs_simulator_event_create(qcs_simulator *sim, bit_t *result);
int qcs_simulator_event_record(qcs_simulator *sim, int event_num);
int qcs_simulator_event_get_elapsed_time(qcs_simulator *sim, event_num_t start_event_num, event_num_t stop_event_num, double *result);
#ifdef __cplusplus
}
#endif

#ifdef __cplusplus
/* Preserve the original C symbol and one-argument source/ABI compatibility. */
inline int qcs_simulator_allocate_memory(qcs_simulator *sim, qcs_initial_state initial_state)
{
    return qcs_simulator_allocate_memory_with_initial_state(sim, initial_state);
}
#endif
