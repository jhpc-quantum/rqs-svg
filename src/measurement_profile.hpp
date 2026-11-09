#pragma once
// Ranges label host enqueue work; use correlated CUDA/MPI trace events for time.
// Compile-time opt-in: normal builds have no tracing calls or dependencies.
#ifdef QCS_MEASUREMENT_PROFILE
#include <nvtx3/nvToolsExt.h>
namespace qcs {
struct measurement_profile_range {
    explicit measurement_profile_range(const char* name) { nvtxRangePushA(name); }
    void next(const char* name) { nvtxRangePop(); nvtxRangePushA(name); }
    ~measurement_profile_range() { nvtxRangePop(); }
};
}
#define QCS_MEASURE_RANGE(name) qcs::measurement_profile_range measurement_range(name)
#define QCS_MEASURE_STAGE(name) measurement_range.next(name)
#else
#define QCS_MEASURE_RANGE(name) ((void)0)
#define QCS_MEASURE_STAGE(name) ((void)0)
#endif
