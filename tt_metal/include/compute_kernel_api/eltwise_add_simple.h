

#include "compute_kernel_api/common.h"

namespace ckernel {

ALWI void eltwise_add_simple_init(uint32_t icb0, uint32_t icb1, uint32_t ocb) {
    UNPACK((llk_unpack_AB_simple_init))
    
    MATH((llk_math_eltwise_binary_init<eltwise_binary_type, NONE, MATH_FIDELITY>(0 /*transpose*/, false)));
    MATH((llk_math_pack_sync_init<DST_ACCUM_MODE>()));
    MATH((llk_math_hw_configure_disaggregated(icb0, icb1)));

    PACK((llk_pack_hw_configure_disaggregated<false, DST_ACCUM_MODE>(ocb)));
    PACK((llk_pack_init(ocb)));
    PACK((llk_pack_dest_init<false, DST_ACCUM_MODE>()));

    



}

ALWI void eltwise_add_simple(uint32_t icb0, uint32_t icb1, uint32_t itile0, uint32_t itile1, uint32_t idst) {
    UNPACK((llk_unpack_AB_simple(icb0, icb1, itile0, itile1)));

}

}
