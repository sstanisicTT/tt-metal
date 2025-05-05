
#include "compute_kernel_api/eltwise_add_simple.h"


namespace NAMESPACE {
void MAIN {
    uint32_t per_core_block_cnt = get_arg_val<uint32_t>(0);
    uint32_t per_core_block_size = get_arg_val<uint32_t>(1);

    constexpr auto cb_src0 = tt::CBIndex::c_0;
    constexpr auto cb_src1 = tt::CBIndex::c_1;
    constexpr auto cb_dst = tt::CBIndex::c_16;

    eltwise_add_simple_init(cb_src0, cb_src1, cb_dst);

    for(uint32_t block = 0; block < per_core_block_cnt; ++block) {
        cb_wait_front(cb_src0, per_core_block_size);
        cb_wait_front(cb_src1, per_core_block_size);
        cb_reserve_back(cb_dst, per_core_block_size);

        tile_regs_aquire();

        for(uint32_t i = 0; i < per_core_block_size; ++i) {
            eltwise_add_simple(cb_src0, cb_src1, i, i, i);
        }

        tile_regs_commit();

        tile_regs_wait();
        for(uint32_t i = 0; i < per_core_block_size; ++i) {
            pack_tile(i, cb_out0);      // todo : explore
        }
        tile_regs_release();

        cb_pop_front(cb_src0, per_core_block_size);
        cb_pop_front(cb_src1, per_core_block_size);
        cb_push_back(cb_dst, per_core_block_size);
    }

}
}

