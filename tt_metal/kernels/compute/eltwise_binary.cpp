// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "compute_kernel_api/eltwise_binary.h"

#include <cstdint>

#include "compute_kernel_api/eltwise_unary/sfpu_split_includes.h"
#include "compute_kernel_api/tile_move_copy.h"
namespace NAMESPACE {
void MAIN {
    uint32_t acc_to_dst = get_arg_val<uint32_t>(2);

    constexpr auto cb_in0 = tt::CBIndex::c_0;
    constexpr auto cb_in1 = tt::CBIndex::c_1;
    constexpr auto cb_inp0 = cb_in0;
    constexpr auto cb_inp1 = cb_in1;
    constexpr auto cb_out0 = tt::CBIndex::c_16;

    constexpr auto cb_in2 = tt::CBIndex::c_2;

    binary_op_init_common(cb_inp0, cb_inp1, cb_out0);

#if not defined ELTWISE_DEST_REUSE_TYPE
#ifdef FULL_INIT
    binary_tiles_init<true, ELTWISE_OP_TYPE>(cb_in0, cb_in1);
#else
    binary_tiles_init<false, ELTWISE_OP_TYPE>(cb_in0, cb_in1);
#endif
#endif

    for (uint32_t i = 0; i < 2048; ++i) {
            pack_tile(i, cb_out0);
    }
}
}  // namespace NAMESPACE
