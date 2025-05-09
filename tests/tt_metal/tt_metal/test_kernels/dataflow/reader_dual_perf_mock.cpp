// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <stdint.h>
#include "dataflow_api.h"

void kernel_main() {
    // same arg indices as in reader_binary_diff_lenghts for compat
    uint32_t src0_num_tiles = get_arg_val<uint32_t>(2);
    uint32_t src1_num_tiles = get_arg_val<uint32_t>(5);

    constexpr uint32_t cb_id_in0 = 0;
    constexpr uint32_t cb_id_in1 = 1;

    uint32_t ublock_size_tiles = 1;

    uint32_t num_tiles = src0_num_tiles > src1_num_tiles ? src0_num_tiles : src1_num_tiles;

    // read ublocks from src0/src1 to CB0/CB1, then push ublocks to compute (unpacker)
    for (uint32_t i = 0; i < num_tiles; i += ublock_size_tiles) {
        if (i < src0_num_tiles) {
            cb_reserve_back(cb_id_in0, ublock_size_tiles);
            cb_push_back(cb_id_in0, ublock_size_tiles);
        }

        if (i < src1_num_tiles) {
            cb_reserve_back(cb_id_in1, ublock_size_tiles);
            cb_push_back(cb_id_in1, ublock_size_tiles);
        }
    }
}
