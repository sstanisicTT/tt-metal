// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include "dataflow_api.h"

void kernel_main() {
    uint32_t num_tiles = get_arg_val<uint32_t>(2);  // Index 2 to match with regular writer_unary

    constexpr uint32_t cb_id_out0 = 16;
    uint32_t ublock_size_tiles = 1;

    for (uint32_t i = 0; i < num_tiles; i++) {
        cb_wait_front(cb_id_out0, ublock_size_tiles);
        cb_pop_front(cb_id_out0, ublock_size_tiles);
    }
}
