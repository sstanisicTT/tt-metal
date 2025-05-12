// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#pragma once
#include "llk_math_common_api.h"
#include "llk_math_eltwise_binary.h"

/*************************************************************************
 * LLK ELTWISE BINARY
 *************************************************************************/

// Version with no operand
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init(const std::uint32_t transpose = 0, const std::uint32_t acc_to_dest = 0) {

    // math thread will only be clearing valid bits to push unpacker to maximum theoretical throughput
    ckernel_template tmp(127, 127, TT_OP_CLEARDVALID(0b11, 0));
    tmp.program(instrn_buffer);

    // const std::uint32_t num_faces = 4;

    // _llk_math_eltwise_binary_init_<eltwise_binary_type, src_b_bcast_type, NUM_FIDELITY_PHASES, binary_reuse_dest>(
    //     num_faces, transpose, acc_to_dest);
}

// Version with operands
template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE>
inline void llk_math_eltwise_binary_init_with_operands(
    const std::uint32_t operand_A,
    const std::uint32_t operand_B,
    const std::uint32_t transpose = 0,
    const std::uint32_t acc_to_dest = 0) {
    const std::uint32_t operand_id =
        get_operand_id(operand_A);  // operand_id is used to extract tile dim data which is the same for both operands
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    _llk_math_eltwise_binary_init_<eltwise_binary_type, src_b_bcast_type, NUM_FIDELITY_PHASES, binary_reuse_dest>(
        num_faces, transpose, acc_to_dest);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary(uint dst_index, const bool clear_fp32_dst_acc = true) {
    const std::uint32_t num_faces = 4;

    _llk_math_eltwise_binary_<
        eltwise_binary_type,
        src_b_bcast_type,
        DST_SYNC_MODE,
        NUM_FIDELITY_PHASES,
        binary_reuse_dest,
        is_fp32_dest_acc_en>(num_faces, dst_index, clear_fp32_dst_acc);
}

template <
    EltwiseBinaryType eltwise_binary_type,
    BroadcastType src_b_bcast_type,
    int NUM_FIDELITY_PHASES = 0,
    EltwiseBinaryReuseDestType binary_reuse_dest = EltwiseBinaryReuseDestType::NONE,
    bool is_fp32_dest_acc_en = false>
inline void llk_math_eltwise_binary(
    const std::uint32_t operand_A,
    const std::uint32_t operand_B,
    uint dst_index,
    const bool clear_fp32_dst_acc = true) {
    const std::uint32_t operand_id = get_operand_id(operand_A);  // both operands must have same number of faces
    const std::uint32_t num_faces = get_operand_num_faces(operand_id);

    // for iterations of mop  should be enough for the unpacker to finish unpacking 512 tiles
    for(int i=0; i<4; i++) {
        ckernel_template::run(instrn_buffer);
    }

    // _llk_math_eltwise_binary_<
    //     eltwise_binary_type,
    //     src_b_bcast_type,
    //     DST_SYNC_MODE,
    //     NUM_FIDELITY_PHASES,
    //     binary_reuse_dest,
    //     is_fp32_dest_acc_en>(num_faces, dst_index, clear_fp32_dst_acc);
}
