// SPDX-FileCopyrightText: © 2023 Tenstorrent Inc.
//
// SPDX-License-Identifier: Apache-2.0

#include <chrono>
#include <errno.h>
#include <fmt/base.h>
#include <stdint.h>
#include <stdlib.h>
#include <tt-metalium/bfloat16.hpp>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>
#include <algorithm>
#include <array>
#include <cstring>
#include <exception>
#include <functional>
#include <map>
#include <memory>
#include <string>
#include <utility>
#include <variant>
#include <vector>

#include <tt-metalium/assert.hpp>
#include <tt-metalium/buffer.hpp>
#include <tt-metalium/buffer_types.hpp>
#include <tt-metalium/circular_buffer_config.hpp>
#include <tt-metalium/core_coord.hpp>
#include <tt-metalium/data_types.hpp>
#include "hostdevcommon/kernel_structs.h"
#include <tt-metalium/kernel_types.hpp>
#include <tt-metalium/logger.hpp>
#include <tt-metalium/program.hpp>
#include <tt_stl/span.hpp>
#include "test_gold_impls.hpp"
#include <tt-metalium/tt_backend_api_types.hpp>

#include <tt-metalium/tt_metal.hpp>

namespace tt {
namespace tt_metal {
class CommandQueue;
}  // namespace tt_metal
}  // namespace tt

using std::vector;
using namespace tt;
using namespace tt::tt_metal;

int main(int argc, char** argv) {
    if (getenv("TT_METAL_SLOW_DISPATCH_MODE") != nullptr) {
        TT_THROW("Test not supported w/ slow dispatch, exiting");
    }

    int device_id = 0;
    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    CommandQueue& cq = device->command_queue();

    Program program = tt_metal::CreateProgram();

    log_info(LogTest, "===========================================");
    log_info(LogTest, "======= Running eltwise_binary_sfpu =======");

    try {
        ////////////////////////////////////////////////////////////////////////////
        //                      Application Setup
        ////////////////////////////////////////////////////////////////////////////
       
        CoreCoord core = {0, 0};

        uint32_t tile_size = 1024 * 2;
        uint32_t tile_count = 512;
        
        uint32_t dram_buffer_size = tile_count * tile_size;
        
        tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = tile_size,
            .buffer_type = tt_metal::BufferType::DRAM};

        vector<uint32_t> kernel_args = {tile_count, 1};
        std::map<std::string, std::string> kernel_defines = {
            {"SFPU_OP_CHAIN_0", "gelu_tile_init(); gelu_tile(0);"},
            {"SFPU_OP_GELU_INCLUDE", "1"},
            {"SFPU_OP_RELU_FAMILY_INCLUDE", "1"},
            {"SFPU_OP_COMPUTE_KERNEL_API_INCLUDE", "1"},
        };

        auto eltwise_sfpu_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_sfpu.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = kernel_args, .defines = kernel_defines});

        SetRuntimeArgs(program, eltwise_sfpu_kernel, core, kernel_args);
        EnqueueProgram(cq, program, false);
       
    } catch (const std::exception& e) {
        // Capture the exception error message
        log_error(LogTest, "{}", e.what());
        // Capture system call errors that may have returned from driver/kernel
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    detail::DumpDeviceProfileResults(device);
    tt_metal::CloseDevice(device);

    return 0;
}
