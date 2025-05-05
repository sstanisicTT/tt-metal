
#include <cstdint>
#include <exception>
#include <tt-metalium/device.hpp>
#include <tt-metalium/host_api.hpp>

namespace tt {
namespace tt_metal {
class CommandQueue;
}  // namespace tt_metal
}  // namespace tt

using namespace tt;
using namespace tt_metal;

int main(int argc, char** argv) {
    bool pass = true;

    int device_id = 0;
    tt_metal::IDevice* device = tt_metal::CreateDevice(device_id);

    CommandQueue& cq = device->command_queue();
    Program program = tt_metal::CreateProgram();

    log_info("====================================================================");
    log_info("======= Running eltwise_add_simple test");

    try {
        CoreCoord core = {0, 0};
        uint32_t single_tile_size = 2 * 1024;  // 1024 fp16b per tile
        uint32_t num_tiles = 2048;             // 2048 tiles
        uint32_t dram_buffer_size = single_tile_size * num_tiles;

        tt_metal::InterleavedBufferConfig dram_config{
            .device = device,
            .size = dram_buffer_size,
            .page_size = dram_buffer_size,  // no interleaving
            .buffer_type = tt_metal::BufferType::DRAM};

        // START setup DRAM buffers
        auto dram_buffer_src0 = CreateBuffer(dram_config);
        uint32_t dram_buffer_src0_addr = dram_buffer_src0->address();

        auto dram_buffer_src1 = CreateBuffer(dram_config);
        uint32_t dram_buffer_src1_addr = dram_buffer_src1->address();

        auto dram_buffer_dst = CreateBuffer(dram_config);
        uint32_t dram_buffer_dst_addr = dram_buffer_dst->address();
        // END setup DRAM buffers

        // START setup circular buffers
        uint32_t num_input_tiles = 2;
        uint32_t num_output_tiles = 2;

        uint32_t cb_src0_idx = tt::CBIndex::c_0;
        tt_metal::CircularBufferConfig cb_src0_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{cb_src0_idx, tt::DataFormat::Float16_b}})
                .set_page_size(cb_src0_idx, single_tile_size);
        auto cb_src0 = tt_metal::CreateCircularBuffer(program, core, cb_src0_config);

        uint32_t cb_src1_idx = tt::CBIndex::c_1;
        tt_metal::CircularBufferConfig cb_src1_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{cb_src1_idx, tt::DataFormat::Float16_b}})
                .set_page_size(cb_src1_idx, single_tile_size);
        auto cb_src1 = tt_metal::CreateCircularBuffer(program, core, cb_src1_config);

        uint32_t cb_dst_idx = tt::CBIndex::c_16;
        tt_metal::CircularBufferConfig cb_dst_config =
            tt_metal::CircularBufferConfig(
                num_input_tiles * single_tile_size, {{cb_dst_idx, tt::DataFormat::Float16_b}})
                .set_page_size(cb_dst_idx, single_tile_size);
        auto cb_dst = tt_metal::CreateCircularBuffer(program, core, cb_dst_config);
        // END setup circular buffers

        auto binary_reader_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/reader_binary_diff_lengths.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_1, .noc = tt_metal::NOC::RISCV_1_default});

        auto unary_writer_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/dataflow/writer_unary.cpp",
            core,
            tt_metal::DataMovementConfig{
                .processor = tt_metal::DataMovementProcessor::RISCV_0, .noc = tt_metal::NOC::RISCV_0_default});

        std::vector<uint32_t> compute_kernel_args = {};

        std::map<std::string, std::string> binary_defines = {
            {"ELTWISE_OP", "add_tiles"}, {"ELTWISE_OP_TYPE", "EltwiseBinaryType::ELWADD"}};
        
        auto eltwise_binary_kernel = tt_metal::CreateKernel(
            program,
            "tt_metal/kernels/compute/eltwise_binary.cpp",
            core,
            tt_metal::ComputeConfig{.compile_args = compute_kernel_args, .defines = binary_defines});
        SetRuntimeArgs(program, eltwise_binary_kernel, core, {2048, 1});

        std::vector<uint32_t> src0_vec = create_random_vector_of_bfloat16(
            dram_buffer_size, 100, std::chrono::system_clock::now().time_since_epoch().count());
        EnqueueWriteBuffer(cq, std::ref(dram_buffer_src0), src0_vec, false);

        std::vector<uint32_t> src1_vec = create_constant_vector_of_bfloat16(dram_buffer_size, 0.0f);
        EnqueueWriteBuffer(cq, std::ref(dram_buffer_src0), src1_vec, false);

        const std::array<uint32_t, 7> reader_args = {
            dram_buffer_src0_addr, 0, num_tiles, dram_buffer_src1_addr, 0, num_tiles, 0};

        const std::array<uint32_t, 3> writer_args = {dram_buffer_dst_addr, 0, num_tiles};

        SetRuntimeArgs(program, unary_writer_kernel, core, writer_args);
        SetRuntimeArgs(program, binary_reader_kernel, core, reader_args);

        EnqueueProgram(cq, program, false);
        std::vector<uint32_t> result_vec;
        EnqueueReadBuffer(cq, dram_buffer_dst, result_vec, true);

        for(uint32_t idx=0; idx<result_vec.size(); idx++) {
            if(src0_vec[idx] != result_vec[idx])
                log_info("Missmatch: index={} src_val={} res_val={}", idx, src0_vec[idx], result_vec[idx]);
            if(idx> 2100) break;
        }

        
        pass &= (src0_vec == result_vec);


    } catch (const std::exception& ex) {
        pass = false;
        log_error(LogTest, "{}", ex.what());
        log_error(LogTest, "System error message: {}", std::strerror(errno));
    }

    pass &= tt_metal::CloseDevice(device);

    if (!pass) {
        TT_THROW("Test Failed");
    }

    log_info(LogTest, "Test Passed");

    return 0;
}
