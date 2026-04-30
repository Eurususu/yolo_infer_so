#pragma once

#include <vector>
#include <cstdint>
#include <cuda_runtime_api.h>

// 声明我们要调用的 Host 端 CUDA 包装函数
void launch_preprocess_cuda(
    const std::vector<const unsigned char*>& host_img_ptrs,
    const std::vector<int>& img_widths, 
    const std::vector<int>& img_heights,
    float* d_dst_blob, 
    int dst_w, int dst_h,
    const std::vector<uint8_t*>& d_img_buffers, 
    uint8_t** d_img_ptrs,
    int* d_image_widths,  // 🌟 新增：由外部传入预分配好的显存！
    int* d_image_heights, // 🌟 新增：由外部传入预分配好的显存！
    std::vector<float>& out_scales,
    std::vector<int>& out_dws,
    std::vector<int>& out_dhs,
    cudaStream_t stream
);
