#pragma once

#include <vector>
#include <opencv2/opencv.hpp>
#include <cuda_runtime_api.h>

// 声明我们要调用的 Host 端 CUDA 包装函数
void launch_preprocess_cuda(
    const std::vector<cv::Mat>& image_list,
    float* d_dst_blob, 
    int dst_w, int dst_h,
    const std::vector<uint8_t*>& d_img_buffers, 
    uint8_t** d_img_ptrs,
    std::vector<float>& out_scales,
    std::vector<int>& out_dws,
    std::vector<int>& out_dhs,
    cudaStream_t stream
);
