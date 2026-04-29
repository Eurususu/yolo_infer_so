#include "preprocess.h"
#include <iostream>


__global__ void preprocess_kernel(
    const uint8_t** src_imgs, // 指针的指针，指向当前 batch 每个图像的独立显存地址
    float* dst_blob,          // 输出的 NCHW 张量 (TensorRT input)
    const int* src_widths,    // 每个图像的宽
    const int* src_heights,   // 每个图像的高
    int dst_w, int dst_h,     // 目标宽高 (如 640x640)
    int batch_size            // 依然需要，用于防止越界
) {
    int x = blockIdx.x * blockDim.x + threadIdx.x;
    int y = blockIdx.y * blockDim.y + threadIdx.y;
    int b = blockIdx.z * blockDim.z + threadIdx.z;

    if (x >= dst_w || y >= dst_h || b >= batch_size) return;

    // 1. 获取当前图片的宽高和独立数据指针
    int src_w = src_widths[b];
    int src_h = src_heights[b];
    const uint8_t* img_b = src_imgs[b];

    // 2. 在 GPU 内部毫秒级推导 scale, dw, dh
    float scale = fminf((float)dst_w / src_w, (float)dst_h / src_h);
    int new_w = roundf(src_w * scale);
    int new_h = roundf(src_h * scale);
    int dw = (dst_w - new_w) / 2; //偏移量 也就是padding
    int dh = (dst_h - new_h) / 2;

    // 默认背景色归一化值 (114 / 255.0)
    float r = 114.0f / 255.0f, g = 114.0f / 255.0f, b_c = 114.0f / 255.0f;

    // 3. 判断是否在有效缩放区域内 (双线性插值)
    if (x >= dw && x < dw + new_w && y >= dh && y < dh + new_h) {

        //中心对齐
        float scale_x = (float)src_w / new_w;
        float scale_y = (float)src_h / new_h;

        float src_x = (x - dw + 0.5f) * scale_x - 0.5f;
        float src_y = (y - dh + 0.5f) * scale_y - 0.5f;

        // //左上角对齐
        // float src_x = (x - dw) / scale;
        // float src_y = (y - dh) / scale;

        int x1 = floorf(src_x);
        int y1 = floorf(src_y);
        int x2 = min(x1 + 1, src_w - 1);
        int y2 = min(y1 + 1, src_h - 1);

        x1 = max(0, min(x1, src_w - 1));
        y1 = max(0, min(y1, src_h - 1));

        float dx = src_x - x1;
        float dy = src_y - y1;

        int stride = src_w * 3;
        
        float b11 = img_b[y1 * stride + x1 * 3 + 0], g11 = img_b[y1 * stride + x1 * 3 + 1], r11 = img_b[y1 * stride + x1 * 3 + 2];
        float b12 = img_b[y1 * stride + x2 * 3 + 0], g12 = img_b[y1 * stride + x2 * 3 + 1], r12 = img_b[y1 * stride + x2 * 3 + 2];
        float b21 = img_b[y2 * stride + x1 * 3 + 0], g21 = img_b[y2 * stride + x1 * 3 + 1], r21 = img_b[y2 * stride + x1 * 3 + 2];
        float b22 = img_b[y2 * stride + x2 * 3 + 0], g22 = img_b[y2 * stride + x2 * 3 + 1], r22 = img_b[y2 * stride + x2 * 3 + 2];

        float b_val = (1 - dx) * (1 - dy) * b11 + dx * (1 - dy) * b12 + (1 - dx) * dy * b21 + dx * dy * b22;
        float g_val = (1 - dx) * (1 - dy) * g11 + dx * (1 - dy) * g12 + (1 - dx) * dy * g21 + dx * dy * g22;
        float r_val = (1 - dx) * (1 - dy) * r11 + dx * (1 - dy) * r12 + (1 - dx) * dy * r21 + dx * dy * r22;

        // BGR 转 RGB 并归一化
        r = r_val / 255.0f;
        g = g_val / 255.0f;
        b_c = b_val / 255.0f;
    }

    // 4. 直接写入 NCHW 内存
    int area = dst_w * dst_h;
    int b_offset = b * 3 * area;
    
    dst_blob[b_offset + 0 * area + y * dst_w + x] = r;
    dst_blob[b_offset + 1 * area + y * dst_w + x] = g;
    dst_blob[b_offset + 2 * area + y * dst_w + x] = b_c;
}


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
)
{
    int batch_size = image_list.size();

    out_scales.resize(batch_size);
    out_dws.resize(batch_size);
    out_dhs.resize(batch_size);

    std::vector<int>h_image_widths(batch_size);
    std::vector<int>h_image_heights(batch_size);
    // std::vector<uint8_t*>h_img_ptrs(batch_size);

    for (int b = 0; b < batch_size; ++b){
        int src_w = image_list[b].cols;
        int src_h = image_list[b].rows;

        h_image_widths[b] = src_w;
        h_image_heights[b] = src_h;

        float scale = std::min((float)dst_h / src_h, (float)dst_w / src_w);
        int new_w = static_cast<int>(std::round(src_w * scale));
        int new_h = static_cast<int>(std::round(src_h * scale));
        out_scales[b] = scale;
        out_dws[b] = (dst_w - new_w) / 2;
        out_dhs[b] = (dst_h - new_h) / 2;

        // 分配这帧图的目标地址 (假设 d_img_buffers 是二维数组/内存池)
        // h_img_ptrs[b] = d_img_buffers[b];

        // 拷贝图片数据到gpu
        cudaMemcpyAsync(
            d_img_buffers[b],
            image_list[b].ptr<uint8_t>(),
            src_w * src_h * 3,
            cudaMemcpyHostToDevice,
            stream
        );

    }

    int *d_image_widths = nullptr, *d_image_heights = nullptr;
    cudaMalloc((void**)&d_image_widths, batch_size * sizeof(int));
    cudaMalloc((void**)&d_image_heights, batch_size * sizeof(int));

    // 将宽、高数组，以及指针目录拷贝进 GPU
    cudaMemcpyAsync(d_image_widths, h_image_widths.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    cudaMemcpyAsync(d_image_heights, h_image_heights.data(), batch_size * sizeof(int), cudaMemcpyHostToDevice, stream);
    // 拷贝显存地址到gpu
    cudaMemcpyAsync(d_img_ptrs, d_img_buffers.data(), batch_size * sizeof(uint8_t*), cudaMemcpyHostToDevice, stream);

    // 启动 Kernel

    dim3 block(16, 16, 1);
    dim3 grid((dst_w + block.x - 1) / block.x,
            (dst_h + block.y - 1) / block.y,
             batch_size);


    preprocess_kernel<<<grid, block, 0, stream>>>(
        (const uint8_t**)d_img_ptrs, d_dst_blob, 
        d_image_widths, d_image_heights, 
        dst_w, dst_h, batch_size
    );
    cudaStreamSynchronize(stream);

    // 释放临时元数据显存...
    // 彻底销毁/归还 显存
    cudaFreeAsync(d_image_widths, stream);
    cudaFreeAsync(d_image_heights, stream);

}
