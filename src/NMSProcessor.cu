#include "NMSProcessor.h"
#include "efficientNMSInference.cuh"
#include <cstdio>

// template <typename T>
// __global__ void YOLODecodeKernel(
//     const T* __restrict__ input,  // GPU: [batchSize, numAnchors, 5 + numClasses]
//     T* __restrict__ boxes,        // GPU: [batchSize, numAnchors, 4]
//     T* __restrict__ scores,       // GPU: [batchSize, numAnchors, numClasses]
//     int32_t numAnchors,
//     int32_t numClasses,
//     int32_t batchSize)
// {
//     int32_t anchorIdx = static_cast<int32_t>(blockDim.x * blockIdx.x + threadIdx.x);
//     int32_t batchIdx  = static_cast<int32_t>(blockIdx.y);

//     if (anchorIdx >= numAnchors || batchIdx >= batchSize)
//     {
//         return;
//     }

//     // 计算当前 anchor 在输入张量中的起始偏移 每个anchor之间的偏移量
//     int32_t stride = 5 + numClasses;
//     // 記憶體全部都是「排成一長條」的一維陣列（1D Array） input就是[batchSize, numAnchors, stride]的起始地址
//     // src就是当前anchor的起始地址，也就是 这个框的cx
//     const T* src = input + (batchIdx * numAnchors + anchorIdx) * stride;

//     // ---- 框解码：(cx, cy, w, h) → BoxCorner (x1, y1, x2, y2) ----
//     // 使用显式的中间变量避免 FP16 精度丢失
//     T cx = src[0];
//     T cy = src[1];
//     T w  = src[2];
//     T h  = src[3];
//     T hw = w * (T) 0.5F; // 半宽
//     T hh = h * (T) 0.5F; // 半高

//     T* dstBox = boxes + (batchIdx * numAnchors + anchorIdx) * 4;
//     dstBox[0] = cx - hw; // x1 = cx - w/2
//     dstBox[1] = cy - hh; // y1 = cy - h/2
//     dstBox[2] = cx + hw; // x2 = cx + w/2
//     dstBox[3] = cy + hh; // y2 = cy + h/2

//     // ---- 分数融合：score[c] = obj_conf * class_prob[c] ----
//     // 结果用于 EfficientNMS 的 scoresInput（[batch, anchors, numClasses]）
//     T objConf = src[4];
//     T* dstScore = scores + (batchIdx * numAnchors + anchorIdx) * numClasses;
//     for (int32_t c = 0; c < numClasses; ++c)
//     {
//         dstScore[c] = objConf * src[5 + c];
//     }
// }


// template <typename T>
// __global__ void YOLODecodeKernel_ultralytics(
//     const T* __restrict__ input,  // GPU: [batchSize, 4 + numClasses, numAnchors]
//     T* __restrict__ boxes,        // GPU: [batchSize, numAnchors, 4]
//     T* __restrict__ scores,       // GPU: [batchSize, numAnchors, numClasses]
//     int32_t numAnchors,
//     int32_t numClasses,
//     int32_t batchSize
// )
// {
//     int32_t anchorIdx = static_cast<int32_t>(blockDim.x * blockIdx.x + threadIdx.x);
//     int32_t batchIdx = static_cast<int32_t>(blockIdx.y);

//     if (anchorIdx >= numAnchors || batchIdx >= batchSize){
//         return;
//     }

//     int32_t numChannels = 4 + numClasses;

//     // ---- 1. 内存寻址：定位到当前 Batch 的起始地址 ----
//     const T* batchInput = input + batchIdx * (numAnchors * numChannels);

//     // ---- 2. 跨步读取：获取 cx, cy, w, h ----
//     T cx = batchInput[0 * numAnchors + anchorIdx]; // 第 0 个通道
//     T cy = batchInput[1 * numAnchors + anchorIdx]; // 第 1 个通道
//     T w  = batchInput[2 * numAnchors + anchorIdx]; // 第 2 个通道
//     T h  = batchInput[3 * numAnchors + anchorIdx]; // 第 3 个通道

//     T hw = w * (T) 0.5F;
//     T hh = h * (T) 0.5F;

//     // ---- 3. 写入输出 Box ----
//     // NMS 插件需要的 boxes 格式依然是连续的 [x1, y1, x2, y2]，所以输出的目标地址计算方式不变
//     T* dstBox = boxes + (batchIdx * numAnchors + anchorIdx) * 4;
//     dstBox[0] = cx - hw;
//     dstBox[1] = cy - hh;
//     dstBox[2] = cx + hw;
//     dstBox[3] = cy + hh;

//     // ---- 4. 读取并写入类别分数 ----
//     // 从第 4 个通道开始，后面全都是类别分数。不再有 obj_conf，直接读取即可。
//     T* dstScore = scores + (batchIdx * numAnchors + anchorIdx) * numClasses;
//     for (int32_t c = 0; c < numClasses; ++c){
//         dstScore[c] = batchInput[(4 + c) * numAnchors + anchorIdx];
//     }
// }

template <typename T>
__global__ void YOLODecodeKernel(
    const T* __restrict__ input,  // GPU: [batchSize, numAnchors, 5 + numClasses]
    T* __restrict__ boxes,        // GPU: [batchSize, numAnchors, 4]
    T* __restrict__ scores,       // GPU: [batchSize, numAnchors, numClasses]
    int32_t numAnchors,
    int32_t numClasses,
    int32_t batchSize)
{
    int32_t anchorIdx = static_cast<int32_t>(blockDim.x * blockIdx.x + threadIdx.x);
    int32_t batchIdx  = static_cast<int32_t>(blockIdx.y);

    if (anchorIdx >= numAnchors || batchIdx >= batchSize) return;

    int32_t stride = 5 + numClasses;
    const T* src = input + (batchIdx * numAnchors + anchorIdx) * stride;

    // 强制转换为 float 进行运算，防止 FP16 精度丢失和溢出
    float fcx = static_cast<float>(src[0]);
    float fcy = static_cast<float>(src[1]);
    float fw  = static_cast<float>(src[2]);
    float fh  = static_cast<float>(src[3]);

    T* dstBox = boxes + (batchIdx * numAnchors + anchorIdx) * 4;
    T* dstScore = scores + (batchIdx * numAnchors + anchorIdx) * numClasses;

    // ===================================================================
    // 🛡️ 第一道防线：坐标白名单清洗 (免疫 NaN/Inf 和长宽畸变框)
    // ===================================================================
    bool is_box_valid = (fw > 0.01f && fw < 5000.0f) && 
                        (fh > 0.01f && fh < 5000.0f) &&
                        (fcx > -5000.0f && fcx < 5000.0f) && 
                        (fcy > -5000.0f && fcy < 5000.0f);

    if (!is_box_valid) {
        dstBox[0] = (T)0.0f; dstBox[1] = (T)0.0f; dstBox[2] = (T)0.0f; dstBox[3] = (T)0.0f;
        for (int32_t c = 0; c < numClasses; ++c) dstScore[c] = (T)0.0f;
        return; 
    }

    // 坐标解码：Center-Size -> Corner
    dstBox[0] = (T)(fcx - fw * 0.5f); 
    dstBox[1] = (T)(fcy - fh * 0.5f); 
    dstBox[2] = (T)(fcx + fw * 0.5f); 
    dstBox[3] = (T)(fcy + fh * 0.5f); 

    // ===================================================================
    // 🛡️ 第二道防线：Top-1 分数提取 (阻断多类别假目标爆仓)
    // ===================================================================
    // float f_obj = static_cast<float>(src[4]);
    // float max_score = -1e10f; // 极小值兜底，兼容原始 Logit 的负数输出
    // int max_class_idx = -1;

    // for (int32_t c = 0; c < numClasses; ++c) {
    //     float f_prob = static_cast<float>(src[5 + c]);
    //     float final_score = f_obj * f_prob;
        
    //     // 正向过滤：只有在安全数值范围内的数才参与比较，自动免疫 NaN/Inf
    //     if (final_score > -10000.0f && final_score < 10000.0f) {
    //         if (final_score > max_score) {
    //             max_score = final_score;
    //             max_class_idx = c;
    //         }
    //     }
    //     // 同步将该类的输出初始化为 0，确保每个锚框只有 1 个非零分数
    //     dstScore[c] = (T)0.0f;
    // }

    // // 仅保留唯一的最高分写入目标位置
    // if (max_class_idx != -1) {
    //     dstScore[max_class_idx] = (T)max_score;
    // }
    T objConf = src[4];
    for (int32_t c = 0; c < numClasses; ++c)
    {
        dstScore[c] = objConf * src[5 + c];
    }
}


template <typename T>
__global__ void YOLODecodeKernel_ultralytics(
    const T* __restrict__ input,  // GPU: [batchSize, 4 + numClasses, numAnchors]
    T* __restrict__ boxes,        // GPU: [batchSize, numAnchors, 4]
    T* __restrict__ scores,       // GPU: [batchSize, numAnchors, numClasses]
    int32_t numAnchors,
    int32_t numClasses,
    int32_t batchSize
)
{
    int32_t anchorIdx = static_cast<int32_t>(blockDim.x * blockIdx.x + threadIdx.x);
    int32_t batchIdx = static_cast<int32_t>(blockIdx.y);

    if (anchorIdx >= numAnchors || batchIdx >= batchSize) return;

    int32_t numChannels = 4 + numClasses;
    const T* batchInput = input + batchIdx * (numAnchors * numChannels);

    // 跨步寻址获取坐标 (Ultralytics 模型在最后两个维度是转置的)
    float fcx = static_cast<float>(batchInput[0 * numAnchors + anchorIdx]); 
    float fcy = static_cast<float>(batchInput[1 * numAnchors + anchorIdx]); 
    float fw  = static_cast<float>(batchInput[2 * numAnchors + anchorIdx]); 
    float fh  = static_cast<float>(batchInput[3 * numAnchors + anchorIdx]); 

    T* dstBox = boxes + (batchIdx * numAnchors + anchorIdx) * 4;
    T* dstScore = scores + (batchIdx * numAnchors + anchorIdx) * numClasses;

    // ===================================================================
    // 🛡️ 第一道防线：坐标白名单清洗
    // ===================================================================
    bool is_box_valid = (fw > 0.01f && fw < 5000.0f) && 
                        (fh > 0.01f && fh < 5000.0f) &&
                        (fcx > -5000.0f && fcx < 5000.0f) && 
                        (fcy > -5000.0f && fcy < 5000.0f);

    if (!is_box_valid) {
        dstBox[0] = (T)0.0f; dstBox[1] = (T)0.0f; dstBox[2] = (T)0.0f; dstBox[3] = (T)0.0f;
        for (int32_t c = 0; c < numClasses; ++c) dstScore[c] = (T)0.0f;
        return; 
    }

    dstBox[0] = (T)(fcx - fw * 0.5f);
    dstBox[1] = (T)(fcy - fh * 0.5f);
    dstBox[2] = (T)(fcx + fw * 0.5f);
    dstBox[3] = (T)(fcy + fh * 0.5f);

    // ===================================================================
    // 🛡️ 第二道防线：Top-1 分数提取 (无 obj_conf 版本)
    // ===================================================================
    // float max_score = -1e10f; 
    // int max_class_idx = -1;

    // for (int32_t c = 0; c < numClasses; ++c){
    //     float f_score = static_cast<float>(batchInput[(4 + c) * numAnchors + anchorIdx]);
        
    //     if (f_score > -10000.0f && f_score < 10000.0f) {
    //         if (f_score > max_score) {
    //             max_score = f_score;
    //             max_class_idx = c;
    //         }
    //     }
    //     dstScore[c] = (T)0.0f;
    // }

    // if (max_class_idx != -1) {
    //     dstScore[max_class_idx] = (T)max_score;
    // }

    for (int32_t c = 0; c < numClasses; ++c){
        dstScore[c] = batchInput[(4 + c) * numAnchors + anchorIdx];
    }
}

// ============================================================================
// NMSProcessor 基类实现
// ============================================================================

NMSProcessor::NMSProcessor(EfficientNMSParams defaultParams)
    : params_(defaultParams)
{
}

void NMSProcessor::configure(int32_t batchSize, int32_t numAnchors, int32_t numClasses)
{
    params_.batchSize        = batchSize;
    params_.numAnchors       = numAnchors;
    params_.numClasses       = numClasses;
    params_.numScoreElements = numAnchors * numClasses;
    params_.numBoxElements   = numAnchors * (params_.shareLocation ? 1 : numClasses) * 4;
}

size_t NMSProcessor::getWorkspaceSize() const
{
    return efficientNMSWorkspaceSize(
        params_.batchSize,
        params_.numScoreElements,
        params_.numClasses,
        params_.datatype);
}

cudaError_t NMSProcessor::run(
    const void*  networkOutput,
    void*        workspace,
    int32_t*     numDetections,
    void*        detBoxes,
    void*        detScores,
    int32_t*     detClasses,
    cudaStream_t stream)
{
    cudaError_t status = preprocess(networkOutput, stream);
    if (status != cudaSuccess)
    {
        return status;
    }
    return runNMS(workspace, numDetections, detBoxes, detScores, detClasses, stream);
}

cudaError_t NMSProcessor::runNMS(
    void*        workspace,
    int32_t*     numDetections,
    void*        detBoxes,
    void*        detScores,
    int32_t*     detClasses,
    cudaStream_t stream)
{
    // outputONNXIndices 模式由外部直接使用 efficientNMSInference 处理；
    // NMSProcessor 固定使用标准输出模式（4 个缓冲区）。
    EfficientNMSParams p  = params_;
    p.outputONNXIndices   = false;

    return efficientNMSInference(
        p,
        decodedBoxes_,
        decodedScores_,
        anchors_,
        numDetections,
        detBoxes,
        detScores,
        detClasses,
        nullptr, // nmsIndicesOutput：标准模式不使用
        workspace,
        stream);
}


// ============================================================================
// YOLONMSProcessor 实现
// ============================================================================

YOLONMSProcessor::YOLONMSProcessor(EfficientNMSParams params)
    : NMSProcessor(params)
{
    // YOLO 解码后的 boxes 始终为 BoxCorner 格式，禁用内置解码器
    params_.shareLocation = true;
    params_.boxDecoder    = false;
    params_.boxCoding     = 0; // BoxCorner
}

YOLONMSProcessor::~YOLONMSProcessor()
{
    freeBuffers();
}

void YOLONMSProcessor::freeBuffers()
{
    if (boxesBuf_ != nullptr)
    {
        cudaFree(boxesBuf_);
        boxesBuf_ = nullptr;
    }
    if (scoresBuf_ != nullptr)
    {
        cudaFree(scoresBuf_);
        scoresBuf_ = nullptr;
    }
    allocBatchSize_  = 0;
    allocNumAnchors_ = 0;
    allocNumClasses_ = 0;
}

void YOLONMSProcessor::configure(int32_t batchSize, int32_t numAnchors, int32_t numClasses)
{
    NMSProcessor::configure(batchSize, numAnchors, numClasses);

    // 仅在尺寸发生变化时才重新分配 GPU 缓冲区，避免不必要的 cudaFree/cudaMalloc
    if (allocBatchSize_ == batchSize && allocNumAnchors_ == numAnchors && allocNumClasses_ == numClasses)
    {
        return;
    }

    freeBuffers();

    size_t dtSize = (params_.datatype == NMSDataType::kHALF) ? sizeof(__half) : sizeof(float);
    cudaError_t err;

    err = cudaMalloc(&boxesBuf_, static_cast<size_t>(batchSize) * numAnchors * 4 * dtSize);
    if (err != cudaSuccess)
    {
        // 分配失败时 boxesBuf_ 保持 nullptr，后续 preprocess/run 返回错误
        fprintf(stderr, "[YOLONMSProcessor] cudaMalloc boxesBuf_ failed: %s\n", cudaGetErrorString(err));
        return;
    }

    err = cudaMalloc(&scoresBuf_, static_cast<size_t>(batchSize) * numAnchors * numClasses * dtSize);
    if (err != cudaSuccess)
    {
        fprintf(stderr, "[YOLONMSProcessor] cudaMalloc scoresBuf_ failed: %s\n", cudaGetErrorString(err));
        cudaFree(boxesBuf_);
        boxesBuf_ = nullptr;
        return;
    }

    allocBatchSize_  = batchSize;
    allocNumAnchors_ = numAnchors;
    allocNumClasses_ = numClasses;

    // 更新基类指针（固定指向内部缓冲区）
    decodedBoxes_  = boxesBuf_;
    decodedScores_ = scoresBuf_;
    anchors_       = nullptr; // YOLO 无需 anchor（已在 kernel 中直接解码为绝对坐标）
}

cudaError_t YOLONMSProcessor::preprocess(const void* networkOutput, cudaStream_t stream)
{
    if (boxesBuf_ == nullptr || scoresBuf_ == nullptr)
    {
        return cudaErrorMemoryAllocation;
    }

    // 每线程处理一个 (batch, anchor)
    const uint32_t threadsPerBlock = 256;
    const uint32_t anchorBlocks = (static_cast<uint32_t>(params_.numAnchors) + threadsPerBlock - 1) / threadsPerBlock;
    const dim3 grid(anchorBlocks, static_cast<uint32_t>(params_.batchSize), 1);

    if (params_.datatype == NMSDataType::kFLOAT)
    {
        if (params_.ultralytics)
        {
            YOLODecodeKernel_ultralytics<float><<<grid, threadsPerBlock, 0, stream>>>(
            static_cast<const float*>(networkOutput),
            static_cast<float*>(boxesBuf_),
            static_cast<float*>(scoresBuf_),
            params_.numAnchors,
            params_.numClasses,
            params_.batchSize);
        }
        else
        {
            YOLODecodeKernel<float><<<grid, threadsPerBlock, 0, stream>>>(
            static_cast<const float*>(networkOutput),
            static_cast<float*>(boxesBuf_),
            static_cast<float*>(scoresBuf_),
            params_.numAnchors,
            params_.numClasses,
            params_.batchSize);
        }
    }
    else // kHALF
        if (params_.ultralytics)
        {
            YOLODecodeKernel_ultralytics<__half><<<grid, threadsPerBlock, 0, stream>>>(
            static_cast<const __half*>(networkOutput),
            static_cast<__half*>(boxesBuf_),
            static_cast<__half*>(scoresBuf_),
            params_.numAnchors,
            params_.numClasses,
            params_.batchSize);
        }
        else
        {
            YOLODecodeKernel<__half><<<grid, threadsPerBlock, 0, stream>>>(
            static_cast<const __half*>(networkOutput),
            static_cast<__half*>(boxesBuf_),
            static_cast<__half*>(scoresBuf_),
            params_.numAnchors,
            params_.numClasses,
            params_.batchSize);
        }
    return cudaGetLastError();
}