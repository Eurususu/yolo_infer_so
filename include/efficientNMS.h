/*
 * SPDX-FileCopyrightText: Copyright (c) 1993-2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
 * SPDX-License-Identifier: Apache-2.0
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 * http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

/**
 * @file efficientNMS.h
 * @brief 独立的 EfficientNMS CUDA 函数接口
 *
 * 本头文件提供脱离 TensorRT 插件系统的 EfficientNMS 推理接口。
 * 仅依赖 CUDA Runtime 和 CUB（CUDA Toolkit >= 11.0 已内置，无需额外安装）。
 *
 * ============================================================================
 * 算法简介
 * ============================================================================
 *
 * EfficientNMS 分 4 个阶段在 GPU 上完成非极大值抑制：
 *
 *   阶段 1：过滤（Filter）
 *     对所有 (anchor, class) 组合的置信度分数进行阈值筛选，
 *     仅保留分数 >= scoreThreshold 的候选框，并记录其索引、类别、anchor 编号。
 *     当 scoreThreshold < 0.007 时走"稠密路径"（全量拷贝），否则走"稀疏路径"（原子写入）。
 *
 *   阶段 2：排序（Sort）
 *     使用 CUB DeviceSegmentedRadixSort::SortPairsDescending 对每张图的
 *     保留候选框按置信度分数降序排列，同时维护到原始 boxes 的索引映射。
 *     每张图（而非整个 batch）的候选框数量上限为 numSelectedBoxes（默认 4096）。
 *
 *   阶段 3：NMS（Non-Maximum Suppression）
 *     对每个 batch，以 tileSize 个线程并行迭代所有候选框：
 *       - 每次迭代由一个"领头线程"决定当前框是保留还是抑制
 *       - 其余线程计算与领头框的 IoU，超过 iouThreshold 的框被标记为抑制
 *       - 当保留框数量达到 numOutputBoxes 时提前退出
 *     支持 class-agnostic 和 per-class NMS，支持 per-class 数量限制。
 *
 *   阶段 4：输出打包（Pack）
 *     将有效检测结果写入 4 个输出缓冲区（detections/boxes/scores/classes），
 *     或在 ONNX 模式下写入索引并用最后一个有效结果填充剩余槽位。
 *
 * ============================================================================
 * 调用流程
 * ============================================================================
 *
 *   1. 填写 EfficientNMSParams 的所有字段（尤其是"张量配置"字段）
 *   2. 调用 efficientNMSWorkspaceSize() 查询所需 workspace 字节数
 *   3. cudaMalloc 分配 workspace
 *   4. 按各参数说明准备 GPU 输入/输出缓冲区
 *   5. 调用 efficientNMSInference()
 *   6. 从输出缓冲区读取检测结果（通常需先 cudaMemcpy 到 host）
 */

#pragma once

#include <cstdint>
#include <cuda_runtime_api.h>

// ============================================================================
// 枚举：数据类型
// ============================================================================

/**
 * @brief EfficientNMS 输入/输出张量的浮点精度
 */
enum class NMSDataType
{
    kFLOAT = 0, ///< 32 位单精度浮点（float，4 字节/元素）
    kHALF  = 1, ///< 16 位半精度浮点（__half / FP16，2 字节/元素）
};

// ============================================================================
// 参数结构体
// ============================================================================

/**
 * @brief EfficientNMS 的全部运行参数
 *
 * 分为三类：
 *   - NMS 超参数：由用户在构造时决定，整个推理会话中通常保持不变
 *   - NMS 内部参数：控制性能与精度权衡，一般保持默认即可
 *   - 张量配置：描述输入张量的形状，必须在每次调用 efficientNMSInference() 前按实际输入填好
 */
struct EfficientNMSParams
{
    // ------------------------------------------------------------------
    // NMS 超参数
    // ------------------------------------------------------------------

    /**
     * IoU（Intersection over Union）阈值。
     * 两框重叠率超过此值时，低分框将被抑制（丢弃）。
     * 取值范围 [0, 1]；值越小，保留的框越少，越严格。
     */
    float iouThreshold = 0.5F;

    /**
     * 置信度分数阈值。
     * 低于此值的候选框在过滤阶段直接丢弃，不参与排序和 NMS。
     * 取值范围 [0, 1]（scoreSigmoid=false 时）或任意实数（scoreSigmoid=true 时为 logit 阈值）。
     */
    float scoreThreshold = 0.5F;

    /**
     * 每张图最多输出的检测框数量。
     * 输出缓冲区 nmsBoxesOutput / nmsScoresOutput / nmsClassesOutput 的第 2 维必须与此值一致。
     */
    int32_t numOutputBoxes = 100;

    /**
     * 每个类别最多输出的框数。
     * -1 表示不限制（全局由 numOutputBoxes 控制）。
     * > 0 时启用 per-class 数量限制，配合 padOutputBoxesPerClass 使用。
     */
    int32_t numOutputBoxesPerClass = -1;

    /**
     * 是否对 per-class 输出进行对齐填充。
     * 仅当 numOutputBoxesPerClass > 0 时有效。
     * true：当某类检测框不足 numOutputBoxesPerClass 时，用最后一个有效框补齐。
     * false：不填充，该类实际检测数可能少于 numOutputBoxesPerClass。
     */
    bool padOutputBoxesPerClass = false;

    /**
     * 背景类 ID（Background class index）。
     * 匹配此 ID 的类别得分在过滤阶段会被忽略，不会被选为检测结果。
     * -1 表示无背景类（所有类别均参与 NMS）。
     */
    int32_t backgroundClass = -1;

    /**
     * 是否对输入分数应用 Sigmoid 激活。
     * true ：scoresInput 为原始 logit（未归一化），在写出结果时计算 sigmoid(score) 作为输出分数。
     *         同时 scoreThreshold 会被自动转换为对应的 logit 阈值（内部处理，调用者无需换算）。
     * false：scoresInput 已经是 [0, 1] 范围内的概率值，直接使用。
     */
    bool scoreSigmoid = false;

    /**
     * 是否将输出框坐标裁剪到归一化范围 [0, 1]。
     * true：输出的 nmsBoxesOutput 中每个坐标值都裁剪到 [0, 1]。
     *        适用于坐标已归一化到图像尺寸（即坐标 ∈ [0,1]）的场景。
     * false：不裁剪，坐标可能超出 [0, 1]。
     */
    bool clipBoxes = false;

    /**
     * 输入框的编码格式（Box coding format）：
     *   0 = BoxCorner 格式：每框 4 个值为 (x1, y1, x2, y2)，即左上角 + 右下角坐标
     *                       等价的 (y1, x1, y2, x2) 对 NMS/IoU 计算结果相同
     *   1 = BoxCenterSize 格式：每框 4 个值为 (cx, cy, h, w)，即中心点 + 高宽
     *                           等价的 (cy, cx, h, w) 对 NMS/IoU 计算结果相同
     *
     * 注意：无论输入格式如何，nmsBoxesOutput 的输出坐标始终为 BoxCorner (x1, y1, x2, y2) 格式。
     */
    int32_t boxCoding = 0;

    /**
     * 是否执行类别无关（class-agnostic）NMS。
     * true ：不同类别的框之间也会互相抑制（IoU 超阈值即抑制，无视类别差异）。
     * false：仅同类别的框之间互相抑制（不同类别的框不互相影响）。
     */
    bool classAgnostic = false;

    // ------------------------------------------------------------------
    // NMS 内部参数（通常保持默认值）
    // ------------------------------------------------------------------

    /**
     * 每张图进入 NMS kernel 的候选框数量上限（per-image 限制，而非 batch 级别）。
     *
     * 过滤阶段会为每张图维护独立计数器 topNumData[imageIdx]，原子递增直至达到此上限。
     * 排序阶段以 topNumData[imageIdx] 为每段实际长度（上限为 numSelectedBoxes）。
     * NMS kernel 的处理量为 min(topNumData[imageIdx], numSelectedBoxes)，每张图独立。
     *
     * 因此，即使同一 batch 内某张图所有候选框分数均低于 scoreThreshold，
     * 该图的 topNumData 为 0，numDetectionsOutput[i] = 0，完全不影响 batch 内其他图像。
     *
     * 值越大，漏检率越低，但 NMS kernel 耗时越长（NMS 复杂度约为 O(N²)）。
     * 推荐范围：2000～5000。
     */
    int32_t numSelectedBoxes = 4096;

    /**
     * FP16 模式下的分数尾数精度控制位数（Score bits optimization）。
     * -1：禁用此优化，使用完整 FP16 精度进行排序。
     * > 0（推荐范围 5～10）：仅使用高 scoreBits 位的尾数进行基数排序，
     *                         可加速排序但略微降低分数精度。
     * 仅在 datatype == NMSDataType::kHALF 时生效；kFLOAT 模式下此参数被强制为 -1。
     */
    int32_t scoreBits = -1;

    /**
     * 输出格式选择：
     * false（默认）：标准输出模式，输出 numDetections + boxes + scores + classes 四组缓冲区。
     * true          ：ONNX 兼容模式，输出 ONNX NonMaxSuppression 算子格式的索引，
     *                 见 nmsIndicesOutput 参数说明。
     */
    bool outputONNXIndices = false;

    // ------------------------------------------------------------------
    // 张量配置（每次调用 efficientNMSInference() 前必须填写）
    // ------------------------------------------------------------------

    /** 当前批次的图像数量（Batch size）。 */
    int32_t batchSize = -1;

    /** 检测类别数（不含背景类）。 */
    int32_t numClasses = 1;

    /**
     * boxesInput 张量的总元素数（含 batch 维，不含字节数）。
     *   shareLocation=true ：numBoxElements = batchSize * numAnchors * 4
     *   shareLocation=false：numBoxElements = batchSize * numAnchors * numClasses * 4
     */
    int32_t numBoxElements = -1;

    /**
     * scoresInput 张量的总元素数（含 batch 维）。
     * numScoreElements = batchSize * numAnchors * numClasses。
     * 此值也决定了过滤/排序阶段中间缓冲区（topScoresData 等）的分配大小。
     */
    int32_t numScoreElements = -1;

    /** 每张图的 anchor（先验框 / 候选框位置）数量。 */
    int32_t numAnchors = -1;

    /**
     * 位置共享标志（shareLocation）。
     * true ：boxesInput 形状为 [batchSize, numAnchors, 1, 4]，
     *         所有类别共用同一套 box 坐标（节省显存，适用于大多数检测器）。
     * false：boxesInput 形状为 [batchSize, numAnchors, numClasses, 4]，
     *         每个类别有独立的 box 坐标。
     */
    bool shareLocation = true;

    /**
     * Anchor 共享标志（shareAnchors）。仅在 boxDecoder=true 时有效。
     * true ：anchorsInput 形状为 [1, numAnchors, 4]，所有 batch 共用同一套 anchors。
     * false：anchorsInput 形状为 [batchSize, numAnchors, 4]，每张图有独立的 anchors。
     */
    bool shareAnchors = true;

    /**
     * 是否启用内置框解码器（Fused Box Decoder）。
     * false：boxesInput 已经是解码后的绝对坐标，直接送入 NMS 计算，anchorsInput 被忽略。
     * true ：boxesInput 为相对于 anchor 的偏移量预测（delta），在 NMS kernel 内部结合
     *         anchorsInput 进行解码后再做 NMS。解码方式由 boxCoding 决定：
     *           boxCoding=0（BoxCorner）：decoded = anchor + delta（加法解码）
     *           boxCoding=1（BoxCenterSize）：decoded.cy = delta.cy * anchor.h + anchor.cy，
     *                                         decoded.h  = anchor.h * exp(delta.h)，等
     */
    bool boxDecoder = false;


    bool ultralytics = false; // 是否是ultralytics模型

    /**
     * 输入/输出数据的浮点精度。
     * 同时决定 boxesInput、scoresInput、anchorsInput、nmsBoxesOutput、nmsScoresOutput
     * 的元素类型（float 或 __half）。
     */
    NMSDataType datatype = NMSDataType::kFLOAT;
};

// ============================================================================
// 公共函数接口
// ============================================================================

/**
 * @brief 计算 efficientNMSInference() 所需的 GPU workspace 字节数
 *
 * workspace 内部布局（所有缓冲区均按 256 字节对齐）：
 *   [计数器区]  (3 + 1 + numClasses) * batchSize 个 int32
 *               含：topNumData（过滤计数）、topOffsetsStart/End（排序段偏移）、
 *                   outputIndexData（ONNX 模式计数器）、outputClassData（per-class 计数器）
 *   [topIndexData]   batchSize * numScoreElements 个 int32（过滤后的位置索引）
 *   [topClassData]   batchSize * numScoreElements 个 int32（类别索引）
 *   [topAnchorsData] batchSize * numScoreElements 个 int32（anchor 索引）
 *   [sortedIndexData]batchSize * numScoreElements 个 int32（排序后的位置索引）
 *   [topScoresData]  batchSize * numScoreElements 个 T（过滤后的分数）
 *   [sortedScoresData] batchSize * numScoreElements 个 T（排序后的分数）
 *   [CUB 排序缓冲区]  大小由 CUB 内部决定（cub::DeviceSegmentedRadixSort 所需临时空间）
 *
 * @param batchSize         批次大小
 * @param numScoreElements  每张图的分数元素数，等于 numAnchors * numClasses
 * @param numClasses        类别数
 * @param datatype          浮点精度类型
 * @return 所需字节数（已按 256 字节向上对齐）
 */
size_t efficientNMSWorkspaceSize(
    int32_t     batchSize,
    int32_t     numScoreElements,
    int32_t     numClasses,
    NMSDataType datatype);

/**
 * @brief 执行 EfficientNMS CUDA 推理
 *
 * 调用者负责：
 *   1. 按正确格式在 GPU 上分配并填充所有输入缓冲区
 *   2. 调用 efficientNMSWorkspaceSize() 并分配足够大的 workspace
 *   3. 分配所有需要的输出缓冲区
 *   4. 调用本函数（异步提交到 stream）
 *   5. 根据需要进行 cudaStreamSynchronize 后读取结果
 *
 * 所有指针均为 GPU 设备指针（cudaMalloc 分配）。
 *
 * =========================================================================
 * 参数说明
 * =========================================================================
 *
 * @param params
 *   完整的推理参数（见 EfficientNMSParams），所有"张量配置"字段必须已填写。
 *
 * -------------------------------------------------------------------------
 * @param boxesInput  [GPU 输入] 候选框坐标
 *
 *   元素类型：由 params.datatype 决定（float 或 __half）
 *   内存布局（行主序，C contiguous）：
 *
 *   (a) params.shareLocation=true，params.boxDecoder=false（最常见情况）：
 *       形状 [batchSize, numAnchors, 4]
 *       坐标格式由 params.boxCoding 决定（BoxCorner 或 BoxCenterSize）
 *       每个 anchor 对应唯一一套 box 坐标，所有类别共用
 *
 *   (b) params.shareLocation=false，params.boxDecoder=false：
 *       形状 [batchSize, numAnchors, numClasses, 4]
 *       每个 anchor 的每个类别有独立的 box 坐标
 *
 *   (c) params.boxDecoder=true：
 *       布局同 (a) 或 (b)，但值为相对于 anchor 的偏移量预测（delta），
 *       由内置解码器结合 anchorsInput 解码为绝对坐标后再做 NMS
 *
 * -------------------------------------------------------------------------
 * @param scoresInput  [GPU 输入] 分类置信度分数
 *
 *   元素类型：由 params.datatype 决定
 *   内存布局（行主序）：
 *       形状 [batchSize, numAnchors, numClasses]
 *       内存步长 [numAnchors*numClasses, numClasses, 1]
 *
 *   值语义：
 *     params.scoreSigmoid=false：值为归一化概率 [0, 1]，直接与 scoreThreshold 比较
 *     params.scoreSigmoid=true ：值为原始 logit（任意实数），内部自动换算阈值，
 *                                 最终输出时应用 sigmoid 转换
 *
 * -------------------------------------------------------------------------
 * @param anchorsInput  [GPU 输入，条件可选] 先验框坐标
 *
 *   params.boxDecoder=false 时：传 nullptr，本参数被忽略。
 *
 *   params.boxDecoder=true，params.shareAnchors=true：
 *       形状 [1, numAnchors, 4]
 *       内存步长 [numAnchors*4, 4, 1]
 *       所有 batch 共用同一套 anchors（显存高效）
 *
 *   params.boxDecoder=true，params.shareAnchors=false：
 *       形状 [batchSize, numAnchors, 4]
 *       内存步长 [numAnchors*4, 4, 1]
 *       每张图有独立的 anchors（适用于动态 anchor 的检测器）
 *
 *   坐标格式与 params.boxCoding 一致（BoxCorner 或 BoxCenterSize）。
 *
 * -------------------------------------------------------------------------
 * @param numDetectionsOutput  [GPU 输出] 每张图的有效检测数量
 *
 *   元素类型：int32
 *   内存布局：形状 [batchSize, 1]，即 batchSize 个独立计数器
 *   值域：[0, params.numOutputBoxes]
 *   第 i 张图的有效检测数为 numDetectionsOutput[i]，后续槽位的数据填 0
 *
 *   params.outputONNXIndices=true 时：传 nullptr（结果通过 nmsIndicesOutput 输出）。
 *
 * -------------------------------------------------------------------------
 * @param nmsBoxesOutput  [GPU 输出] NMS 筛选后的检测框坐标
 *
 *   元素类型：由 params.datatype 决定
 *   内存布局（行主序）：
 *       形状 [batchSize, numOutputBoxes, 4]
 *       内存步长 [numOutputBoxes*4, 4, 1]
 *
 *   坐标格式：始终为 BoxCorner (x1, y1, x2, y2)，与 params.boxCoding 的输入格式无关。
 *   排序：同一张图内按置信度分数降序排列（第 0 个框分数最高）。
 *   填充：有效条目数由 numDetectionsOutput[i] 给出，后续槽位清零（cudaMemsetAsync 保证）。
 *   裁剪：若 params.clipBoxes=true，坐标已裁剪到 [0, 1]。
 *
 *   params.outputONNXIndices=true 时：传 nullptr。
 *
 * -------------------------------------------------------------------------
 * @param nmsScoresOutput  [GPU 输出] NMS 筛选后每框的置信度分数
 *
 *   元素类型：由 params.datatype 决定
 *   内存布局（行主序）：
 *       形状 [batchSize, numOutputBoxes]
 *       内存步长 [numOutputBoxes, 1]
 *
 *   值语义：
 *     params.scoreSigmoid=false：输出原始分数值（与 scoresInput 精度相同）
 *     params.scoreSigmoid=true ：输出经过 sigmoid 转换的概率值
 *   填充：超出有效检测数的槽位清零。
 *
 *   params.outputONNXIndices=true 时：传 nullptr。
 *
 * -------------------------------------------------------------------------
 * @param nmsClassesOutput  [GPU 输出] NMS 筛选后每框的类别 ID
 *
 *   元素类型：int32
 *   内存布局：形状 [batchSize, numOutputBoxes]
 *   值域：[0, numClasses)，与 scoresInput 的最后一维下标一致
 *   填充：超出有效检测数的槽位清零。
 *
 *   params.outputONNXIndices=true 时：传 nullptr。
 *
 * -------------------------------------------------------------------------
 * @param nmsIndicesOutput  [GPU 输出，ONNX 模式专用] ONNX 兼容检测索引
 *
 *   params.outputONNXIndices=false 时：传 nullptr，本参数被忽略。
 *
 *   params.outputONNXIndices=true：
 *     元素类型：int32
 *     内存布局：形状 [batchSize * numOutputBoxes, 3]
 *     每行 3 个值为 [batch_idx, class_idx, anchor_idx]
 *     对应 ONNX NonMaxSuppression 算子 (opset 10+) 的 selected_indices 输出格式
 *     不足 batchSize * numOutputBoxes 行时，用最后一个有效结果填充（PadONNXResult kernel 保证）
 *
 * -------------------------------------------------------------------------
 * @param workspace  [GPU] 临时工作缓冲区
 *   大小（字节）：不得低于 efficientNMSWorkspaceSize() 的返回值。
 *   内容：调用结束后内容无效，可安全复用于下一次调用。
 *
 * @param stream
 *   CUDA stream。所有 GPU 操作（kernel launch、cudaMemsetAsync 等）均在此 stream 上
 *   异步提交，调用者需在读取输出前自行同步（cudaStreamSynchronize 或 event）。
 *
 * =========================================================================
 * @return cudaSuccess          成功
 * @return cudaErrorNotSupported params.datatype 不是 kFLOAT 或 kHALF
 * @return 其他 cudaError_t     某步 CUDA 操作失败
 */
cudaError_t efficientNMSInference(
    EfficientNMSParams params,
    const void*        boxesInput,
    const void*        scoresInput,
    const void*        anchorsInput,
    int32_t*           numDetectionsOutput,
    void*              nmsBoxesOutput,
    void*              nmsScoresOutput,
    int32_t*           nmsClassesOutput,
    int32_t*           nmsIndicesOutput,
    void*              workspace,
    cudaStream_t       stream);
