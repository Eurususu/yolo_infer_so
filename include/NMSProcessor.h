#pragma once

#include "efficientNMS.h"
#include <cstdint>
#include <cuda_runtime_api.h>



struct EfficientNMSInput
{
    /**
     * 框坐标（已解码）GPU 指针。
     *
     *   shareLocation=true ：形状 [batchSize, numAnchors, 4]
     *   shareLocation=false：形状 [batchSize, numAnchors, numClasses, 4]
     *   坐标格式由 EfficientNMSParams::boxCoding 决定。
     */
    const void* boxes;

    /**
     * 分类置信度分数 GPU 指针。
     * 形状 [batchSize, numAnchors, numClasses]，行主序。
     */
    const void* scores;

    /**
     * 先验框坐标 GPU 指针（可选）。
     *   boxDecoder=false 时设为 nullptr。
     *   boxDecoder=true，shareAnchors=true ：形状 [1, numAnchors, 4]
     *   boxDecoder=true，shareAnchors=false：形状 [batchSize, numAnchors, 4]
     */
    const void* anchors = nullptr;
};


class NMSProcessor
{
public:
    /**
     * @brief 构造函数
     * @param defaultParams NMS 超参数初始值（张量配置字段稍后通过 configure() 填写）
     */
    explicit NMSProcessor(EfficientNMSParams defaultParams);

    virtual ~NMSProcessor() = default;

    /**
     * @brief 根据实际输入张量维度更新内部 EfficientNMSParams
     *
     * 必须在 run() 之前调用。输入维度发生变化时须重新调用。
     * 子类可重写以进行额外的维度相关初始化（如重新分配中间缓冲区）。
     *
     * @param batchSize   批次大小
     * @param numAnchors  每张图的 anchor（候选框）数量
     * @param numClasses  检测类别数（不含背景）
     */
    virtual void configure(int32_t batchSize, int32_t numAnchors, int32_t numClasses);

    /**
     * @brief 查询所需 GPU workspace 字节数
     *
     * configure() 之后调用才返回正确值。
     * @return workspace 大小（字节）
     */
    size_t getWorkspaceSize() const;

    /**
     * @brief 执行完整的检测后处理（preprocess + NMS）
     *
     * 流程：
     *   1. 调用 preprocess(networkOutput, stream)，将网络输出转换为标准格式
     *   2. 调用内部 runNMS()，执行 EfficientNMS 推理
     *
     * @param networkOutput  检测网络的原始 GPU 输出
     *                       （具体格式由各子类约定，参见各子类说明）
     * @param workspace      GPU workspace 缓冲区，大小 >= getWorkspaceSize()
     * @param numDetections  [输出] GPU buffer，形状 [batchSize, 1]，int32
     *                       每张图有效检测框数量，上限为 params_.numOutputBoxes
     * @param detBoxes       [输出] GPU buffer，形状 [batchSize, numOutputBoxes, 4]
     *                       坐标格式：BoxCorner (x1, y1, x2, y2)，元素类型由 datatype 决定
     * @param detScores      [输出] GPU buffer，形状 [batchSize, numOutputBoxes]
     *                       检测框置信度，元素类型由 datatype 决定
     * @param detClasses     [输出] GPU buffer，形状 [batchSize, numOutputBoxes]，int32
     *                       检测框类别 ID（0-based）
     * @param stream         CUDA stream
     * @return cudaSuccess 或对应错误码
     */
    cudaError_t run(
        const void*  networkOutput,
        void*        workspace,
        int32_t*     numDetections,
        void*        detBoxes,
        void*        detScores,
        int32_t*     detClasses,
        cudaStream_t stream);

protected:
    /**
     * @brief 将网络原始输出转换为 EfficientNMS 标准输入格式（纯虚，子类实现）
     *
     * 实现此函数时，须将以下成员指向有效的 GPU buffer：
     *   decodedBoxes_  ← boxes（形状见 EfficientNMSParams::shareLocation 说明）
     *   decodedScores_ ← scores（形状 [batch, anchors, classes]）
     *   anchors_       ← anchors（boxDecoder=false 时保持 nullptr）
     *
     * 若需要在 GPU 上执行预处理计算（如 YOLO 解码），使用 stream 提交。
     *
     * @param networkOutput 网络原始 GPU 输出（具体类型由子类决定）
     * @param stream        CUDA stream
     * @return cudaSuccess 或对应错误码
     */
    virtual cudaError_t preprocess(const void* networkOutput, cudaStream_t stream) = 0;

    /** NMS 推理参数，configure() 中更新张量配置字段 */
    EfficientNMSParams params_;

    /**
     * 预处理结果：指向 GPU 内的 boxes buffer（由子类在 preprocess() 中设置）
     * 直接对应 efficientNMSInference() 的 boxesInput 参数
     */
    const void* decodedBoxes_ = nullptr;

    /**
     * 预处理结果：指向 GPU 内的 scores buffer（由子类在 preprocess() 中设置）
     * 直接对应 efficientNMSInference() 的 scoresInput 参数
     */
    const void* decodedScores_ = nullptr;

    /**
     * 预处理结果：指向 GPU 内的 anchors buffer（由子类在 preprocess() 中设置）
     * boxDecoder=false 时应保持 nullptr
     */
    const void* anchors_ = nullptr;

private:
    /** 调用底层 efficientNMSInference()，不对外暴露 */
    cudaError_t runNMS(
        void*        workspace,
        int32_t*     numDetections,
        void*        detBoxes,
        void*        detScores,
        int32_t*     detClasses,
        cudaStream_t stream);
};


class YOLONMSProcessor : public NMSProcessor
{
public:
    explicit YOLONMSProcessor(EfficientNMSParams params);

    /**
     * @brief 析构函数，释放内部分配的 GPU 缓冲区
     */
    ~YOLONMSProcessor() override;

    /**
     * @brief 更新张量配置，必要时重新分配内部 GPU 缓冲区
     * @param batchSize   批次大小
     * @param numAnchors  anchor 数量
     * @param numClasses  类别数
     */
    void configure(int32_t batchSize, int32_t numAnchors, int32_t numClasses) override;

protected:
    /**
     * 在 GPU 上执行 YOLODecodeKernel：解码 boxes 并融合 scores。
     * @param networkOutput 指向 GPU 上 [batch, anchors, 5+C] 数据的设备指针
     */
    cudaError_t preprocess(const void* networkOutput, cudaStream_t stream) override;

private:
    void freeBuffers();

    void*   boxesBuf_  = nullptr; ///< GPU: [batch, anchors, 4]，内部分配
    void*   scoresBuf_ = nullptr; ///< GPU: [batch, anchors, numClasses]，内部分配

    // 记录已分配的尺寸，避免不必要的重新分配
    int32_t allocBatchSize_  = 0;
    int32_t allocNumAnchors_ = 0;
    int32_t allocNumClasses_ = 0;
};
