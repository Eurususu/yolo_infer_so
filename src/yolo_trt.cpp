#include "yolo_trt.h"

#include <NvInfer.h>
#include <cuda_runtime_api.h>
#include <NvInferPlugin.h>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>
#include <fstream>
#include <filesystem>

#include "preprocess.h"
#include "NMSProcessor.h"

namespace fs = std::filesystem;


namespace {
    // CUDA 锁页内存分配器
    template <typename T>
    struct CudaPinnedAllocator {
        using value_type = T;

        CudaPinnedAllocator() = default;
        template <class U> constexpr CudaPinnedAllocator(const CudaPinnedAllocator<U>&) noexcept {}

        T* allocate(std::size_t n) {
            T* ptr = nullptr;
            cudaError_t err = cudaMallocHost((void**)&ptr, n * sizeof(T));
            if (err != cudaSuccess) {
                throw std::bad_alloc();
            }
            return ptr;
        }

        void deallocate(T* p, std::size_t /*n*/) noexcept {
            cudaFreeHost(p);
        }
    };

    template <class T, class U>
    bool operator==(const CudaPinnedAllocator<T>&, const CudaPinnedAllocator<U>&) { return true; }
    template <class T, class U>
    bool operator!=(const CudaPinnedAllocator<T>&, const CudaPinnedAllocator<U>&) { return false; }
    template <typename T>
    using PinnedVector = std::vector<T, CudaPinnedAllocator<T>>;

    struct PipelineData {
        std::vector<cv::Mat> frames;
        std::vector<yolo::BatchResult> dets;
        std::vector<float> prof;
        double batch_time = 0.0;
        bool is_last = false;
    };

    // 线程安全的阻塞队列
    template <typename T>
    class SafeQueue {
    private:
        std::queue<T> q;
        std::mutex m;
        std::condition_variable cv_push, cv_pop;
        size_t max_size;
        bool stop_flag = false;
    public:
        SafeQueue(size_t max_size = 3) : max_size(max_size) {} // 队列长度设为3足以缓冲，太大吃内存

        // void push(T val) {
        //     std::unique_lock<std::mutex> lock(m);
        //     cv_push.wait(lock, [this] { return q.size() < max_size || stop_flag; }); // 等待队列有空间或者收到停止信号
        //     if (stop_flag) return;
        //     q.push(std::move(val)); // 等待成功则进行push
        //     cv_pop.notify_one(); // 通知在等待消费的线程
        // }


        // 支持右值，避免不必要的拷贝
        bool push (T&& val){
            /*
            尝试获得互斥锁，如果没有其他线程持有锁，则成功获得锁，并继续执行后续代码；
            如果其他线程已经持有锁，那么当前线程就会阻塞，什么都做不了，这个线程处于等待锁的状态，知道它获得锁。
            */
            std::unique_lock<std::mutex> lock(m);
            /*
            只有成功获得锁之后，才会去检查lambda函数，如果条件不满足，线程就会释放锁，进入沉睡，如果条件满足就继续执行后续代码。

            处于睡眠状态（Sleep）的线程，是不消耗 CPU 资源的，它也完全失去了执行代码的能力。 
            它根本无法去检查那个 lambda 表达式（q.size() < max_size）是否已经变成了 true。
            它就像一个深度昏迷的人。如果没人唤醒就一直睡下去，即使条件已经满足了也不会自己醒来。
            这个时候需要通过cv_push.notify_one()来唤醒它，否则他就一直睡下去，这就是死锁。
            */
            cv_push.wait(lock, [this] {return q.size() < max_size || stop_flag;});

            if (stop_flag) return false;

            q.push(std::move(val));
            lock.unlock();
            cv_pop.notify_one();
            return true;
        }


        // 支持左值
        bool push(const T& val){
            std::unique_lock<std::mutex> lock(m);
            cv_push.wait(lock, [this] {return q.size() < max_size || stop_flag;});

            if (stop_flag) return false;

            q.push(val);
            lock.unlock();
            cv_pop.notify_one();
            return true;
        }

        bool pop(T& val) {
            std::unique_lock<std::mutex> lock(m);
            cv_pop.wait(lock, [this] { return !q.empty() || stop_flag; }); // 等待队列非空或者收到停止信号
            if (stop_flag && q.empty()) return false;
            val = std::move(q.front()); // 取出队列头元素
            q.pop(); // pop 后才释放锁，确保生产者在 push 后能第一时间看到队列状态的改变
            lock.unlock();
            cv_push.notify_one(); // 条件变量中的 lambda 表达式，并不是在后台时刻不停地被监控着 所以需要pop之后手动通知生产者线程，唤醒他们继续生产
            return true;
        }

        void stop() {
            std::unique_lock<std::mutex> lock(m);
            stop_flag = true; // 设置停止标志，通知所有等待线程
            lock.unlock();
            cv_push.notify_all(); // 唤醒所有等待生产的线程
            cv_pop.notify_all(); // 唤醒所有等待消费的线程
        }
    };


    // TRT 日志器
    class TrtLogger : public nvinfer1::ILogger {
        void log(Severity severity, const char* msg) noexcept override {
            if (severity <= Severity::kWARNING){
                std::cout << "[TensorRT] " << msg << std::endl;
            }
        }
    };

    // 为 TensorRT 对象专属定制的智能指针 Deleter
    struct TRTDeleter {
        template <typename T>
        void operator()(T* obj) const {
            if (obj) {
                delete obj; // TRT 8.0+ 标准销毁方式
            }
        }
    };

    // 存储张量信息的结构体
    struct TensorInfo {
        std::string name;
        bool is_input;
        nvinfer1::DataType dtype;
        std::vector<int64_t> max_shape;     // 分配显存时用的最大形状
        std::vector<int64_t> actual_shape;  // 推理后得到的真实形状
        size_t size_bytes;                  // 显存分配大小
        void* dev_ptr = nullptr;            // GPU 指针
        // std::vector<float> host_buffer;     // CPU 接收缓存 (仅输出需要)
        PinnedVector<float> host_buffer;
    };


    static const std::vector<std::string> COCO_NAMES = {
        "person", "bicycle", "car", "motorcycle", "airplane", "bus", "train", "truck", "boat", "traffic light",
        "fire hydrant", "stop sign", "parking meter", "bench", "bird", "cat", "dog", "horse", "sheep", "cow",
        "elephant", "bear", "zebra", "giraffe", "backpack", "umbrella", "handbag", "tie", "suitcase", "frisbee",
        "skis", "snowboard", "sports ball", "kite", "baseball bat", "baseball glove", "skateboard", "surfboard",
        "tennis racket", "bottle", "wine glass", "cup", "fork", "knife", "spoon", "bowl", "banana", "apple",
        "sandwich", "orange", "broccoli", "carrot", "hot dog", "pizza", "donut", "cake", "chair", "couch",
        "potted plant", "bed", "dining table", "toilet", "tv", "laptop", "mouse", "remote", "keyboard", "cell phone",
        "microwave", "oven", "toaster", "sink", "refrigerator", "book", "clock", "vase", "scissors", "teddy bear",
        "hair drier", "toothbrush"
    };

    static const std::vector<std::string> IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"};


    static TrtLogger gLogger;


};


namespace yolo {
    // 继承自对外接口的真正实现类
    class YoloDetectorImpl : public IYoloDetector {
        private:
            Config config_;
            float conf_thres;
            float iou_thres;
            int num_classes;
            std::vector<std::string> class_names;
            int max_batch_size;
            int opt_batch_size;
            int max_det;

            std::unique_ptr<nvinfer1::IRuntime, TRTDeleter> runtime;
            std::unique_ptr<nvinfer1::ICudaEngine, TRTDeleter> engine;
            std::unique_ptr<nvinfer1::IExecutionContext, TRTDeleter> context;
            cudaStream_t stream = nullptr;

            std::vector<TensorInfo> io_tensors;
            std::vector<uint8_t*> d_img_buffers;  // 指向每张图原数据显存的指针列表
            uint8_t** d_img_ptrs = nullptr;  // GPU 端的指针目录
            int max_src_bytes = 0; 
            int input_width;
            int input_height;
            int input_channels;

            bool noend;
            bool is_ultralytics;
            std::unique_ptr<YOLONMSProcessor> nms_processor;
            // GPU 端 NMS 显存指针 (持久化分配)
            void* d_nms_workspace = nullptr;
            int32_t* d_nms_num_det = nullptr;
            float* d_nms_boxes = nullptr;
            float* d_nms_scores = nullptr;
            int32_t* d_nms_classes = nullptr;

            // 使用 cudaMallocHost（页锁定内存）替代普通的 vector，能大幅提升 PCIe 传输效率并实现真正的异步
            int32_t* h_nms_num_det_pinned = nullptr;
            float* h_nms_boxes_pinned = nullptr;
            float* h_nms_scores_pinned = nullptr;
            int32_t* h_nms_classes_pinned = nullptr;
            // std::vector<int32_t> h_nms_num_det;
            // std::vector<float>   h_nms_boxes;
            // std::vector<float>   h_nms_scores;
            // std::vector<int32_t> h_nms_classes;

            cudaEvent_t event_start = nullptr, event_h2d_preprocess = nullptr, event_comp = nullptr, event_d2h = nullptr;
            bool profile = false;

            void cleanup() {
                for (auto& t : io_tensors) {
                    if(t.dev_ptr) { cudaFree(t.dev_ptr); t.dev_ptr = nullptr; }
                }
                for (auto& ptr: d_img_buffers) {
                    if (ptr) { cudaFree(ptr); ptr = nullptr; }
                }
                if (d_img_ptrs) { cudaFree(d_img_ptrs); d_img_ptrs = nullptr; }

                if (stream) { cudaStreamDestroy(stream); stream = nullptr; }

                if (d_nms_workspace) { cudaFree(d_nms_workspace); d_nms_workspace = nullptr; }
                if (d_nms_num_det)   { cudaFree(d_nms_num_det); d_nms_num_det = nullptr; }
                if (d_nms_boxes)     { cudaFree(d_nms_boxes); d_nms_boxes = nullptr; }
                if (d_nms_scores)    { cudaFree(d_nms_scores); d_nms_scores = nullptr; }
                if (d_nms_classes)   { cudaFree(d_nms_classes); d_nms_classes = nullptr; }

                if (h_nms_num_det_pinned) { cudaFreeHost(h_nms_num_det_pinned); h_nms_num_det_pinned = nullptr; }
                if (h_nms_boxes_pinned)   { cudaFreeHost(h_nms_boxes_pinned); h_nms_boxes_pinned = nullptr; }
                if (h_nms_scores_pinned)  { cudaFreeHost(h_nms_scores_pinned); h_nms_scores_pinned = nullptr; }
                if (h_nms_classes_pinned) { cudaFreeHost(h_nms_classes_pinned); h_nms_classes_pinned = nullptr; }

                if (profile) {
                    if (event_start) { cudaEventDestroy(event_start); event_start = nullptr; }
                    if (event_h2d_preprocess) { cudaEventDestroy(event_h2d_preprocess); event_h2d_preprocess = nullptr; }
                    if (event_comp) { cudaEventDestroy(event_comp); event_comp = nullptr; }
                    if (event_d2h) { cudaEventDestroy(event_d2h); event_d2h = nullptr; }
                }
            }
        public:
            // 构造函数：实现你之前的引擎加载、显存分配逻辑
            YoloDetectorImpl(const Config& config) : config_(config){
                conf_thres = config_.conf_thres;
                iou_thres = config_.iou_thres;
                noend = !config_.efficient_end2end && !config_.end2end_model && !config_.end2end;
                is_ultralytics = config_.ultralytics;
                profile = config_.profile;
                num_classes = config_.num_classes;
                max_batch_size = config_.max_batch_size;

                class_names = COCO_NAMES;
                max_det = 300;


                try{
                    opt_batch_size = (config_.opt_batch_size > 0) ? config_.opt_batch_size : max_batch_size;

                    if (this->class_names.size() != static_cast<size_t>(this->num_classes)){
                        std::cerr << "[警告] 传入的类别名称数量 (" << this->class_names.size() 
                                << ") 不等于 num_classes (" << this->num_classes << ")! 画框时可能会越界。" << std::endl;
                    }

                    // 1. 加载 engine 二进制文件
                    std::ifstream file(config_.engine_path, std::ios::binary);
                    if (!file.good()){
                        throw std::runtime_error("无法打开 Engine 文件: " + config_.engine_path);
                    }

                    file.seekg(0, file.end);
                    size_t size = file.tellg();
                    file.seekg(0, file.beg);

                    std::vector<char> engine_data(size);
                    file.read(engine_data.data(), size);
                    file.close();
                    
                    // 2. 实例化 TRT 对象，并使用 .reset() 交给智能指针接管
                    runtime.reset(nvinfer1::createInferRuntime(gLogger));
                    if (!runtime) throw std::runtime_error("创建 TRT Runtime 失败");

                    initLibNvInferPlugins(&gLogger, "");

                    engine.reset(runtime->deserializeCudaEngine(engine_data.data(), size));
                    if (!engine) throw std::runtime_error("反序列化 CudaEngine 失败");

                    context.reset(engine->createExecutionContext());
                    if (!context) throw std::runtime_error("创建 ExecutionContext 失败");

                    cudaStreamCreate(&stream);

                    if (profile){
                        cudaEventCreate(&event_start);
                        cudaEventCreate(&event_h2d_preprocess);
                        cudaEventCreate(&event_comp);
                        cudaEventCreate(&event_d2h);
                    }

                    // 3. 动态显存分配
                    int num_io_tensors = engine->getNbIOTensors();
                    for (int i = 0; i < num_io_tensors; ++i){
                        const char* name = engine->getIOTensorName(i);
                        bool is_input = engine->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT;
                        nvinfer1::Dims dims = engine->getTensorShape(name);

                        std::vector<int64_t> shape(dims.d, dims.d + dims.nbDims);

                        if (is_input){
                            if (shape[0] >= 1){
                                max_batch_size = shape[0];
                                if (config_.opt_batch_size <= 0) opt_batch_size = shape[0];
                            }
                            input_channels = shape[1];
                            input_height = shape[2];
                            input_width = shape[3];

                            // 防止模型是全动态形状 [-1, 3, -1, -1] 导致宽高也是 -1
                            if (input_height == -1 || input_width == -1){
                                std::cout << "[警告] 检测到输入宽高为动态(-1)，强制使用默认 640x640！" << std::endl;
                                input_height = 640;
                                input_width = 640;
                            }

                        }
                        // IMSLayer output 显存分配
                        if (shape[0] == -1 && !is_input && shape.size() > 1 && shape[1] == 7){
                            shape[0] = max_det * max_batch_size;
                        } else if (shape[0] == -1) {
                            shape[0] = max_batch_size;
                        }

                        // 遍历其余维度，处理其他动态维度 (例如动态NMS插件的输出可能是 [batch, -1, 4])
                        for (size_t j = 1; j < shape.size(); ++j){
                            if (shape[j] == -1) shape[j] = max_det;
                        }

                        size_t vol = 1;
                        for (auto s : shape) vol *= s;
                        size_t bytes = vol * sizeof(float);

                        void* ptr = nullptr;
                        if (cudaMalloc(&ptr, bytes) != cudaSuccess){
                            throw std::runtime_error("CUDA Malloc 失败");
                        }

                        // TRT V3 API 绑定地址
                        context->setTensorAddress(name, ptr);

                        TensorInfo info;
                        info.name = name;
                        info.is_input = is_input;
                        info.dtype = engine->getTensorDataType(name);
                        info.max_shape = shape;
                        info.size_bytes = bytes;
                        info.dev_ptr = ptr;
                        if (!is_input){
                            info.host_buffer.resize(vol);
                        }
                        io_tensors.push_back(info);

                    }
                    // cudaMalloc((void **)&d_img_ptrs, max_batch_size * sizeof(uint8_t*));
                    // d_img_buffers.resize(max_batch_size, nullptr);

                    max_src_bytes = 1920 * 1080 * 3;
                    if (cudaMalloc((void **)&d_img_ptrs, max_batch_size * sizeof(uint8_t*)) != cudaSuccess) throw std::runtime_error("d_img_ptrs 失败");
                    d_img_buffers.resize(max_batch_size, nullptr);
                    for (int i = 0; i < max_batch_size; i++) {
                        if (cudaMalloc((void**)&d_img_buffers[i], max_src_bytes) != cudaSuccess) throw std::runtime_error("d_img_buffers 失败");
                    }

                    TensorInfo* out_info = nullptr;
                    for (auto& t : io_tensors) {
                        if (!t.is_input) { out_info = &t; }
                    }
                    if (noend){
                        int ndim = out_info->max_shape.size();
                        int num_anchors, channels;

                        if (is_ultralytics) {
                            channels = out_info->max_shape[ndim - 2];
                            num_anchors = out_info->max_shape[ndim - 1];
                        } else {
                            num_anchors = out_info->max_shape[ndim - 2];
                            channels = out_info->max_shape[ndim - 1];
                        }

                        // 计算真实类别数 (校验与传参是否一致)
                        int expected_classes = channels - (is_ultralytics ? 4 : 5);
                        if (expected_classes != this->num_classes) {
                            std::cerr << "[警告] 模型解析到的类别数 (" << expected_classes 
                            << ") 与传入的 (" << this->num_classes << ") 不一致！" << std::endl;
                        }

                        // 1. 设置参数并实例化 Processor
                        EfficientNMSParams nms_params;
                        nms_params.iouThreshold   = this->iou_thres;
                        nms_params.scoreThreshold = this->conf_thres;
                        nms_params.numOutputBoxes = this->max_det;
                        nms_params.datatype       = NMSDataType::kFLOAT;
                        nms_params.ultralytics    = this->is_ultralytics;

                        nms_processor = std::make_unique<YOLONMSProcessor>(nms_params);
                        nms_processor->configure(max_batch_size, num_anchors, this->num_classes);

                        // 2. 一次性分配 GPU 显存
                        size_t wsSize = nms_processor->getWorkspaceSize();
                        if (cudaMalloc(&d_nms_workspace, wsSize) != cudaSuccess) throw std::runtime_error("NMS Workspace 分配失败");
                        cudaMalloc((void**)&d_nms_num_det, max_batch_size * sizeof(int32_t));
                        cudaMalloc((void**)&d_nms_boxes, max_batch_size * max_det * 4 * sizeof(float));
                        cudaMalloc((void**)&d_nms_scores, max_batch_size * max_det * sizeof(float));
                        cudaMalloc((void**)&d_nms_classes, max_batch_size * max_det * sizeof(int32_t));
                        
                        // 3. 一次性分配 CPU 接收内存池 在构造函数中分配页锁定内存 (替换原有的 resize)
                        cudaMallocHost((void**)&h_nms_num_det_pinned, max_batch_size * sizeof(int32_t));
                        cudaMallocHost((void**)&h_nms_boxes_pinned, max_batch_size * max_det * 4 * sizeof(float));
                        cudaMallocHost((void**)&h_nms_scores_pinned, max_batch_size * max_det * sizeof(float));
                        cudaMallocHost((void**)&h_nms_classes_pinned, max_batch_size * max_det * sizeof(int32_t));
                        // h_nms_num_det.resize(max_batch_size);
                        // h_nms_boxes.resize(max_batch_size * max_det * 4);
                        // h_nms_scores.resize(max_batch_size * max_det);
                        // h_nms_classes.resize(max_batch_size * max_det);
                    }
                } catch (const std::exception& e){
                    // 💥 捕获到异常：立刻召唤 cleanup 打扫战场，然后再把异常往外抛！
                    std::cerr << "\n[致命错误] YoloTRTRunner 初始化失败: " << e.what() << std::endl;
                    std::cerr << "正在安全释放已分配的显存资源...\n";
                    cleanup();
                    throw; // 继续抛出，阻止程序运行
                } catch (...) {
                    std::cerr << "\n[致命错误] YoloTRTRunner 运行时发生未知异常！\n";
                    cleanup();
                    throw; // 继续抛出，阻止程序运行

                }
            }

            ~YoloDetectorImpl() override {
                cleanup();
            }

            std::vector<BatchResult> process_output(int real_batch_size, const std::vector<float>& scales, 
                const std::vector<int>& dws, const std::vector<int>& dhs){
                std::vector<BatchResult> batch_dets(real_batch_size);
                /*
                给所有的 batch 预留一个“经验值”空间（比如 100），避免在后续的 push_back 时频繁触发 vector 的扩容机制（重新分配内存并复制数据），从而提升性能。1
                */
                for (int b = 0; b < real_batch_size; b++) {
                    batch_dets[b].reserve(100);
                }
                
                // 收集所有的输出 Tensor
                std::vector<TensorInfo*> outputs;
                for (auto& t : io_tensors){
                    if (!t.is_input) outputs.push_back(&t);
                }
                if (outputs.empty()) return batch_dets;

                if (config_.efficient_end2end){
                    if (outputs.size() < 4){
                        std::cerr << "错误: efficient_end2end 需要模型有 4 个输出节点!" << std::endl;
                        return batch_dets;
                    }

                    // TRT 插件标准输出顺序：num_dets(0), boxes(1), scores(2), classes(3)
                    TensorInfo* t_num = outputs[0];
                    TensorInfo* t_box = outputs[1];
                    TensorInfo* t_score = outputs[2];
                    TensorInfo* t_cls = outputs[3];

                    int max_det = t_box->actual_shape[1]; // [batch, max_det, 4]

                    // 智能读取工具：因为 num_dets 和 classes 可能是 int32 或 float32
                    auto get_int_val = [](TensorInfo* t, int index) -> int {
                        if (t->dtype == nvinfer1::DataType::kINT32) {
                            return reinterpret_cast<const int32_t*>(t->host_buffer.data())[index];
                        } else {
                            return static_cast<int>(t->host_buffer[index]);
                        }
                    };
                    
                    const float* boxes_ptr = t_box->host_buffer.data();
                    const float* scores_ptr = t_score->host_buffer.data();

                    for (int b = 0; b < real_batch_size; b++){
                        // 读取当前帧有效框的数量
                        int valid_count = get_int_val(t_num, b);
                        batch_dets[b].reserve(valid_count);

                        for (int i =0; i < valid_count; i++){
                            float score = scores_ptr[b * max_det + i];
                            if (score > conf_thres){
                                float inv_scale = 1.0f / scales[b];
                                const float* box_ptr = boxes_ptr + (b * max_det + i) * 4;
                                float x1 = (box_ptr[0] - dws[b]) * inv_scale;
                                float y1 = (box_ptr[1] - dhs[b]) * inv_scale;
                                float x2 = (box_ptr[2] - dws[b]) * inv_scale;
                                float y2 = (box_ptr[3] - dhs[b]) * inv_scale;

                                int cls = get_int_val(t_cls, b * max_det + i);
                                Box bbox {x1, y1, x2, y2};
                                BoundingBox boundingbox {bbox, score, cls};
                                batch_dets[b].push_back(boundingbox);
                            }
                        }
                    }
                }

                else{
                    TensorInfo* out_tensor = outputs[0];
                    const float* output_data = out_tensor->host_buffer.data();
                    const auto& actual_shape = out_tensor->actual_shape;
                    int ndim = actual_shape.size();

                    if (config_.end2end){
                        int num_dets = actual_shape[0];
                        int dim = actual_shape[1]; // 7
                        for (int i = 0; i < num_dets; i++){
                            const float* row = output_data + i * dim;
                            int b = static_cast<int>(row[0]);
                            // batch 不能大于real_batch 不能小于0
                            if (b < 0 || b >= real_batch_size) continue;
                            float score = row[5];
                            if (score > conf_thres){
                                float inv_scale = 1.0f / scales[b];
                                float x1 = (row[1] - dws[b]) * inv_scale;
                                float y1 = (row[2] - dhs[b]) * inv_scale;
                                float x2 = (row[3] - dws[b]) * inv_scale;
                                float y2 = (row[4] - dhs[b]) * inv_scale;
                                int cls = static_cast<int>(row[6]);
                                Box bbox {x1, y1, x2, y2};
                                BoundingBox boundingbox {bbox, score, cls};
                                batch_dets[b].push_back(boundingbox);
                            }
                        }
                    }
                    else if (config_.end2end_model){
                        int num_anchors = actual_shape[ndim - 2];
                        int dim = actual_shape[ndim - 1];
                        for (int b = 0; b < real_batch_size; b++){
                            const float* batch_ptr = output_data + b * num_anchors * dim;
                            for (int i = 0; i < num_anchors; i++){
                                const float* row = batch_ptr + i * dim;
                                float score = row[4];
                                if (score > conf_thres) {
                                    float inv_scale = 1.0f / scales[b];
                                    float x1 = (row[0] - dws[b]) * inv_scale;
                                    float y1 = (row[1] - dhs[b]) * inv_scale;
                                    float x2 = (row[2] - dws[b]) * inv_scale;
                                    float y2 = (row[3] - dhs[b]) * inv_scale;
                                    int cls = static_cast<int>(row[5]);

                                    Box bbox {x1, y1, x2, y2};
                                    BoundingBox boundingbox {bbox, score, cls};
                                    batch_dets[b].push_back(boundingbox);
                                }
                            }
                        }
                    }
                    else{


                        // 1. 获取 TRT 输出在 GPU 上的地址
                        float* dYolo = reinterpret_cast<float*>(out_tensor->dev_ptr);

                        // 防卫性清零！哪怕 NMS 罢工，读回来的框数量也是 0，绝不是垃圾值
                        cudaMemsetAsync(d_nms_num_det, 0, real_batch_size * sizeof(int32_t), stream);

                        int ndim = actual_shape.size();
                        int num_anchors = config_.ultralytics ? actual_shape[ndim - 1] : actual_shape[ndim - 2];
                        nms_processor->configure(real_batch_size, num_anchors, this->num_classes);


                        // 2. 取 TRT 输出在 GPU 上的地址
                        nms_processor->run(dYolo, d_nms_workspace, d_nms_num_det, 
                        d_nms_boxes, d_nms_scores, d_nms_classes, stream);
                        
                        // 3. 异步拷贝 NMS 过滤后的结果到 CPU
                        // cudaMemcpyAsync(h_nms_num_det.data(), d_nms_num_det, real_batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
                        // cudaMemcpyAsync(h_nms_boxes.data(),   d_nms_boxes,   real_batch_size * max_det * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
                        // cudaMemcpyAsync(h_nms_scores.data(),  d_nms_scores,  real_batch_size * max_det * sizeof(float), cudaMemcpyDeviceToHost, stream);
                        // cudaMemcpyAsync(h_nms_classes.data(), d_nms_classes, real_batch_size * max_det * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);

                        cudaMemcpyAsync(h_nms_num_det_pinned, d_nms_num_det, real_batch_size * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);
                        cudaMemcpyAsync(h_nms_boxes_pinned,   d_nms_boxes,   real_batch_size * max_det * 4 * sizeof(float), cudaMemcpyDeviceToHost, stream);
                        cudaMemcpyAsync(h_nms_scores_pinned,  d_nms_scores,  real_batch_size * max_det * sizeof(float), cudaMemcpyDeviceToHost, stream);
                        cudaMemcpyAsync(h_nms_classes_pinned, d_nms_classes, real_batch_size * max_det * sizeof(int32_t), cudaMemcpyDeviceToHost, stream);

                        // 4. 等待所有流完成（等待拷贝和计算结束）
                        cudaStreamSynchronize(stream);

                        // 5. 将精简后的结果进行原图坐标还原
                        for (int b = 0; b < real_batch_size; b++){
                            int valid_count = h_nms_num_det_pinned[b]; // 获取当前 batch 保留的框数量
                            if (valid_count < 0) valid_count = 0;
                            if (valid_count > max_det) valid_count = max_det;
                            // 一键预分配确切内存，极致压榨 CPU 性能
                            batch_dets[b].reserve(valid_count);

                            float inv_scale = 1.0f / scales[b];
                            float dw = dws[b];
                            float dh = dhs[b];

                            for (int i = 0; i < valid_count; i++) {
                                int base_idx = b * max_det + i;
                                
                                float score = h_nms_scores_pinned[base_idx];
                                int cls = h_nms_classes_pinned[base_idx];
                                const float* box = &h_nms_boxes_pinned[base_idx * 4];

                                // 还原坐标 (x1, y1, x2, y2)
                                float x1 = (box[0] - dw) * inv_scale;
                                float y1 = (box[1] - dh) * inv_scale;
                                float x2 = (box[2] - dw) * inv_scale;
                                float y2 = (box[3] - dh) * inv_scale;

                                Box bbox {x1, y1, x2, y2};
                                BoundingBox boundingbox {bbox, score, cls};
                                batch_dets[b].push_back(boundingbox);
                            }
                        }
                    }
                }
                return batch_dets;
            }

            // draw rectangle
            void draw_results(cv::Mat& img, const BatchResult& res){
                for (size_t i = 0; i < res.size(); i++){
                    // static_cast 杜绝隐式转换警告
                    int x1 = static_cast<int>(std::round(res[i].box.x1));
                    int y1 = static_cast<int>(std::round(res[i].box.y1));
                    int x2 = static_cast<int>(std::round(res[i].box.x2));
                    int y2 = static_cast<int>(std::round(res[i].box.y2));

                    x1 = std::max(0, std::min(x1, img.cols));
                    y1 = std::max(0, std::min(y1, img.rows));
                    x2 = std::max(0, std::min(x2, img.cols));
                    y2 = std::max(0, std::min(y2, img.rows));

                    int cls_id = res[i].class_id;
                    float score = res[i].score;

                    cv::Scalar color( (cls_id * 50) % 255, (cls_id * 100) % 255, (cls_id * 150) % 255 );
                    cv::rectangle(img, cv::Point(x1, y1), cv::Point(x2, y2), color, 2);

                    char score_str[8];
                    snprintf(score_str, sizeof(score_str), "%.2f", score);
                    // std::string label = (cls_id < (int)class_names.size() ? class_names[cls_id] : std::to_string(cls_id)) + ": " + score_str;
                    std::string label = (cls_id >= 0 && cls_id < (int)class_names.size() ? class_names[cls_id] : std::to_string(cls_id)) + ": " + score_str;
                    int baseLine;
                    cv::Size labelSize = cv::getTextSize(label, cv::FONT_HERSHEY_SIMPLEX, 0.5, 1, &baseLine);
                    // 如果顶部空间不够，就把标签挪到框的内部去画
                    int label_y = (y1 - labelSize.height - 3 < 0) ? y1 + labelSize.height + 3 : y1;
                    cv::rectangle(img, cv::Point(x1, label_y - labelSize.height - 3), cv::Point(x1 + labelSize.width, label_y), color, cv::FILLED);
                    cv::putText(img, label, cv::Point(x1, label_y - 2), cv::FONT_HERSHEY_SIMPLEX, 0.5, cv::Scalar(255, 255, 255), 1, cv::LINE_AA);
                }
            }

            // 实现接口 1：推理
            std::pair<std::vector<BatchResult>, std::vector<float>> infer_batch(const std::vector<cv::Mat>& images) override {
                int real_batch_size = images.size();
                std::vector<float> scales;
                std::vector<int> dws, dhs;

                // 1. 预处理
                // cv::Mat blob = preprocess_batch(img_list, scales, dws, dhs);
                if (profile) cudaEventRecord(event_start, stream);

                // 2. H2D
                TensorInfo* input_tensor = nullptr;
                for (auto& t : io_tensors){
                    if (t.is_input){
                        input_tensor = &t;
                        // 设置输入维度
                        nvinfer1::Dims4 input_dims {real_batch_size, input_channels, input_height, input_width};
                        context->setInputShape(t.name.c_str(), input_dims);

                        // // 仅拷贝真实的 batch size 数据
                        // size_t input_bytes = real_batch_size * 3 * input_height * input_width * sizeof(float);
                        // cudaMemcpyAsync(t.dev_ptr, blob.ptr<float>(), input_bytes, cudaMemcpyHostToDevice, stream);
                    } else {
                        // 防止INMSlayer的多余框
                        cudaMemsetAsync(t.dev_ptr, 0, t.size_bytes, stream);
                    }
                }
                float* trt_input_ptr = static_cast<float*>(input_tensor->dev_ptr);

                // --- 2. 动态维护原图显存池 (懒加载机制) ---
                int max_current_bytes = 0;
                std::vector<cv::Mat> continuous_imgs;
                continuous_imgs.reserve(real_batch_size);

                for (int b = 0; b < real_batch_size; b++) {
                    cv::Mat img = images[b];
                    
                    // 必须检查！打破 OpenCV 的内存 Padding 陷阱，强制内存连续
                    if (!img.isContinuous()) {
                        img = img.clone();
                    }
                    continuous_imgs.push_back(img);
                    
                    int bytes = img.cols * img.rows * input_channels;
                    if (bytes > max_current_bytes) {
                        max_current_bytes = bytes;
                    }
                }

                // 如果遇到比以前更大的图，扩容 GPU 显存池
                // 如果是比之前小或者一样的图片，则不需要再次分配显存，因为现在的显存足够大了
                // 因为分配大块显存极其耗时，我们要尽可能白嫖已经开辟好的大空间。
                if (max_current_bytes > max_src_bytes) {
                    for (auto ptr : d_img_buffers) { if(ptr) cudaFree(ptr); }
                    for (int i = 0; i < max_batch_size; i++) {
                        cudaMalloc((void**)&d_img_buffers[i], max_current_bytes); 
                    }
                    max_src_bytes = max_current_bytes;
                }

                // d_img_buffers 存在显存复用， 做到零显存分配
                launch_preprocess_cuda(
                    continuous_imgs,
                    trt_input_ptr,
                    input_width, input_height,
                    d_img_buffers,
                    d_img_ptrs,
                    scales, dws, dhs,   // <--- 这里接收返回值
                    stream
                );



                if (profile) cudaEventRecord(event_h2d_preprocess, stream);

                // 3. infer(V3接口)
                context->enqueueV3(stream);

                if (profile) cudaEventRecord(event_comp, stream);

                // 4. D2H & 获取真实维度 (零拷贝视图的等效实现)
                bool need_full_d2h = true; // 默认需要拷回全量数据
                if (noend){need_full_d2h = false;} // 触发极致零拷贝模式！只有端到端模型需要全量数据，其他情况都不需要，直接在 GPU 上处理完后只拷回有效结果即可
                for (auto& t : io_tensors){
                    if (!t.is_input){
                        nvinfer1::Dims actual_dims = context->getTensorShape(t.name.c_str());

                        bool has_dynamic = false;
                        size_t actual_vol = 1;
                        for (int j = 0; j < actual_dims.nbDims; ++j) {
                            if (actual_dims.d[j] < 0){
                                has_dynamic = true;
                                break;
                            }
                            actual_vol *= actual_dims.d[j];
                        }

                        size_t bytes_to_copy = has_dynamic ? t.size_bytes : actual_vol * sizeof(float);
                        if (has_dynamic){
                            t.actual_shape = t.max_shape;
                        } else {
                            // 使用assign赋值，避免重新新建一个Vector
                            t.actual_shape.assign(actual_dims.d, actual_dims.d + actual_dims.nbDims);
                        }
                        // t.actual_shape = has_dynamic ? t.max_shape : std::vector<int64_t>(actual_dims.d, actual_dims.d + actual_dims.nbDims);

                        // 仅拷回有效数据，剔除无用显存，极大幅度提升 D2H 速度 但是 INMSlayer 拷贝所有数据
                        if (need_full_d2h){
                            cudaMemcpyAsync(t.host_buffer.data(), t.dev_ptr, bytes_to_copy, cudaMemcpyDeviceToHost, stream);
                        }
                    }
                }

                if (profile) cudaEventRecord(event_d2h, stream);
                if (need_full_d2h || profile){
                    cudaStreamSynchronize(stream);
                }

                // 5. 后处理
                auto t_post_start = std::chrono::high_resolution_clock::now();
                std::vector<BatchResult> batch_dets = process_output(real_batch_size, scales, dws, dhs);
                auto t_post_end = std::chrono::high_resolution_clock::now();

                std::vector<float> prof_times(4, 0.0f);
                if (profile) {
                    cudaEventElapsedTime(&prof_times[0], event_start, event_h2d_preprocess);
                    cudaEventElapsedTime(&prof_times[1], event_h2d_preprocess, event_comp);
                    cudaEventElapsedTime(&prof_times[2], event_comp, event_d2h);
                    prof_times[3] = std::chrono::duration<float, std::milli>(t_post_end - t_post_start).count();
                }

                return {batch_dets, prof_times};
            }

            // 实现接口 2：多线程流水线
            void run(){
                std::string source = config_.source;
                int batch_size = opt_batch_size; // 按最优批次走
                std::string save_dir = config_.save_dir;

                if (config_.save) {
                    fs::create_directories(save_dir);
                }

                if (fs::is_directory(source)){
                    // === 模式 1: 目录图片多批次攒帧推理 保持串行，补充画框逻辑===
                    std::vector<std::string> img_paths;
                    for (const auto& entry : fs::directory_iterator(source)){
                        std::string ext = entry.path().extension().string();
                        std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                        if (std::find(IMAGE_EXTS.begin(), IMAGE_EXTS.end(), ext) != IMAGE_EXTS.end()){
                            img_paths.push_back(entry.path().string());
                        }
                    }

                    std::sort(img_paths.begin(), img_paths.end());

                    std::cout << "找到 " << img_paths.size() << " 张图片，按照 opt_batch_size=" << batch_size << " 开始推理...\n";

                    for (size_t i = 0; i < img_paths.size(); i += batch_size){
                        std::vector<cv::Mat> valid_imgs;
                        std::vector<std::string> valid_names;

                        for (size_t j = i; j < std::min(i + batch_size, img_paths.size()); ++j){
                            cv::Mat img = cv::imread(img_paths[j]);
                            if (!img.empty()){
                                valid_imgs.push_back(img);
                                valid_names.push_back(fs::path(img_paths[j]).filename().string());
                            }
                        }

                        if (valid_imgs.empty()) continue;

                        auto t1 = std::chrono::high_resolution_clock::now();
                        auto [batch_dets, prof] = infer_batch(valid_imgs);
                        auto t2 = std::chrono::high_resolution_clock::now();
                        double t = std::chrono::duration<double, std::milli>(t2 - t1).count();

                        // 手动画框并保存
                        for (size_t k = 0; k < valid_imgs.size(); ++k){
                            if (!config_.no_draw) draw_results(valid_imgs[k], batch_dets[k]);
                            if (config_.save) cv::imwrite((fs::path(save_dir)/valid_names[k]).string(), valid_imgs[k]);
                        }


                        // if (args.profile){
                        //     printf("[Profile] H2D: %.2fms | Compute: %.2fms | D2H: %.2fms\n", prof[0], prof[1], prof[2]);
                        // }
                        if (profile){
                            if (noend) {
                                printf("[Profile] Preprocess(H2D+Kernel): %.2fms | Compute: %.2fms | Postprocess(D2H+kernel): %.2fms\n", 
                                prof[0], prof[1], prof[3]);
                            }
                            else {
                                printf("[Profile] Preprocess(H2D+Kernel): %.2fms | Compute: %.2fms | D2H: %.2fms | Postprocess: %.2fms\n", 
                                prof[0], prof[1], prof[2], prof[3]);
                            }
                        }
                        std::cout << "已处理进度: " << std::min(i + batch_size, img_paths.size()) << "/" << img_paths.size()
                                << " | Batch总耗时: " << std::fixed << std::setprecision(2) << t << "ms\n";
                    }
                    std::cout << "✅ 目录处理完成。\n";
                }else{
                    std::string ext = fs::path(source).extension().string();
                    std::transform(ext.begin(), ext.end(), ext.begin(), ::tolower);
                    bool is_image = std::find(IMAGE_EXTS.begin(), IMAGE_EXTS.end(), ext) != IMAGE_EXTS.end();

                    if (is_image){
                        // === 模式 2: 单张图片推理 (保持串行)===
                        cv::Mat img = cv::imread(source);
                        if (img.empty()) return;

                        auto t1 = std::chrono::high_resolution_clock::now();
                        auto [batch_dets, prof] = infer_batch({img});
                        auto t2 = std::chrono::high_resolution_clock::now();
                        double t = std::chrono::duration<double, std::milli>(t2 - t1).count();
                        if (!config_.no_draw) draw_results(img, batch_dets[0]);
                        if (config_.save) cv::imwrite((fs::path(save_dir)/fs::path(source).filename()).string(), img);

                        // if (args.profile) printf("[Profile] H2D: %.2fms | Compute: %.2fms | D2H: %.2fms\n", prof[0], prof[1], prof[2]);
                        if (profile){
                            if (noend) {
                                printf("[Profile] Preprocess(H2D+Kernel): %.2fms | Compute: %.2fms | Postprocess(D2H+kernel): %.2fms\n", 
                                prof[0], prof[1], prof[3]);
                            }
                            else {
                                printf("[Profile] Preprocess(H2D+Kernel): %.2fms | Compute: %.2fms | D2H: %.2fms | Postprocess: %.2fms\n", 
                                prof[0], prof[1], prof[2], prof[3]);
                            }
                        }
                        std::cout << "推理时间: " << t << "ms, 结果已保存\n";
                    }
                    else {
                        // === 模式 3: 视频/RTSP 攒帧加速推理 ===
                        cv::VideoCapture cap;
                        bool is_digit = !source.empty() && std::all_of(source.begin(), source.end(), ::isdigit);
                        if (is_digit) cap.open(std::stoi(source));
                        else cap.open(source);
                        if (!cap.isOpened()) return;

                        int width = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_WIDTH));
                        int height = static_cast<int>(cap.get(cv::CAP_PROP_FRAME_HEIGHT));
                        // double fps = cap.get(cv::CAP_PROP_FPS);
                        // if (fps == 0.0) fps = 25.0;

                        double fps = cap.get(cv::CAP_PROP_FPS);
                        if (fps <= 0.0 || std::isnan(fps) || std::isinf(fps)){
                            std::cout << "[警告] 无法获取真实 FPS，强制使用默认值 25.0\n";
                            fps = 25.0;
                        }

                        cv::VideoWriter out_writer;
                        bool is_file = fs::exists(source);
                        if (is_file && config_.save){
                            std::string save_path = (fs::path(save_dir)/fs::path(source).filename()).string();
                            out_writer.open(save_path, cv::VideoWriter::fourcc('m', 'p', '4', 'v'), fps, cv::Size(width, height));
                            std::cout << "视频开始处理，按 opt_batch=" << batch_size << " 攒批...\n";
                        }


                        /*
                        为什么使用 std::atomic<bool>？
                        在多线程环境中，多个线程可能会同时访问和修改同一个变量。如果这个变量是一个普通的 bool 类型，那么在没有适当的同步机制（如互斥锁）的情况下，可能会导致数据竞争
                        另外编译阶段使用了-O3优化选项，编译器可能会对普通的 bool 变量进行寄存器优化，每次循环只检查寄存器里的值。导致一个线程修改了这个变量的值，但其他线程可能无法及时看到这个更新，从而无法正确响应停止信号。
                        atomic 告诉编译器：“这个变量随时可能被别的线程暗改！你绝对不许把它优化到寄存器里。每次读它，都必须老老实实去主内存（或多核共享缓存）里拿最新的一手数据！
                        */
                        std::atomic<bool> pipeline_stop{false}; // 线程安全的停止信号
                        // 创建两个管道缓冲队列 容量为3
                        SafeQueue<PipelineData> in_queue(3);
                        SafeQueue<PipelineData> out_queue(3);

                        // ----------------------------------------------------
                        // 🧵 线程 1: Reader (专职读图，IO 密集型)
                        // ----------------------------------------------------
                        std::thread reader_thread([&](){
                            std::vector<cv::Mat> batch_frames;
                            cv::Mat frame;
                            while (!pipeline_stop && cap.read(frame)){ // 只要读到帧且没有停止信号，就不断的往队列里面放数据
                                batch_frames.push_back(frame.clone());
                                if (batch_frames.size() == static_cast<size_t>(batch_size)){ // 凑够一个批次的图片就推送入队列
                                    PipelineData data;
                                    // 使用swap而不是直接赋值
                                    // data.frames = batch_frames;
                                    std::swap(data.frames, batch_frames);
                                    // 使用右值传入，触发移动语义，避免不必要的复制
                                    // in_queue.push(data);
                                    in_queue.push(std::move(data));
                                    // batch_frames.clear();
                                    // swap之后batch_frames是空的，直接reserve就行了
                                    batch_frames.reserve(batch_size);
                                }
                            }
                            if (!batch_frames.empty() && !pipeline_stop){ // 尾部不足一个批次的残余帧也送入队列
                                PipelineData data;
                                // data.frames = batch_frames;
                                std::swap(data.frames, batch_frames);
                                // 右值传人，同理
                                in_queue.push(std::move(data));
                            }

                            // 读完了，发个空包当结束信号

                            if (!pipeline_stop){
                                PipelineData end_data;
                                end_data.is_last = true;
                                in_queue.push(std::move(end_data));
                            }
                        });


                        // ----------------------------------------------------
                        // 🧵 线程 2: Worker (专职调用 TRT，GPU 计算密集型)
                        // ----------------------------------------------------
                        std::thread worker_thread([&](){
                            PipelineData data;
                            while (in_queue.pop(data)){
                                if (pipeline_stop) break;
                                if (data.is_last){
                                    out_queue.push(std::move(data)); // 击鼓传花，把结束信号传给主线程
                                    break;
                                }
                                auto t1 = std::chrono::high_resolution_clock::now();
                                auto [dets, prof] = infer_batch(data.frames);
                                auto t2 = std::chrono::high_resolution_clock::now();

                                data.dets = std::move(dets);
                                data.prof = std::move(prof);
                                data.batch_time = std::chrono::duration<double, std::milli>(t2 - t1).count();
                                out_queue.push(std::move(data));
                            }
                        });

                        // ----------------------------------------------------
                        // 🧵 主线程: Writer (专职画框、显示、写硬盘)
                        // ----------------------------------------------------


                        int frame_count = 0;
                        PipelineData out_data;
                        
                        auto last_pop_time = std::chrono::high_resolution_clock::now();
                        auto global_start_time = last_pop_time;

                        while (out_queue.pop(out_data)){
                            if (out_data.is_last) break;

                            auto current_pop_time = std::chrono::high_resolution_clock::now();
                            double pipeline_batch_time = std::chrono::duration<double, std::milli>(current_pop_time - last_pop_time).count();
                            last_pop_time = current_pop_time; // 更新打点

                            // 1. 真实的端到端流水线 FPS
                            double true_fps = 1000.0 / (pipeline_batch_time / out_data.frames.size());
                            // 2. GPU 纯算力 FPS
                            double gpu_fps = 1000.0 / (out_data.batch_time / out_data.frames.size());


                            if (profile){
                                const auto& prof = out_data.prof;
                                if (noend) {
                                    printf("[Profile] Preprocess(H2D+Kernel): %.2fms | Compute: %.2fms | Postprocess(D2H+kernel): %.2fms\n", 
                                    prof[0], prof[1], prof[3]);
                                }
                                else {
                                    printf("[Profile] Preprocess(H2D+Kernel): %.2fms | Compute: %.2fms | D2H: %.2fms | Postprocess: %.2fms\n", 
                                    prof[0], prof[1], prof[2], prof[3]);
                                }
                            }

                            for (size_t i = 0; i < out_data.frames.size(); i++){
                                if (!config_.no_draw){
                                    // 1. 画框
                                    draw_results(out_data.frames[i], out_data.dets[i]);

                                    // 2. 显示FPS
                                    char fps_text[128];
                                    snprintf(fps_text, sizeof(fps_text), "SYS FPS: %.1f | GPU FPS: %.1f", true_fps, gpu_fps);
                                    cv::putText(out_data.frames[i], fps_text, cv::Point(20, 40),
                                                cv::FONT_HERSHEY_SIMPLEX, 0.7, cv::Scalar(0, 0, 255), 2);
                                }
                                
                                // 3. 写入视频与显示
                                if (out_writer.isOpened()) out_writer.write(out_data.frames[i]);
                                if (!config_.no_show){
                                    cv::imshow("TRT C++ Pipeline", out_data.frames[i]);
                                    if (cv::waitKey(1) == 'q') {
                                        pipeline_stop = true;
                                        in_queue.stop(); out_queue.stop(); // 优雅关闭线程
                                        break;
                                    }
                                }
                            }

                            if (pipeline_stop) break;

                            frame_count += out_data.frames.size();
                            if (frame_count % (batch_size * 5) == 0) {
                                std::cout << "已处理 " << frame_count << " 帧 | "
                                        << "GPU 耗时: " << std::fixed << std::setprecision(1) << out_data.batch_time << "ms | "
                                        << "流水线节拍耗时: " << pipeline_batch_time << "ms\n";
                            }
                        }

                        // ----------------------------------------------------
                        // 🪦 打扫战场
                        // ----------------------------------------------------
                        auto global_end_time = std::chrono::high_resolution_clock::now();
                        double total_seconds = std::chrono::duration<double>(global_end_time - global_start_time).count();

                        reader_thread.join();
                        worker_thread.join();
                        cap.release();
                        if (out_writer.isOpened()) out_writer.release();
                        cv::destroyAllWindows();
                        std::cout << "\n=========================================\n";
                        std::cout << "✅ 视频多线程检测完毕。\n";
                        std::cout << "处理总帧数: " << frame_count << " 帧\n";
                        std::cout << "系统平均总吞吐量: " << frame_count / total_seconds << " FPS\n";
                        std::cout << "=========================================\n";
                    }
                }
            }

    };


    // 实现工厂函数：对外唯一暴露的符号！
    std::shared_ptr<IYoloDetector> create_detector(const Config& config) {
        try{
            return std::make_shared<YoloDetectorImpl>(config);
        }catch (const std::exception& e){
            std::cerr << "创建 YoloDetector 失败: " << e.what() << std::endl;
            return nullptr;
        }
    }

};