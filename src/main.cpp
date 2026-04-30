#include "yolo_trt.h" // 现在这个头文件非常干净，没有第三方依赖
#include <opencv2/opencv.hpp>
#include <iostream>
#include <chrono>
#include <thread>
#include <mutex>
#include <condition_variable>
#include <queue>
#include <atomic>

#include <filesystem>

namespace fs = std::filesystem;

static const std::vector<std::string> IMAGE_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"};

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


// draw rectangle
void draw_results(cv::Mat& img, const yolo::BatchResult& res, const std::vector<std::string>& class_names = COCO_NAMES) {
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


void run(const yolo::Config& config){
    auto detector = yolo::create_detector(config);
    if (!detector) {
        return;
    }
    std::string source = config.source;
    bool noend = !config.end2end && !config.efficient_end2end && !config.end2end_model;
    auto opt_batch_size = (config.opt_batch_size > 0) ? config.opt_batch_size : config.max_batch_size;

    int batch_size = opt_batch_size; // 按最优批次走
    std::string save_dir = config.save_dir;

    if (config.save) {
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

            std::vector<yolo::ImageView> input_views;

            for (size_t j = i; j < std::min(i + batch_size, img_paths.size()); ++j){
                cv::Mat img = cv::imread(img_paths[j]);
                // 【核心防御】：脱离了 OpenCV，底层 CUDA 前处理极其依赖连续内存！
                // 很多裁剪过的 ROI 图像在内存里是断片的，必须强行拉平成一块连续内存。
                if (!img.isContinuous()) {
                    img = img.clone();
                } 
                if (!img.empty()){
                    valid_imgs.push_back(img);
                    valid_names.push_back(fs::path(img_paths[j]).filename().string());
                }

                // 【新增】：将 cv::Mat 的信息“翻译”成纯 C++ 的 ImageView
                yolo::ImageView view;
                view.data = img.data;         // 提取 BGR 裸数据指针
                view.width = img.cols;        // 提取宽度
                view.height = img.rows;       // 提取高度
                view.channels = img.channels(); // 提取通道数 (通常是 3)

                input_views.push_back(view);
            }

            if (valid_imgs.empty()) continue;

            auto t1 = std::chrono::high_resolution_clock::now();
            auto [batch_dets, prof] = detector->infer_batch(input_views);
            auto t2 = std::chrono::high_resolution_clock::now();
            double t = std::chrono::duration<double, std::milli>(t2 - t1).count();

            // 手动画框并保存
            for (size_t k = 0; k < valid_imgs.size(); ++k){
                if (!config.no_draw) draw_results(valid_imgs[k], batch_dets[k]);
                if (config.save) cv::imwrite((fs::path(save_dir)/valid_names[k]).string(), valid_imgs[k]);
            }


            // if (args.profile){
            //     printf("[Profile] H2D: %.2fms | Compute: %.2fms | D2H: %.2fms\n", prof[0], prof[1], prof[2]);
            // }
            if (config.profile){
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

            // 【新增】：将 cv::Mat 的信息“翻译”成纯 C++ 的 ImageView
            yolo::ImageView view;
            view.data = img.data;         // 提取 BGR 裸数据指针
            view.width = img.cols;        // 提取宽度
            view.height = img.rows;       // 提取高度
            view.channels = img.channels(); // 提取通道数 (通常是 3)

            auto t1 = std::chrono::high_resolution_clock::now();
            auto [batch_dets, prof] = detector->infer_batch({view});
            auto t2 = std::chrono::high_resolution_clock::now();
            double t = std::chrono::duration<double, std::milli>(t2 - t1).count();
            if (!config.no_draw) draw_results(img, batch_dets[0]);
            if (config.save) cv::imwrite((fs::path(save_dir)/fs::path(source).filename()).string(), img);

            // if (args.profile) printf("[Profile] H2D: %.2fms | Compute: %.2fms | D2H: %.2fms\n", prof[0], prof[1], prof[2]);
            if (config.profile){
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
            if (is_file && config.save){
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
                    // 🌟 核心适配：将 OpenCV 的 cv::Mat 翻译给底层 SDK 认识的 ImageView
                    std::vector<yolo::ImageView> input_views;
                    input_views.reserve(data.frames.size());
                    for (auto& img : data.frames) {
                        // 必须保证内存连续！
                        if (!img.isContinuous()) {
                            img = img.clone(); 
                        }
                        yolo::ImageView view;
                        view.data = img.data;
                        view.width = img.cols;
                        view.height = img.rows;
                        view.channels = img.channels();
                        input_views.push_back(view);
                    }
                    auto t1 = std::chrono::high_resolution_clock::now();
                    auto [dets, prof] = detector->infer_batch(input_views);
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


                if (config.profile){
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
                    if (!config.no_draw){
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
                    if (!config.no_show){
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


void print_usage(const char* program_name) {
    std::cout << "Usage: " << program_name << " [options]\n"
              << "Options:\n"
              << "  -e, --engine <path>    TRT engine path (required)\n"
              << "  -i, --image <path>     Input image path\n"
              << "  -o, --output <path>    Output image/video path (default: result.jpg/result.mp4)\n"
              << "  -v, --video <path>     Input video path\n"
              << "  --conf <float>         Confidence threshold (default: 0.25)\n"
              << "  --iou <float>          NMS IoU threshold (default: 0.7)\n"
              << "  --end2end              Use end2end engine\n"
              << "  --efficient_end2end    Use efficient_end2end engine\n"
              << "  --ultralytics          Use ultralytics model\n"
              << "  --end2end_model        Use end2end model\n"
              << "  -h, --help             Show this help message\n"
              << std::endl;
}

int main(int argc, char* argv[]) {
    if (argc < 2) {
        print_usage(argv[0]);
        return 1;
    }

    yolo::Config args;

    // 1. 简单 CLI 解析
    for (int i = 1; i < argc; i++){
        std::string arg = argv[i];
        if (arg == "--engine" && i + 1 < argc) args.engine_path = argv[++i];
        else if (arg == "--source" && i + 1 < argc) args.source = argv[++i];
        else if (arg == "--opt_batch_size" && i + 1 < argc) args.opt_batch_size = std::stoi(argv[++i]);
        else if (arg == "--max_batch_size" && i + 1 < argc) args.max_batch_size = std::stoi(argv[++i]);
        else if (arg == "--save_dir" && i + 1 < argc) args.save_dir = argv[++i];
        else if (arg == "--conf" && i + 1 < argc) args.conf_thres = std::stof(argv[++i]);
        else if (arg == "--iou" && i + 1 < argc) args.iou_thres = std::stof(argv[++i]);
        else if (arg == "--classes" && i + 1 < argc) args.num_classes = std::stoi(argv[++i]);
        else if (arg == "--save") args.save = true;
        else if (arg == "--efficient_end2end") args.efficient_end2end = true;
        else if (arg == "--end2end") args.end2end = true;
        else if (arg == "--end2end_model") args.end2end_model = true;
        else if (arg == "--ultralytics") args.ultralytics = true;
        else if (arg == "--no_show") args.no_show = true;
        else if (arg == "--no_draw") args.no_draw = true;
        else if (arg == "--profile") args.profile = true;
        else if (arg == "-h" || arg == "--help") {
            std::cout << "用法: " << argv[0] << " [选项]\n"
                      << "  --engine <path>     引擎路径 (默认: weights/yolov7-tiny.engine)\n"
                      << "  --source <path>     输入文件/目录/摄像头\n"
                      << "  --opt_batch_size    推理批量大小 (默认: 1)\n"
                      << "  --max_batch_size    最大允许推理batch\n"
                      << "  --classes           类别数量\n"
                      << "  --save_dir <path>   保存结果目录\n"
                      << "  --save              是否保存结果\n"
                      << "  --profile           开启 CUDA 测速\n"
                      << "  --efficient_end2end 使用efficient_nms 插件\n"
                      << "  --end2end           使用 INMSlayer 插件\n"
                      << "  --end2end_model     使用端到端模型, 如yolo26\n"
                      << "  --ultralytics       使用 Ultralytics 模型\n"
                      << "  --no_show           是否显示画面,默认显示\n";
            return 0;
        }
    }

    run(args);

    return 0;
}