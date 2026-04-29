#pragma once

#include <vector>
#include <string>
#include <memory>

#include <opencv2/opencv.hpp>


namespace yolo{
    struct Config {
        std::string engine_path;
        std::string source;
        int opt_batch_size = 1;
        int max_batch_size = 32;
        bool save = false;
        std::string save_dir;
        float conf_thres = 0.25f;
        float iou_thres = 0.7f;
        int num_classes = 80;
        bool end2end = false;
        bool efficient_end2end = false;
        bool end2end_model = false;
        bool ultralytics = false;
        bool no_show = false;
        bool no_draw = false;
        bool profile = false;
    };

    struct Box {
        float x1, y1, x2, y2;
    };

    struct BoundingBox {
        Box box;
        float score;
        int class_id;
    };


    using BatchResult = std::vector<BoundingBox>;

    class IYoloDetector {
        public:
            virtual ~IYoloDetector() = default;

            // 核心接口 1：传入多张图片，返回检测结果
            virtual std::pair<std::vector<BatchResult>, std::vector<float>> infer_batch(const std::vector<cv::Mat>& images) = 0;

            // 核心接口 2：处理视频流流水线 (传入视频路径/摄像头索引)
            virtual void run() = 0;

    };

    // 对外暴露的工厂函数：用于创建实例
    // 使用 shared_ptr 自动管理内存，用户无需手动 delete
    std::shared_ptr<IYoloDetector> create_detector(const Config& config);

}// namespace yolo