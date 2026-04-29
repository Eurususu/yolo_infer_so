#include "yolo_trt.h" // 现在这个头文件非常干净，没有第三方依赖
#include <opencv2/opencv.hpp>
#include <iostream>

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

    // 2. 创建推理引擎实例
    auto detector = yolo::create_detector(args);
    if (!detector) {
        return -1;
    }

    // 3. 极简调用：视频多线程流水线
    detector->run();

    return 0;
}