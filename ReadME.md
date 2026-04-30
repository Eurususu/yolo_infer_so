### cv::mat 依赖版本
构建
`./build.sh`
运行
`./build/yolo --engine weights/yolo11s.engine --source /home/jia/project/export_yolos/data/video1.mp4 --opt_batch_size 16 --ultralytics --conf 0.25 --profile`

### 纯cuda + tensorrrt
构建
`./build.sh`
运行
`./build/yolo --engine weights/yolo11s.engine --source /home/jia/project/export_yolos/data/video1.mp4 --opt_batch_size 16 --ultralytics --conf 0.25 --profile`