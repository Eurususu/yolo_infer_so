cmake_minimum_required(VERSION 3.18)
project(trt_infer LANGUAGES CXX CUDA)

set(CMAKE_BUILD_TYPE "Release")
set(CMAKE_CXX_FLAGS "-O3 -Wall -ffast-math")
set(CMAKE_CXX_FLAGS_RELEASE "-O3 -DNDEBUG")

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# 强制生成位置无关代码 (-fPIC)，这是编译动态链接库 (.so) 必须的
set(CMAKE_POSITION_INDEPENDENT_CODE ON)

# ==========================================
# 1. TensorRT 设置
# ==========================================
set(TENSORRT_ROOT "/opt/TensorRT" CACHE PATH "TensorRT root directory")
set(TENSORRT_INCLUDE_DIR "${TENSORRT_ROOT}/include" CACHE PATH "TensorRT include directory")
set(TENSORRT_LIB_DIR "${TENSORRT_ROOT}/lib" CACHE PATH "TensorRT library directory")

# 查找 TensorRT 主库 (nvinfer)
find_library(TENSORRT_LIBRARY 
    NAMES nvinfer
    HINTS ${TENSORRT_LIB_DIR}
)
if(NOT TENSORRT_LIBRARY)
    message(FATAL_ERROR "Could not find libnvinfer.so in ${TENSORRT_LIB_DIR}")
endif()

# 查找 TensorRT 插件库 (nvinfer_plugin)，用于 initLibNvInferPlugins
find_library(TENSORRT_PLUGIN_LIBRARY 
    NAMES nvinfer_plugin
    HINTS ${TENSORRT_LIB_DIR}
)
if(NOT TENSORRT_PLUGIN_LIBRARY)
    message(FATAL_ERROR "Could not find libnvinfer_plugin.so in ${TENSORRT_LIB_DIR}")
endif()



# ==========================================
# 2. 查找其他依赖包
# ==========================================
# OpenCV
find_package(OpenCV REQUIRED)

# 【修复 2】使用现代 CMake 查找 CUDA 的方式
find_package(CUDAToolkit REQUIRED)

# ==========================================
# 3. 编译动态库
# ==========================================
set(SOURCES
    src/efficientNMSInference.cu
    src/NMSProcessor.cu
    src/preprocess.cu
    src/yolo_trt.cpp
)

add_library(yolo_infer SHARED ${SOURCES})


# 设置头文件包含路径
target_include_directories(yolo_infer 
    PUBLIC  
        # 【修改提示】如果你的 .h 文件和 .cpp 都在 src 目录下，这里应该写 ${CMAKE_CURRENT_SOURCE_DIR}/src
        # 如果你建了一个 include 目录放 .h，那就写 ${CMAKE_CURRENT_SOURCE_DIR}/include
        ${CMAKE_CURRENT_SOURCE_DIR}/include 
    PRIVATE 
        ${OpenCV_INCLUDE_DIRS}
        ${TENSORRT_INCLUDE_DIR}
)

# 设置链接库
target_link_libraries(yolo_infer 
    PRIVATE 
        ${OpenCV_LIBS}
        CUDA::cudart
        ${TENSORRT_LIBRARY}
        ${TENSORRT_PLUGIN_LIBRARY}
        pthread
)

set_target_properties(yolo_infer PROPERTIES
    LIBRARY_OUTPUT_DIRECTORY ${CMAKE_CURRENT_SOURCE_DIR}/lib
)

add_executable(yolo src/main.cpp)
# 我们的 CLI 工具仅仅链接了 yolo_infer 这一个动态库！
# 它不需要链接 OpenCV，也不需要链接 TensorRT 和 CUDA。
# 因为所有复杂的脏活累活，都被我们的 .so 屏蔽在底层了
target_link_libraries(yolo 
                    PRIVATE yolo_infer ${OpenCV_LIBS})

set_target_properties(yolo PROPERTIES 
    INSTALL_RPATH "${CMAKE_CURRENT_SOURCE_DIR}/lib"
    BUILD_WITH_INSTALL_RPATH TRUE
)