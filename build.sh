#!/bin/bash

# 遇到任何错误立即退出脚本
set -e

# 如果你想在报错时有个明显的提示，可以加上这一行（可选）
trap 'echo "❌ 构建过程中出现错误，已中止！"' ERR

echo "🚀 开始构建项目..."

# 使用 -p 参数：如果 build 目录已经存在就不会报错，不存在则创建
mkdir -p build
cd build

echo "⚙️  正在运行 cmake .. "
cmake ..

echo "🔨 正在编译 (make -j8) ... "
make -j8

cd ..

echo "✅ 所有步骤执行完毕，构建成功！"