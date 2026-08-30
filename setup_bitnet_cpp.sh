#!/usr/bin/env bash
# Automated build & setup script for bitnet.cpp on Apple Silicon / CPU
set -e

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$SCRIPT_DIR"

echo "=================================================="
echo "⚡ Building bitnet.cpp C++ inference engine..."
echo "=================================================="

if [ ! -d "bitnet_cpp" ]; then
    echo "Cloning Microsoft BitNet repository..."
    git clone --recursive https://github.com/microsoft/BitNet.git bitnet_cpp
fi

cd bitnet_cpp

if [ ! -f "3rdparty/llama.cpp/CMakeLists.txt" ]; then
    echo "Initializing submodules..."
    git submodule update --init --recursive || git clone --depth 1 https://github.com/isHuangXin/llama.cpp.git 3rdparty/llama.cpp
fi

echo "Configuring with CMake..."
cmake -B build \
    -DBITNET_ARM_TL1=OFF \
    -DCMAKE_C_COMPILER=clang \
    -DCMAKE_CXX_COMPILER=clang++ \
    -DLLAMA_BUILD_TOOLS=ON \
    -DLLAMA_BUILD_EXAMPLES=ON \
    -DLLAMA_BUILD_COMMON=ON

echo "Compiling binaries (llama-cli, llama-quantize)..."
cmake --build build --target llama-cli llama-quantize --config Release -j 4

echo ""
echo "🎉 bitnet.cpp built successfully! Run with: python3 chat_cpp.py"
