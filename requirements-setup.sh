#!/bin/bash

if [[ $(lshw -C display | grep vendor) =~ Nvidia ]]; then
    CMAKE_ARGS="-DLLAMA_CUBLAS=on" FORCE_CMAKE=1 pip install llama-cpp-python --no-cache-dir
    git clone https://github.com/ggerganov/llama.cpp.git
    cd llama.cpp
    export CUDA_HOME=/usr/local/cuda
    export LLAMA_CUBLAS=on
    source ~/.bashrc
    make libllama.so
    sudo cp libllama.so /usr/local/lib
    cd ..
    sudo rm -rf llama.cpp
else
    pip install llama-cpp-python --no-cache-dir
fi

pip install -r requirements.txt