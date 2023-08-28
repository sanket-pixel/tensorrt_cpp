## TensorRT meets C++
This repository is the source code for the blog [TensorRT meets C++](https://sanket-pixel.github.io/blog/2023/tensorrt-meets-cpp/)

### Project Setup and Execution

To set up and run the project on your machine, follow these steps:

1. Open the `tools/environment.sh` script and adjust the paths for `TensorRT` and `CUDA` libraries as per your system configuration:
    ```bash
    export TensorRT_Lib=/path/to/TensorRT/lib
    export TensorRT_Inc=/path/to/TensorRT/include
    export TensorRT_Bin=/path/to/TensorRT/bin

    export CUDA_Lib=/path/to/CUDA/lib64
    export CUDA_Inc=/path/to/CUDA/include
    export CUDA_Bin=/path/to/CUDA/bin
    export CUDA_HOME=/path/to/CUDA

    export MODE=inference

    export CONDA_ENV=tensorrt
    ```
    Set the `MODE` to `build_engine` for building the TensorRT engine or make it `inference` for running inference on the sample image with the engine.

2. Run the `tools/run.sh` script to execute the PyTorch inference and save its output:

    ```bash
    bash tools/run.sh
    ```

    Upon executing the above steps, you'll observe an informative output similar to the one below, detailing both PyTorch and TensorRT C++ inference results:

    ```plaintext
    ===============================================================
    ||  MODE: inference
    ||  TensorRT: /path/to/TensorRT/lib
    ||  CUDA: /path/to/CUDA
    ===============================================================
    Configuration done!
    =================== STARTING PYTORCH INFERENCE===============================
    class: hotdog, hot dog, red hot, confidence: 60.50566864013672 %, index: 934
    Saved Pytorch output in torch_stuff/output.txt
    Average Latency for 10 iterations: 5.42 ms
    =============================================================================
    -- Configuring done
    -- Generating done
    -- Build files have been written to: /path/to/project/build
    =================== STARTING C++ TensorRT INFERENCE==========================
    class: hotdog, hot dog, red hot, confidence: 59.934%, index: 934
    Mean Absolute Difference in Pytorch and TensorRT C++: 0.0121075
    Average Latency for 10 iterations: 2.19824 ms
    =====================================SUMMARY=================================
    Pytorch Latency: 5.42 ms
    TensorRT in C++ Latency: 2.19824 ms
    Speedup by Quantization: 2.46561x
    Mean Absolute Difference in Pytorch and TensorRT C++: 0.0121075
    =============================================================================
    ```

    With this guide, you can effortlessly set up and run the project on your local machine, leveraging the power of TensorRT in C++ inference and comparing it with PyTorch's results.

