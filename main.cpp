#include "inference.hpp"
#include <iostream>
#include <cstring>
#include <fstream>
#include <vector>
#include <cmath>
#include <sstream>
#include <chrono>

void printHelp() {
    std::cout << "Usage: ./main [--build_engine] [--inference] [--help]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --build_engine   Build the TensorRT engine" << std::endl;
    std::cout << "  --inference      Perform inference using a pre-built engine" << std::endl;
    std::cout << "  --help           Display this help message" << std::endl;
}

std::vector<float> readPythonOutput(const std::string& filePath) {
    std::ifstream python_output_file(filePath);
    std::vector<float> python_output;
    std::string line;
    
    while (std::getline(python_output_file, line)) {
        std::istringstream iss(line);
        float value;
        
        while (iss >> value) {
            python_output.push_back(value);
            
            if (iss.peek() == ',') {
                iss.ignore();
            }
        }
    }
    
    return python_output;
}

float calculateMeanAbsoluteDifference(const float* host_output, const std::vector<float>& python_output) {
    float mean_absolute_difference = 0.0f;

    for (size_t i = 0; i < python_output.size(); ++i) {
        float diff = std::abs(host_output[i] - python_output[i]);
        mean_absolute_difference += diff;
    }

    mean_absolute_difference /= python_output.size();
    return mean_absolute_difference;
}


int main(int argc, char* argv[]) {
    if (argc == 1 || std::strcmp(argv[1], "--help") == 0) {
        printHelp();
        return 0;
    }

    Params params;
    Inference Inference(params); 

    // Check command line arguments
    for (int i = 1; i < argc; ++i) {
        if (std::strcmp(argv[i], "--build_engine") == 0) {
            Inference.build();
        } else if (std::strcmp(argv[i], "--inference") == 0) {
            std::cout<< "=================== STARTING C++ TensorRT INFERENCE===============================" << std::endl;
            Inference.buildFromSerializedEngine();
            Inference.initialize_inference();
            // compute difference between python and C++
            Inference.verbose = true;
            Inference.do_inference();
            float *host_output = Inference.host_output;
            std::vector<float> python_output = readPythonOutput("../torch_stuff/output.txt");
            float mean_absolute_difference = calculateMeanAbsoluteDifference(host_output, python_output);
            std::cout << "Mean Absolute Difference in Pytorch and TensorRT C++ : " << mean_absolute_difference << std::endl;
            // measure speedup
            int num_iterations = 10;
            float total_latency = 0.0f;
            Inference.verbose = false;
            for (int i = 0; i < 10; ++i) {
                Inference.do_inference();
                total_latency += Inference.latency;
            }

            float tensorrt_latency = total_latency / num_iterations;
            std::cout << "Average Latency for " << num_iterations << " iterations: " << tensorrt_latency << " ms" << std::endl;
            std::ifstream pytorch_latency_file("../torch_stuff/latency.txt");
            float pytorch_latency;
            pytorch_latency_file >> pytorch_latency;
            std::cout<< "=====================================SUMMARY=================================" << std::endl;
            float speedup = pytorch_latency / tensorrt_latency;   
            std::cout << "Pytorch Latency: " << pytorch_latency << " ms" << std::endl;
            std::cout << "TensorRT in C++  Latency: " << tensorrt_latency << " ms" << std::endl;
            std::cout << "Speedup by Quantization: " << speedup << "x" << std::endl;
            std::cout << "Mean Absolute Difference in Pytorch and TensorRT C++ : " << mean_absolute_difference << std::endl;
            std::cout<< "============================================================================" << std::endl;
        }
    }

    return 0;
}