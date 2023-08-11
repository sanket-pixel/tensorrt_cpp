#include "inference.hpp"
#include <iostream>
#include <cstring>

void printHelp() {
    std::cout << "Usage: ./your_program_name [--build_engine] [--inference] [--help]" << std::endl;
    std::cout << "Options:" << std::endl;
    std::cout << "  --build_engine   Build the TensorRT engine" << std::endl;
    std::cout << "  --inference      Perform inference using a pre-built engine" << std::endl;
    std::cout << "  --help           Display this help message" << std::endl;
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
            Inference.buildFromSerializedEngine();
            Inference.initialize_inference();
            Inference.do_inference();
        }
    }

    return 0;
}