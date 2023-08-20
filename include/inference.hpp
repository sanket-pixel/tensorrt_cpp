#pragma once
/*!
 @file Inference.hpp
 @author Sanket Rajendra Shah (sanket.shah@motor-ai.com)
 @brief 
 @version 0.1
 @date 2023-05-11
 
 @copyright Copyright (c) 2023
 
 */
// #include "parserOnnxConfig.h"
#include "NvOnnxParser.h"
#include <opencv2/opencv.hpp>
#include "NvInfer.h"
#include <cuda_runtime_api.h>
#include <vector>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <chrono>
#include <NvInferRuntime.h>
#include "preprocessor.hpp"
#include "postprocessor.hpp"
using namespace nvinfer1;

struct Params{
    struct EngineParams{
    std::string OnnxFilePath = "../deploy_tools/resnet.onnx";
    std::string SerializedEnginePath = "../deploy_tools/resnet.engine";

    // Input Output Names
    std::string inputTensorName = "input";
    std::string outputTensorName = "output";

    // Input Output Paths
    std::string savePath ;
    std::vector<std::string>  filePaths;
    
    // Attrs
    int dlaCore = -1;
    bool fp16 = true;
    bool int8 = false;
    bool load_engine = true;
    int batch_size = 1;

    } engineParams;

    struct IOPathsParams{
        std::string image_path = "../data/hotdog.jpg";    
        std::string classes_path = "../data/imagenet-classes.txt";
    } ioPathsParams;

    struct ModelParams{
    int resized_image_size_height = 224;
    int resized_image_size_width = 336;
    int num_classes = 1000;
    } modelParams;
    
};


class InferLogger : public nvinfer1::ILogger {
public:
    void log(nvinfer1::ILogger::Severity severity, const char* msg) noexcept override {
         if (severity == nvinfer1::ILogger::Severity::kERROR) {
            // Print only error messages
            std::cout << "TRT_Logs : " << msg << std::endl;
        }
    }
};


//! \brief  The Inference class implements the MFFD model
//!
//! \details It creates the network using an ONNX model
//!

class Inference{
    public:
        Inference(const Params params)
        : mParams(params)
        ,BATCH_SIZE_(params.engineParams.batch_size)
        ,mRuntime(nullptr)
        ,mEngine(nullptr)
        {
        } 
        // Engine Building Functions
        std::shared_ptr<nvinfer1::ICudaEngine> build();
        bool buildFromSerializedEngine();
        bool initialize_inference();
        void do_inference();
        float *host_input, *device_input;
        float *host_output, *device_output;
        float latency;
        bool verbose;

    private:
        Params mParams;             //!< The parameters for the sample.
        int BATCH_SIZE_ = 1;        // batch size
        
        size_t input_size_in_bytes, output_size_in_bytes;
        void *bindings[2] ;
        const cudaStream_t& stream = 0;

        std::shared_ptr<nvinfer1::IRuntime> mRuntime; //!< The TensorRT runtime used to deserialize the engine
        std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network
        std::unique_ptr<nvinfer1::IExecutionContext> mContext;

        InferLogger mLogger; 

        // Parses an ONNX model for Inference and creates a TensorRT network
        bool constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
            std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config,
            std::unique_ptr<nvonnxparser::IParser>& parser);
        Preprocessor mPreprocess{mParams.modelParams.resized_image_size_width, mParams.modelParams.resized_image_size_height};    
        Postprocessor mPostprocess{mParams.ioPathsParams.classes_path};     

        // Inference related functions
        cv::Mat read_image(std::string image_path); 
        bool preprocess(cv::Mat img, cv::Mat &preprocessed_img );
        bool enqueue_input(float* host_buffer, cv::Mat preprocessed_img);
      
};
