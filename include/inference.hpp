#pragma once
/*!
 @file Inference.hpp
 @author Sanket Rajendra Shah (sanket.shah@motor-ai.com)
 @brief 
 @version 0.1
 @date 2023-05-11
 
 @copyright Copyright (c) 2023
 
 */
#include "argsParser.h"
#include "buffers.h"
#include "common.h"
#include "logger.h"
#include "parserOnnxConfig.h"
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

using namespace nvinfer1;

struct Params{
    struct EngineParams{
    std::string OnnxFilePath = "../deploy_tools/resnet.onnx";
    std::string SerializedEnginePath = "../deploy_tools/resnet.engine";

    // Input Output Names
    std::vector<std::string> InputTensorNames;
    std::vector<std::string> OutputTensorNames;

    // Input Output Paths
    std::string savePath ;
    std::vector<std::string>  filePaths;
    
    // Attrs
    int dlaCore = -1;
    bool fp16 = false;
    bool int8 = false;
    bool load_engine = true;
    int batch_size = 1;

    } engineParams;

    struct IOPathsParams{
        std::string image_path = "../data/hotdog.jpg";    
    } ioPathsParams;

    struct ModelParams{
    int resized_image_size_height = 224;
    int resized_image_size_width = 336;
    int num_classes = 1000;
    } modelParams;
    
};

using samplesCommon::SampleUniquePtr;


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
        bool engineInitlization();
        void get_bindings();

        // std::vector<std::vector<float>> get_bindings();

    private:
        Params mParams;             //!< The parameters for the sample.
        int BATCH_SIZE_ = 1;        // batch size
        nvinfer1::Dims mInputDims;  //!< The dimensions of the input to the network.
        nvinfer1::Dims mOutputDims; //!< The dimensions of the output to the network.

        std::shared_ptr<nvinfer1::IRuntime> mRuntime; //!< The TensorRT runtime used to deserialize the engine
        std::shared_ptr<nvinfer1::ICudaEngine> mEngine; //!< The TensorRT engine used to run the network

        // Parses an ONNX model for Inference and creates a TensorRT network
        bool constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
            SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
            SampleUniquePtr<nvonnxparser::IParser>& parser);
        Preprocessor mPreprocess{mParams.modelParams.resized_image_size_width, mParams.modelParams.resized_image_size_height};        
        // Inference related functions
        cv::Mat read_image(std::string image_path); 
        bool preprocess(cv::Mat img, cv::Mat &preprocessed_img );
        bool enqueue_input(float* host_buffer, cv::Mat preprocessed_img);
      
};
