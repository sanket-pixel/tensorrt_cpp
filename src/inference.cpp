/*!
 @file Inference.cpp
 @author Sanket Rajendra Shah (sanketshah812@gmail.com)
 @brief 
 @version 0.1
 @date 2023-05-11
 
 @copyright Copyright (c) 2023
 
 */
#include "inference.hpp"
#include <memory>
#include <chrono>

//!
//! \brief Uses a ONNX parser to create the Onnx Inference Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx Inference network
//!
//! \param builder Pointer to the engine builder
//!

bool Inference::constructNetwork(std::unique_ptr<nvinfer1::IBuilder>& builder,
    std::unique_ptr<nvinfer1::INetworkDefinition>& network, std::unique_ptr<nvinfer1::IBuilderConfig>& config,
    std::unique_ptr<nvonnxparser::IParser>& parser)
{
    
    auto parsed = parser->parseFromFile(mParams.engineParams.OnnxFilePath.c_str(),
        1);
    if (!parsed)
    {
        mLogger.log(nvinfer1::ILogger::Severity::kERROR,  "Onnx model cannot be parsed ! ");
        return false;
    }
    builder->setMaxBatchSize(BATCH_SIZE_);
    if (mParams.engineParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    return true;
}



//!
//! \brief Creates the network, configures the builder and creates the network engine
//!
//! \details This function creates the Onnx Inference network by parsing the Onnx model and builds
//!          the engine that will be used to run Inference (mEngine)
//!
//! \return true if the engine was created successfully and false otherwise
//!
std::shared_ptr<nvinfer1::ICudaEngine> Inference::build()
{
    auto builder = std::unique_ptr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(mLogger));
    if (!builder)
    {
        return nullptr;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return nullptr;
    }

    auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return nullptr;
    }

    auto parser
        = std::unique_ptr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, mLogger));
    if (!parser)
    {
        return nullptr;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return nullptr;
    }else{
         mLogger.log(nvinfer1::ILogger::Severity::kERROR,  "Network made ! ");
    }
    std::unique_ptr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        mLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to build Network.");
        return nullptr;
    }
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(mLogger));
    if (!mRuntime)
    {
        return nullptr;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(mRuntime->deserializeCudaEngine(plan->data(), plan->size()));
    if (!mEngine)
    {
        mLogger.log(nvinfer1::ILogger::Severity::kERROR, "Failed to build Engine.");
        return nullptr;
    }
    
    std::ofstream engineFile(mParams.engineParams.SerializedEnginePath, std::ios::binary);
    engineFile.write(static_cast<const char*>(plan->data()), plan->size());
    engineFile.close();

    return mEngine;
}

bool Inference::buildFromSerializedEngine(){

    // Load serialized engine from file
    std::ifstream engineFileStream(mParams.engineParams.SerializedEnginePath, std::ios::binary);
    engineFileStream.seekg(0, engineFileStream.end);
    const size_t engineSize = engineFileStream.tellg();
    engineFileStream.seekg(0, engineFileStream.beg);
    std::unique_ptr<char[]> engineData(new char[engineSize]);
    engineFileStream.read(engineData.get(), engineSize);
    engineFileStream.close();
    // Create the TensorRT runtime
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(mLogger));   
     // Deserialize the TensorRT engine
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            mRuntime->deserializeCudaEngine(engineData.get(), engineSize));   
          // Create the execution mContext
    mContext = std::unique_ptr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    return true;
}


cv::Mat Inference::read_image(std::string image_path){
    return cv::imread(image_path,cv::IMREAD_COLOR);
}

bool Inference::preprocess(cv::Mat img, cv::Mat &preprocessed_img ){
    mPreprocess.resize(img, preprocessed_img);
    mPreprocess.normalization(preprocessed_img, preprocessed_img);
}

bool Inference::enqueue_input(float* host_buffer, cv::Mat image){
    nvinfer1::Dims input_dims = mEngine->getBindingDimensions(0);
    for (size_t batch = 0; batch < 1; ++batch) {
  
        int offset = input_dims.d[1] * input_dims.d[2] * input_dims.d[3] * batch;
        int r = 0 , g = 0, b = 0;
        
        for (int i = 0; i < input_dims.d[1] * input_dims.d[2] * input_dims.d[3]; ++i) {
            if (i % 3 == 0) {
                host_buffer[offset + r++] = *(reinterpret_cast<float*>(image.data) + i);
            } else if (i % 3 == 1) {
                host_buffer[offset + g++ + input_dims.d[2] * input_dims.d[3]] = *(reinterpret_cast<float*>(image.data) + i);
            } else {
                host_buffer[offset + b++ + input_dims.d[2] * input_dims.d[3] * 2] = *(reinterpret_cast<float*>(image.data) + i);
            }
        }
    }    
}


inline uint32_t getElementSize(nvinfer1::DataType t) noexcept
{
    switch (t)
    {
    case nvinfer1::DataType::kINT32: return 4;
    case nvinfer1::DataType::kFLOAT: return 4;
    case nvinfer1::DataType::kHALF: return 2;
    case nvinfer1::DataType::kBOOL:
    case nvinfer1::DataType::kUINT8:
    case nvinfer1::DataType::kINT8:
    case nvinfer1::DataType::kFP8: return 1;
    }
    return 0;
}

bool Inference::initialize_inference(){      
        //  input buffers
        int input_idx = mEngine->getBindingIndex("input");
        auto input_dims = mContext->getBindingDimensions(input_idx);
        nvinfer1::DataType input_type = mEngine->getBindingDataType(input_idx);
        size_t input_vol = 1;
        for(int i=0; i < input_dims.nbDims;i++){
            input_vol*=input_dims.d[i];
        }
        input_size_in_bytes = input_vol*getElementSize(input_type);
        cudaMalloc((void**)&device_input, input_size_in_bytes);
        host_input = (float*)malloc(input_size_in_bytes);
        bindings[input_idx] = device_input;

        //  output buffers
        int output_idx = mEngine->getBindingIndex("output");
        auto output_dims = mContext->getBindingDimensions(output_idx);
        nvinfer1::DataType output_type = mEngine->getBindingDataType(output_idx);
        size_t output_vol = 1;
        for(int i=0; i < output_dims.nbDims;i++){
            output_vol*=output_dims.d[i];
        }
        output_size_in_bytes = output_vol*getElementSize(output_type);
        cudaMalloc((void**)&device_output, output_size_in_bytes);
        host_output = (float*)malloc(output_size_in_bytes);
        bindings[output_idx] = device_output;

                 
}


void Inference::do_inference(){

    cv::Mat img = read_image(mParams.ioPathsParams.image_path);
    cv::Mat preprocessed_image;
    Inference::preprocess(img, preprocessed_image);
    // populate host buffer with input image.
    auto start_time = std::chrono::high_resolution_clock::now();
    enqueue_input(host_input, preprocessed_image);
    // copy input from host to device
    cudaMemcpy(device_input, host_input, input_size_in_bytes, cudaMemcpyHostToDevice);
       // perform inference
    bool status_0 = mContext->executeV2(bindings); 
    // copy input from device to host
    cudaMemcpy(host_output, device_output, output_size_in_bytes, cudaMemcpyDeviceToHost);
    auto end_time = std::chrono::high_resolution_clock::now();
    std::chrono::duration<float, std::milli> duration = end_time - start_time;
    latency = duration.count();
    // apply softmax to output and get prediction
    float* class_flattened = static_cast<float*>(host_output);
    std::vector<float> predictions(class_flattened, class_flattened + mParams.modelParams.num_classes);
    mPostprocess.softmax_classify(predictions, verbose);

}