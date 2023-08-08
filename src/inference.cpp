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


//!
//! \brief Uses a ONNX parser to create the Onnx Inference Network and marks the
//!        output layers
//!
//! \param network Pointer to the network that will be populated with the Onnx Inference network
//!
//! \param builder Pointer to the engine builder
//!

bool Inference::constructNetwork(SampleUniquePtr<nvinfer1::IBuilder>& builder,
    SampleUniquePtr<nvinfer1::INetworkDefinition>& network, SampleUniquePtr<nvinfer1::IBuilderConfig>& config,
    SampleUniquePtr<nvonnxparser::IParser>& parser)
{
    auto parsed = parser->parseFromFile(mParams.engineParams.OnnxFilePath.c_str(),
        static_cast<int>(sample::gLogger.getReportableSeverity()));
    if (!parsed)
    {
        sample::gLogError<< "Onnx model cannot be parsed ! " << std::endl;
        return false;
    }
    builder->setMaxBatchSize(BATCH_SIZE_);
    // config->setMaxWorkspaceSize(2_GiB); //8_GiB);

    if (mParams.engineParams.fp16)
    {
        config->setFlag(BuilderFlag::kFP16);
    }
    if (mParams.engineParams.int8)
    {
        config->setFlag(BuilderFlag::kINT8);
        samplesCommon::setAllDynamicRanges(network.get(), 127.0F, 127.0F);
    }
    if (mParams.engineParams.dlaCore >=0 ){
    samplesCommon::enableDLA(builder.get(), config.get(), mParams.engineParams.dlaCore);
    sample::gLogInfo << "Deep Learning Acclerator (DLA) was enabled . \n";
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
    auto builder = SampleUniquePtr<nvinfer1::IBuilder>(nvinfer1::createInferBuilder(sample::gLogger.getTRTLogger()));
    if (!builder)
    {
        return nullptr;
    }

    const auto explicitBatch = 1U << static_cast<uint32_t>(NetworkDefinitionCreationFlag::kEXPLICIT_BATCH);
    auto network = SampleUniquePtr<nvinfer1::INetworkDefinition>(builder->createNetworkV2(explicitBatch));
    if (!network)
    {
        return nullptr;
    }

    auto config = SampleUniquePtr<nvinfer1::IBuilderConfig>(builder->createBuilderConfig());
    if (!config)
    {
        return nullptr;
    }

    auto parser
        = SampleUniquePtr<nvonnxparser::IParser>(nvonnxparser::createParser(*network, sample::gLogger.getTRTLogger()));
    if (!parser)
    {
        return nullptr;
    }

    auto constructed = constructNetwork(builder, network, config, parser);
    if (!constructed)
    {
        return nullptr;
    }

    // CUDA stream used for profiling by the builder.
    auto profileStream = samplesCommon::makeCudaStream();
    if (!profileStream)
    {
        return nullptr;
    }
    config->setProfileStream(*profileStream);

    SampleUniquePtr<IHostMemory> plan{builder->buildSerializedNetwork(*network, *config)};
    if (!plan)
    {
        return nullptr;
    }

    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(createInferRuntime(sample::gLogger.getTRTLogger()));
    if (!mRuntime)
    {
        return nullptr;
    }

    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
        mRuntime->deserializeCudaEngine(plan->data(), plan->size()), samplesCommon::InferDeleter());
    if (!mEngine)
    {
        sample::gLogError << "Failed to create engine \n";
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
    mRuntime = std::shared_ptr<nvinfer1::IRuntime>(nvinfer1::createInferRuntime(sample::gLogger.getTRTLogger()));   
     // Deserialize the TensorRT engine
    mEngine = std::shared_ptr<nvinfer1::ICudaEngine>(
            mRuntime->deserializeCudaEngine(engineData.get(), engineSize));   

    std::cout << "Input Image " << mEngine->getBindingDimensions(0) << std::endl; 
    std::cout << "Output  " << mEngine->getBindingDimensions(1) << std::endl; 
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

void Inference::get_bindings(){
   
    // Create the execution context
    auto context = SampleUniquePtr<nvinfer1::IExecutionContext>(mEngine->createExecutionContext());
    
    //  input buffers
    auto dims = context->getBindingDimensions(0);
    nvinfer1::DataType type = mEngine->getBindingDataType(0);
    size_t vol = 1;
    for(int i=0; i < dims.nbDims;i++){
        vol*=dims.d[i];
    }
    size_t input_size_in_bytes = vol*getElementSize(type);
    float* device_input;
    cudaMalloc((void**)&device_input, input_size_in_bytes);
    float* host_input = (float*)malloc(input_size_in_bytes);

    //  output buffers
    dims = context->getBindingDimensions(1);
    type = mEngine->getBindingDataType(1);
    vol = 1;
    for(int i=0; i < dims.nbDims;i++){
        vol*=dims.d[i];
    }
    size_t output_size_in_bytes = vol*getElementSize(type);
    float* device_output;
    cudaMalloc((void**)&device_output, output_size_in_bytes);
    float* host_output = (float*)malloc(output_size_in_bytes);

    // make array of pointers
    void* bindings[2] = {device_input, device_output};

    cv::Mat img = read_image(mParams.ioPathsParams.image_path);
    cv::Mat preprocessed_image;
    Inference::preprocess(img, preprocessed_image);

    // Populate host buffer with input image.
    enqueue_input(host_input, preprocessed_image);
    const cudaStream_t& stream = 0;
    cudaMemcpyAsync(device_input, host_input, input_size_in_bytes, cudaMemcpyHostToDevice, stream);

     // Perform inference
    bool status_0 = context->executeV2(bindings); 
    cudaMemcpyAsync(host_output, device_output, output_size_in_bytes, cudaMemcpyDeviceToHost, stream);
    std::cout << host_input[0] << std::endl;

     // convert boxes to vector
    float* class_flattened = static_cast<float*>(host_output);
    std::cout << host_output[934] << std::endl;
    int num_predictions = 1000;
    std::vector<float> predictions(class_flattened, class_flattened + num_predictions);
    mPostprocess.softmax_classify(predictions);

    }