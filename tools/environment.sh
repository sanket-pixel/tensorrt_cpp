# enter your own TensorRT paths here
export TensorRT_Lib=/home/sanket/libs/TensorRT-8.6.1.6/lib
export TensorRT_Inc=/home/sanket/libs/TensorRT-8.6.1.6/include
export TensorRT_Bin=/home/sanket/libs/TensorRT-8.6.1.6/bin

# enter your own CUDA paths here
export CUDA_Lib=/usr/local/cuda/lib64
export CUDA_Inc=/usr/local/cuda/include

export CUDA_Bin=/usr/local/cuda/bin
export CUDA_HOME=/usr/local/cuda

# build_engine or inference
export MODE=inference

# Write your conda env name here
export CONDA_ENV=tensorrt


# check the configuration path
# clean the configuration status
export ConfigurationStatus=Failed
if [ ! -f "${TensorRT_Bin}/trtexec" ]; then
    echo "Can not find ${TensorRT_Bin}/trtexec, there may be a mistake in the directory you configured."
    return
fi

if [ ! -f "${CUDA_Bin}/nvcc" ]; then
    echo "Can not find ${CUDA_Bin}/nvcc, there may be a mistake in the directory you configured."
    return
fi

echo "=========================================================="
echo "||  MODEL: $DEBUG_MODEL"
echo "||  TensorRT: $TensorRT_Lib"
echo "||  CUDA: $CUDA_HOME"
echo "=========================================================="


export PATH=$TensorRT_Bin:$CUDA_Bin:$PATH
export LD_LIBRARY_PATH=$TensorRT_Lib:$CUDNN_Lib:$BuildDirectory:$LD_LIBRARY_PATH
export ConfigurationStatus=Success


echo Configuration done!