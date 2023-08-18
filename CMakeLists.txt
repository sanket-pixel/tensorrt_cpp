cmake_minimum_required(VERSION 3.10.2)
project(Inference)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_INSTALL_RPATH "$ORIGIN")

# include headers
include_directories(
    ${PROJECT_SOURCE_DIR}/include
)

#opencv
find_package(OpenCV REQUIRED)
include_directories(${OpenCV_INCLUDE_DIRS})

#cuda:
find_package(CUDA 11.1 REQUIRED)
include_directories(${CUDA_INCLUDE_DIRS})
link_directories(${CUDA_LIBRARY_DIRS})

# enable cuda support
set(CMAKE_CUDA_COMPILER /usr/local/cuda/bin/nvcc)
set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -arch=sm_50 --expt-extended-lambda -D_MWAITXINTRIN_H_INCLUDED")

# tensorrt
set(TensorRT_DIR "/home/sanket/libs/TensorRT-8.6.1.6")
link_directories( ${TensorRT_DIR}/lib)
include_directories( ${TensorRT_DIR}/include)

# add preprocessor library
add_library(pre_processor  src/preprocessor.cpp)
set_target_properties(pre_processor PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(pre_processor PUBLIC include)


# add postprocessor library
add_library(post_processor  src/postprocess.cpp)
set_target_properties(post_processor PROPERTIES POSITION_INDEPENDENT_CODE ON)
target_include_directories(post_processor PUBLIC include)
 
# add common 
# file(GLOB common_SRC "${TensorRT_DIR}/samples/common/*.cpp")
# add_library(common ${common_SRC})
# set_target_properties(common PROPERTIES POSITION_INDEPENDENT_CODE ON)
# target_include_directories(common PUBLIC  ${TensorRT_DIR}/include ${TensorRT_DIR}/samples/common)

# make Inference library
add_library(Inference SHARED src/inference.cpp)
target_include_directories(Inference PUBLIC ${TensorRT_DIR}/include)
target_link_libraries(Inference PUBLIC pre_processor post_processor  nvinfer nvonnxparser stdc++fs  ${CUDA_LIBRARIES} ${OpenCV_LIBS} common)
set_target_properties(Inference PROPERTIES INSTALL_RPATH "$ORIGIN")

add_executable(main main.cpp)
target_link_libraries(main PUBLIC Inference)