. tool/environment.sh

python tools/torch_inference

mkdir -p build

cd build
cmake ..
make 
./main --$MODE