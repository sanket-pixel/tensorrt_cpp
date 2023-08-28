. tools/environment.sh

conda activate $CONDA_ENV
python tools/torch_inference.py

mkdir -p build

cd build
cmake ..
make > /dev/null
./main --$MODE