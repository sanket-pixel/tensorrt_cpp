#include "inference.hpp"

int main(){
    
    Params params;
    Inference Inference(params); 
    // Inference.build();
    Inference.buildFromSerializedEngine();
    Inference.get_bindings();


}