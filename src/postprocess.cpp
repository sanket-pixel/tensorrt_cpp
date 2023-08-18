#include "postprocessor.hpp"
#include <iostream>
#include <fstream>
#include <algorithm>
#include <cmath>

Postprocessor::Postprocessor(const std::string& class_file_path) : _class_file_path(class_file_path) {
}

void Postprocessor::softmax_classify(const std::vector<float>& outputs, bool verbose) {
    // Read class names from the provided class file
    std::ifstream class_file(_class_file_path);
    std::vector<std::string> classes;
    std::string line;
    while (std::getline(class_file, line)) {
        classes.push_back(line);
    }
    
    // Assuming you have outputs and confidences defined
    std::vector<float> confidences(outputs.size()); // Make sure confidences has the same size as outputs

    // Calculate the sum of exp(outputs)
    float exp_sum = 0.0f;
    for (float output : outputs) {
        exp_sum += std::exp(output);
    }

    // Transform and normalize the confidences
    for (size_t i = 0; i < outputs.size(); ++i) {
        confidences[i] = (std::exp(outputs[i]) / exp_sum) * 100;
    };

    // Find top predicted classes
    std::vector<int> indices(outputs.size());
    for (int i = 0; i < indices.size(); ++i) {
        indices[i] = i;
    }
    std::sort(indices.begin(), indices.end(), [&confidences](int i1, int i2) {
        return confidences[i1] > confidences[i2];
    });

    // Print the top classes predicted by the model
    if(verbose){
        int i = 0;
        while (confidences[indices[i]] > 50) {
        int class_idx = indices[i];
        std::cout << "class: " << classes[class_idx] << ", confidence: " << confidences[class_idx] << "%, index: " << class_idx << std::endl;
        ++i;
    }

    }
    
}