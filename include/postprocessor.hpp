#pragma once

#include <string>
#include <vector>

class Postprocessor {
private:
    std::string _class_file_path;

public:
    Postprocessor(const std::string& class_file_path);

    void softmax_classify(const std::vector<float>& outputs, bool verbose);
};