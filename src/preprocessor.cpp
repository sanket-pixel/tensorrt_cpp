#include "preprocessor.hpp"


void Preprocessor::resize(cv::Mat input_image, cv::Mat &output_image){
        cv::resize(input_image, output_image, cv::Size(_resized_width, _resized_height), 0, 0, cv::INTER_LINEAR);
}


void Preprocessor::normalization(cv::Mat input_image, cv::Mat &output_image){

        // Convert to float image and scale to [0, 1] range
        cv::Mat float_image;
        input_image.convertTo(float_image, CV_32FC3, 1.0 / 255.0);

        // Define the mean and standard deviation values
        cv::Scalar mean(0.485, 0.456, 0.406);
        cv::Scalar stdDev(0.229, 0.224, 0.225);

        // Subtract the mean from each channel
        cv::Mat subtracted_image;
        cv::subtract(float_image, mean, subtracted_image);

        // Divide the subtracted image by the standard deviation
        cv::Mat normalized_image;
        cv::divide(subtracted_image, stdDev, normalized_image);

        output_image = normalized_image;

}



