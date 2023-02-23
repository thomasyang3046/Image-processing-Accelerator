#pragma once

void GammaCorrection_serial(cv::Mat img, float gamma);

cv::Mat Add_serial(cv::Mat img1, cv::Mat img2);

cv::Mat Sub_serial(cv::Mat img1, cv::Mat img2);

cv::Mat Convolution_serial(cv::Mat img, cv::Mat kernel);


