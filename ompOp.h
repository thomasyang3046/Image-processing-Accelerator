#pragma once

cv::Mat GammaCorrection_omp(cv::Mat img, float gamma);

cv::Mat Add_omp(cv::Mat img1, cv::Mat img2);
cv::Mat Sub_omp(cv::Mat img1, cv::Mat img2);
cv::Mat Convolution_omp(cv::Mat img, cv::Mat kernel);
