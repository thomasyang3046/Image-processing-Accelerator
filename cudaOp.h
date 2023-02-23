#pragma once
cv::Mat Add_Cuda(cv::Mat* inImg1, cv::Mat* inImg2);

cv::Mat Sub_Cuda(cv::Mat* inImg1, cv::Mat* inImg2);

cv::Mat GammaCorrection_Cuda(cv::Mat inImg, float gamma);

cv::Mat GammaCorrection_Cuda(cv::Mat inImg, float gamma);

cv::Mat Convolution_Cuda(cv::Mat inImg, cv::Mat inKernel);
