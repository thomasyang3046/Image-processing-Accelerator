#pragma once

typedef struct
{
	int thread_id;
	int start;
	int end;
	int gamma;
	cv::Mat* inImg1, *inImg2;
	cv::Mat* outImg;
	cv::Mat* kernel;
} Arg;

void GammaCorrection_pthread(cv::Mat img, float gamma);

cv::Mat Add_pthread(cv::Mat img1, cv::Mat img2);
cv::Mat Sub_pthread(cv::Mat img1, cv::Mat img2);
cv::Mat Convolution_pthread(cv::Mat img, cv::Mat kernel);