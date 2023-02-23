#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include "serialOp.h"
#include "pthreadOp.h"
#include "ompOp.h"
#include "simdOp.h"
#include "cudaOp.h"
#include <iostream>
#include <time.h>

using namespace cv;


int main() {
	Mat img = imread("download.png", IMREAD_GRAYSCALE);
	int pixels = 20000000;
	int imgSize = sqrt(pixels);
	resize(img, img, Size(imgSize, imgSize));
	namedWindow("SRC", WINDOW_AUTOSIZE);
	imshow("SRC", (img));

	srand(time(0));

	//跑3x3, 5x5, 7x7, 9x9 kernel的convolution
	for (int c = 1; c < 5; c++)
	{
		//計算Kernel size並產生kernel
		int size = 1+c*2;
		Mat kernel = Mat_<float>(size, size);
		float sum = 0;

		//亂數產生kernel
		for (int i = 0; i < size * size; i++)
		{
			float tmp = rand() % 256;
			kernel.at<float>(i) = tmp;
			sum += tmp;
		}
		for (int i = 0; i < size * size; i++)
		{
			kernel.at<float>(i) /= sum;
		}

		//printf Convolution kernel
		printf("Convolution %d: %dx%d kernel\n", c, size, size);
		for (int i = 0; i < size; i++)
		{
			for (int j = 0; j < size; j++)
			{
				printf("%.2f ", kernel.at<float>(i*size+j));
			}
			printf("\n");
		}
		printf("\n");


		int it = 1;
		clock_t start, stop;
		start = clock();

		for (size_t i = 0; i < it; i++)
		{
			cv::filter2D(img, img, -1, kernel);
			//img = Convolution_serial(img, kernel);
			//img = Convolution_simd(img, kernel);
			//img = Convolution_pthread(img, kernel);
			//img = Convolution_omp(img, kernel);
			//img = Convolution_Cuda(img, kernel);
		}
		stop = clock();
		printf("%f ms\n", (float)(stop - start) * 1000 / (CLOCKS_PER_SEC) / it);

	}

	namedWindow("DST", WINDOW_AUTOSIZE);
	imshow("DST", (img));
	waitKey(0);

	return 0;
}