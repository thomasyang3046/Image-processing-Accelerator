#include "cuda_runtime.h"
#include "device_launch_parameters.h"
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>

#include "cudaOp.h"
#include <stdio.h>
#include <string.h>

using namespace cv;

#define MaxBlockNum 1024
#define ThreadNum 1024

__global__ void Partial_Add_Cuda(uchar* outImg, uchar* inImg1, uchar* inImg2, int size)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int step = gridDim.x * blockDim.x;
    for (int i = id; i < size; i += step) {
        short tmp = (short)inImg1[i] + (short)inImg2[i];
        if (tmp > 255)
        {
            outImg[i] = 255;
        }
        else
        {
            outImg[i] = inImg1[i] + inImg2[i];
        }
    }

}

__global__ void Partial_Sub_Cuda(uchar* outImg, uchar* inImg1, uchar* inImg2, int size)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int step = gridDim.x * blockDim.x;
    for (int i = id; i < size; i += step) {
        if (inImg1[i] > inImg2[i])
        {
            outImg[i] = inImg1[i] - inImg2[i];
        }
        else
        {
            outImg[i] = inImg2[i] - inImg1[i];
        }
    }

}

__global__ void Partial_GammaCorrection_Cuda(uchar* outImg, uchar* inImg, float gamma, int size)
{
    int id = threadIdx.x + blockIdx.x * blockDim.x;
    int step = gridDim.x * blockDim.x;
    for (int i = id; i < size; i += step) {
        float pixel = pow(((float)inImg[i]) / 255.0, gamma) * 255.0f;
        if (pixel > 255)
        {
            outImg[i] = 255;
        }
        else
        {
            outImg[i] = (uchar)(pixel);
        }
    }

}

__global__ void Partial_Convolution_Cuda(uchar* outImg, uchar* inImg, float* inKernel, int colBias, int imgRows, int imgCols, int imgChannels, int kernelRows, int kernelCols)
{
    int step = imgRows / blockDim.x;
    int id = threadIdx.x + blockIdx.x * blockDim.x;

    for (int c = 0; c < imgCols; c += blockDim.x)
    {
        int i = blockIdx.x, j = threadIdx.x + c;
        //開始做捲積
        float sum = 0;
        for (int curr_row = 0; curr_row < kernelRows; curr_row++)
            for (int curr_col = 0; curr_col < kernelCols; curr_col++)
            {
                sum += (inKernel[curr_row * kernelCols + curr_col] * inImg[(curr_row + i) * (imgCols + colBias) + (curr_col + j)]);
            }
        outImg[i * imgCols + j] = (uchar)sum;
    }

}

Mat Add_Cuda(Mat* inImg1, Mat* inImg2)
{
    Mat outImg = inImg1->clone();

    int size = inImg1->rows * inImg1->cols * inImg1->channels();
    uchar* host_inImg1 = inImg1->ptr<uchar>(0);
    uchar* host_inImg2 = inImg2->ptr<uchar>(0);
    uchar* host_outImg = outImg.ptr<uchar>(0);
    uchar* device_inImg1, * device_inImg2;
    uchar* device_outImg;

    cudaMalloc((void**)&device_inImg1, sizeof(uchar) * size);
    cudaMalloc((void**)&device_inImg2, sizeof(uchar) * size);
    cudaMalloc((void**)&device_outImg, sizeof(uchar) * size);

    cudaMemcpy(device_inImg1, host_inImg1, sizeof(uchar) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_inImg2, host_inImg2, sizeof(uchar) * size, cudaMemcpyHostToDevice);

    int BlockNum = min(size / ThreadNum, MaxBlockNum);
    Partial_Add_Cuda << <BlockNum, ThreadNum >> > (device_outImg, device_inImg1, device_inImg2, size);

    cudaMemcpy(host_outImg, device_outImg, sizeof(uchar) * size, cudaMemcpyDeviceToHost);

    cudaFree(device_inImg1);
    cudaFree(device_inImg2);
    cudaFree(host_outImg);

    return outImg;
}

Mat Sub_Cuda(Mat* inImg1, Mat* inImg2)
{
    Mat outImg = inImg1->clone();

    int size = inImg1->rows * inImg1->cols * inImg1->channels();
    uchar* host_inImg1 = inImg1->ptr<uchar>(0);
    uchar* host_inImg2 = inImg2->ptr<uchar>(0);
    uchar* host_outImg = outImg.ptr<uchar>(0);
    uchar* device_inImg1, * device_inImg2;
    uchar* device_outImg;

    cudaMalloc((void**)&device_inImg1, sizeof(uchar) * size);
    cudaMalloc((void**)&device_inImg2, sizeof(uchar) * size);
    cudaMalloc((void**)&device_outImg, sizeof(uchar) * size);

    cudaMemcpy(device_inImg1, host_inImg1, sizeof(uchar) * size, cudaMemcpyHostToDevice);
    cudaMemcpy(device_inImg2, host_inImg2, sizeof(uchar) * size, cudaMemcpyHostToDevice);

    int BlockNum = min(size / ThreadNum, MaxBlockNum);
    Partial_Sub_Cuda << <BlockNum, ThreadNum >> > (device_outImg, device_inImg1, device_inImg2, size);

    cudaMemcpy(host_outImg, device_outImg, sizeof(uchar) * size, cudaMemcpyDeviceToHost);

    cudaFree(device_inImg1);
    cudaFree(device_inImg2);
    cudaFree(host_outImg);

    return outImg;
}

Mat GammaCorrection_Cuda(Mat inImg, float gamma)
{
    Mat outImg = inImg.clone();

    int size = inImg.rows * inImg.cols * inImg.channels();
    uchar* host_inImg = inImg.ptr<uchar>(0);
    uchar* host_outImg = outImg.ptr<uchar>(0);
    uchar* device_inImg;
    uchar* device_outImg;

    cudaMalloc((void**)&device_inImg, sizeof(uchar) * size);
    cudaMalloc((void**)&device_outImg, sizeof(uchar) * size);

    cudaMemcpy(device_inImg, host_inImg, sizeof(uchar) * size, cudaMemcpyHostToDevice);

    int BlockNum = min(size / ThreadNum, MaxBlockNum);
    Partial_GammaCorrection_Cuda << <BlockNum, ThreadNum >> > (device_outImg, device_inImg, gamma, size);

    cudaMemcpy(host_outImg, device_outImg, sizeof(uchar) * size, cudaMemcpyDeviceToHost);

    cudaFree(device_inImg);
    cudaFree(device_outImg);

    return outImg;
}


//Convolution using pthread
extern void padding(const Mat& img, Mat& result_image, int bios);

Mat Convolution_Cuda(Mat inImg, Mat inKernel)
{
    Mat outImg = Mat::zeros(inImg.rows, inImg.cols, inImg.type());

    int bios = (inKernel.cols - 1) / 2;//算出中心點
    Mat after_padding_img;
    padding(inImg, after_padding_img, bios);//padding
    //Mat outImg = Mat::zeros(inImg.rows, inImg.cols, inImg.type());


    int imgSize = inImg.rows * inImg.cols * inImg.channels();
    int imgPaddedSize = after_padding_img.rows * after_padding_img.cols * after_padding_img.channels();
    int kernelSize = inKernel.rows * inKernel.cols;

    uchar* host_inImg = after_padding_img.ptr<uchar>(0);
    float* host_inKernel = inKernel.ptr<float>(0);
    uchar* host_outImg = outImg.ptr<uchar>(0);
    uchar* device_inImg;
    float* device_inKernel;
    uchar* device_outImg;

    cudaMalloc((void**)&device_inImg, sizeof(uchar) * imgPaddedSize);
    cudaMalloc((void**)&device_inKernel, sizeof(float) * kernelSize);
    cudaMalloc((void**)&device_outImg, sizeof(uchar) * imgSize);

    cudaMemcpy(device_inImg, host_inImg, sizeof(uchar) * imgPaddedSize, cudaMemcpyHostToDevice);
    cudaMemcpy(device_inKernel, host_inKernel, sizeof(float) * kernelSize, cudaMemcpyHostToDevice);

    int BlockNum = after_padding_img.rows;
    //BlockNum = (BlockNum == 0)? 1 : BlockNum;
    Partial_Convolution_Cuda << <BlockNum, min(ThreadNum, after_padding_img.cols) >> > (device_outImg, device_inImg, device_inKernel, bios * 2, inImg.rows, inImg.cols, inImg.channels(), inKernel.rows, inKernel.cols);

    cudaMemcpy(host_outImg, device_outImg, sizeof(uchar) * imgSize, cudaMemcpyDeviceToHost);

    cudaFree(device_inImg);
    cudaFree(device_inKernel);
    cudaFree(device_outImg);

    return outImg;
}


