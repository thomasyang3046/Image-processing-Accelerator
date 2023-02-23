#include <opencv2/highgui.hpp>
#include "ompOp.h"
#include<omp.h>
using namespace cv;
using namespace std;
Mat GammaCorrection_omp(Mat img, float gamma)
{
    Mat des = img.clone();
    #pragma omp parallel for
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
            for (int k = 0; k < img.channels(); k++)
            {
                float pixel = (float)(img.at<Vec3b>(i, j)[k]) / 255.0;
                des.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(pow(pixel, gamma) * 255.0f);
            }
    return des;
}

//image add
Mat Add_omp(Mat img1, Mat img2)
{
    Mat des = img1.clone();
    #pragma omp parallel for
    for (int i = 0; i < img1.rows; i++)
        for (int j = 0; j < img1.cols; j++)
            for (int k = 0; k < img1.channels(); k++)
            {
                des.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(img1.at<Vec3b>(i, j)[k] + img2.at<Vec3b>(i, j)[k]);
            }
    return des;
}

//image sub
Mat Sub_omp(Mat img1, Mat img2)
{
    Mat des = img1.clone();
    #pragma omp parallel for
    for (int i = 0; i < img1.rows; i++)
        for (int j = 0; j < img1.cols; j++)
            for (int k = 0; k < img1.channels(); k++)
            {
                des.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(img1.at<Vec3b>(i, j)[k] - img2.at<Vec3b>(i, j)[k]);
            }
    return des;
}

//convolution
extern void padding(const Mat& img, Mat& result_image, int bios);

Mat Convolution_omp(Mat img, Mat kernel)
{
    int bios = (kernel.cols - 1) / 2;
    Mat after_padding_img;
    padding(img, after_padding_img, bios);//padding
    Mat result_image = Mat::zeros(img.rows, img.cols, img.type());
    int channels = img.channels();
    float sum = 0;
    if (channels == 3)//RGB image
        #pragma omp parallel for reduction(+:sum)
        for (int c = 0; c < channels; c++)
            for (int i = 0; i < result_image.rows; i++)
                for (int j = 0; j < result_image.cols; j++)
                {
 
                    sum = 0;
                    for (int curr_row = 0; curr_row < kernel.rows; curr_row++)
                        for (int curr_col = 0; curr_col < kernel.cols; curr_col++)
                        {
                            sum += (kernel.at<float>(curr_row, curr_col) * after_padding_img.at<Vec3b>(curr_row + i, curr_col + j)[c]);
                        }
                    result_image.at<Vec3b>(i, j)[c] = (int)sum;
                }
    else if (channels == 1)
        #pragma omp parallel for reduction(+:sum)
        for (int i = 0; i < result_image.rows; i++)
            for (int j = 0; j < result_image.cols; j++)
            {
                sum = 0;
                for (int curr_row = 0; curr_row < kernel.rows; curr_row++)
                    for (int curr_col = 0; curr_col < kernel.cols; curr_col++)
                    {
                        sum += (kernel.at<float>(curr_row, curr_col) * after_padding_img.at<uchar>(curr_row + i, curr_col + j));
                    }
                if (sum < 0)
                {
                    result_image.at<uchar>(i, j) = 0;
                }
                else
                {
                    result_image.at<uchar>(i, j) = (uchar)sum;
                }
            }
    return result_image;

}


