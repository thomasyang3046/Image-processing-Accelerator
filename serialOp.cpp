#include <opencv2/highgui.hpp>
#include "serialOp.h"
using namespace cv;

void GammaCorrection_serial(Mat img, float gamma)
{
    for (int i = 0; i < img.rows; i++)
        for (int j = 0; j < img.cols; j++)
        {
            float pixel = (float)(img.at<uchar>(i, j)) / 255.0;
            img.at<uchar>(i, j) = saturate_cast<uchar>(pow(pixel, gamma) * 255.0f);
        }
}

//image add
Mat Add_serial(Mat img1, Mat img2)
{
    Mat des = img1.clone();
    for (int i = 0; i < img1.rows; i++)
        for (int j = 0; j < img1.cols; j++)
            
                des.at<uchar>(i, j) = img1.at<uchar>(i, j) + img2.at<uchar>(i, j);
            
    return des;
}

//image sub
Mat Sub_serial(Mat img1, Mat img2)
{
    Mat des = img1.clone();
    for (int i = 0; i < img1.rows; i++)
        for (int j = 0; j < img1.cols; j++)
            for (int k = 0; k < img1.channels(); k++)
            {
                des.at<Vec3b>(i, j)[k] = saturate_cast<uchar>(img1.at<Vec3b>(i, j)[k] - img2.at<Vec3b>(i, j)[k]);
            }
    return des;
}

//convolution
void padding(const Mat& img, Mat& result_image, int bios)
{
    result_image = Mat::zeros(2 * bios + img.rows, 2 * bios + img.cols, img.type());
    Rect ROI_image = Rect(bios, bios, img.cols, img.rows);
    Mat after_padding = result_image(ROI_image);
    img.copyTo(result_image(ROI_image));
}

//img為輸入的圖片，kernel為看要哪種filer
Mat Convolution_serial(Mat img, Mat kernel)
{
    int bios = (kernel.cols - 1) / 2;//算出中心點
    Mat after_padding_img;
    padding(img, after_padding_img, bios);//padding
    Mat result_image = Mat::zeros(img.rows, img.cols, img.type());
    int channels = img.channels();
    if (channels == 3)//RGB image
        for (int c = 0; c < channels; c++)
            for (int i = 0; i < result_image.rows; i++)
                for (int j = 0; j < result_image.cols; j++)//這三層為image的長,寬,通道數
                {
                    //開始做捲積
                    float sum = 0;
                    for (int curr_row = 0; curr_row < kernel.rows; curr_row++)
                        for (int curr_col = 0; curr_col < kernel.cols; curr_col++)
                        {
                            sum += (kernel.at<float>(curr_row, curr_col) * after_padding_img.at<Vec3b>(curr_row + i, curr_col + j)[c]);
                        }
                    if (sum < 0)
                    {
                        result_image.at<Vec3b>(i, j)[c] = 0;
                    }
                    else
                    {
                        result_image.at<Vec3b>(i, j)[c] = (uchar)sum;
                    }
                }
    else if (channels == 1)//灰階image
        for (int i = 0; i < result_image.rows; i++)
            for (int j = 0; j < result_image.cols; j++)//這兩層為image的長,寬
            {
                //開始做捲積
                float sum = 0;
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


