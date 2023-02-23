#include <immintrin.h>
#include <opencv2/highgui.hpp>
#include <opencv2/opencv.hpp>
#include "simdOp.h"
//#include "serialOp.h"
using namespace cv;

extern void padding(const Mat& img, Mat& result_image, int bios);

Mat Convolution_simd(Mat img, Mat kernel)
{
    int bios = (kernel.cols - 1) / 2;
    Mat after_padding_img;
    padding(img, after_padding_img, bios);
    Mat result_image = Mat::zeros(img.rows, img.cols, img.type());
    int channels = img.channels();
    if (channels == 3) // RGB image
    {
        for (int c = 0; c < channels; c++){
            for (int i = 0; i < result_image.rows; i++)
            {
                for (int j = 0; j < result_image.cols; j += 8) // process eight pixels at a time
                {
                    __m256 sum = _mm256_setzero_ps(); // set sum to zero
                    for (int curr_row = 0; curr_row < kernel.rows; curr_row++)
                    {
                        for (int curr_col = 0; curr_col < kernel.cols; curr_col++)
                        {
                            // load eight pixels
                            __m256 img_pixels = _mm256_set_ps(
                                after_padding_img.at<Vec3b>(curr_row + i, curr_col + j + 7)[c],
                                after_padding_img.at<Vec3b>(curr_row + i, curr_col + j + 6)[c],
                                after_padding_img.at<Vec3b>(curr_row + i, curr_col + j + 5)[c],
                                after_padding_img.at<Vec3b>(curr_row + i, curr_col + j + 4)[c],
                                after_padding_img.at<Vec3b>(curr_row + i, curr_col + j + 3)[c],
                                after_padding_img.at<Vec3b>(curr_row + i, curr_col + j + 2)[c],
                                after_padding_img.at<Vec3b>(curr_row + i, curr_col + j + 1)[c],
                                after_padding_img.at<Vec3b>(curr_row + i, curr_col + j)[c]
                            );
                            __m256 kern_val = _mm256_set1_ps(kernel.at<float>(curr_row, curr_col)); // set kern_val to the same value for all eight pixels
                            sum = _mm256_add_ps(sum, _mm256_mul_ps(img_pixels, kern_val)); // sum += img_pixels * kern_val
                        }
                    }
                    // store sum back to the result image
                    __m256i sum_i = _mm256_cvtps_epi32(sum);
                    // store the values in the result image
                    result_image.at<Vec3b>(i, j)[c] = cv::saturate_cast<uchar>(_mm256_cvtsi256_si32(sum_i));
                    result_image.at<Vec3b>(i, j + 1)[c] = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 1));
                    result_image.at<Vec3b>(i, j + 2)[c] = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 2));
                    result_image.at<Vec3b>(i, j + 3)[c] = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 3));
                    result_image.at<Vec3b>(i, j + 4)[c] = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 4));
                    result_image.at<Vec3b>(i, j + 5)[c] = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 5));
                    result_image.at<Vec3b>(i, j + 6)[c] = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 6));
                    result_image.at<Vec3b>(i, j + 7)[c] = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 7));
                }
            }
        }
    }
    else if (channels == 1) // grayscale image
    {
        for (int i = 0; i < result_image.rows; i++)
        {
            for (int j = 0; j < result_image.cols; j += 8) // process eight pixels at a time
            {
                __m256 sum = _mm256_setzero_ps(); // set sum to zero
                for (int curr_row = 0; curr_row < kernel.rows; curr_row++)
                {
                    for (int curr_col = 0; curr_col < kernel.cols; curr_col++)
                    {
                        __m256 img_pixels = _mm256_set_ps(
                            after_padding_img.at<uchar>(curr_row + i, curr_col + j + 7),
                            after_padding_img.at<uchar>(curr_row + i, curr_col + j + 6),
                            after_padding_img.at<uchar>(curr_row + i, curr_col + j + 5),
                            after_padding_img.at<uchar>(curr_row + i, curr_col + j + 4),
                            after_padding_img.at<uchar>(curr_row + i, curr_col + j + 3),
                            after_padding_img.at<uchar>(curr_row + i, curr_col + j + 2),
                            after_padding_img.at<uchar>(curr_row + i, curr_col + j + 1),
                            after_padding_img.at<uchar>(curr_row + i, curr_col + j)
                        );
                        __m256 kern_val = _mm256_set1_ps(kernel.at<float>(curr_row, curr_col)); // set kern_val to the same value for all eight pixels
                        sum = _mm256_add_ps(sum, _mm256_mul_ps(img_pixels, kern_val)); // sum += img_pixels * kern_val
                    }
                }
                // store sum back to the result image
                __m256i sum_i = _mm256_cvtps_epi32(sum);
                // store the values in the result image
                result_image.at<uchar>(i, j) = cv::saturate_cast<uchar>(_mm256_cvtsi256_si32(sum_i));
                result_image.at<uchar>(i, j + 1) = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 1));
                result_image.at<uchar>(i, j + 2) = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 2));
                result_image.at<uchar>(i, j + 3) = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 3));
                result_image.at<uchar>(i, j + 4) = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 4));
                result_image.at<uchar>(i, j + 5) = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 5));
                result_image.at<uchar>(i, j + 6) = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 6));
                result_image.at<uchar>(i, j + 7) = cv::saturate_cast<uchar>(_mm256_extract_epi32(sum_i, 7));
            }
        }
    }
    return result_image;
}
/*
int main()
{
    Mat img = imread("lena.jpg", IMREAD_GRAYSCALE);
    Mat kernel = (Mat_<float>(3, 3) << -1, -1, -1, -1, 8, -1, -1, -1, -1);
    Mat result_image = Convolution_simd(img, kernel);
    imwrite("result.jpg", result_image);
    return 0;
}*/
