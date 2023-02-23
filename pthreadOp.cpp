#include <pthread.h>
#include <thread>
#include <opencv2/highgui.hpp>
#include "pthreadOp.h"
#include <time.h>

using namespace cv;

//GammaCorrection using pthread
void* Partial_GammaCorrection(void* arg)
{
    Arg* args = (Arg*)arg;
    for (int i = args->start; i < args->end; i++)
        for (int j = 0; j < args->inImg1->cols; j++)
            for (int k = 0; k < args->inImg1->channels(); k++)
            {
                float pixel = (float)args->inImg1->ptr<Vec3b>(i)[j][k] / 255.0;
                args->inImg1->ptr<Vec3b>(i)[j][k] = saturate_cast<uchar>(pow(pixel, args->gamma) * 255.0f);
            }
    return NULL;
}

void GammaCorrection_pthread(cv::Mat img, float gamma)
{
    const int MAXTHREADSCOUNT = std::thread::hardware_concurrency();

    pthread_t *threads = new pthread_t[MAXTHREADSCOUNT]();

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    Arg *arg = new Arg[MAXTHREADSCOUNT]();

    int part = img.rows / MAXTHREADSCOUNT;
    for (int i = 0; i < MAXTHREADSCOUNT; i++)
    {
        arg[i].thread_id = i;
        arg[i].start = part * i;
        arg[i].end = part * (i + 1);
        arg[i].gamma = gamma;
        arg[i].inImg1 = &img;
    }
    arg[MAXTHREADSCOUNT - 1].end = img.rows;

    for (int i = 1; i < MAXTHREADSCOUNT; i++)
    {
        pthread_create(&threads[i], &attr, Partial_GammaCorrection, (void*)&arg[i]);
    }
    //Master thread
    Partial_GammaCorrection((void*)arg);

    pthread_attr_destroy(&attr);

    void* status;
    for (int i = 1; i < MAXTHREADSCOUNT; i++)
    {
        // 等待每一個 thread 執行完畢
        pthread_join(threads[i], &status);
    }
}

//Add using pthread
void* Partial_Add(void* arg)
{
    Arg* args = (Arg*)arg;
    for (int i = args->start; i < args->end; i++)
        for (int j = 0; j < args->inImg1->cols; j++)
            for (int k = 0; k < args->inImg1->channels(); k++)
            {
                args->outImg->ptr<Vec3b>(i)[j][k] = saturate_cast<uchar>(args->inImg1->ptr<Vec3b>(i)[j][k] + args->inImg2->ptr<Vec3b>(i)[j][k]);
            }
    return NULL;
}

Mat Add_pthread(Mat img1, Mat img2)
{
    Mat des = img1.clone();
    const int MAXTHREADSCOUNT = std::thread::hardware_concurrency();

    pthread_t* threads = new pthread_t[MAXTHREADSCOUNT]();

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    Arg* arg = new Arg[MAXTHREADSCOUNT]();

    int part = img1.rows / MAXTHREADSCOUNT;
    for (int i = 0; i < MAXTHREADSCOUNT; i++)
    {
        arg[i].thread_id = i;
        arg[i].start = part * i;
        arg[i].end = part * (i + 1);
        arg[i].inImg1 = &img1;
        arg[i].inImg2 = &img2;
        arg[i].outImg = &des;
    }
    arg[MAXTHREADSCOUNT - 1].end = img1.rows;

    for (int i = 1; i < MAXTHREADSCOUNT; i++)
    {
        pthread_create(&threads[i], &attr, Partial_Add, (void*)&arg[i]);
    }
    //Master thread
    Partial_Add((void*)arg);

    pthread_attr_destroy(&attr);

    void* status;
    for (int i = 1; i < MAXTHREADSCOUNT; i++)
    {
        // 等待每一個 thread 執行完畢
        pthread_join(threads[i], &status);
    }
    return des;
}

//Sub using pthread
void* Partial_Sub(void* arg)
{
    Arg* args = (Arg*)arg;
    for (int i = args->start; i < args->end; i++)
        for (int j = 0; j < args->inImg1->cols; j++)
            for (int k = 0; k < args->inImg1->channels(); k++)
            {
                args->outImg->ptr<Vec3b>(i)[j][k] = saturate_cast<uchar>(args->inImg1->ptr<Vec3b>(i)[j][k] - args->inImg2->ptr<Vec3b>(i)[j][k]);
            }
    return NULL;
}

Mat Sub_pthread(Mat img1, Mat img2)
{
    Mat des = img1.clone();
    const int MAXTHREADSCOUNT = std::thread::hardware_concurrency();

    pthread_t* threads = new pthread_t[MAXTHREADSCOUNT]();

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    Arg* arg = new Arg[MAXTHREADSCOUNT]();

    int part = img1.rows / MAXTHREADSCOUNT;
    for (int i = 0; i < MAXTHREADSCOUNT; i++)
    {
        arg[i].thread_id = i;
        arg[i].start = part * i;
        arg[i].end = part * (i + 1);
        arg[i].inImg1 = &img1;
        arg[i].inImg2 = &img2;
        arg[i].outImg = &des;
    }
    arg[MAXTHREADSCOUNT - 1].end = img1.rows;

    for (int i = 1; i < MAXTHREADSCOUNT; i++)
    {
        pthread_create(&threads[i], &attr, Partial_Sub, (void*)&arg[i]);
    }
    //Master thread
    Partial_Sub((void*)arg);

    pthread_attr_destroy(&attr);

    void* status;
    for (int i = 1; i < MAXTHREADSCOUNT; i++)
    {
        // 等待每一個 thread 執行完畢
        pthread_join(threads[i], &status);
    }
    return des;
}

//Convolution using pthread
extern void padding(const Mat& img, Mat& result_image, int bios);

void* Partial_Convolution(void* arg)
{
    Mat* kernel = ((Arg*)arg)->kernel;
    Mat* after_padding_img = ((Arg*)arg)->inImg1;

    int start = ((Arg*)arg)->start;
    int end = ((Arg*)arg)->end;

    Mat* result_image = ((Arg*)arg)->outImg;

    int channels = result_image->channels();
    if (channels == 3)//RGB image
        for (int c = 0; c < channels; c++)
            for (int i = start; i < end; i++)
                for (int j = 0; j < result_image->cols; j++)//這三層為image的長,寬,通道數
                {
                    //開始做捲積
                    float sum = 0;
                    for (int curr_row = 0; curr_row < kernel->rows; curr_row++)
                        for (int curr_col = 0; curr_col < kernel->cols; curr_col++)
                        {
                            sum += (kernel->at<float>(curr_row, curr_col) * after_padding_img->at<Vec3b>(curr_row + i, curr_col + j)[c]);
                        }
                    if (sum < 0)
                    {
                        result_image->at<Vec3b>(i, j)[c] = 0;
                    }
                    else
                    {
                        result_image->at<Vec3b>(i, j)[c] = (uchar)sum;
                    }
                }
    else if (channels == 1)//灰階image
        for (int i = start; i < end; i++)
            for (int j = 0; j < result_image->cols; j++)//這兩層為image的長,寬
            {
                //開始做捲積
                float sum = 0;
                for (int curr_row = 0; curr_row < kernel->rows; curr_row++)
                    for (int curr_col = 0; curr_col < kernel->cols; curr_col++)
                    {
                        sum += (kernel->at<float>(curr_row, curr_col) * after_padding_img->at<uchar>(curr_row + i, curr_col + j));
                    }
                if (sum < 0)
                {
                    result_image->at<uchar>(i, j) = 0;
                }
                else
                {
                    result_image->at<uchar>(i, j) = (uchar)sum;
                }
            }
    return NULL;
}

Mat Convolution_pthread(Mat img, Mat kernel)
{
    int bios = (kernel.cols - 1) / 2;//算出中心點
    Mat after_padding_img;
    padding(img, after_padding_img, bios);//padding
    Mat result_image = Mat::zeros(img.rows, img.cols, img.type());

    const int MAXTHREADSCOUNT = std::thread::hardware_concurrency();

    pthread_t* threads = new pthread_t[MAXTHREADSCOUNT]();

    pthread_attr_t attr;
    pthread_attr_init(&attr);
    pthread_attr_setdetachstate(&attr, PTHREAD_CREATE_JOINABLE);

    Arg* arg = new Arg[MAXTHREADSCOUNT]();

    int part = img.rows / MAXTHREADSCOUNT;
    for (int i = 0; i < MAXTHREADSCOUNT; i++)
    {
        arg[i].thread_id = i;
        arg[i].start = part * i;
        arg[i].end = part * (i + 1);
        arg[i].inImg1 = &after_padding_img;
        arg[i].kernel = &kernel;
        arg[i].outImg = &result_image;
    }
    arg[MAXTHREADSCOUNT - 1].end = img.rows;

    for (int i = 1; i < MAXTHREADSCOUNT; i++)
    {
        pthread_create(&threads[i], &attr, Partial_Convolution, (void*)&arg[i]);
    }
    //Master thread
    Partial_Convolution((void*)arg);

    void* status;
    for (int i = 1; i < MAXTHREADSCOUNT; i++)
    {
        // 等待每一個 thread 執行完畢
        pthread_join(threads[i], &status);
    }
    pthread_attr_destroy(&attr);
    return result_image;
}

