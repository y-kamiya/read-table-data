#include <stdio.h>
#include <opencv2/opencv.hpp>

int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    auto imageName = argv[1];
    auto image = cv::imread(imageName, 0);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }

    cv::Mat bw;
    cv::adaptiveThreshold(~image, bw, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, -2);

    cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    cv::imshow("Display Image", bw);
    cv::waitKey(0);
    return 0;
}
