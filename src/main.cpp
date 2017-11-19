#include <stdio.h>
#include <opencv2/opencv.hpp>


cv::Mat getVerticalMask(cv::Mat &imageBinary) {
    auto vertical = imageBinary.clone();

    int scale = 10; // play with this variable in order to increase/decrease the amount of lines to be detected

    // Specify size on vertical axis
    int verticalsize = vertical.rows / scale;

    // Create structure element for extracting vertical lines through morphology operations
    auto verticalStructure = getStructuringElement(cv::MORPH_RECT, cv::Size( 1,verticalsize));

    // Apply morphology operations
    erode(vertical, vertical, verticalStructure, cv::Point(-1, -1));
    dilate(vertical, vertical, verticalStructure, cv::Point(-1, -1));

    return vertical;
}

cv::Mat getHorizontalMask(cv::Mat &imageBinary) {
    auto horizontal = imageBinary.clone();

    int scale = 10; // play with this variable in order to increase/decrease the amount of lines to be detected

    // Specify size on horizontal axis
    int horizontalsize = horizontal.cols / scale;

    // Create structure element for extracting horizontal lines through morphology operations
    auto horizontalStructure = getStructuringElement(cv::MORPH_RECT, cv::Size(horizontalsize,1));

    // Apply morphology operations
    erode(horizontal, horizontal, horizontalStructure, cv::Point(-1, -1));
    dilate(horizontal, horizontal, horizontalStructure, cv::Point(-1, -1));

    return horizontal;
}

void displayLinesByCanny(cv::Mat &image) {
    cv::Mat edges;
    std::vector<cv::Vec2f> lines;
    cv::Canny(image, edges, 100, 300);
    cv::imshow("edges", edges);

    cv::HoughLines(edges, lines, 1, 3.14/180, 100);
    auto z = image.cols;
    for (auto &line : lines) {
        printf("%f, %f\n", line[0], line[1]);
        auto rho = line[0];
        auto theta = line[1];
        auto ct = cos(theta);
        auto st = sin(theta);
        auto start = cv::Point(rho * ct - z * st, rho * st + z * ct);
        auto end   = cv::Point(rho * ct + z * st, rho * st - z * ct);
        cv::line(image, start, end, cv::Scalar(0,0,255));
    }
    cv::imshow("image", image);
    cv::waitKey(0);
}



int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    auto imageName = argv[1];
    auto imageIn = cv::imread(imageName, 1);

    cv::Mat image, imageResized;
    cv::resize(imageIn, imageResized, cv::Size(800,900));
    cv::cvtColor(imageResized, image, CV_BGR2GRAY);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    cv::Mat imageBinary;
    cv::adaptiveThreshold(~image, imageBinary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, -10);

    auto horizontal = getHorizontalMask(imageBinary);
    auto vertical = getVerticalMask(imageBinary);

    cv::Mat mask = horizontal + vertical;
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(mask, lines, 1, 3.14/180, 200);

    // 近い値の直線をまとめる
    // p < 10, theta < 0.1くらい？
    // std::vector<cv::Vec2f> acc;
    // for (auto &line : lines) {
    // }

    auto imageH = image.clone();
    imageH.setTo(cv::Scalar(0,0,0));
    auto imageV = image.clone();
    imageV.setTo(cv::Scalar(0,0,0));

    auto z = 1000;
    for (auto &line : lines) {
        // printf("%f, %f\n", line[0], line[1]);
        auto rho = line[0];
        auto theta = line[1];
        auto ct = cos(theta);
        auto st = sin(theta);
        auto start = cv::Point(rho * ct - z * st, rho * st + z * ct);
        auto end   = cv::Point(rho * ct + z * st, rho * st - z * ct);
        if (theta <= 0.1 || M_PI - 0.1 < theta) {
            cv::line(imageV, start, end, cv::Scalar(255,255,255));
        } else if (M_PI / 2 - 0.1 <= theta && theta <= M_PI / 2 + 0.1) {
            cv::line(imageH, start, end, cv::Scalar(255,255,255));
        }
    }

    cv::Mat joints;
    bitwise_and(imageH, imageV, joints);
    // cv::imshow("joints", joints);

    cv::Mat imageMask = imageH + imageV;
    cv::imshow("imageMask", imageMask);



    // Find external contours from the mask, which most probably will belong to tables or to images
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    // cv::findContours(imageBinary, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    cv::findContours(imageMask, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    // for (auto contour = contours.begin(); contour != contours.end(); contour++){
    //     cv::polylines(imageResized, *contour, true, cv::Scalar(0, 255, 0), 2);
    // }   
    // cv::imshow("imageResized", imageResized);
    // printf("%ld\n", contours.size());




    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    std::vector<cv::Rect> boundRect( contours.size() );
    std::vector<cv::Mat> rois;

    for (size_t i = 0; i < contours.size(); i++)
    {
        // find the area of each contour
        double area = contourArea(contours[i]);

//        // filter individual lines of blobs that might exist and they do not represent a table
        if(area < 100) // value is randomly chosen, you will need to find that by yourself with trial and error procedure
            continue;

        approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 0.01 * cv::arcLength(contours[i], true), true );
        boundRect[i] = boundingRect( cv::Mat(contours_poly[i]) );

        printf("%ld\n", contours_poly[i].size());
        printf("%lf\n", area);
        // find the number of joints that each table has
        auto roi = joints(boundRect[i]);

        std::vector<std::vector<cv::Point> > joints_contours;
        findContours(roi, joints_contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);

        // if the number is not more than 5 then most likely it not a table
        if(joints_contours.size() <= 4)
            continue;

        rois.push_back(image(boundRect[i]).clone());

//        drawContours( image, contours, i, Scalar(0, 0, 255), CV_FILLED, 8, std::vector<Vec4i>(), 0, Point() );
        rectangle( imageResized, boundRect[i].tl(), boundRect[i].br(), cv::Scalar(0, 255, 0), 1, 8, 0 );
    }

    std::cout << rois.size() << std::endl;
    for(size_t i = 0; i < rois.size(); ++i)
    {
        /* Now you can do whatever post process you want
         * with the data within the rectangles/tables. */
        // cv::imshow("roi", rois[i]);
        // cv::waitKey();
    }




    // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    // cv::imshow("Display Image", imageBinary);
    // cv::imshow("horizontal", horizontal);
    // cv::imshow("vertical", vertical);
    // cv::imshow("mask", mask);
    // cv::imshow("joints", joints);
    cv::imshow("imageResized", imageResized);
    cv::waitKey(0);
    return 0;
}
