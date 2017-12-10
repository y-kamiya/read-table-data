#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

int SCALE_FOR_MASK = 20;
int THRESHOLD_SAME_ROW = 50;

float CELL_SIZE = 8000.0f;
float RATIO_CELL_SIZE = 1.0f;
float THRESHOLD_SAME_LINE_RADIAN = 0.01f;
float THRESHOLD_SAME_LINE_RHO = 25.0f;
float RATIO_IMAGE_SIZE = 0.4f;
float RADIAN_PER_DEGREE = M_PI / 180.0f;

cv::Mat getVerticalMask(cv::Mat &imageBinary) {
    auto vertical = imageBinary.clone();

    // Specify size on vertical axis
    int verticalsize = vertical.rows / SCALE_FOR_MASK;

    // Create structure element for extracting vertical lines through morphology operations
    auto verticalStructure = getStructuringElement(cv::MORPH_RECT, cv::Size( 1,verticalsize));

    // Apply morphology operations
    erode(vertical, vertical, verticalStructure, cv::Point(-1, -1));
    dilate(vertical, vertical, verticalStructure, cv::Point(-1, -1));

    return vertical;
}

cv::Mat getHorizontalMask(cv::Mat &imageBinary) {
    auto horizontal = imageBinary.clone();

    // Specify size on horizontal axis
    int horizontalsize = horizontal.cols / SCALE_FOR_MASK;

    // Create structure element for extracting horizontal lines through morphology operations
    auto horizontalStructure = getStructuringElement(cv::MORPH_RECT, cv::Size(horizontalsize,1));

    // Apply morphology operations
    erode(horizontal, horizontal, horizontalStructure, cv::Point(-1, -1));
    dilate(horizontal, horizontal, horizontalStructure, cv::Point(-1, -1));

    return horizontal;
}

cv::Mat displayLinesByCanny(cv::Mat &image) {
    cv::Mat edges;
    cv::Canny(image, edges, 100, 300);

    auto kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(3,3));
    dilate(edges, edges, kernel);
    erode(edges, edges, kernel);

    return edges;
}

struct AverageBuilder {
    float _rho = 0.0f;
    float _theta = 0.0f;
    int _count = 0;

    AverageBuilder(float rho, float theta) {
        _rho = rho;
        _theta = theta;
        _count = 1;
    }

    void add(float rho, float theta) {
        _rho = (_rho * _count + rho) / (_count + 1);
        _theta = (_theta * _count + theta) / (_count + 1);
        _count += 1;
    }

};

bool isLineHorizontal(float theta, float dTheta = THRESHOLD_SAME_LINE_RADIAN) {
    return M_PI / 2 - dTheta <= theta && theta <= M_PI / 2 + dTheta;
}

bool isLineVertical(float theta, float dTheta = THRESHOLD_SAME_LINE_RADIAN) {
    return theta <= dTheta || M_PI - dTheta < theta;
}

bool isTableLine(float theta, float dTheta = THRESHOLD_SAME_LINE_RADIAN) {
    return isLineHorizontal(theta, dTheta) || isLineVertical(theta, dTheta);
}

float translateTheta(float theta) {
    return M_PI - THRESHOLD_SAME_LINE_RADIAN <= theta ? theta - M_PI : theta;
}

bool isSimilarLineHorizontal(float rho, float theta, float dRho, float dTheta, AverageBuilder &entry) {
    return rho - dRho < entry._rho && entry._rho < rho + dRho
        && theta - dTheta < entry._theta && entry._theta < theta + dTheta;
}

bool isSimilarLineVertical(float rho, float theta, float dRho, float dTheta, AverageBuilder &entry) {
    auto _rho = fabs(rho);
    auto _theta = translateTheta(theta);
    return _rho - dRho < entry._rho && entry._rho < _rho + dRho
        && _theta - dTheta < entry._theta && entry._theta < _theta + dTheta;
}

std::vector<cv::Vec2f> extractTableLines(std::vector<cv::Vec2f> &lines, float dRho, float dTheta) {
    std::vector<AverageBuilder> linesUnited;
    for (auto &line : lines) {
        // printf("raw lines: %f, %f\n", line[0], line[1]);
        auto rho = fabs(line[0]);
        auto theta = translateTheta(line[1]);
        if (!isTableLine(theta)) {
            continue;
        }

        auto it = std::find_if(linesUnited.begin(), linesUnited.end(),
            [rho, theta, dRho, dTheta](AverageBuilder &entry) {
                return isSimilarLineHorizontal(rho, theta, dRho, dTheta, entry)
                    && isSimilarLineVertical(rho, theta, dRho, dTheta, entry);
        });

        if (it == linesUnited.end()) {
            AverageBuilder builder(rho, theta);
            linesUnited.push_back(std::move(builder));
        } else {
            auto &builder = *it;
            builder.add(rho, theta);
        }
    }

    std::vector<cv::Vec2f> ret;
    for (auto &builder : linesUnited) {
        cv::Vec2f entry;
        entry[0] = builder._rho;
        entry[1] = builder._theta;
        if (entry[1] < 0) {
            entry[0] = -1 * entry[0];
            entry[1] = M_PI + entry[1];
        }
        ret.push_back(std::move(entry));
        printf("united lines: %f, %f\n", entry[0], entry[1]);
    }
    return ret;
}

cv::Mat drawTableLinesOnBlack(cv::Mat &image, std::vector<cv::Vec2f> &lines, float dTheta, int thickness = 1) {
    auto img = image.clone();
    img.setTo(cv::Scalar(0,0,0));

    auto z = std::max(img.size().width, img.size().height);
    for (auto &line : lines) {
        auto rho = line[0];
        auto theta = line[1];

        auto ct = cos(theta);
        auto st = sin(theta);
        auto start = cv::Point(rho * ct - z * st, rho * st + z * ct);
        auto end   = cv::Point(rho * ct + z * st, rho * st - z * ct);
        if (isTableLine(theta, dTheta)) {
            cv::line(img, start, end, cv::Scalar(255,255,255), thickness);
        }
    }
    return img;
}

struct Cell {
    int x;
    int y;
    cv::Mat data;

    Cell() {}
    Cell(int x, int y, cv::Mat data): x(x), y(y), data(data) {
    }

};


std::vector<std::vector<Cell>> sortCells(std::vector<Cell> &cells) {
    std::sort(cells.begin(), cells.end(), [](Cell &a, Cell &b) {
        return a.y < b.y;
    });

    std::vector<std::vector<Cell>> groups;
    Cell cellPrev;
    cellPrev.y = -10000;
    auto index = -1;
    for (auto &cell : cells) {
        if (THRESHOLD_SAME_ROW < cell.y - cellPrev.y) {
            groups.push_back(std::vector<Cell>());
            cellPrev = cell;
            index++;
            // printf("index: %d, y: %d\n", index, cell.y);
        }
        groups[index].push_back(cell);
    }

    for (auto &group : groups) {
        std::sort(group.begin(), group.end(), [](Cell &a, Cell &b) {
            return a.x < b.x;
        });
    }
    return groups;
}

int getNumberFromCell(Cell &cell, tesseract::TessBaseAPI *api) {
    auto img = cell.data;
    cv::imwrite("output.jpg", img);

    std::vector<cv::Vec2f> lines;
    auto threshold = static_cast<int>(std::min(img.rows, img.cols)*0.7f);
    // auto threshold = static_cast<int>(std::min(img.rows, img.cols)*0.9);
    cv::HoughLines(img, lines, 1, RADIAN_PER_DEGREE * 5, threshold);
    auto imageLines = drawTableLinesOnBlack(img, lines, 3);
    cv::imwrite("output_line.jpg", imageLines);

    img = img - imageLines;
    erode(img, img,  getStructuringElement(cv::MORPH_RECT, cv::Size(2,2)));
    dilate(img, img, getStructuringElement(cv::MORPH_RECT, cv::Size(3,3)));
    cv::imwrite("output_final.jpg", img);

    char *outText;

    // Open input image with leptonica library
    Pix *image = pixRead("/Users/uj/gws/read-table-data/build/output_final.jpg");
    api->SetImage(image);
    // Get OCR result
    outText = api->GetUTF8Text();
    auto str = std::string(outText);
    auto num = 0;
    try {
        num = std::stoi(str);
    } catch (std::exception e) { }

    printf("OCR output num: %d\n", num);
    printf("OCR output txt: %s\n", outText);

    delete [] outText;
    pixDestroy(&image);

    cv::imshow("roi", img);
    // cv::waitKey();
    return num;
}

cv::Mat modifyAngle(cv::Mat &image) {
    auto edges = displayLinesByCanny(image);

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1, RADIAN_PER_DEGREE/10, 300);
    float angle = 0.0f;
    int count = 0;
    for (auto &line : lines) {
        auto theta = line[1];
        if (isLineHorizontal(theta)) {
            printf("angle: %f\n", theta);
            ++count;
            angle += theta;
        }
    }

    auto averageDegree = 0.0f;
    if (count != 0) {
        averageDegree = angle / count * 180 / M_PI;
        printf("average angle: %f\n", averageDegree);
    }

    auto center = cv::Point(image.cols / 2, image.rows / 2);
    auto mat = cv::getRotationMatrix2D(center, averageDegree - 90.0f, 1.0f);
    cv::Mat imgDst;
    cv::warpAffine(image, imgDst, mat, image.size());

    // cv::imshow("bbbbbbbb", imgDst);
    // cv::waitKey();

    // std::vector<cv::Vec4i> lines;
    // cv::HoughLinesP(edges, lines, 1, RADIAN_PER_DEGREE, 100, imageResized.cols / 5, 20);
    // for( size_t i = 0; i < lines.size(); i++ ) {
    //     line( imageResized, cv::Point(lines[i][0], lines[i][1]), cv::Point(lines[i][2], lines[i][3]), cv::Scalar(0,0,255), 3, 8 );
    // }

    return imgDst;
}

int main(int argc, char** argv )
{
    if ( argc != 3 )
    {
        printf("usage: DisplayImage.out <Image_Path>\n");
        return -1;
    }

    auto imageName = argv[1];
    auto configCellSize = atof(argv[2]); 
    auto imageIn = cv::imread(imageName, 1);

    cv::Mat image, imageResized;
    cv::Size size(imageIn.size().width * RATIO_IMAGE_SIZE, imageIn.size().height * RATIO_IMAGE_SIZE);
    cv::resize(imageIn, imageResized, size);
    cv::cvtColor(imageResized, image, CV_BGR2GRAY);
    if ( !image.data )
    {
        printf("No image data \n");
        return -1;
    }
    cv::Mat imageBinary;
    cv::adaptiveThreshold(~image, imageBinary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, -10);

    imageBinary = modifyAngle(imageBinary);
    // cv::imshow("modifiedImage", imageBinary );
    // cv::waitKey(0);
    // return 0;

    auto horizontal = getHorizontalMask(imageBinary);
    auto vertical = getVerticalMask(imageBinary);

    cv::Mat mask = horizontal + vertical;
    std::vector<cv::Vec2f> lines;
    cv::HoughLines(mask, lines, 1, RADIAN_PER_DEGREE * 5, 200);

    cv::imshow("mask", mask);
    // cv::waitKey(0);
    // return 0;
    auto tableLines = extractTableLines(lines, THRESHOLD_SAME_LINE_RHO, THRESHOLD_SAME_LINE_RADIAN);
    // tableLines = lines;

    auto imageMask = drawTableLinesOnBlack(imageBinary, tableLines, THRESHOLD_SAME_LINE_RADIAN);
    // cv::imshow("imageMask", imageMask);
    // cv::imshow("binary", imageBinary);
    // cv::waitKey(0);


    // Find external contours from the mask, which most probably will belong to tables or to images
    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    // cv::findContours(imageBinary, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));
    cv::findContours(imageMask, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    // for (auto contour = contours.begin(); contour != contours.end(); contour++){
    //     cv::polylines(imageResized, *contour, true, cv::Scalar(0, 255, 0), 2);
    // }
    // printf("%ld\n", contours.size());
    // for (size_t i = 0; i < contours.size(); i++) {
    //     auto img = image.clone();
    //     img.setTo(cv::Scalar(0,0,0));
    //     cv::polylines(img, contours[i], true, cv::Scalar(255, 255, 255), 2);
    //     cv::imshow("img", img);
    //     cv::waitKey(0);
    // }
    //
    // cv::imshow("imageResized", imageResized);
    // cv::waitKey(0);
    // return 0;


    auto kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(1,1));
    erode(imageBinary, imageBinary, kernel );
    dilate(imageBinary, imageBinary, kernel);

    // auto edges = displayLinesByCanny(imageBinary);
    // edges = edges - imageMask;
    // imageBinary = imageBinary - imageMask;
    // imshow("edges", edges);
    // imshow("binary", imageBinary);
    // imshow("imageMask", imageMask);
    // cv::waitKey();


    std::vector<std::vector<cv::Point> > contours_poly( contours.size() );
    std::vector<cv::Rect> boundRect( contours.size() );
    std::vector<Cell> rois;

    for (size_t i = 0; i < contours.size(); i++) {
        // find the area of each contour
        double area = contourArea(contours[i]);

        printf("%lf\n", area);
        // if(area < 1000 || 3000 < area) {
        if(area < configCellSize * 0.8f || configCellSize * 1.2f < area) {
            continue;
        }

        approxPolyDP( cv::Mat(contours[i]), contours_poly[i], 0.01 * cv::arcLength(contours[i], true), true );
        boundRect[i] = boundingRect( cv::Mat(contours_poly[i]) );

        // find the number of joints that each table has
        // auto roi = joints(boundRect[i]);
        //
        // std::vector<std::vector<cv::Point> > joints_contours;
        // findContours(roi, joints_contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        //
        // printf("%ld\n", joints_contours.size());

        // cv::Mat img = cv::Mat(imageBinary, boundRect[i]);
        auto rect = boundRect[i];
        rect.x += rect.width * (1.0f - RATIO_CELL_SIZE) / 2;
        rect.y += rect.height * (1.0f - RATIO_CELL_SIZE) / 2;
        rect.width *= RATIO_CELL_SIZE;
        rect.height *= RATIO_CELL_SIZE;

        Cell cell(rect.x, rect.y, imageBinary(rect).clone());
        rois.push_back(std::move(cell));

        cv::rectangle(imageResized, cv::Point(rect.x, rect.y), cv::Point(rect.x+rect.width, rect.y+rect.height), cv::Scalar(0,0,255), 3);

//        drawContours( image, contours, i, Scalar(0, 0, 255), CV_FILLED, 8, std::vector<Vec4i>(), 0, Point() );
        // rectangle( imageResized, rect.tl(), rect.br(), cv::Scalar(0, 255, 0), 1, 8, 0 );
    }


    cv::imshow("image", imageResized);
    cv::waitKey();

    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    if (api->Init(nullptr, "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }
    // api->SetVariable("tessedit_char_whitelist", "0123456789");

    std::ofstream ofs("data.csv");

    auto groups = sortCells(rois);
    for (auto &group : groups) {
        for (auto &cell : group) {
            // printf("cell x: %d, y: %d\n", cell.x, cell.y);
            auto num = getNumberFromCell(cell, api);
            ofs << num << ',';
        }
        ofs << std::endl;
    }

    api->End();


    // cv::namedWindow("Display Image", cv::WINDOW_AUTOSIZE );
    // cv::imshow("Display Image", imageBinary);
    // cv::imshow("horizontal", horizontal);
    // cv::imshow("vertical", vertical);
    // cv::imshow("mask", mask);
    // cv::imshow("joints", joints);
    // cv::imshow("imageResized", imageResized);
    // cv::waitKey(0);
    return 0;
}



