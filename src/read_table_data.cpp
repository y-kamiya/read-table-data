#include "read_table_data.hpp"

const int ReadTableData::SCALE_FOR_MASK = 20;
const int ReadTableData::THRESHOLD_SAME_ROW = 50;
const int ReadTableData::LENGTH_FOR_ANGLE_MODIFICATION = 500;

const float ReadTableData::CELL_SIZE = 8000.0f;
const float ReadTableData::RATIO_CELL_SIZE = 1.0f;
const float ReadTableData::THRESHOLD_SAME_LINE_RADIAN = 0.01f;
const float ReadTableData::THRESHOLD_SAME_LINE_RHO = 25.0f;
const float ReadTableData::RATIO_IMAGE_SIZE = 0.5f;
const float ReadTableData::RADIAN_PER_DEGREE = M_PI / 180.0f;

void ReadTableData::execute(const std::string &imageName) {
    auto imageBinary = getImageBinary(imageName);
    auto imageMask = getTableMask(imageBinary);

    std::vector<cv::Vec4i> hierarchy;
    std::vector<std::vector<cv::Point> > contours;
    cv::findContours(imageMask, contours, hierarchy, CV_RETR_LIST, CV_CHAIN_APPROX_SIMPLE, cv::Point(0, 0));

    auto kernel = getStructuringElement(cv::MORPH_RECT, cv::Size(1,1));
    erode(imageBinary, imageBinary, kernel );
    dilate(imageBinary, imageBinary, kernel);

    auto cells = createCells(contours, imageBinary, getImageResized(imageName));

    readCells(cells);
}

cv::Mat ReadTableData::getImageResized(const std::string &imageName) const {
    auto imageIn = cv::imread(imageName, 1);
    cv::Mat imageResized;
    cv::Size size(imageIn.size().width * RATIO_IMAGE_SIZE, imageIn.size().height * RATIO_IMAGE_SIZE);
    cv::resize(imageIn, imageResized, size);
    return imageResized;
}

cv::Mat ReadTableData::getImageBinary(const std::string &imageName) {
    auto imageResized = getImageResized(imageName);
    cv::imshow("before angle", imageResized);

    cv::Mat imageGray;
    imageResized = modifyAngle(imageResized);
    cv::cvtColor(imageResized, imageGray, CV_BGR2GRAY);
    if ( !imageGray.data )
    {
        printf("No image data \n");
        return cv::Mat();
    }
    cv::Mat imageBinary;
    cv::adaptiveThreshold(~imageGray, imageBinary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, -10);
    return imageBinary;
}

cv::Mat ReadTableData::getTableMask(cv::Mat &imageBinary) {
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

    return drawTableLinesOnBlack(imageBinary, tableLines, THRESHOLD_SAME_LINE_RADIAN);
}

std::vector<ReadTableData::Cell> ReadTableData::createCells(const std::vector<std::vector<cv::Point>> &contours, const cv::Mat &imageBinary, const cv::Mat &imageResized) {
    auto size = contours.size(); 
    std::vector<Cell> cells;
    cells.reserve(size);

    for (size_t i = 0; i < size; i++) {
        // find the area of each contour
        double area = contourArea(contours[i]);

        printf("%lf\n", area);
        if(50000< area) {
            continue;
        }

        std::vector<cv::Point> contours_poly;
        approxPolyDP( cv::Mat(contours[i]), contours_poly, 0.01 * cv::arcLength(contours[i], true), true );
        auto rect = boundingRect( cv::Mat(contours_poly) );

        // find the number of joints that each table has
        // auto roi = joints(rect);
        //
        // std::vector<std::vector<cv::Point> > joints_contours;
        // findContours(roi, joints_contours, cv::RETR_CCOMP, cv::CHAIN_APPROX_SIMPLE);
        //
        // printf("%ld\n", joints_contours.size());

        // cv::Mat img = cv::Mat(imageBinary, rect);
        rect.x += rect.width * (1.0f - RATIO_CELL_SIZE) / 2;
        rect.y += rect.height * (1.0f - RATIO_CELL_SIZE) / 2;
        rect.width *= RATIO_CELL_SIZE;
        rect.height *= RATIO_CELL_SIZE;

        Cell cell(rect.x, rect.y, imageBinary(rect).clone());
        cells.push_back(std::move(cell));

        cv::rectangle(imageResized, cv::Point(rect.x, rect.y), cv::Point(rect.x+rect.width, rect.y+rect.height), cv::Scalar(0,0,255), 1);
    }

    cv::imshow("image", imageResized);
    cv::waitKey();

    return cells;
}

void ReadTableData::displayImage(std::string name, cv::Mat &image) const {
    cv::imshow(name, image);
    cv::waitKey(0);
}

void ReadTableData::readCells(std::vector<Cell> &cells) const {
    tesseract::TessBaseAPI *api = new tesseract::TessBaseAPI();
    if (api->Init(nullptr, "eng")) {
        fprintf(stderr, "Could not initialize tesseract.\n");
        exit(1);
    }
    // api->SetVariable("tessedit_char_whitelist", "0123456789");

    std::ofstream ofs("data.csv");

    auto groups = sortCells(cells);
    for (auto &group : groups) {
        for (auto &cell : group) {
            // printf("cell x: %d, y: %d\n", cell.x, cell.y);
            auto num = getNumberFromCell(cell, api);
            ofs << num << ',';
        }
        ofs << std::endl;
    }

    api->End();
}

cv::Mat ReadTableData::getVerticalMask(cv::Mat &imageBinary) {
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

cv::Mat ReadTableData::getHorizontalMask(cv::Mat &imageBinary) {
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

cv::Mat ReadTableData::getImageByCanny(cv::Mat &image) {
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

bool ReadTableData::isLineHorizontal(float theta, float dTheta) const {
    return M_PI / 2 - dTheta <= theta && theta <= M_PI / 2 + dTheta;
}

bool ReadTableData::isLineVertical(float theta, float dTheta) const {
    return theta <= dTheta || M_PI - dTheta < theta;
}

bool ReadTableData::isTableLine(float theta, float dTheta) const {
    return isLineHorizontal(theta, dTheta) || isLineVertical(theta, dTheta);
}

float ReadTableData::translateTheta(float theta) const {
    return M_PI - THRESHOLD_SAME_LINE_RADIAN <= theta ? theta - M_PI : theta;
}

bool ReadTableData::isSimilarLineHorizontal(float rho, float theta, float dRho, float dTheta, AverageBuilder &entry) const {
    return rho - dRho < entry._rho && entry._rho < rho + dRho
        && theta - dTheta < entry._theta && entry._theta < theta + dTheta;
}

bool ReadTableData::isSimilarLineVertical(float rho, float theta, float dRho, float dTheta, AverageBuilder &entry) const {
    auto _rho = fabs(rho);
    auto _theta = translateTheta(theta);
    return _rho - dRho < entry._rho && entry._rho < _rho + dRho
        && _theta - dTheta < entry._theta && entry._theta < _theta + dTheta;
}

std::vector<cv::Vec2f> ReadTableData::extractTableLines(std::vector<cv::Vec2f> &lines, float dRho, float dTheta) {
    std::vector<AverageBuilder> linesUnited;
    for (auto &line : lines) {
        // printf("raw lines: %f, %f\n", line[0], line[1]);
        auto rho = fabs(line[0]);
        auto theta = translateTheta(line[1]);
        if (!isTableLine(theta)) {
            continue;
        }

        auto it = std::find_if(linesUnited.begin(), linesUnited.end(),
            [this, rho, theta, dRho, dTheta](AverageBuilder &entry) {
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

cv::Mat ReadTableData::drawTableLinesOnBlack(cv::Mat &image, std::vector<cv::Vec2f> &lines, float dTheta, int thickness) const {
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

std::vector<std::vector<ReadTableData::Cell>> ReadTableData::sortCells(std::vector<Cell> &cells) const {
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

int ReadTableData::getNumberFromCell(Cell &cell, tesseract::TessBaseAPI *api) const {
    auto img = cell.data;

    erode(img, img,  getStructuringElement(cv::MORPH_RECT, cv::Size(2,2)));
    dilate(img, img, getStructuringElement(cv::MORPH_RECT, cv::Size(2,2)));
    cv::imwrite("output.jpg", img);

    std::vector<cv::Vec2f> lines;
    auto threshold = static_cast<int>(std::min(img.rows, img.cols)*0.7f);
    cv::HoughLines(img, lines, 1, RADIAN_PER_DEGREE, threshold);

    auto imageLines = drawTableLinesOnBlack(img, lines, 0.1f, 3);
    cv::imwrite("output_line.jpg", imageLines);

    img = img - imageLines;
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

    // cv::imshow("roi", img);
    // cv::waitKey();
    return num;
}

cv::Mat ReadTableData::modifyAngle(cv::Mat &image) {
    cv::Mat imageBinary;
    cv::cvtColor(image, imageBinary, CV_BGR2GRAY);
    cv::adaptiveThreshold(~imageBinary, imageBinary, 255, CV_ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY, 15, -10);

    auto edges = getImageByCanny(imageBinary);

    std::vector<cv::Vec2f> lines;
    cv::HoughLines(edges, lines, 1, RADIAN_PER_DEGREE/10, LENGTH_FOR_ANGLE_MODIFICATION);
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

    auto center = cv::Point(imageBinary.cols / 2, imageBinary.rows / 2);
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

