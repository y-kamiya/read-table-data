#pragma once

#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <tesseract/baseapi.h>
#include <leptonica/allheaders.h>

class ReadTableData {
private:
    static const int SCALE_FOR_MASK;
    static const int THRESHOLD_SAME_ROW;
    static const int LENGTH_FOR_ANGLE_MODIFICATION;
    static const float CELL_SIZE;
    static const float RATIO_CELL_SIZE;
    static const float THRESHOLD_SAME_LINE_RADIAN;
    static const float THRESHOLD_SAME_LINE_RHO;
    static const float RATIO_IMAGE_SIZE;
    static const float RADIAN_PER_DEGREE;

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

    struct Cell {
        int x;
        int y;
        cv::Mat data;

        Cell() {}
        Cell(int x, int y, cv::Mat data): x(x), y(y), data(data) {
        }

    };

public:
    void execute(const std::string &imageName);

private:
    void readCells(std::vector<Cell> &cells) const;
    cv::Mat getImageResized(const std::string &imageName) const;
    cv::Mat getImageBinary(const std::string &imageName);
    cv::Mat getTableMask(cv::Mat &imageBinary);

    std::vector<Cell> createCells(const std::vector<std::vector<cv::Point>> &contours, const cv::Mat &imageBinary, const cv::Mat &imageResized);

    cv::Mat getVerticalMask(cv::Mat &imageBinary);
    cv::Mat getHorizontalMask(cv::Mat &imageBinary);
    cv::Mat getImageByCanny(cv::Mat &imageBinary);
    bool isLineHorizontal(float theta, float dTheta = THRESHOLD_SAME_LINE_RADIAN) const;
    bool isLineVertical(float theta, float dTheta = THRESHOLD_SAME_LINE_RADIAN) const;
    bool isTableLine(float theta, float dTheta = THRESHOLD_SAME_LINE_RADIAN) const;
    float translateTheta(float theta) const;

    bool isSimilarLineHorizontal(float rho, float theta, float dRho, float dTheta, AverageBuilder &entry) const;
    bool isSimilarLineVertical(float rho, float theta, float dRho, float dTheta, AverageBuilder &entry) const;

    std::vector<cv::Vec2f> extractTableLines(std::vector<cv::Vec2f> &lines, float dRho, float dTheta);
    cv::Mat drawTableLinesOnBlack(cv::Mat &image, std::vector<cv::Vec2f> &lines, float dTheta, int thickness = 1) const;

    // readCels
    std::vector<std::vector<Cell>> sortCells(std::vector<Cell> &cells) const;
    int getNumberFromCell(Cell &cell, tesseract::TessBaseAPI *api) const;

    cv::Mat modifyAngle(cv::Mat &image);

    void displayImage(std::string name, cv::Mat &image) const;
};
