#include "ROIDetector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>

using namespace cv;
using namespace std;

static cv::Rect rectPct(const cv::Mat &img,
                        float x, float y, float w, float h)
{
    int X = int(x * img.cols);
    int Y = int(y * img.rows);
    int W = int(w * img.cols);
    int H = int(h * img.rows);
    return cv::Rect(X, Y, W, H);
}

static std::string detectOMRGrid(const cv::Mat &roiGray,
                                 int rows,
                                 int cols,
                                 char firstLabel = 'A')
{
    cv::Mat blurImg, thr;
    cv::GaussianBlur(roiGray, blurImg, Size(3,3), 0);
    cv::adaptiveThreshold(blurImg, thr, 255,
                          ADAPTIVE_THRESH_MEAN_C,
                          THRESH_BINARY_INV, 21, 7);

    int cellH = roiGray.rows / rows;
    int cellW = roiGray.cols / cols;

    std::string result;
    result.reserve(rows * 2);

    for (int r = 0; r < rows; ++r) {
        int bestCol = -1;
        double bestVal = 0.0;

        for (int c = 0; c < cols; ++c) {
            int x = c * cellW;
            int y = r * cellH;
            cv::Rect cell(x, y, cellW, cellH);
            cv::Mat sub = thr(cell);

            double filled = (double)cv::countNonZero(sub) / (cell.area());
            if (filled > bestVal) {
                bestVal = filled;
                bestCol = c;
            }
        }

        if (bestVal < 0.20) {
            result += "-"; 
        } else {
            char mark = char(firstLabel + bestCol);
            result += mark;
        }

        if (r != rows - 1)
            result += ",";
    }

    return result;
}

static std::string detectSingleColumn(const cv::Mat &roiGray, int rows)
{
    cv::Mat blurImg, thr;
    cv::GaussianBlur(roiGray, blurImg, Size(3,3), 0);
    cv::adaptiveThreshold(blurImg, thr, 255,
                          ADAPTIVE_THRESH_MEAN_C,
                          THRESH_BINARY_INV, 21, 7);

    int cellH = roiGray.rows / rows;
    int cellW = roiGray.cols;

    int bestIdx = -1;
    double bestVal = 0.0;

    for (int r = 0; r < rows; ++r) {
        cv::Rect cell(0, r * cellH, cellW, cellH);
        cv::Mat sub = thr(cell);
        double filled = (double)cv::countNonZero(sub) / (cell.area());
        if (filled > bestVal) {
            bestVal = filled;
            bestIdx = r;
        }
    }

    if (bestVal < 0.20)
        return "-";

    return std::to_string(bestIdx); 
}

BubbleDetector::BubbleDetector()
{
 
    regions_.push_back({
        "tc_kimlik",
       {0.000f, 0.245f, 0.273f, 0.213f},
        11, 10, GRID 
    });

    regions_.push_back({
        "ogrenci_no",
        {0.279f, 0.245f, 0.136f, 0.210f},
        10, 10, GRID 
    });

    regions_.push_back({
        "adi_soyadi",
        {0.000f, 0.459f, 0.507f, 0.541f},
        30, 12, GRID 
    });

    regions_.push_back({
        "turkce",
        {0.53f, 0.23f, 0.12f, 0.37f},
        20, 5, GRID 
    });

    regions_.push_back({
        "sosyal",
        {0.65f, 0.23f, 0.12f, 0.37f},
        20, 5, GRID 
    });

    regions_.push_back({
        "din",
        {0.77f, 0.23f, 0.12f, 0.37f},
        20, 5, GRID 
    });

    regions_.push_back({
        "ingilizce",
        {0.89f, 0.23f, 0.12f, 0.37f},
        20, 5, GRID 
    });

    regions_.push_back({
        "matematik",
        {0.64f, 0.62f, 0.12f, 0.36f},
        20, 5, GRID 
    });

    regions_.push_back({
        "fen",
        {0.76f, 0.62f, 0.12f, 0.36f},
        20, 5, GRID 
    });
}

std::map<std::string, std::string>
BubbleDetector::process(const cv::Mat &warped, cv::Mat &debugOut)
{
    CV_Assert(!warped.empty());
    cv::Mat gray;
    if (warped.channels() == 3)
        cv::cvtColor(warped, gray, COLOR_BGR2GRAY);
    else
        gray = warped.clone();

    debugOut = warped.clone();

    std::map<std::string, std::string> out;

    for (const auto &reg : regions_) { 
        cv::Rect roi = rectPct(gray, reg.rectPct[0], reg.rectPct[1],
                               reg.rectPct[2], reg.rectPct[3]);

        roi &= cv::Rect(0,0, gray.cols, gray.rows);
        cv::Mat sub = gray(roi).clone();

        std::string val;
        if (reg.type == GRID) {
            val = detectOMRGrid(sub, reg.rows, reg.cols);
        } else { 
            val = detectSingleColumn(sub, reg.rows);
        }

        out[reg.name] = val;

        cv::rectangle(debugOut, roi, cv::Scalar(0,255,0), 2);
        cv::putText(debugOut, reg.name, roi.tl() + Point(3,15),
                    FONT_HERSHEY_SIMPLEX, 0.45, Scalar(0,0,0), 1);
    }

    return out;
}