#include "ROIDetector.hpp"
#include <opencv2/opencv.hpp>
#include <iostream>
#include <algorithm>
#include <set>
#include <sstream>

using namespace cv;
using namespace std;

namespace {

cv::Rect rectPct(const cv::Mat& img, float x, float y, float w, float h) {
    int X = static_cast<int>(x * img.cols);
    int Y = static_cast<int>(y * img.rows);
    int W = static_cast<int>(w * img.cols);
    int H = static_cast<int>(h * img.rows);
    return cv::Rect(X, Y, W, H);
}

// Basit grid okuyucu (T.C. kimlik, numara gibi alanlar iÇõin)
std::string detectOMRGrid(const cv::Mat& roiGray,
                          int rows,
                          int cols,
                          double fillThreshold,
                          char firstLabel = 'A') {
    cv::Mat blurImg, thr;
    cv::GaussianBlur(roiGray, blurImg, cv::Size(3, 3), 0);
    cv::adaptiveThreshold(blurImg, thr, 255,
                          cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY_INV, 21, 7);

    int cellH = roiGray.rows / rows;
    int cellW = roiGray.cols / cols;

    std::ostringstream oss;

    for (int r = 0; r < rows; ++r) {
        int bestCol = -1;
        double bestVal = 0.0;

        for (int c = 0; c < cols; ++c) {
            int x = c * cellW;
            int y = r * cellH;
            cv::Rect cell(x, y, cellW, cellH);
            cv::Mat sub = thr(cell);

            int marginX = std::max(1, cellW / 10);
            int marginY = std::max(1, cellH / 10);
            cv::Rect inner(marginX, marginY,
                           std::max(1, cellW - 2 * marginX),
                           std::max(1, cellH - 2 * marginY));
            inner &= cv::Rect(0, 0, sub.cols, sub.rows);

            cv::Mat innerCell = sub(inner);
            double filled = static_cast<double>(cv::countNonZero(innerCell)) /
                            static_cast<double>(innerCell.total());
            if (filled > bestVal) {
                bestVal = filled;
                bestCol = c;
            }
        }

        char mark = '-';
        if (bestVal >= fillThreshold && bestCol >= 0) {
            mark = static_cast<char>(firstLabel + bestCol);
        }

        if (r > 0) oss << ",";
        oss << mark;
    }

    return oss.str();
}

// Tek sÇ¬tunlu iYaretlemeler iÇõin (oturum vb.)
std::string detectSingleColumn(const cv::Mat& roiGray,
                               int rows,
                               double fillThreshold) {
    cv::Mat blurImg, thr;
    cv::GaussianBlur(roiGray, blurImg, cv::Size(3, 3), 0);
    cv::adaptiveThreshold(blurImg, thr, 255,
                          cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY_INV, 21, 7);

    int cellH = roiGray.rows / rows;
    int cellW = roiGray.cols;

    int bestIdx = -1;
    double bestVal = 0.0;

    for (int r = 0; r < rows; ++r) {
        cv::Rect cell(0, r * cellH, cellW, cellH);
        cv::Mat sub = thr(cell);

        int marginY = std::max(1, cellH / 10);
        cv::Rect inner(0, marginY, cellW, std::max(1, cellH - 2 * marginY));
        inner &= cv::Rect(0, 0, sub.cols, sub.rows);

        cv::Mat innerCell = sub(inner);
        double filled = static_cast<double>(cv::countNonZero(innerCell)) /
                        static_cast<double>(innerCell.total());
        if (filled > bestVal) {
            bestVal = filled;
            bestIdx = r;
        }
    }

    if (bestVal < fillThreshold || bestIdx < 0)
        return "-";

    return std::to_string(bestIdx);
}

} // namespace

ROIDetector::ROIDetector()
    : fillThreshold_(0.20),
      bubbleDetector_(fillThreshold_),
      debugMode_(false) {
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

void ROIDetector::setFillThreshold(double threshold) {
    fillThreshold_ = threshold;
    bubbleDetector_.setFillThreshold(threshold);
}

void ROIDetector::setDebugMode(bool enabled) {
    debugMode_ = enabled;
}

cv::Mat ROIDetector::getLastDebugVisualization() const {
    if (lastDebugVis_.empty()) return cv::Mat();
    return lastDebugVis_.clone();
}

bool ROIDetector::isSubjectRegion(const std::string& name) const {
    static const std::set<std::string> subjects = {
        "turkce", "sosyal", "din", "ingilizce", "matematik", "fen"
    };
    return subjects.find(name) != subjects.end();
}

std::string ROIDetector::bubblesToAnswerString(const std::vector<BubbleResult>& results) const {
    std::ostringstream oss;
    for (size_t i = 0; i < results.size(); ++i) {
        char mark = '-';
        const auto& br = results[i];

        if (!br.isValid) {
            mark = 'X';
        } else if (!br.markedAnswer.empty() && br.markedAnswer != "-") {
            mark = br.markedAnswer[0];
        }

        if (i > 0) oss << ",";
        oss << mark;
    }
    return oss.str();
}

std::map<std::string, std::string>
ROIDetector::process(const cv::Mat& warped, cv::Mat& debugOut) {
    CV_Assert(!warped.empty());

    cv::Mat gray;
    if (warped.channels() == 3)
        cv::cvtColor(warped, gray, cv::COLOR_BGR2GRAY);
    else
        gray = warped.clone();

    if (warped.channels() == 3)
        lastDebugVis_ = warped.clone();
    else
        cv::cvtColor(warped, lastDebugVis_, cv::COLOR_GRAY2BGR);

    std::map<std::string, std::string> out;

    for (const auto& reg : regions_) {
        cv::Rect roi = rectPct(gray, reg.rectPct[0], reg.rectPct[1],
                               reg.rectPct[2], reg.rectPct[3]);
        roi &= cv::Rect(0, 0, gray.cols, gray.rows);
        if (roi.width <= 0 || roi.height <= 0) continue;

        cv::Mat sub = gray(roi).clone();

        std::string val;
        if (reg.type == GRID && isSubjectRegion(reg.name)) {
            auto bubbles = bubbleDetector_.detectBubbles(sub, reg.rows, reg.cols, 1, 'A');
            val = bubblesToAnswerString(bubbles);

            if (debugMode_) {
                bubbleDetector_.drawBubbleDebug(lastDebugVis_, roi, bubbles, reg.rows, reg.cols, reg.name);
            }
        } else if (reg.type == GRID) {
            val = detectOMRGrid(sub, reg.rows, reg.cols, fillThreshold_);
        } else {
            val = detectSingleColumn(sub, reg.rows, fillThreshold_);
        }

        out[reg.name] = val;

        cv::rectangle(lastDebugVis_, roi, cv::Scalar(0, 255, 0), 2);
        cv::putText(lastDebugVis_, reg.name, roi.tl() + cv::Point(4, 16),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
    }

    debugOut = lastDebugVis_.clone();
    return out;
}

std::vector<ROIDetector::QuestionDetail>
ROIDetector::analyzeGridWithDetails(
    const cv::Mat& roiGray,
    int rows,
    int cols,
    const std::map<int, char>& correctAnswers,
    std::vector<BubbleResult>* bubbleResults,
    char firstLabel) {

    std::vector<BubbleResult> localResults = bubbleDetector_.detectBubbles(
        roiGray, rows, cols, 1, firstLabel);
    if (bubbleResults) {
        *bubbleResults = localResults;
    }

    std::vector<QuestionDetail> details;
    details.reserve(localResults.size());

    for (size_t i = 0; i < localResults.size(); ++i) {
        const auto& br = localResults[i];

        QuestionDetail qd;
        qd.questionNumber = static_cast<int>(i);
        qd.fillRatio = br.confidence;

        if (!br.isValid) {
            qd.markedAnswer = 'X';
        } else if (br.markedAnswer.empty() || br.markedAnswer == "-") {
            qd.markedAnswer = '-';
        } else {
            qd.markedAnswer = br.markedAnswer[0];
        }

        auto it = correctAnswers.find(static_cast<int>(i));
        qd.correctAnswer = (it != correctAnswers.end()) ? it->second : '-';
        qd.isCorrect = (qd.markedAnswer == qd.correctAnswer &&
                        qd.markedAnswer != '-' &&
                        qd.markedAnswer != 'X');

        details.push_back(qd);
    }

    return details;
}

std::map<std::string, std::vector<ROIDetector::QuestionDetail>>
ROIDetector::processWithDetails(
    const cv::Mat& warped,
    cv::Mat& debugOut,
    const std::map<std::string, std::map<int, char>>& answerKey) {

    CV_Assert(!warped.empty());
    cv::Mat gray;
    if (warped.channels() == 3)
        cv::cvtColor(warped, gray, cv::COLOR_BGR2GRAY);
    else
        gray = warped.clone();

    if (warped.channels() == 3)
        lastDebugVis_ = warped.clone();
    else
        cv::cvtColor(warped, lastDebugVis_, cv::COLOR_GRAY2BGR);

    std::map<std::string, std::vector<QuestionDetail>> allDetails;

    for (const auto& reg : regions_) {
        if (reg.type != GRID || !isSubjectRegion(reg.name))
            continue;

        cv::Rect roi = rectPct(gray, reg.rectPct[0], reg.rectPct[1],
                               reg.rectPct[2], reg.rectPct[3]);
        roi &= cv::Rect(0, 0, gray.cols, gray.rows);
        if (roi.width <= 0 || roi.height <= 0) continue;

        cv::Mat sub = gray(roi).clone();

        std::map<int, char> subjectAnswers;
        auto it = answerKey.find(reg.name);
        if (it != answerKey.end()) {
            subjectAnswers = it->second;
        }

        std::vector<BubbleResult> bubbleResults;
        std::vector<QuestionDetail> details = analyzeGridWithDetails(
            sub, reg.rows, reg.cols, subjectAnswers, &bubbleResults);
        allDetails[reg.name] = details;

        if (debugMode_) {
            bubbleDetector_.drawBubbleDebug(lastDebugVis_, roi, bubbleResults, reg.rows, reg.cols, reg.name);
            cv::rectangle(lastDebugVis_, roi, cv::Scalar(0, 255, 0), 2);
            cv::putText(lastDebugVis_, reg.name, roi.tl() + cv::Point(4, 16),
                        cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
        }
    }

    debugOut = lastDebugVis_.clone();
    return allDetails;
}
