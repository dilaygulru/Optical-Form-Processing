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

static cv::Mat preprocessForFill(const cv::Mat& roiGray) {
    cv::Mat blurImg, thr;
    cv::GaussianBlur(roiGray, blurImg, cv::Size(5, 5), 0);

    // Daha agresif adaptive threshold
    cv::adaptiveThreshold(
        blurImg, thr, 255,
        cv::ADAPTIVE_THRESH_GAUSSIAN_C,
        cv::THRESH_BINARY_INV,
        15, 3
    );

    cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(2, 2));
    cv::morphologyEx(thr, thr, cv::MORPH_OPEN, kernel);

    return thr;
}

static double cellFillRatio(const cv::Mat& bin, const cv::Rect& cell) {
    cv::Rect c = cell & cv::Rect(0, 0, bin.cols, bin.rows);
    if (c.width <= 0 || c.height <= 0) return 0.0;

    cv::Mat sub = bin(c);

    int marginX = std::max(2, c.width / 5);
    int marginY = std::max(2, c.height / 5);
    cv::Rect inner(
        marginX, marginY,
        std::max(1, c.width - 2 * marginX),
        std::max(1, c.height - 2 * marginY)
    );
    inner &= cv::Rect(0, 0, sub.cols, sub.rows);

    cv::Mat innerCell = sub(inner);
    return static_cast<double>(cv::countNonZero(innerCell)) /
           static_cast<double>(innerCell.total());
}

static std::string detectGridByColumns_AsDigits_SkipFirstRow(
    const cv::Mat& roiGray,
    int rows, int cols,
    double fillThreshold
) {
    cv::Mat thr = preprocessForFill(roiGray);

    int cellH = roiGray.rows / rows;
    int cellW = roiGray.cols / cols;

    std::string out;
    out.reserve(cols);

    for (int c = 0; c < cols; ++c) {
        int bestRow = -1;
        double bestVal = 0.0;
        double secondVal = 0.0;
        for (int r = 0; r < rows; ++r) {
            cv::Rect cell(c * cellW, r * cellH, cellW, cellH);
            double filled = cellFillRatio(thr, cell);

            if (filled > bestVal) {
                secondVal = bestVal;
                bestVal = filled;
                bestRow = r;
            } else if (filled > secondVal) {
                secondVal = filled;
            }
        }

        if (bestVal < fillThreshold || bestRow < 0) {
            out.push_back('-');
            continue;
        }
        if (secondVal >= fillThreshold * 0.6 && (bestVal - secondVal) < 0.08) {
            out.push_back('-');
            continue;
        }

        int digit = bestRow; 
        if (digit < 0 || digit > 9) {
            out.push_back('-');
            continue;
        }

        out.push_back(static_cast<char>('0' + digit));
    }

    return out;
}

static std::string detectGridByColumns_AsLetters_SkipFirstRow(
    const cv::Mat& roiGray,
    int rows, int cols,
    double fillThreshold,
    const std::string& alphabet
) {
    cv::Mat thr = preprocessForFill(roiGray);

    int cellH = roiGray.rows / rows;
    int cellW = roiGray.cols / cols;

    std::string out;
    out.reserve(cols);

    for (int c = 0; c < cols; ++c) {
        int bestRow = -1;
        double bestVal = 0.0;
        double secondVal = 0.0;

        for (int r = 0; r < rows; ++r) {
            cv::Rect cell(c * cellW, r * cellH, cellW, cellH);
            double filled = cellFillRatio(thr, cell);

            if (filled > bestVal) {
                secondVal = bestVal;
                bestVal = filled;
                bestRow = r;
            } else if (filled > secondVal) {
                secondVal = filled;
            }
        }

        if (bestVal < fillThreshold || bestRow < 0) {
            out.push_back('-');
            continue;
        }

        if (secondVal >= fillThreshold * 0.6 && (bestVal - secondVal) < 0.08) {
            out.push_back('?');
            continue;
        }

        int alphaIndex = bestRow; 
        if (alphaIndex >= 0 && alphaIndex < (int)alphabet.size())
            out.push_back(alphabet[alphaIndex]);
        else
            out.push_back('-');
    }

    return out;
}


static std::string detectSingleColumn(const cv::Mat& roiGray, int rows, double fillThreshold) {
    cv::Mat thr = preprocessForFill(roiGray);

    int cellH = roiGray.rows / rows;
    int cellW = roiGray.cols;

    int bestIdx = -1;
    double bestVal = 0.0;

    for (int r = 0; r < rows; ++r) {
        cv::Rect cell(0, r * cellH, cellW, cellH);
        double filled = cellFillRatio(thr, cell);
        if (filled > bestVal) {
            bestVal = filled;
            bestIdx = r;
        }
    }

    if (bestVal < fillThreshold || bestIdx < 0) return "-";
    return std::to_string(bestIdx);
}

} 

ROIDetector::ROIDetector()
    : fillThreshold_(0.15),
      bubbleDetector_(fillThreshold_),
      debugMode_(false) {

    regions_.push_back({
        "tc_kimlik",
        {0.0065f, 0.283f, 0.271f, 0.172f},
        10, 11, GRID
    });

    regions_.push_back({
        "ogrenci_no",
        {0.287f, 0.283f, 0.123f, 0.172f},
        10, 5, GRID
    });

    regions_.push_back({
        "adi_soyadi",
        {0.000f, 0.490f, 0.520f, 0.495f},
        29, 21, GRID
    });

    regions_.push_back({ "turkce",    {0.525f, 0.263f, 0.125f, 0.345f}, 20, 4, GRID });
    regions_.push_back({ "sosyal",    {0.640f, 0.263f, 0.125f, 0.345f}, 20, 4, GRID });
    regions_.push_back({ "din",       {0.755f, 0.263f, 0.125f, 0.345f}, 20, 4, GRID });
    regions_.push_back({ "ingilizce", {0.874f, 0.263f, 0.125f, 0.345f}, 20, 4, GRID });

    regions_.push_back({ "matematik", {0.641f, 0.640f, 0.125f, 0.345f}, 20, 4, GRID });
    regions_.push_back({ "fen",       {0.757f, 0.640f, 0.125f, 0.345f}, 20, 4, GRID });
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
    
    const double CONFIDENCE_THRESHOLD = 60.0; 

    for (size_t i = 0; i < results.size(); ++i) {
        char mark = '-'; 
        const auto& br = results[i];

        if (!br.isValid) {
            mark = 'X'; 
        } 
        else {
            if (br.confidence < CONFIDENCE_THRESHOLD) {
                mark = '-'; 
            }
            
            else if (br.secondConfidence > CONFIDENCE_THRESHOLD) {
                 mark = '-'; 
            }

            else if (!br.markedAnswer.empty()) {
                mark = br.markedAnswer[0];
            }
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

    double idThr = std::clamp(fillThreshold_ * 1.2, 0.25, 0.45);

    for (const auto& reg : regions_) {
        cv::Rect roi = rectPct(gray, reg.rectPct[0], reg.rectPct[1],
                               reg.rectPct[2], reg.rectPct[3]);

        if (reg.type == GRID) {
            int dx = std::max(1, static_cast<int>(roi.width * 0.02));
            int dy = std::max(1, static_cast<int>(roi.height * 0.02));
            roi = cv::Rect(roi.x + dx, roi.y + dy,
                           std::max(1, roi.width - 2 * dx),
                           std::max(1, roi.height - 2 * dy));
        }

        if (isSubjectRegion(reg.name)) {

            double questionOffsetRatio = 0.19; 
            double widthScaleFactor = 0.95;    

            int originalW = roi.width;
            
            int offsetX = static_cast<int>(originalW * questionOffsetRatio);
            
            int targetTotalW = static_cast<int>(originalW * widthScaleFactor);
            int newWidth = std::max(1, targetTotalW - offsetX);

            roi.x += offsetX;
            roi.width = newWidth;
        }

        roi &= cv::Rect(0, 0, gray.cols, gray.rows);
        if (roi.width <= 0 || roi.height <= 0) continue;

        cv::Mat sub = gray(roi).clone();
        std::string val;

        if (reg.type == GRID && isSubjectRegion(reg.name)) {
            
            auto bubbles = bubbleDetector_.detectBubblesWithContours(sub, reg.rows, reg.cols, 1, 'A');
            val = bubblesToAnswerString(bubbles);

            if (debugMode_) {
                bubbleDetector_.drawBubbleDebug(lastDebugVis_, roi, bubbles, reg.rows, reg.cols, reg.name);
            }
        }
        else if (reg.name == "tc_kimlik") { 
            
            std::string resultString = "";
            int rows = reg.rows; 
            int cols = reg.cols; 
            
            int cellW = sub.cols / cols;
            int cellH = sub.rows / rows;

            cv::Mat workingImg = sub.clone();

            cv::GaussianBlur(workingImg, workingImg, cv::Size(5, 5), 0);
            
            cv::Mat adaptiveBin;
            cv::adaptiveThreshold(workingImg, adaptiveBin, 255, cv::ADAPTIVE_THRESH_GAUSSIAN_C, cv::THRESH_BINARY_INV, 21, 15); 

            cv::Mat globalBin;
            cv::threshold(workingImg, globalBin, 160, 255, cv::THRESH_BINARY_INV);

            cv::Mat finalBin;
            cv::bitwise_and(adaptiveBin, globalBin, finalBin);
            
            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(finalBin, finalBin, cv::MORPH_OPEN, kernel);

            for (int c = 0; c < cols; ++c) {
                
                double bestVal = 0.0;
                int bestRow = -1;

                for (int r = 0; r < rows; ++r) {
                    
                    int marginX = static_cast<int>(cellW * 0.30);
                    int marginY = static_cast<int>(cellH * 0.30);
                    
                    cv::Rect cell(c * cellW + marginX, r * cellH + marginY, 
                                  cellW - 2*marginX, cellH - 2*marginY);
                    
                    cell &= cv::Rect(0, 0, finalBin.cols, finalBin.rows);
                    if (cell.width <= 0 || cell.height <= 0) continue;

                    cv::Mat binCell = finalBin(cell);
                    
                    double ratio = (double)cv::countNonZero(binCell) / (cell.width * cell.height);

                    if (ratio > bestVal) {
                        bestVal = ratio;
                        bestRow = r;
                    }

                    if (debugMode_) {
                        int centerX = roi.x + (c * cellW) + (cellW / 2);
                        int centerY = roi.y + (r * cellH) + (cellH / 2);
                        int radius = std::min(cellW, cellH) * 0.35;

                        cv::circle(lastDebugVis_, cv::Point(centerX, centerY), radius, 
                                   cv::Scalar(100, 100, 100), 1, cv::LINE_AA);

                        if (ratio > 0.05) {
                            std::string scoreTxt = std::to_string((int)(ratio * 100));
                            cv::putText(lastDebugVis_, scoreTxt,
                                       cv::Point(centerX - 10, centerY + 5), 
                                       cv::FONT_HERSHEY_DUPLEX, 0.40, cv::Scalar(0, 255, 0), 1);
                        }
                    }
                }

                double THRESHOLD = 0.30; 
                char detectedChar = '-'; 
                
                if (bestVal > THRESHOLD && bestRow != -1) {
                    detectedChar = '0' + bestRow;

                    if (debugMode_) {
                        cv::Rect finalCell(roi.x + c * cellW, roi.y + bestRow * cellH, cellW, cellH);
                        
                        cv::rectangle(lastDebugVis_, finalCell, cv::Scalar(0, 255, 0), 2);
                        
                        std::string charTxt(1, detectedChar);
                        cv::putText(lastDebugVis_, charTxt,
                                   cv::Point(finalCell.x + 5, finalCell.y + finalCell.height - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 255, 0), 2);
                    }
                }
                
                resultString += detectedChar;
            }
            
            val = resultString;
        }
        else if (reg.name == "ogrenci_no") { 
            
            std::string resultString = "";
            int rows = reg.rows; 
            int cols = reg.cols; 
            
            int cellW = sub.cols / cols;
            int cellH = sub.rows / rows;

            cv::Mat workingImg = sub.clone();

            cv::GaussianBlur(workingImg, workingImg, cv::Size(7, 7), 0);
            
            cv::Mat adaptiveBin;
            cv::adaptiveThreshold(workingImg, adaptiveBin, 255,
                                  cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv::THRESH_BINARY_INV,
                                  25, 20); 

            cv::Mat globalBin;
            cv::threshold(workingImg, globalBin, 180, 255, cv::THRESH_BINARY_INV);

            cv::Mat finalBin;
            cv::bitwise_and(adaptiveBin, globalBin, finalBin);
            
            cv::Mat kernel_morph = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(finalBin, finalBin, cv::MORPH_OPEN, kernel_morph);

            cv::Mat kernel_erode = cv::getStructuringElement(cv::MORPH_RECT, cv::Size(2, 2));
            cv::erode(finalBin, finalBin, kernel_erode, cv::Point(-1, -1), 1);

            for (int c = 0; c < cols; ++c) {
                
                double bestVal = 0.0;
                int bestRow = -1;

                for (int r = 0; r < rows; ++r) {
                    
                    int marginX = static_cast<int>(cellW * 0.30);
                    int marginY = static_cast<int>(cellH * 0.30);
                    
                    cv::Rect cell(c * cellW + marginX, r * cellH + marginY, 
                                  cellW - 2*marginX, cellH - 2*marginY);
                    
                    cell &= cv::Rect(0, 0, finalBin.cols, finalBin.rows);
                    if (cell.width <= 0 || cell.height <= 0) continue;

                    cv::Mat binCell = finalBin(cell);
                    
                    double ratio = (double)cv::countNonZero(binCell) / (cell.width * cell.height);

                    if (ratio > bestVal) {
                        bestVal = ratio;
                        bestRow = r;
                    }

                    if (debugMode_) {
                        int centerX = roi.x + (c * cellW) + (cellW / 2);
                        int centerY = roi.y + (r * cellH) + (cellH / 2);
                        int radius = std::min(cellW, cellH) * 0.35;

                        cv::circle(lastDebugVis_, cv::Point(centerX, centerY), radius, 
                                   cv::Scalar(100, 100, 100), 1, cv::LINE_AA);
                        
                        if (ratio > 0.05) {
                            std::string scoreTxt = std::to_string((int)(ratio * 100));
                            cv::putText(lastDebugVis_, scoreTxt,
                                       cv::Point(centerX - 10, centerY + 5), 
                                       cv::FONT_HERSHEY_DUPLEX, 0.40, cv::Scalar(0, 255, 0), 1);
                        }
                    }
                }

                double THRESHOLD = 0.10; 
                char detectedChar = '-'; 
                
                if (bestVal > THRESHOLD && bestRow != -1) {
                    detectedChar = '0' + bestRow;

                    if (debugMode_) {
                        cv::Rect finalCell(roi.x + c * cellW, roi.y + bestRow * cellH, cellW, cellH);
                        
                        cv::rectangle(lastDebugVis_, finalCell, cv::Scalar(0, 255, 0), 2);
                        
                        std::string charTxt(1, detectedChar);
                        cv::putText(lastDebugVis_, charTxt,
                                   cv::Point(finalCell.x + 5, finalCell.y + finalCell.height - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 255, 0), 2);
                    }
                }
                
                resultString += detectedChar;
            }
            
            val = resultString;
        }

        else if (reg.name == "adi_soyadi") {
            
            std::vector<std::string> TR_CHARS = {
                "A","B","C","C","D","E","F","G","G","H","I","I","J","K","L","M",
                "N","O","O","P","R","S","S","T","U","U","V","Y","Z"
            };

            std::string resultString = "";
            int rows = reg.rows; 
            int cols = reg.cols; 
            
            int cellW = sub.cols / cols;
            int cellH = sub.rows / rows;

            cv::Mat workingImg = sub.clone();
            
            cv::GaussianBlur(workingImg, workingImg, cv::Size(5, 5), 0);

            cv::Mat adaptiveBin;
            cv::adaptiveThreshold(workingImg, adaptiveBin, 255,
                                  cv::ADAPTIVE_THRESH_GAUSSIAN_C,
                                  cv::THRESH_BINARY_INV,
                                  31, 25);

            cv::Mat globalBin;
            cv::threshold(workingImg, globalBin, 160, 255, cv::THRESH_BINARY_INV);

            cv::Mat finalBin;
            cv::bitwise_and(adaptiveBin, globalBin, finalBin);

            cv::Mat kernel = cv::getStructuringElement(cv::MORPH_ELLIPSE, cv::Size(3, 3));
            cv::morphologyEx(finalBin, finalBin, cv::MORPH_OPEN, kernel);

            for (int c = 0; c < cols; ++c) {
                
                double bestVal = 0.0;
                int bestRow = -1;

                for (int r = 0; r < rows; ++r) {
                    
                    int marginX = static_cast<int>(cellW * 0.30);
                    int marginY = static_cast<int>(cellH * 0.30);
                    
                    cv::Rect cell(c * cellW + marginX, r * cellH + marginY, 
                                  cellW - 2*marginX, cellH - 2*marginY);
                    
                    cell &= cv::Rect(0, 0, finalBin.cols, finalBin.rows);
                    if (cell.width <= 0 || cell.height <= 0) continue;

                    cv::Mat binCell = finalBin(cell);
                    
                    double ratio = (double)cv::countNonZero(binCell) / (cell.width * cell.height);

                    if (ratio > bestVal) {
                        bestVal = ratio;
                        bestRow = r;
                    }

                    if (debugMode_) {
                        int centerX = roi.x + (c * cellW) + (cellW / 2);
                        int centerY = roi.y + (r * cellH) + (cellH / 2);
                        int radius = std::min(cellW, cellH) * 0.35;

                        cv::circle(lastDebugVis_, cv::Point(centerX, centerY), radius, 
                                   cv::Scalar(100, 100, 100), 1, cv::LINE_AA);
                        
                        if (ratio > 0.05) { 
                            std::string scoreTxt = std::to_string((int)(ratio * 100));
                            cv::putText(lastDebugVis_, scoreTxt,
                                       cv::Point(centerX - 10, centerY + 5), 
                                       cv::FONT_HERSHEY_DUPLEX, 0.40, cv::Scalar(0, 255, 0), 1);
                        }
                    }
                }

                double THRESHOLD = 0.40; 
                std::string detectedChar = " "; 
                
                if (bestVal > THRESHOLD && bestRow != -1 && bestRow < (int)TR_CHARS.size()) {
                    detectedChar = TR_CHARS[bestRow];

                    if (debugMode_) {
                        cv::Rect finalCell(roi.x + c * cellW, roi.y + bestRow * cellH, cellW, cellH);
                        cv::rectangle(lastDebugVis_, finalCell, cv::Scalar(0, 255, 0), 2);
                        cv::putText(lastDebugVis_, detectedChar,
                                   cv::Point(finalCell.x + 5, finalCell.y + finalCell.height - 5),
                                   cv::FONT_HERSHEY_SIMPLEX, 0.50, cv::Scalar(0, 255, 0), 2);
                    }
                }
                
                resultString += detectedChar;
            }
            
            size_t lastChar = resultString.find_last_not_of(' ');
            if (lastChar != std::string::npos) {
                resultString = resultString.substr(0, lastChar + 1);
            } else {
                resultString = ""; 
            }

            val = resultString;
        }
        else {
            val = detectSingleColumn(sub, reg.rows, idThr);
        }

        out[reg.name] = val;

        cv::rectangle(lastDebugVis_, roi, cv::Scalar(0, 255, 0), 2);
        cv::putText(lastDebugVis_, reg.name, roi.tl() + cv::Point(4, 16),
                    cv::FONT_HERSHEY_SIMPLEX, 0.6, cv::Scalar(255, 0, 255), 2);
    }

    debugOut = lastDebugVis_.clone();
    return out;
}
