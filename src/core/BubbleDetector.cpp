#include "BubbleDetector.hpp"
#include <algorithm>

using namespace cv;

BubbleDetector::BubbleDetector(double fillThreshold)
    : fillThreshold_(fillThreshold),
      minSeparation_(0.12) {}

double BubbleDetector::calculateFillRatio(const cv::Mat& cellImg) {
    // Kutu kenarlarının etkisini azaltmak için ortadaki bölgeyi kullan
    int marginX = std::max(1, cellImg.cols / 5);  // daha agresif kırpma: dıştaki yazıları at
    int marginY = std::max(1, cellImg.rows / 5);
    int innerW = std::max(1, cellImg.cols - 2 * marginX);
    int innerH = std::max(1, cellImg.rows - 2 * marginY);
    cv::Rect inner(marginX, marginY, innerW, innerH);
    
    Mat blurred, thresh;
    GaussianBlur(cellImg(inner), blurred, Size(3, 3), 0);
    adaptiveThreshold(blurred, thresh, 255,
                      ADAPTIVE_THRESH_MEAN_C,
                      THRESH_BINARY_INV, 21, 7);
    
    Mat kernel = getStructuringElement(MORPH_ELLIPSE, Size(3, 3));
    morphologyEx(thresh, thresh, MORPH_OPEN, kernel, Point(-1,-1), 1);
    morphologyEx(thresh, thresh, MORPH_CLOSE, kernel, Point(-1,-1), 1);
    
    int nonZero = countNonZero(thresh);
    double ratio = static_cast<double>(nonZero) / static_cast<double>(thresh.total());
    return ratio;
}

std::vector<BubbleResult> BubbleDetector::detectBubbles(
    const cv::Mat& roiGray,
    int rows,
    int cols,
    int startQuestionNumber,
    char firstLabel)
{
    std::vector<BubbleResult> results;
    results.reserve(rows);
    
    int cellH = roiGray.rows / rows;
    int cellW = roiGray.cols / cols;
    
    // Gürültüyü hafifletmek için median blur
    cv::Mat preFiltered;
    cv::medianBlur(roiGray, preFiltered, 3);
    
    for (int r = 0; r < rows; ++r) {
        BubbleResult br;
        br.questionNumber = startQuestionNumber + r;
        br.markedAnswer = "-";
        br.confidence = 0.0;
        br.isValid = true;
        
        std::vector<std::pair<int, double>> fillRatios; // (col_index, fill_ratio)
        fillRatios.reserve(cols);
        
        for (int c = 0; c < cols; ++c) {
            int x = c * cellW;
            int y = r * cellH;
            Rect cell(x, y, cellW, cellH);
            
            Mat cellImg = preFiltered(cell);
            double fillRatio = calculateFillRatio(cellImg);
            
            fillRatios.push_back({c, fillRatio});
        }
        
        // En yüksek ve ikinci en yüksek doluluk
        int bestCol = -1;
        double maxFill = -1.0;
        double secondFill = -1.0;
        int aboveThreshold = 0;
        
        for (const auto& [col, fill] : fillRatios) {
            if (fill > maxFill) {
                secondFill = maxFill;
                maxFill = fill;
                bestCol = col;
            } else if (fill > secondFill) {
                secondFill = fill;
            }
            if (fill >= fillThreshold_) aboveThreshold++;
        }
        
        // Sert filtreler: eşiğin altında ise boş, fark küçükse geçersiz
        if (maxFill < fillThreshold_) {
            br.markedAnswer = "-";
            br.confidence = (maxFill < 0) ? 0.0 : maxFill;
        } else if ((maxFill - std::max(0.0, secondFill)) < minSeparation_ || aboveThreshold > 1) {
            br.isValid = false;
            br.markedAnswer = "X";
            br.confidence = maxFill;
        } else {
            br.markedAnswer = std::string(1, char(firstLabel + bestCol));
            br.confidence = maxFill;
        }
        
        results.push_back(br);
    }
    
    return results;
}

void BubbleDetector::drawBubbleDebug(
    cv::Mat& debugImg,
    const cv::Rect& roi,
    const std::vector<BubbleResult>& results,
    int rows,
    int cols,
    const std::string& label)
{
    int cellH = roi.height / rows;
    int cellW = roi.width / cols;
    
    // ROI başlığı
    if (!label.empty()) {
        cv::putText(debugImg, label, roi.tl() + Point(2, -5),
                    FONT_HERSHEY_SIMPLEX, 0.55, Scalar(0, 255, 255), 2);
    }
    
    for (size_t i = 0; i < results.size() && i < (size_t)rows; ++i) {
        const auto& br = results[i];
        int r = static_cast<int>(i);
        
        // Satır numarasını yaz
        Point textPos(roi.x - 35, roi.y + r * cellH + cellH/2 + 5);
        putText(debugImg, std::to_string(br.questionNumber),
                textPos, FONT_HERSHEY_SIMPLEX, 0.4,
                Scalar(255, 255, 255), 1);
        
        // İşaretlenen cevabı vurgula
        if (br.markedAnswer != "-" && br.markedAnswer != "X") {
            int col = br.markedAnswer[0] - 'A';
            if (col >= 0 && col < cols) {
                Rect cell(roi.x + col * cellW, roi.y + r * cellH, cellW, cellH);
                
                Scalar color = br.isValid ? Scalar(0, 255, 0) : Scalar(0, 0, 255);
                rectangle(debugImg, cell, color, 2);
                
                // Confidence değerini göster
                std::string confText = std::to_string((int)(br.confidence * 100)) + "%";
                putText(debugImg, confText,
                        Point(cell.x + 2, cell.y + cell.height - 2),
                        FONT_HERSHEY_SIMPLEX, 0.3, color, 1);
                
                // Seçilen şık harfi
                putText(debugImg, br.markedAnswer,
                        Point(cell.x + cell.width / 2 - 5, cell.y + cell.height / 2 + 4),
                        FONT_HERSHEY_SIMPLEX, 0.4, color, 1);
            }
        }
        else if (br.markedAnswer == "X") {
            // Çoklu işaretleme hatası - tüm satırı kırmızı çerçevele
            Rect rowRect(roi.x, roi.y + r * cellH, roi.width, cellH);
            rectangle(debugImg, rowRect, Scalar(0, 0, 255), 2);
            putText(debugImg, "MULTI", Point(roi.x + roi.width + 5, roi.y + r * cellH + cellH/2),
                    FONT_HERSHEY_SIMPLEX, 0.35, Scalar(0, 0, 255), 1);
        } else {
            // Boş satırları sol tarafa küçük noktalarla işaretle
            Point p1(roi.x - 6, roi.y + r * cellH + cellH / 2);
            circle(debugImg, p1, 1, Scalar(180, 180, 180), FILLED);
        }
    }
}
