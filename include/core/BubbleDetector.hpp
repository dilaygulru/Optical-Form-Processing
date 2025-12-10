#ifndef BUBBLE_DETECTOR_HPP
#define BUBBLE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>
#include <map>

struct BubbleResult {
    int questionNumber;
    std::string markedAnswer;  // A, B, C, D, E veya "-" (boş)
    double confidence;         // 0.0 - 1.0 arası doluluk
    bool isValid;              // Çoklu işaretleme kontrolü
};

struct ScoreResult {
    std::string subject;
    int correct;
    int wrong;
    int empty;
    double score;
    std::vector<BubbleResult> details;
};

class BubbleDetector {
public:
    BubbleDetector(double fillThreshold = 0.28);
    
    // Tek bir grid'den (ders alanı) bubbleları oku
    std::vector<BubbleResult> detectBubbles(
        const cv::Mat& roiGray,
        int rows,
        int cols,
        int startQuestionNumber = 1,
        char firstLabel = 'A'
    );
    
    // Threshold ayarı
    void setFillThreshold(double threshold) { fillThreshold_ = threshold; }
    double getFillThreshold() const { return fillThreshold_; }
    
    // Debug görselleştirme
    void drawBubbleDebug(
        cv::Mat& debugImg,
        const cv::Rect& roi,
        const std::vector<BubbleResult>& results,
        int rows,
        int cols,
        const std::string& label = ""
    );
    
private:
    double fillThreshold_;   // Dolu kabul etme eşiği
    double minSeparation_;   // En dolu ve ikinci dolu arasındaki min fark
    
    // Bir hücrenin doluluk oranını hesapla
    double calculateFillRatio(const cv::Mat& cellImg);
};

#endif
