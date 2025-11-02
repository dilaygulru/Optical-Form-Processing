#ifndef BUBBLE_DETECTOR_HPP
#define BUBBLE_DETECTOR_HPP

#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

class BubbleDetector {
public:
    // Yeni eklendi: Region'un tipi (GRID veya COLUMN)
    enum RegionType {
        GRID,
        COLUMN
    };

    // her bölüm için ROI tanımlamak için kullanıyoruz
    struct RegionDef {
        std::string name;      // "TURKCE", "KURUM KODU" vs.

        // Düzeltildi: 'roi' yerine yüzdelik koordinatlar eklendi.
        float rectPct[4];      // {x, y, w, h} - yüzdelik koordinatlar

        int rows;              // kaç soru / kaç satır
        int cols;              // kaç seçenek (A,B,C,D,E)
        RegionType type;       // Yeni eklendi: GRID mi yoksa COLUMN mu
    };

    // ... (diğer kısımlar aynı) ...

    BubbleDetector(); // ' = default;' kısmı cpp'ye taşındığı için tanımı burada bıraktım.
    // 'setRegions' metodu .cpp dosyasından kaldırılmıştı, onu da sildim.

    // tek frame üzerinde bütün işaretleri oku
    std::map<std::string, std::string> process(const cv::Mat& warped, cv::Mat& debugOut);
    // NOT: .cpp dosyasında 'process' kullanıldığı için burayı da process olarak güncelledim.

private:
    std::vector<RegionDef> regions_;
};

#endif // BUBBLE_DETECTOR_HPP