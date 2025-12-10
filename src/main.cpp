#include <opencv2/opencv.hpp>
#include "core/PerspectiveCorrector.hpp"
#include "ROIDetector.hpp"
#include "AnswerKey.hpp"
#include <iostream>
#include <iomanip>

using namespace cv;
using namespace std;

// Ekrana skor bilgilerini yazdır
void drawScoreOverlay(cv::Mat& frame, const AnswerKey::ScoreResult& score) {
    int startY = 30;
    int lineHeight = 35;
    
    // Siyah yarı-şeffaf arka plan
    cv::Rect bgRect(10, 10, 350, 250);
    cv::Mat roi = frame(bgRect);
    cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::addWeighted(color, 0.6, roi, 0.4, 0, roi);
    
    // Başlık
    cv::putText(frame, "CANLI PUANLAMA SISTEMI", 
                cv::Point(20, startY), 
                cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                cv::Scalar(0, 255, 255), 2);
    
    startY += lineHeight;
    
    // Skor
    std::stringstream ss;
    ss << "SKOR: " << std::fixed << std::setprecision(1) << score.score << " / 100";
    cv::putText(frame, ss.str(), 
                cv::Point(20, startY), 
                cv::FONT_HERSHEY_SIMPLEX, 0.8, 
                cv::Scalar(0, 255, 0), 2);
    
    startY += lineHeight;
    
    // Doğru / Yanlış / Boş
    ss.str("");
    ss << "Dogru: " << score.correctAnswers << " | Yanlis: " << score.wrongAnswers;
    cv::putText(frame, ss.str(), 
                cv::Point(20, startY), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                cv::Scalar(255, 255, 255), 1);
    
    startY += lineHeight - 5;
    
    ss.str("");
    ss << "Bos: " << score.emptyAnswers << " | Toplam: " << score.totalQuestions;
    cv::putText(frame, ss.str(), 
                cv::Point(20, startY), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                cv::Scalar(255, 255, 255), 1);
    
    startY += lineHeight;
    
    // Ders bazında skorlar
    cv::putText(frame, "DERS BAZINDA:", 
                cv::Point(20, startY), 
                cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                cv::Scalar(200, 200, 200), 1);
    
    startY += 25;
    
    for (const auto& subScore : score.subjectScores) {
        ss.str("");
        ss << subScore.first << ": " << subScore.second;
        cv::putText(frame, ss.str(), 
                    cv::Point(30, startY), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                    cv::Scalar(255, 255, 255), 1);
        startY += 22;
    }
}

// Detaylı soru analizi çizimi
void drawQuestionAnalysis(cv::Mat& frame, 
                         const std::map<std::string, std::vector<ROIDetector::QuestionDetail>>& details) {
    int startX = frame.cols - 400;
    int startY = 30;
    int lineHeight = 25;
    
    // Siyah yarı-şeffaf arka plan
    cv::Rect bgRect(startX - 10, 10, 390, std::min(frame.rows - 20, 500));
    cv::Mat roi = frame(bgRect);
    cv::Mat color(roi.size(), CV_8UC3, cv::Scalar(0, 0, 0));
    cv::addWeighted(color, 0.6, roi, 0.4, 0, roi);
    
    // Başlık
    cv::putText(frame, "SORU ANALIZI", 
                cv::Point(startX, startY), 
                cv::FONT_HERSHEY_SIMPLEX, 0.6, 
                cv::Scalar(255, 255, 0), 2);
    
    startY += lineHeight + 5;
    
    for (const auto& subjectPair : details) {
        const std::string& subject = subjectPair.first;
        const auto& questions = subjectPair.second;
        
        // Ders başlığı
        cv::putText(frame, subject + ":", 
                    cv::Point(startX, startY), 
                    cv::FONT_HERSHEY_SIMPLEX, 0.5, 
                    cv::Scalar(100, 200, 255), 1);
        startY += lineHeight - 5;
        
        // İlk 5 soruyu göster
        int count = 0;
        for (const auto& q : questions) {
            if (count >= 5) break;
            if (q.markedAnswer == '-') continue; // Boş soruları atla
            
            std::stringstream ss;
            ss << "  S" << (q.questionNumber + 1) << ": " 
               << q.markedAnswer << " -> " << q.correctAnswer;
            
            cv::Scalar textColor = q.isCorrect ? 
                cv::Scalar(0, 255, 0) : cv::Scalar(0, 0, 255);
            
            cv::putText(frame, ss.str(), 
                        cv::Point(startX + 10, startY), 
                        cv::FONT_HERSHEY_SIMPLEX, 0.45, 
                        textColor, 1);
            
            startY += 20;
            count++;
        }
        
        startY += 10;
        
        if (startY > frame.rows - 50) break;
    }
}

int main(int argc, char** argv) {
    int camIndex = 0;
    if (argc > 1) camIndex = std::atoi(argv[1]);
    
    cv::VideoCapture cap(camIndex);
    if (!cap.isOpened()) {
        std::cerr << "Kamera açılamadı!\n";
        return 1;
    }
    
    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);
    
    core::PerspectiveCorrector pc(1600, 2200);
    ROIDetector detector;
    
    // CEVAP ANAHTARINI YÜKLE
    // Bu örnek cevap anahtarı - kendi cevap anahtarınızı buraya yazın
    AnswerKey answerKey;
    std::vector<AnswerKey::QuestionAnswer> answers;
    
    // Türkçe dersi cevapları (20 soru)
    std::string turkceAnswers = "BACCDABCDDABCCBADCBA";
    for (int i = 0; i < 20; ++i) {
        answers.push_back({"turkce", i, turkceAnswers[i]});
    }
    
    // Sosyal Bilgiler cevapları (20 soru)
    std::string sosyalAnswers = "DCBAEDCBAEABCDEDCBAE";
    for (int i = 0; i < 20; ++i) {
        answers.push_back({"sosyal", i, sosyalAnswers[i]});
    }
    
    // Din Kültürü cevapları (20 soru)
    std::string dinAnswers = "AEBCDAEBCDAEBCDAEBCD";
    for (int i = 0; i < 20; ++i) {
        answers.push_back({"din", i, dinAnswers[i]});
    }
    
    // İngilizce cevapları (20 soru)
    std::string ingilizceAnswers = "EABCDCBADEEDCBAACBDE";
    for (int i = 0; i < 20; ++i) {
        answers.push_back({"ingilizce", i, ingilizceAnswers[i]});
    }
    
    // Matematik cevapları (20 soru)
    std::string matematikAnswers = "CDABECDABECDABECDBAE";
    for (int i = 0; i < 20; ++i) {
        answers.push_back({"matematik", i, matematikAnswers[i]});
    }
    
    // Fen Bilimleri cevapları (20 soru)
    std::string fenAnswers = "BCDEABCDEABCDEABCDEA";
    for (int i = 0; i < 20; ++i) {
        answers.push_back({"fen", i, fenAnswers[i]});
    }
    
    answerKey.loadAnswerKey(answers);
    
    // Cevap anahtarını map formatına çevir (ROIDetector için)
    std::map<std::string, std::map<int, char>> answerKeyMap;
    for (const auto& qa : answers) {
        answerKeyMap[qa.subject][qa.questionNumber] = qa.correctAnswer;
    }
    
    bool showDebug = true;
    bool showAnalysis = true;
    bool showBubbleDebug = true;  // Bubble detection debug modu
    
    std::cout << "=== CANLI OPTIK FORM PUANLAMA SISTEMI ===\n";
    std::cout << "ESC: Çıkış\n";
    std::cout << "d: Debug görünümü aç/kapat\n";
    std::cout << "a: Detaylı analiz aç/kapat\n";
    std::cout << "b: Bubble detection debug aç/kapat\n";
    std::cout << "s: Warped görüntüyü kaydet\n";
    std::cout << "t: Doluluk eşiğini ayarla (varsayılan: 0.20)\n";
    std::cout << "+/-: Eşiği artır/azalt (0.05 adımlarla)\n\n";
    
    cv::namedWindow("Kamera", cv::WINDOW_NORMAL);
    cv::namedWindow("Form - Canli Puanlama", cv::WINDOW_NORMAL);
    cv::namedWindow("Bubble Debug", cv::WINDOW_NORMAL);  // Yeni pencere
    cv::resizeWindow("Form - Canli Puanlama", 800, 1100);
    cv::resizeWindow("Bubble Debug", 400, 800);
    
    cv::Mat lastWarped;
    AnswerKey::ScoreResult lastScore;
    
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;
        
        // Perspektif düzeltme
        auto R = pc.findAndWarp(frame, showDebug);
        
        cv::Mat displayFrame = frame.clone();
        
        if (R.ok && !R.warped.empty()) {
            lastWarped = R.warped.clone();
            
            // Debug modunu ayarla
            detector.setDebugMode(showBubbleDebug);
            
            // Detaylı analiz yap
            cv::Mat omrDebug;
            auto questionDetails = detector.processWithDetails(
                R.warped, omrDebug, answerKeyMap
            );
            
            // Normal işleme ile cevapları al
            std::map<std::string, std::string> studentAnswers = 
                detector.process(R.warped, omrDebug);
            
            // Bubble debug görüntüsünü göster
            if (showBubbleDebug) {
                cv::Mat bubbleDebug = detector.getLastDebugVisualization();
                if (!bubbleDebug.empty()) {
                    cv::imshow("Bubble Debug", bubbleDebug);
                }
            }
            
            // Skoru hesapla
            lastScore = answerKey.calculateScore(studentAnswers);
            
            // Warped görüntüye skor bilgisi ekle
            if (!omrDebug.empty()) {
                cv::Mat scoreOverlay = omrDebug.clone();
                
                // Sol üst köşeye skor yaz
                int yPos = 50;
                std::stringstream ss;
                ss << "SKOR: " << std::fixed << std::setprecision(1) 
                   << lastScore.score << " / 100";
                
                // Arka plan
                cv::rectangle(scoreOverlay, 
                             cv::Point(10, 10), 
                             cv::Point(400, 100),
                             cv::Scalar(0, 0, 0), -1);
                
                // Skor
                cv::putText(scoreOverlay, ss.str(), 
                           cv::Point(20, yPos), 
                           cv::FONT_HERSHEY_SIMPLEX, 1.2, 
                           cv::Scalar(0, 255, 0), 3);
                
                yPos += 35;
                ss.str("");
                ss << "D:" << lastScore.correctAnswers 
                   << " Y:" << lastScore.wrongAnswers 
                   << " B:" << lastScore.emptyAnswers;
                cv::putText(scoreOverlay, ss.str(), 
                           cv::Point(20, yPos), 
                           cv::FONT_HERSHEY_SIMPLEX, 0.7, 
                           cv::Scalar(255, 255, 255), 2);
                
                cv::imshow("Form - Canli Puanlama", scoreOverlay);
            }
        }
        
        // Kamera görüntüsüne skor ve analiz ekle
        if (lastScore.totalQuestions > 0) {
            drawScoreOverlay(displayFrame, lastScore);
        }
        
        if (showDebug && !R.debug.empty()) {
            cv::imshow("Kamera", R.debug);
        } else {
            cv::imshow("Kamera", displayFrame);
        }
        
        int k = cv::waitKey(1) & 0xFF;
        if (k == 27) break; // ESC
        
        if (k == 'd' || k == 'D') {
            showDebug = !showDebug;
            std::cout << "Debug görünümü: " << (showDebug ? "AÇIK" : "KAPALI") << "\n";
        }
        
        if (k == 'a' || k == 'A') {
            showAnalysis = !showAnalysis;
            std::cout << "Detaylı analiz: " << (showAnalysis ? "AÇIK" : "KAPALI") << "\n";
        }
        
        if (k == 'b' || k == 'B') {
            showBubbleDebug = !showBubbleDebug;
            detector.setDebugMode(showBubbleDebug);
            std::cout << "Bubble debug: " << (showBubbleDebug ? "AÇIK" : "KAPALI") << "\n";
            if (!showBubbleDebug) {
                cv::destroyWindow("Bubble Debug");
            }
        }
        
        // + tuşu ile eşiği artır
        if (k == '+' || k == '=') {
            double current = detector.getFillThreshold();
            detector.setFillThreshold(current + 0.05);
            std::cout << "Doluluk eşiği: " << (current + 0.05) << "\n";
        }
        
        // - tuşu ile eşiği azalt
        if (k == '-' || k == '_') {
            double current = detector.getFillThreshold();
            detector.setFillThreshold(std::max(0.05, current - 0.05));
            std::cout << "Doluluk eşiği: " << std::max(0.05, current - 0.05) << "\n";
        }
        
        if ((k == 's' || k == 'S') && !lastWarped.empty()) {
            static int saved = 0;
            std::string name = "warped_score_" + std::to_string(saved++) + ".png";
            cv::imwrite(name, lastWarped);
            std::cout << "Kaydedildi: " << name 
                     << " (Skor: " << lastScore.score << ")\n";
        }
        
        if (k == 't' || k == 'T') {
            std::cout << "Yeni doluluk eşiğini girin (0.0 - 1.0, varsayılan 0.20): ";
            double newThreshold;
            std::cin >> newThreshold;
            if (newThreshold >= 0.0 && newThreshold <= 1.0) {
                detector.setFillThreshold(newThreshold);
                std::cout << "Doluluk eşiği güncellendi: " << newThreshold << "\n";
            } else {
                std::cout << "Geçersiz değer!\n";
            }
        }
    }
    
    cap.release();
    cv::destroyAllWindows();
    
    std::cout << "\n=== SON SKOR ===\n";
    std::cout << "Toplam Puan: " << lastScore.score << " / 100\n";
    std::cout << "Doğru: " << lastScore.correctAnswers << "\n";
    std::cout << "Yanlış: " << lastScore.wrongAnswers << "\n";
    std::cout << "Boş: " << lastScore.emptyAnswers << "\n";
    std::cout << "\nDers Bazında Doğrular:\n";
    for (const auto& sub : lastScore.subjectScores) {
        std::cout << "  " << sub.first << ": " << sub.second << "\n";
    }
    
    return 0;
}