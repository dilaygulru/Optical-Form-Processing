#include <opencv2/opencv.hpp>
#include "core/PerspectiveCorrector.hpp"
#include <iostream>
#include "core/BubbleDetector.hpp"

using namespace cv;
using namespace core;


std::vector<core::Bubble> buildFormTemplate() {
    std::vector<core::Bubble> bubbles;
    
  
    const int startX = 200;    
    const int startY = 480;    
    const int stepY = 14;     
    const int stepX = 17;      
    const int colStepX = 260;  
    const int bubbleSize = 10; 
    
 
    for (int col = 0; col < 4; ++col) {
        for (int q = 0; q < 25; ++q) {
            int questionIdx = col * 25 + q + 1; 
            
            int currentX = startX + col * colStepX;
            int currentY = startY + q * stepY;
            
            for (int option = 0; option < 5; ++option) {
                core::Bubble b;
                b.questionIdx = questionIdx;
                b.optionIdx = option; 

                b.roi = cv::Rect(currentX + option * stepX, currentY, bubbleSize, bubbleSize);
                
                bubbles.push_back(b);
            }
        }
    }
    
    
    
    return bubbles;
}


int main(int argc, char** argv) {
    int camIndex = 0;
    if (argc > 1) camIndex = std::atoi(argv[1]);

    cv::VideoCapture cap(camIndex);
    if (!cap.isOpened()) {
        std::cerr << "Camera open failed\n";
        return 1;
    }

    cap.set(cv::CAP_PROP_FRAME_WIDTH, 1280);
    cap.set(cv::CAP_PROP_FRAME_HEIGHT, 720);

    core::PerspectiveCorrector pc(1600, 2200);


    std::vector<core::Bubble> bubbles = buildFormTemplate();
    core::BubbleParams bparams;
    bool showDebug = true;
    std::cout << "ESC: exit | d: debug toggle | s: save warped\n";

    cv::namedWindow("camera", cv::WINDOW_NORMAL);
    cv::namedWindow("warped", cv::WINDOW_NORMAL);
    cv::resizeWindow("warped", 600, 850);

    cv::Mat lastWarped;
    
    while (true) {
        cv::Mat frame;
        if (!cap.read(frame) || frame.empty()) break;

        auto R = pc.findAndWarp(frame, showDebug);

        if (showDebug && !R.debug.empty())
            cv::imshow("camera", R.debug);
        else
            cv::imshow("camera", frame);

        if (R.ok && !R.warped.empty()) {
           
            lastWarped = R.warped.clone();

            std::vector<core::BubbleRead> result = core::BubbleDetector::readBubbles(
                lastWarped, 
                bubbles, 
                bparams
            );

            for(const auto& br : result) {
                if (br.filled) {
                    cv::rectangle(lastWarped, br.bubble.roi, cv::Scalar(0, 255, 0), 2);
                } else {
                    
                }
            }
        }

        if (!lastWarped.empty())
            cv::imshow("warped", lastWarped);

        int k = cv::waitKey(1) & 0xFF;
        if (k==27) break;
        if (k=='d'||k=='D') showDebug = !showDebug;
        if ((k=='s'||k=='S') && !lastWarped.empty()) {
            static int saved=0;
            std::string name = "warped_" + std::to_string(saved++) + ".png";
            cv::imwrite(name, lastWarped);
            std::cout << "Saved: " << name << "\n";
        }
    }

    cap.release();
    cv::destroyAllWindows();
    return 0;
}