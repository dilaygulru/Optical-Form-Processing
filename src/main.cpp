#include <opencv2/opencv.hpp>
#include "core/PerspectiveCorrector.hpp"
#include <iostream>

using namespace cv;
using namespace core;

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