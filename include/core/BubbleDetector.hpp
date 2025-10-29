#pragma once
#include <opencv2/opencv.hpp>
#include <string>
#include <vector>

namespace core {

struct Bubble {
  cv::Rect roi;       
  int questionIdx;    
  int optionIdx;      
};

struct BubbleRead {
  Bubble bubble;
  bool filled;        
  double confidence;  
  double darkness;    
};
struct BubbleParams {

  int blockSize = 31;          
  int C = 10;                  
  int minArea = 40;            
  int maxArea = 5000;         
  double fillRatio = 0.45;    
  bool useOtsu = false;        
};

class BubbleDetector {
public:
  static std::vector<BubbleRead> readBubbles(
      const cv::Mat& alignedImage,
      const std::vector<Bubble>& bubbles,
      const BubbleParams& params = {});
};

}