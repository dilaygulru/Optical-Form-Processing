#include "core/BubbleDetector.hpp"
#include <algorithm> // std::clamp

static cv::Mat toGray(const cv::Mat& img) {
  if (img.channels() == 1) return img.clone();
  cv::Mat g; cv::cvtColor(img, g, cv::COLOR_BGR2GRAY);
  return g;
}

std::vector<core::BubbleRead> core::BubbleDetector::readBubbles(
    const cv::Mat& alignedImage,
    const std::vector<core::Bubble>& bubbles,
    const core::BubbleParams& p)
{
  std::vector<core::BubbleRead> out;
  if (alignedImage.empty() || bubbles.empty()) return out;

  cv::Mat gray = toGray(alignedImage), bin;
  if (p.useOtsu) {
    cv::threshold(gray, bin, 0, 255, cv::THRESH_BINARY_INV | cv::THRESH_OTSU);
  } else {
    cv::adaptiveThreshold(gray, bin, 255, cv::ADAPTIVE_THRESH_MEAN_C,
                          cv::THRESH_BINARY_INV, p.blockSize, p.C);
  }

  for (const auto& b : bubbles) {
    cv::Rect r = b.roi & cv::Rect(0, 0, gray.cols, gray.rows);
    if (r.area() <= 0) continue;

    cv::Mat mask = bin(r), proc;
    cv::morphologyEx(mask, proc, cv::MORPH_CLOSE,
                     cv::getStructuringElement(cv::MORPH_ELLIPSE, {3,3}));

    const double black = static_cast<double>(cv::countNonZero(proc));
    const double fill  = black / static_cast<double>(r.area());
    const double conf  = std::clamp((fill - p.fillRatio) / (1.0 - p.fillRatio), 0.0, 1.0);

    core::BubbleRead br;
    br.bubble     = b;
    br.filled     = (fill >= p.fillRatio);
    br.confidence = conf;
    br.darkness   = fill;
    out.push_back(br);
  }
  return out;
}
