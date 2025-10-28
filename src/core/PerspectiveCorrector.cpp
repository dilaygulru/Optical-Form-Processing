#include "core/PerspectiveCorrector.hpp"

using namespace cv;

namespace core {

PerspectiveCorrector::PerspectiveCorrector(int outW, int outH)
    : outW_(outW), outH_(outH), finder_(outW, outH) {}

WarpResult PerspectiveCorrector::findAndWarp(const cv::Mat& bgr, bool wantDebug) const {
    WarpResult R;
    if (bgr.empty()) return R;

    CornerResult C = finder_.processFrame(bgr, wantDebug);

    if (wantDebug) R.debug = C.debug_bgr.empty() ? bgr.clone() : C.debug_bgr;

    if (!C.paper_ok) {
        R.ok = false;
        return R;
    }

    cv::Mat warpedBgr;
    cv::cvtColor(C.warped_gray, warpedBgr, cv::COLOR_GRAY2BGR);
    R.warped = warpedBgr;
    R.corners = C.markers_orig;
    R.ok = !R.warped.empty();
    
    return R;
}

} 