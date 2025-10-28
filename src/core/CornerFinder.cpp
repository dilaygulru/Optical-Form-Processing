#include "core/CornerFinder.hpp"
#include <algorithm>
#include <cmath>

using namespace cv;

namespace core {

std::vector<Point2f> CornerFinder::orderTLTRBRBL(const std::vector<Point2f>& pts, Point2f C) const {
    std::array<Point2f,4> o{};
    int found = 0;
    if (pts.size() > 4) {}
    for (auto& p : pts) {
        if (p.x < C.x && p.y < C.y) { o[0] = p; found++; }
        else if (p.x >= C.x && p.y < C.y) { o[1] = p; found++; }
        else if (p.x >= C.x && p.y >= C.y) { o[2] = p; found++; }
        else if (p.x < C.x && p.y >= C.y) { o[3] = p; found++; }
    }
    std::vector<Point2f> sortedPts = pts;
    std::sort(sortedPts.begin(), sortedPts.end(), [](const Point2f& a, const Point2f& b) {
        return (a.x + a.y) < (b.x + b.y);
    });
    o[0] = sortedPts.front();
    o[2] = sortedPts.back();
    std::sort(sortedPts.begin(), sortedPts.end(), [](const Point2f& a, const Point2f& b) {
        return (a.y - a.x) < (b.y - a.x);
    });
    o[1] = sortedPts.front();
    o[3] = sortedPts.back();
    return {o[0], o[1], o[2], o[3]};
}

bool CornerFinder::findCornerSquares(const Mat& gray, std::vector<Point2f>& corners, Mat* dbg) const {
    Mat th;
    GaussianBlur(gray, th, Size(5, 5), 0);
    adaptiveThreshold(th, th, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 21, 5);
    Mat kernel = getStructuringElement(MORPH_RECT, Size(3, 3));
    morphologyEx(th, th, MORPH_OPEN, kernel, Point(-1,-1), 2);
    morphologyEx(th, th, MORPH_CLOSE, kernel, Point(-1,-1), 2);
    if (dbg) cvtColor(gray, *dbg, COLOR_GRAY2BGR);
    std::vector<std::vector<Point>> contours;
    findContours(th, contours, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);
    std::vector<std::vector<Point>> candidateContours;
    for (auto& c : contours) {
        double area = contourArea(c);
        if (area < 100 || area > 10000) continue;
        Rect br = boundingRect(c);
        float ar = (float)br.width / br.height;
        if (ar < 0.8f || ar > 1.2f) continue;
        double extent = area / br.area();
        if (extent < 0.7) continue;
        candidateContours.push_back(c);
    }
    if (dbg) drawContours(*dbg, candidateContours, -1, Scalar(0,255,255), 2);
    if (candidateContours.size() < 4) {
        if (dbg) putText(*dbg, "Yeterli aday bulunamadi (<4)", {20,40}, FONT_HERSHEY_SIMPLEX, 1, {0,0,255}, 2);
        return false;
    }
    std::sort(candidateContours.begin(), candidateContours.end(), [](auto& a, auto& b) {
        return contourArea(a) > contourArea(b);
    });
    std::vector<std::vector<Point>> finalFour(candidateContours.begin(), candidateContours.begin() + 4);
    if (dbg) drawContours(*dbg, finalFour, -1, Scalar(0,255,0), 2);
    std::vector<Point2f> centers;
    for (auto& c : finalFour) {
        Moments m = moments(c);
        centers.push_back(Point2f(m.m10 / m.m00, m.m01 / m.m00));
    }
    corners = orderTLTRBRBL(centers, Point2f(gray.cols / 2.f, gray.rows / 2.f));
    if (dbg) {
        for (int i=0; i<4; ++i) {
            circle(*dbg, corners[i], 8, Scalar(0,0,255), FILLED);
            putText(*dbg, std::to_string(i), corners[i], FONT_HERSHEY_SIMPLEX, 1, {255,0,0}, 2);
        }
    }
    return true;
}

CornerResult CornerFinder::processFrame(const Mat& bgr, bool debug_on) const {
    CornerResult R;
    if (bgr.empty()) return R;
    Mat gray;
    cvtColor(bgr, gray, COLOR_BGR2GRAY);
    std::vector<Point2f> srcPoints;
    Mat dbgImg;
    R.paper_ok = findCornerSquares(gray, srcPoints, debug_on ? &dbgImg : nullptr);
    if (debug_on) R.debug_bgr = dbgImg;
    if (!R.paper_ok) return R;
    std::vector<Point2f> dstPoints = {
        {0, 0},
        {(float)outW_ - 1, 0},
        {(float)outW_ - 1, (float)outH_ - 1},
        {0, (float)outH_ - 1}
    };
    Mat H = getPerspectiveTransform(srcPoints, dstPoints);
    warpPerspective(gray, R.warped_gray, H, Size(outW_, outH_), INTER_LINEAR, BORDER_REPLICATE);
    for(int i=0; i<4; ++i) R.markers_orig[i] = srcPoints[i];
    return R;
}

} 