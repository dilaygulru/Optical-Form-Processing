#include "core/CornerFinder.hpp"
#include <algorithm>
#include <cmath>

using namespace cv;

namespace core {

std::vector<Point2f> CornerFinder::orderTLTRBRBL(const std::vector<Point2f>& pts, Point2f C) const {
    std::array<Point2f,4> o{};
    for (auto&p:pts){
        if (p.x<C.x && p.y<C.y) o[0]=p;            // TL
        else if (p.x>=C.x && p.y<C.y) o[1]=p;      // TR
        else if (p.x>=C.x && p.y>=C.y) o[2]=p;     // BR
        else o[3]=p;                                // BL
    }
    return {o[0],o[1],o[2],o[3]};
}


bool CornerFinder::findPaperQuad(const Mat& gray, std::vector<Point2f>& quad, Mat* dbg) const {
    Mat g; GaussianBlur(gray, g, Size(5,5), 0);
    Mat edges; Canny(g, edges, 50, 150);
    morphologyEx(edges, edges, MORPH_CLOSE, getStructuringElement(MORPH_RECT, Size(7,7)));
    dilate(edges, edges, getStructuringElement(MORPH_RECT, Size(5,5)));

    std::vector<std::vector<Point>> cs;
    findContours(edges, cs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

    double bestA=0; std::vector<Point> best;
    for (auto& c: cs){
        std::vector<Point> approx;
        approxPolyDP(c, approx, 0.02*arcLength(c,true), true);
        if (approx.size()!=4 || !isContourConvex(approx)) continue;
        double a = std::fabs(contourArea(approx));
        if (a>bestA){ bestA=a; best=approx; }
    }
    if (best.empty()){
        if (dbg){
            cvtColor(gray, *dbg, COLOR_GRAY2BGR);
            putText(*dbg,"Paper not found", {20,40}, FONT_HERSHEY_SIMPLEX,1,{0,0,255},2);
        }
        return false;
    }

    std::vector<Point2f> pts; pts.reserve(4);
    for (auto&p:best) pts.push_back(Point2f((float)p.x,(float)p.y));
    pts = orderTLTRBRBL(pts, Point2f(gray.cols*0.5f, gray.rows*0.5f));
    quad = pts;

    if (dbg){
        cvtColor(gray, *dbg, COLOR_GRAY2BGR);
        polylines(*dbg, std::vector<Point>(best.begin(), best.end()), true, Scalar(0,255,0), 2);
    }
    return true;
}

void CornerFinder::detectCornerSquaresInWarp(const Mat& warpedGray,
                                             std::array<Point2f,4>& markers_warp,
                                             Mat* dbgWarp) const
{
    CV_Assert(warpedGray.type()==CV_8UC1);
    const int W = warpedGray.cols, H = warpedGray.rows;
    for (auto& p: markers_warp) p = Point2f(-1,-1);

    Rect roiTL(0, 0, W/6, H/6);
    Rect roiTR(W*5/6, 0, W/6, H/6);
    Rect roiBR(W*5/6, H*5/6, W/6, H/6);
    Rect roiBL(0, H*5/6, W/6, H/6);

    std::array<Rect,4> ROIs{roiTL,roiTR,roiBR,roiBL};

    
    Mat eq; { Ptr<CLAHE> clahe = createCLAHE(2.0, Size(8,8)); clahe->apply(warpedGray, eq); }
    Mat th; adaptiveThreshold(eq, th, 255, ADAPTIVE_THRESH_GAUSSIAN_C, THRESH_BINARY_INV, 41, 8);

    Mat hor, ver;
    morphologyEx(th, hor, MORPH_OPEN, getStructuringElement(MORPH_RECT,{25,1}));
    morphologyEx(th, ver, MORPH_OPEN, getStructuringElement(MORPH_RECT,{1,25}));
    Mat noLines; subtract(th, hor|ver, noLines);

    if (dbgWarp) cvtColor(warpedGray, *dbgWarp, COLOR_GRAY2BGR);

    for (int idx=0; idx<4; ++idx){
        Rect R = ROIs[idx];
        Mat roi = warpedGray(R);
        Mat roiNoLines = noLines(R);

        Mat bin = roiNoLines.clone();
        morphologyEx(bin, bin, MORPH_OPEN, getStructuringElement(MORPH_RECT,{3,3}));
        morphologyEx(bin, bin, MORPH_CLOSE, getStructuringElement(MORPH_RECT,{5,5}));

        std::vector<std::vector<Point>> cs;
        findContours(bin, cs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

        const double A = double(R.width)*R.height;
        double amin = A * 0.0005;   
        double amax = A * 0.0200;   

        double bestScore=-1; Point2f best(-1,-1);

        for (auto& c : cs){
            double a = std::fabs(contourArea(c));
            if (a<amin || a>amax) continue;

            Rect br = boundingRect(c);
            double ar = double(br.width)/br.height;
            if (ar<0.85 || ar>1.15) continue;          

            double per = arcLength(c,true);
            if (per<=1) continue;
            double compact = (4.0*CV_PI*a)/(per*per);  
            double extent  = a/double(br.area());      
            if (compact<0.70 || extent<0.80) continue;

            Mat mask(bin.size(), CV_8U, Scalar(0));
            drawContours(mask, std::vector<std::vector<Point>>{c}, -1, Scalar(255), FILLED);
            Scalar meanInside = mean(roi, mask);
            if (meanInside[0] > 80) continue;

            
            int pad = std::max(3, std::min(br.width,br.height)/6);
            Rect outer = br;
            outer.x = std::max(0, outer.x - pad);
            outer.y = std::max(0, outer.y - pad);
            outer.width  = std::min(roi.cols - outer.x, outer.width + 2*pad);
            outer.height = std::min(roi.rows - outer.y, outer.height + 2*pad);

            Mat ringMask = Mat::zeros(roi.size(), CV_8U);
            rectangle(ringMask, outer, Scalar(255), FILLED);
            rectangle(ringMask, br,    Scalar(0),   FILLED);

            Scalar meanRing = mean(roi, ringMask);
            if (meanRing[0] < 185) continue;
            if ((meanRing[0]-meanInside[0]) < 110) continue;

            
            Moments m = moments(c);
            Point2f cc(float(m.m10/m.m00), float(m.m01/m.m00));
            double dx = (idx==0 || idx==3) ? cc.x : (R.width - cc.x);
            double dy = (idx==0 || idx==1) ? cc.y : (R.height - cc.y);
            double gateX = 0.12 * R.width;
            double gateY = 0.12 * R.height;
            if (dx > gateX || dy > gateY) continue;

            double score = (a * (255.0 - meanInside[0])) / (1.0 + dx*dx + dy*dy);
            if (score>bestScore){
                bestScore=score;
                best = Point2f(cc.x + R.x, cc.y + R.y);
            }
        }

        if (best.x>=0){
            markers_warp[idx]=best;
            if (dbgWarp){
                rectangle(*dbgWarp, R, Scalar(0,255,255), 2);
                circle(*dbgWarp, best, 8, Scalar(0,0,255), FILLED);
            }
        } else if (dbgWarp){
            rectangle(*dbgWarp, R, Scalar(0,255,255), 1);
            putText(*dbgWarp, "no marker", {R.x+5,R.y+18}, FONT_HERSHEY_SIMPLEX, 0.5, {0,0,255}, 1);
        }
    }

    bool haveBottom = (markers_warp[2].x >= 0 && markers_warp[3].x >= 0);
    if (haveBottom) {
        float avgY = (markers_warp[2].y + markers_warp[3].y) / 2.0f;
        float shiftY = H - avgY;

        Point2f predTL(markers_warp[3].x, markers_warp[3].y - 2*shiftY);
        Point2f predTR(markers_warp[2].x, markers_warp[2].y - 2*shiftY);

        auto clamp = [&](Point2f p){
            p.x = std::clamp(p.x, 10.0f, (float)(W-10));
            p.y = std::clamp(p.y, 10.0f, (float)(H-10));
            return p;
        };
        predTL = clamp(predTL);
        predTR = clamp(predTR);

        auto searchLocal = [&](Point2f guess)->Point2f {
            int win = 40;
            Rect R(guess.x - win/2, guess.y - win/2, win, win);
            R &= Rect(0,0,W,H);
            if (R.width < 10 || R.height < 10) return Point2f(-1,-1);

            Mat roi = warpedGray(R);
            Mat bin; threshold(roi, bin, 0,255,THRESH_BINARY_INV|THRESH_OTSU);
            std::vector<std::vector<Point>> cs;
            findContours(bin, cs, RETR_EXTERNAL, CHAIN_APPROX_SIMPLE);

            double bestA=0; Point2f best(-1,-1);
            for (auto& c: cs){
                double a = fabs(contourArea(c));
                if (a < 50) continue;
                Rect br = boundingRect(c);
                double ar = double(br.width)/br.height;
                if (ar<0.8 || ar>1.2) continue;
                Moments m = moments(c);
                Point2f cc(float(m.m10/m.m00)+R.x, float(m.m01/m.m00)+R.y);
                if (a > bestA) { bestA = a; best = cc; }
            }
            return best;
        };

        if (markers_warp[0].x < 0) markers_warp[0] = searchLocal(predTL);
        if (markers_warp[1].x < 0) markers_warp[1] = searchLocal(predTR);

        if (dbgWarp) {
            if (markers_warp[0].x >= 0) circle(*dbgWarp, markers_warp[0], 6, Scalar(255,0,0), 2);
            if (markers_warp[1].x >= 0) circle(*dbgWarp, markers_warp[1], 6, Scalar(255,0,0), 2);
        }
    }
}


CornerResult CornerFinder::processFrame(const Mat& bgr, bool debug_on) const {
    CornerResult R;
    if (bgr.empty()) return R;

    Mat gray; cvtColor(bgr, gray, COLOR_BGR2GRAY);

  
    std::vector<Point2f> paperQuad;
    Mat dbgPaper;
    R.paper_ok = findPaperQuad(gray, paperQuad, debug_on? &dbgPaper: nullptr);
    if (!R.paper_ok){
        if (debug_on) R.debug_bgr = dbgPaper;
        return R;
    }

   
    std::vector<Point2f> dst = { {0,0},{(float)outW_-1,0},{(float)outW_-1,(float)outH_-1},{0,(float)outH_-1} };
    Mat H = getPerspectiveTransform(paperQuad, dst);
    warpPerspective(gray, R.warped_gray, H, Size(outW_,outH_), INTER_LINEAR, BORDER_REPLICATE);

   
    Mat dbgWarp;
    detectCornerSquaresInWarp(R.warped_gray, R.markers_warp, debug_on? &dbgWarp: nullptr);

    
    Mat Hinv; invert(H, Hinv);
    std::vector<Point2f> inPts, outPts;
    std::array<int,4> mapIdx{{-1,-1,-1,-1}};
    for (int i=0,k=0;i<4;++i){
        if (R.markers_warp[i].x>=0){ inPts.push_back(R.markers_warp[i]); mapIdx[i]=k++; }
    }
    if (!inPts.empty()){
        outPts.resize(inPts.size());
        perspectiveTransform(inPts, outPts, Hinv);
        for (int i=0;i<4;++i)
            if (mapIdx[i]>=0) R.markers_orig[i]=outPts[ mapIdx[i] ];
    }
    R.markers_ok = true;

    if (debug_on){
        if (dbgPaper.empty()) cvtColor(gray, R.debug_bgr, COLOR_GRAY2BGR);
        else R.debug_bgr = dbgPaper;

        
        polylines(R.debug_bgr, std::vector<Point>(paperQuad.begin(), paperQuad.end()), true, Scalar(0,255,0), 2);
        
        for (int i=0;i<4;++i){
            if (R.markers_orig[i].x>=0) circle(R.debug_bgr, R.markers_orig[i], 8, Scalar(0,0,255), FILLED);
        }
    }
    return R;
}

} 
