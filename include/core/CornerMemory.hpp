#pragma once
#include <opencv2/opencv.hpp>
#include <array>

namespace core {

struct CornerMemory {
    std::array<cv::Point2f,4> pts{{
        cv::Point2f(-1,-1), cv::Point2f(-1,-1),
        cv::Point2f(-1,-1), cv::Point2f(-1,-1)
    }};
    std::array<int,4> age{{9999,9999,9999,9999}}; 

    void reset(){
        pts = { cv::Point2f(-1,-1), cv::Point2f(-1,-1),
                cv::Point2f(-1,-1), cv::Point2f(-1,-1) };
        age = {9999,9999,9999,9999};
    }

    void integrate(const std::array<cv::Point2f,4>& det, int max_age=20){
        for (int i=0;i<4;++i){
            if (det[i].x>=0){ pts[i]=det[i]; age[i]=0; }
            else            { age[i]=std::min(age[i]+1, max_age+100); }
            if (age[i]>max_age) pts[i]=cv::Point2f(-1,-1);
        }
    }

    void complete_if_three(){
        int miss=-1, cnt=0;
        for (int i=0;i<4;++i){ if (pts[i].x<0) miss=i; else cnt++; }
        if (cnt!=3) return;
        auto have=[&](int i){return pts[i].x>=0;};
        if (miss==0 && have(1)&&have(2)&&have(3)) pts[0] = pts[3] + pts[1] - pts[2]; // TL
        if (miss==1 && have(0)&&have(2)&&have(3)) pts[1] = pts[0] + pts[2] - pts[3]; // TR
        if (miss==2 && have(0)&&have(1)&&have(3)) pts[2] = pts[1] + pts[3] - pts[0]; // BR
        if (miss==3 && have(0)&&have(1)&&have(2)) pts[3] = pts[0] + pts[2] - pts[1]; // BL
    }

    bool has_all() const {
        for (int i=0;i<4;++i) if (pts[i].x<0) return false;
        return true;
    }
};

}
