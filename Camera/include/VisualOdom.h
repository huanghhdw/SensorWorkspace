//
// Created by huanghh on 2020/10/25.
//

#ifndef VISUALODOM_H
#define VISUALODOM_H

#include "common.h"
#include <opencv2/opencv.hpp>
#include <Eigen/Core>
#include <unistd.h>
#include <string>

namespace ImageProcess {
    class VisualOdom {
    public:
        void InitCamInfo(std::string camInfoPath);
        void GetCurrentPose(Eigen::Matrix<double, 4, 4> &pose);
        void ProcessImage(cv::Mat &leftImage, cv::Mat &rightImage);

    private:
        Eigen::Vector3d currentT_;
        Eigen::Matrix3d currentR_;
        Util::CameraIntrisic cameraIntrisicLeft_;
        Util::CameraIntrisic cameraIntrisicRight_;
        Util::CameraPose leftToRight_;
    };
}
#endif
