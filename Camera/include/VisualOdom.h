//
// Created by huanghh on 2020/10/25.
//

#ifndef VISUALODOM_H
#define VISUALODOM_H

#include "common.h"
#include <Eigen/Core>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include <unistd.h>
#include <string>

namespace ImageProcess {
    class VisualOdom {
    public:
        void InitCamInfo(std::string camInfoPath);
        void GetCurrentPose(Eigen::Matrix<double, 4, 4> &pose);
        void ProcessImage(cv::Mat &leftImage, cv::Mat &rightImage);
        void ProcessStereoImg(bool isFirst);

    private:
        Util::CameraInfo cameraLeftInfo_;
        Util::CameraInfo cameraRightInfo_;
        cv::Mat currentLeftImg_;
        cv::Mat currentRightImg_;
        cv::Mat lastLeftImg_;
        cv::Mat lastRightImg_;
        bool isFirstFrame_ = true;
        Eigen::Vector3f StereoT_;
        Eigen::Matrix3f StereoR_;


        std::vector<cv::Point2d> currentLeft2DPoints_;
        std::vector<cv::Point2d> currentRight2DPoints_;
        std::vector<cv::Point3d> current3DPoint_;

        std::vector<cv::Point2d> lastLeft2DPoints_;
        std::vector<cv::Point2d> lastRight2DPoints_;
        std::vector<cv::Point3d> last3DPoint_;
    };
}
#endif
