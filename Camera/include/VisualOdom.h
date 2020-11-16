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
        VisualOdom();
        void InitCamInfo(std::string camInfoPath);
        void GetCurrentPose(Eigen::Matrix<double, 4, 4> &pose);
        void ProcessImage(cv::Mat &leftImage, cv::Mat &rightImage);
        void ProcessStereoImg(bool isFirst);
        cv::Point3f uv2xyz(cv::Point2f uvLeft, cv::Point2f uvRight);
        void find_feature_matches (const cv::Mat& img_1, const cv::Mat& img_2,
                std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                cv::Mat &descriptors_1, cv::Mat &descriptors_2,std::vector< cv::DMatch >& matches);

    private:
        Util::CameraInfo cameraLeftInfo_;
        Util::CameraInfo cameraRightInfo_;
        cv::Mat currentLeftImg_;
        cv::Mat currentRightImg_;
        cv::Mat lastLeftImg_;
        cv::Mat lastRightImg_;
        bool isFirstFrame_ = true;

        std::vector<cv::Point2d> currentLeft2DPoints_;
        std::vector<cv::Point2d> currentRight2DPoints_;
        std::vector<cv::Point3d> current3DPoint_;

        std::vector<cv::Point2d> lastLeft2DPoints_;
        std::vector<cv::Point2d> lastRight2DPoints_;
        std::vector<cv::Point3d> last3DPoint_;

        Eigen::Vector3d StereoT_; //双目外参
        Eigen::Matrix3d StereoR_;
        Eigen::Vector3d PoseT_;   //以左相机作为参考
        Eigen::Matrix3d PoseR_;
    };
}
#endif
