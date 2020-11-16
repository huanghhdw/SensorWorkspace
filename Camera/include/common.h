#ifndef  COMMON_H
#define  COMMON_H

#include<Eigen/Core>
#include<Eigen/Dense>
#include <opencv2/opencv.hpp>
#include <opencv2/core/eigen.hpp>
#include<vector>

namespace  Util {

const double PI = 3.141592653;

enum CameraModel{
    RAN_TAN,
    FISH_EYE
};

class CameraIntrisic {
public:
    int imageRow_;
    int imageCol_;
    float fx_;
    float fy_;
    float cx_;
    float cy_;
    CameraModel cameraModel_;
    cv::Mat cameraIntrisic_;
    cv::Mat distortion_;
};

class CameraPose {
public:
    Eigen::Matrix3d r_;
    Eigen::Vector3d t_;
    Eigen::Vector3d eulerAngle_;    // yaw, pitch, roll, 弧度
    Eigen::MatrixXd transMatrix_ = Eigen::MatrixXd::Zero(4, 4);
    void UpdateAllWithEulerT();
} ;

class CameraInfo {
public:
    CameraIntrisic cameraIntrisic_;
    CameraPose cameraPose_;
};
}
#endif