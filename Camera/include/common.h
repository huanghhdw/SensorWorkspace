#ifndef  COMMON_H
 #define COMMON_H

#include<Eigen/Core>
#include<Eigen/Dense>
#include<vector>

namespace  Util {
enum CameraModel{
    RAN_TAN,
    FISH_EYE
};

class CameraIntrisic {
public:
    double fx_;
    double fy_;
    double cx_;
    double cy_;
    CameraModel cameraModel_;
    std::vector<double> distortion_;
} ;

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
} ;
}
#endif