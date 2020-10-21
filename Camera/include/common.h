#ifndef  COMMON_H
 #define COMMON_H

#include<Eigen/Core>
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
    Eigen::Vector3d eulerAngle_;    // yaw, pitch, roll
    Eigen::MatrixXd transMatrix_ = Eigen::MatrixXd::Zero(4, 4);
    void UpdateAllWithEulerT();
} ;

class CameraInfo {
public:
    CameraIntrisic cameraIntrisic_;
    CameraPose cameraPose_;
} ;

// 通过欧拉角和平移向量更新ｒ
void CameraPose::UpdateAllWithEulerT()
{
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle_[2], Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle_[1], Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle_[0], Eigen::Vector3d::UnitZ()));
    r_ = yawAngle * pitchAngle * rollAngle;   // 右乘绕旋转轴
    transMatrix_.block<3, 3>(0, 0) = r_;
    transMatrix_.block<3, 1>(0, 3) = t_;
}
}
#endif