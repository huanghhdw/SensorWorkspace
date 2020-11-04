#ifndef  COMMON_H
#define  COMMON_H

#include<Eigen/Core>
#include<Eigen/Dense>
#include<vector>
#include <opencv2/opencv.hpp>

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
    std::vector<double> distortion_;
};

class CameraPose {
public:
    Eigen::Matrix3f r_;
    Eigen::Vector3f t_;
    Eigen::Vector3f eulerAngle_;    // yaw, pitch, roll, 弧度
    Eigen::MatrixXf transMatrix_ = Eigen::MatrixXf::Zero(4, 4);
    void UpdateAllWithEulerT();
} ;

class CameraInfo {
public:
    CameraIntrisic cameraIntrisic_;
    CameraPose cameraPose_;
};

void find_feature_matches( const cv::Mat& img_1, const cv::Mat& img_2,
                            std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                            cv::Mat &descriptors_1, cv::Mat &descriptors_2,
                            std::vector< cv::DMatch >& matches );

cv::Point3f uv2xyz(cv::Point2f uvLeft,cv::Point2f uvRight,
                   float leftIntrinsic[3][3], float leftRotation[3][3], float leftTranslation[1][3],
                   float rightIntrinsic[3][3], float rightRotation[3][3], float rightTranslation[1][3]);

void ICP(const std::vector<Eigen::Vector3f>& pts1, const std::vector<Eigen::Vector3f>& pts2, Eigen::Matrix3f &R_12, Eigen::Vector3f &t_12);
}
#endif