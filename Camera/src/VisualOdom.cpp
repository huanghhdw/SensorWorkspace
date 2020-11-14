//
// Created by huanghh on 2020/10/25.
//

#include "VisualOdom.h"
#include <stdio.h>
#include <map>
using namespace std;
using namespace Eigen;
using namespace cv;
using namespace Util;


void ImageProcess::VisualOdom::GetCurrentPose(Eigen::Matrix<double, 4, 4> &pose)
{

}

void ImageProcess::VisualOdom::ProcessStereoImg(bool isFirst)
{
    //step1: 前一帧左目图像中已经三角化的坐标点与当前帧左目图像进行光流跟踪，得到3D-2D对应关系，进行PNP解算。
    if (!isFirstFrame_) {
        vector<cv::Point2d> currentKeypoints;
        vector<cv::Point2d> keypoints2DValid;
        vector<cv::Point3d> keypoints3DValid;
        vector<unsigned char> status;   // 关键点跟踪状态标志
        vector<float> error;            //信息
        cv::calcOpticalFlowPyrLK(lastLeftImg_, currentLeftImg_, lastLeft2DPoints_, currentKeypoints, status, error );
        for (int i = 0; i < lastLeft2DPoints_.size(); i++)
        {
            if ( status[i] == 1 )//状态为0  跟踪 失败
            {
                keypoints2DValid.push_back(currentKeypoints[i]);
                keypoints3DValid.push_back(last3DPoint_[i]);
            }
        }
        cv::Mat rvec, tvec, rotationMat;
        Eigen::Matrix3f deltaR, deltaT;
        cv::solvePnP(keypoints3DValid, keypoints2DValid, cameraLeftInfo_.cameraIntrisic_.cameraIntrisic_,
                     cameraLeftInfo_.cameraIntrisic_.distortion_, rvec, tvec); //TODO
        cv::Rodrigues(rvec, rotationMat);
        cv::cv2eigen(rotationMat, deltaR);
        cv::cv2eigen(tvec, deltaT);
    }

    //step2: 每一帧左右图像orb进行特点匹配，并进行三角化。
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(currentLeftImg_, currentRightImg_, keypoints_1, keypoints_2,descriptors_1, descriptors_2,matches);
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;
    //-- 把匹配点转换为vector<Point2f>的形式

    float leftIntrinsic[3][3] = {cameraLeftInfo_.cameraIntrisic_.fx_, 0.0, cameraLeftInfo_.cameraIntrisic_.cx_,
                                 0.0, cameraLeftInfo_.cameraIntrisic_.fy_, cameraLeftInfo_.cameraIntrisic_.cy_,
                                 0.0, 0.0, 1.0};

    float rightIntrinsic[3][3] = {cameraRightInfo_.cameraIntrisic_.fx_, 0.0, cameraRightInfo_.cameraIntrisic_.cx_,
                                 0.0, cameraRightInfo_.cameraIntrisic_.fy_, cameraRightInfo_.cameraIntrisic_.cy_,
                                 0.0, 0.0, 1.0};

    float leftTranslation[1][3] = {0.0, 0.0, 0.0};
    float leftRotation[3][3] = {1.0,0.0,0.0,
                                0.0,1.0,0.0,
                                0.0,0.0,1.0};
    float rightTranslation[1][3] = {StereoT_.x(), StereoT_.y(), StereoT_.z()};
    float rightRotation[3][3] = {StereoR_(0,0),StereoR_(0,1),StereoR_(0,2),
                                 StereoR_(1,0),StereoR_(1,1),StereoR_(1,2),
                                 StereoR_(2,0),StereoR_(2,1),StereoR_(2,2)};

    for(int i = 0; i < (int)matches.size(); i++) {
        lastLeft2DPoints_.push_back (keypoints_1[matches[i].queryIdx].pt);
        lastRight2DPoints_.push_back (keypoints_2[matches[i].trainIdx].pt);
        last3DPoint_.push_back(uv2xyz(currentLeft2DPoints_[i], currentRight2DPoints_[i],
               leftIntrinsic, leftRotation, leftTranslation,
               rightIntrinsic, rightRotation, rightTranslation));
    }
}

void ImageProcess::VisualOdom::ProcessImage(cv::Mat &leftImage, cv::Mat &rightImage)
{
    if (isFirstFrame_) {
        isFirstFrame_ = false;
        lastLeftImg_ = currentLeftImg_ = leftImage;
        lastRightImg_ =  currentRightImg_ = rightImage;
        ProcessStereoImg(true);
        return;
    }
    currentLeftImg_ = leftImage;
    currentRightImg_ = rightImage;
    ProcessStereoImg(false);
    lastLeftImg_ = currentLeftImg_;
    lastRightImg_ = currentRightImg_;
}

void ImageProcess::VisualOdom::InitCamInfo(std::string camInfoPath)
{
    FILE *fh = fopen(camInfoPath.c_str(),"r");
    if (fh == NULL) {
        return;
    }
    fclose(fh);
    cv::FileStorage fsSettings(camInfoPath.c_str(), cv::FileStorage::READ);
    if (!fsSettings.isOpened()) {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    fsSettings["left_cam"]["image_width"] >> cameraLeftInfo_.cameraIntrisic_.imageCol_;
    fsSettings["left_cam"]["image_height"] >> cameraLeftInfo_.cameraIntrisic_.imageRow_;
    fsSettings["left_cam"]["projection_parameters"]["fx"] >>cameraLeftInfo_.cameraIntrisic_.fx_;
    fsSettings["left_cam"]["projection_parameters"]["fy"] >> cameraLeftInfo_.cameraIntrisic_.fy_;

    fsSettings["right_cam"]["image_width"] >> cameraRightInfo_.cameraIntrisic_.imageCol_;
    fsSettings["right_cam"]["image_height"] >> cameraRightInfo_.cameraIntrisic_.imageRow_;
    fsSettings["right_cam"]["projection_parameters"]["fx"] >> cameraRightInfo_.cameraIntrisic_.fx_;
    fsSettings["right_cam"]["projection_parameters"]["fy"] >> cameraRightInfo_.cameraIntrisic_.fy_;

    StereoT_.x() = 0.6;
    StereoT_.y() = 0.0;
    StereoT_.z() = 0.0;
    StereoR_ << 1.0,0.0,0.0,
                0.0,1.0,0.0,
                0.0,0.0,1.0;
}