//
// Created by huanghh on 2020/10/25.
//

#include "VisualOdom.h"
#include <stdio.h>

using namespace std;
using namespace Eigen;
using namespace cv;
using namespace Util;


void ImageProcess::VisualOdom::GetCurrentPose(Eigen::Matrix<double, 4, 4> &pose)
{

}

void ImageProcess::VisualOdom::FindKeypointAndTriangulation()
{
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(currentLeftImg_, currentRightImg_, keypoints_1, keypoints_2, matches);
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;
    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> leftPoints;
    vector<Point2f> rightPoints;
    vector<Point3f> worldPoints;
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
                                 StereoR_(2,0),StereoR_(2,1),StereoR_(2,2),};

    for(int i = 0; i < (int)matches.size(); i++) {
        leftPoints.push_back (keypoints_1[matches[i].queryIdx].pt);
        rightPoints.push_back (keypoints_2[matches[i].trainIdx].pt);
        worldPoints.push_back(uv2xyz(leftPoints[i], rightPoints[i],
               leftIntrinsic, leftRotation, leftTranslation,
               rightIntrinsic, rightRotation, rightTranslation));
    }
}

void ImageProcess::VisualOdom::MatchPointAndICP()
{

}

void ImageProcess::VisualOdom::ProcessImage(cv::Mat &leftImage, cv::Mat &rightImage)
{
    if (isFirstFrame_) {
        isFirstFrame_ = false;
        lastLeftImg_ = leftImage;
        currentLeftImg_ = leftImage;
        lastRightImg_ = rightImage;
        currentRightImg_ = rightImage;
        FindKeypointAndTriangulation();
        return;
    }
    currentLeftImg_ = leftImage;
    currentRightImg_ = rightImage;
    FindKeypointAndTriangulation();
    MatchPointAndICP();
    lastLeftImg_ = currentLeftImg_;
    lastRightImg_ = currentRightImg_;
    //usleep(100000);
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

