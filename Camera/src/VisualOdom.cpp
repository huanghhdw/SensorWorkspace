//
// Created by huanghh on 2020/10/25.
//

#include "VisualOdom.h"
#include <stdio.h>

using namespace std;

void ImageProcess::VisualOdom::GetCurrentPose(Eigen::Matrix<double, 4, 4> &pose)
{

}

void ImageProcess::VisualOdom::FindKeypointAndTriangulation()
{

}

void ImageProcess::VisualOdom::MatchPointAndSolvePnP()
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
    MatchPointAndSolvePnP();
    lastLeftImg_ = currentLeftImg_;
    lastRightImg_ = currentRightImg_;
    usleep(100000);
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
}


//左相机内参数矩阵
float leftIntrinsic[3][3] = {4037.82450,			 0,		947.65449,
                             0,	3969.79038,		455.48718,
                             0,			 0,				1};

//左相机旋转矩阵
float leftRotation[3][3] = {0.912333,		-0.211508,		 0.350590,
                            0.023249,		-0.828105,		-0.560091,
                            0.408789,		 0.519140,		-0.750590};
//左相机平移向量
float leftTranslation[1][3] = {-127.199992, 28.190639, 1471.356768};

//右相机内参数矩阵
float rightIntrinsic[3][3] = {3765.83307,			 0,		339.31958,
                              0,	3808.08469,		660.05543,
                              0,			 0,				1};

//右相机旋转矩阵
float rightRotation[3][3] = {-0.134947,		 0.989568,		-0.050442,
                             0.752355,		 0.069205,		-0.655113,
                             -0.644788,		-0.126356,		-0.753845};
//右相机平移向量
float rightTranslation[1][3] = {50.877397, -99.796492, 1507.312197};

cv::Point3f uv2xyz(cv::Point2f uvLeft,cv::Point2f uvRight)
{
    //  [u1]      |X|					  [u2]      |X|
    //Z*|v1| = Ml*|Y|					Z*|v2| = Mr*|Y|
    //  [ 1]      |Z|					  [ 1]      |Z|
    //			  |1|								|1|
    cv::Mat mLeftRotation = cv::Mat(3,3,CV_32F,leftRotation);
    cv::Mat mLeftTranslation = cv::Mat(3, 1, CV_32F, leftTranslation);
    cv::Mat mLeftRT = cv::Mat(3, 4, CV_32F);//左相机M矩阵
    hconcat(mLeftRotation,mLeftTranslation,mLeftRT);
    cv::Mat mLeftIntrinsic = cv::Mat(3,3,CV_32F,leftIntrinsic);
    cv::Mat mLeftM = mLeftIntrinsic * mLeftRT;
    //cout<<"左相机M矩阵 = "<<endl<<mLeftM<<endl;

    cv::Mat mRightRotation = cv::Mat(3,3,CV_32F,rightRotation);
    cv::Mat mRightTranslation = cv::Mat(3,1,CV_32F,rightTranslation);
    cv::Mat mRightRT = cv::Mat(3,4,CV_32F);//右相机M矩阵
    hconcat(mRightRotation,mRightTranslation,mRightRT);
    cv::Mat mRightIntrinsic = cv::Mat(3,3,CV_32F,rightIntrinsic);
    cv::Mat mRightM = mRightIntrinsic * mRightRT;
    //cout<<"右相机M矩阵 = "<<endl<<mRightM<<endl;

    //最小二乘法A矩阵
    cv::Mat A = cv::Mat(4,3,CV_32F);
    A.at<float>(0,0) = uvLeft.x * mLeftM.at<float>(2,0) - mLeftM.at<float>(0,0);
    A.at<float>(0,1) = uvLeft.x * mLeftM.at<float>(2,1) - mLeftM.at<float>(0,1);
    A.at<float>(0,2) = uvLeft.x * mLeftM.at<float>(2,2) - mLeftM.at<float>(0,2);

    A.at<float>(1,0) = uvLeft.y * mLeftM.at<float>(2,0) - mLeftM.at<float>(1,0);
    A.at<float>(1,1) = uvLeft.y * mLeftM.at<float>(2,1) - mLeftM.at<float>(1,1);
    A.at<float>(1,2) = uvLeft.y * mLeftM.at<float>(2,2) - mLeftM.at<float>(1,2);

    A.at<float>(2,0) = uvRight.x * mRightM.at<float>(2,0) - mRightM.at<float>(0,0);
    A.at<float>(2,1) = uvRight.x * mRightM.at<float>(2,1) - mRightM.at<float>(0,1);
    A.at<float>(2,2) = uvRight.x * mRightM.at<float>(2,2) - mRightM.at<float>(0,2);

    A.at<float>(3,0) = uvRight.y * mRightM.at<float>(2,0) - mRightM.at<float>(1,0);
    A.at<float>(3,1) = uvRight.y * mRightM.at<float>(2,1) - mRightM.at<float>(1,1);
    A.at<float>(3,2) = uvRight.y * mRightM.at<float>(2,2) - mRightM.at<float>(1,2);

    //最小二乘法B矩阵
    cv::Mat B = cv::Mat(4,1,CV_32F);
    B.at<float>(0,0) = mLeftM.at<float>(0,3) - uvLeft.x * mLeftM.at<float>(2,3);
    B.at<float>(1,0) = mLeftM.at<float>(1,3) - uvLeft.y * mLeftM.at<float>(2,3);
    B.at<float>(2,0) = mRightM.at<float>(0,3) - uvRight.x * mRightM.at<float>(2,3);
    B.at<float>(3,0) = mRightM.at<float>(1,3) - uvRight.y * mRightM.at<float>(2,3);

    cv::Mat XYZ = cv::Mat(3,1,CV_32F);
    //采用SVD最小二乘法求解XYZ
    solve(A,B,XYZ,cv::DECOMP_SVD);

    //cout<<"空间坐标为 = "<<endl<<XYZ<<endl;

    //世界坐标系中坐标
    cv::Point3f world;
    world.x = XYZ.at<float>(0,0);
    world.y = XYZ.at<float>(1,0);
    world.z = XYZ.at<float>(2,0);

    return world;
}