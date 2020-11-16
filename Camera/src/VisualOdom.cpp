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

ImageProcess::VisualOdom::VisualOdom()
{
    PoseR_ = Matrix3d::Identity();
    PoseT_ = Vector3d::Zero();
}

void ImageProcess::VisualOdom::GetCurrentPose(Eigen::Matrix<double, 4, 4> &pose)
{
    pose = Matrix4d::Identity();
    pose.block<3, 3>(0, 0) = PoseR_;
    pose.block<3, 1>(3, 0) = PoseT_;
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
        if(lastLeft2DPoints_.size() >= 5U) {
            cv::calcOpticalFlowPyrLK(lastLeftImg_, currentLeftImg_, lastLeft2DPoints_, currentKeypoints, status, error );
            for (uint32_t i = 0; i < lastLeft2DPoints_.size(); i++)
            {
                if ( status[i] == 1 )//状态为0  跟踪 失败
                {
                    keypoints2DValid.push_back(currentKeypoints[i]);
                    keypoints3DValid.push_back(last3DPoint_[i]);
                }
            }
            cv::Mat rvec, tvec, rotationMat;
            Eigen::Matrix3d deltaR;
            Eigen::Vector3d deltaT;
            cv::solvePnP(keypoints3DValid, keypoints2DValid, cameraLeftInfo_.cameraIntrisic_.cameraIntrisic_,
                         cameraLeftInfo_.cameraIntrisic_.distortion_, rvec, tvec); //TODO
            cv::Rodrigues(rvec, rotationMat);
            cv::cv2eigen(rotationMat, deltaR);
            cv::cv2eigen(tvec, deltaT);
            PoseT_ = PoseT_ - PoseR_ * deltaR.transpose() * deltaT;
            PoseR_ = PoseR_ * deltaR.transpose();
        } else {
            cout << "lastLeft2DPoints_.size(): " << lastLeft2DPoints_.size() << endl;
        }
    }

    //step2: 每一帧左右图像orb进行特点匹配，并进行三角化。
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(currentLeftImg_, currentRightImg_, keypoints_1, keypoints_2,descriptors_1, descriptors_2,matches);
    cout << "一共找到了"<< matches.size() << "组匹配点"<<endl;
    //-- 把匹配点转换为vector<Point2f>的形式

    for(int i = 0; i < (int)matches.size(); i++) {
        lastLeft2DPoints_.push_back (keypoints_1[matches[i].queryIdx].pt);
        lastRight2DPoints_.push_back (keypoints_2[matches[i].trainIdx].pt);
        last3DPoint_.push_back(uv2xyz(currentLeft2DPoints_[i], currentRight2DPoints_[i]));
    }
}

void ImageProcess::VisualOdom::ProcessImage(cv::Mat &leftImage, cv::Mat &rightImage)
{
    usleep(50000);
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

cv::Point3f ImageProcess::VisualOdom::uv2xyz(cv::Point2f uvLeft, cv::Point2f uvRight)
{
    Matrix3d mLeftRotationEigen = Matrix3d::Identity();
    Vector3d mLeftTranslationEigen = Vector3d::Zero();
    cv::Mat mLeftRotation;
    cv::eigen2cv(mLeftRotationEigen, mLeftRotation);
    cv::Mat mLeftTranslation;
    cv::eigen2cv(mLeftTranslationEigen, mLeftTranslation);
    cv::Mat mLeftRT = cv::Mat(3,4, CV_64F);//左相机M矩阵
    hconcat(mLeftRotation,mLeftTranslation,mLeftRT);
    cv::Mat mLeftIntrinsic = cameraLeftInfo_.cameraIntrisic_.cameraIntrisic_;
    cv::Mat mLeftM = mLeftIntrinsic * mLeftRT;

    Matrix3d mRightRotationEigen = Matrix3d::Identity();
    Vector3d mRightTranslationEigen = Vector3d::Zero();
    cv::Mat mRightRotation;
    cv::eigen2cv(mRightRotationEigen, mRightRotation);
    cv::Mat mRightTranslation;
    cv::eigen2cv(mRightTranslationEigen, mRightTranslation);
    cv::Mat mRightRT = cv::Mat(3,4, CV_64F);//左相机M矩阵
    hconcat(mRightRotation,mRightTranslation,mRightRT);
    cv::Mat mRightIntrinsic = cameraRightInfo_.cameraIntrisic_.cameraIntrisic_;
    cv::Mat mRightM = mRightIntrinsic * mRightRT;

    //最小二乘法A矩阵
    cv::Mat A = cv::Mat(4,3,CV_64F);
    A.at<double>(0,0) = uvLeft.x * mLeftM.at<double>(2,0) - mLeftM.at<double>(0,0);
    A.at<double>(0,1) = uvLeft.x * mLeftM.at<double>(2,1) - mLeftM.at<double>(0,1);
    A.at<double>(0,2) = uvLeft.x * mLeftM.at<double>(2,2) - mLeftM.at<double>(0,2);

    A.at<double>(1,0) = uvLeft.y * mLeftM.at<double>(2,0) - mLeftM.at<double>(1,0);
    A.at<double>(1,1) = uvLeft.y * mLeftM.at<double>(2,1) - mLeftM.at<double>(1,1);
    A.at<double>(1,2) = uvLeft.y * mLeftM.at<double>(2,2) - mLeftM.at<double>(1,2);

    A.at<double>(2,0) = uvRight.x * mRightM.at<double>(2,0) - mRightM.at<double>(0,0);
    A.at<double>(2,1) = uvRight.x * mRightM.at<double>(2,1) - mRightM.at<double>(0,1);
    A.at<double>(2,2) = uvRight.x * mRightM.at<double>(2,2) - mRightM.at<double>(0,2);

    A.at<double>(3,0) = uvRight.y * mRightM.at<double>(2,0) - mRightM.at<double>(1,0);
    A.at<double>(3,1) = uvRight.y * mRightM.at<double>(2,1) - mRightM.at<double>(1,1);
    A.at<double>(3,2) = uvRight.y * mRightM.at<double>(2,2) - mRightM.at<double>(1,2);

    //最小二乘法B矩阵
    cv::Mat B = cv::Mat(4,1,CV_64F);
    B.at<double>(0,0) = mLeftM.at<double>(0,3) - uvLeft.x * mLeftM.at<double>(2,3);
    B.at<double>(1,0) = mLeftM.at<double>(1,3) - uvLeft.y * mLeftM.at<double>(2,3);
    B.at<double>(2,0) = mRightM.at<double>(0,3) - uvRight.x * mRightM.at<double>(2,3);
    B.at<double>(3,0) = mRightM.at<double>(1,3) - uvRight.y * mRightM.at<double>(2,3);

    cv::Mat XYZ = cv::Mat(3,1,CV_64F);
    //采用SVD最小二乘法求解XYZ
    solve(A,B,XYZ,cv::DECOMP_SVD);

    //cout<<"空间坐标为 = "<<endl<<XYZ<<endl;

    //世界坐标系中坐标
    cv::Point3f world;
    world.x = XYZ.at<double>(0,0);
    world.y = XYZ.at<double>(1,0);
    world.z = XYZ.at<double>(2,0);

    return world;
}

void ImageProcess::VisualOdom::find_feature_matches( const cv::Mat& img_1, const cv::Mat& img_2,
                                  std::vector<cv::KeyPoint>& keypoints_1, std::vector<cv::KeyPoint>& keypoints_2,
                                  cv::Mat &descriptors_1, cv::Mat &descriptors_2,
                                  std::vector< cv::DMatch >& matches)
{
    //-- 初始化
    // used in OpenCV3
    cv::Ptr<cv::FeatureDetector> detector = cv::ORB::create();
    cv::Ptr<cv::DescriptorExtractor> descriptor = cv::ORB::create();
    // use this if you are in OpenCV2
    // Ptr<FeatureDetector> detector = FeatureDetector::create ( "ORB" );
    // Ptr<DescriptorExtractor> descriptor = DescriptorExtractor::create ( "ORB" );
    cv::Ptr<cv::DescriptorMatcher> matcher  = cv::DescriptorMatcher::create ( "BruteForce-Hamming" );
    //-- 第一步:检测 Oriented FAST 角点位置
    detector->detect ( img_1,keypoints_1 );
    detector->detect ( img_2,keypoints_2 );

    //-- 第二步:根据角点位置计算 BRIEF 描述子
    descriptor->compute ( img_1, keypoints_1, descriptors_1 );
    descriptor->compute ( img_2, keypoints_2, descriptors_2 );

    //-- 第三步:对两幅图像中的BRIEF描述子进行匹配，使用 Hamming 距离
    std::vector<cv::DMatch> match;
    //BFMatcher matcher ( NORM_HAMMING );
    matcher->match ( descriptors_1, descriptors_2, match );

    //-- 第四步:匹配点对筛选
    double min_dist=10000, max_dist=0;

    //找出所有匹配之间的最小距离和最大距离, 即是最相似的和最不相似的两组点之间的距离
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    printf ( "-- Max dist : %f \n", max_dist );
    printf ( "-- Min dist : %f \n", min_dist );

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if (match[i].distance <= std::max( 2*min_dist, 30.0 )) {
            matches.push_back (match[i]);
        }
    }
}

//double leftIntrinsic[3][3] = {cameraLeftInfo_.cameraIntrisic_.fx_, 0.0, cameraLeftInfo_.cameraIntrisic_.cx_,
//                              0.0, cameraLeftInfo_.cameraIntrisic_.fy_, cameraLeftInfo_.cameraIntrisic_.cy_,
//                              0.0, 0.0, 1.0};
//
//double rightIntrinsic[3][3] = {cameraRightInfo_.cameraIntrisic_.fx_, 0.0, cameraRightInfo_.cameraIntrisic_.cx_,
//                               0.0, cameraRightInfo_.cameraIntrisic_.fy_, cameraRightInfo_.cameraIntrisic_.cy_,
//                               0.0, 0.0, 1.0};
//
//double leftTranslation[1][3] = {0.0, 0.0, 0.0};
//double leftRotation[3][3] = {1.0,0.0,0.0,
//                             0.0,1.0,0.0,
//                             0.0,0.0,1.0};
//double rightTranslation[1][3] = {StereoT_.x(), StereoT_.y(), StereoT_.z()};
//double rightRotation[3][3] = {StereoR_(0,0),StereoR_(0,1),StereoR_(0,2),
//                              StereoR_(1,0),StereoR_(1,1),StereoR_(1,2),
//                              StereoR_(2,0),StereoR_(2,1),StereoR_(2,2)};