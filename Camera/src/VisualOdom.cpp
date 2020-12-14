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
    if (!isFirst) {
        vector<cv::Point2f> currentKeypoints;
        vector<cv::Point2f> keypoints2DValid;
        vector<cv::Point3f> keypoints3DValid;
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
            cout << "PoseT_: x = " << PoseT_.x() << "  y:" << PoseT_.y() << " z:" << PoseT_.z() << endl;
        } else {
            cout << "lastLeft2DPoints_.size(): " << lastLeft2DPoints_.size() << endl;
        }
    }

    //step2: 每一帧左右图像orb进行特点匹配，并进行三角化。
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(currentLeftImg_, currentRightImg_, keypoints_1, keypoints_2,descriptors_1, descriptors_2,matches);
    for(int i = 0; i < (int)matches.size(); i++) {
        cv::Point2f a = keypoints_1[matches[i].queryIdx].pt;
        cv::Point2f b = keypoints_2[matches[i].trainIdx].pt;
        cv::Point3f c = uv2xyz(a, b);
        if(c.z > 0) {
            lastLeft2DPoints_.push_back (a);
            lastRight2DPoints_.push_back (b);
            last3DPoint_.push_back(c);
        }
    }
}

void ImageProcess::VisualOdom::ProcessImage(cv::Mat &leftImage, cv::Mat &rightImage)
{
    usleep(100000);
    if (isFirstFrame_) {
        isFirstFrame_ = false;
        lastLeftImg_ = leftImage;
        currentLeftImg_ = leftImage;
        lastRightImg_ = rightImage;
        currentRightImg_ = rightImage;
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
    cv::Mat cam_k;

    fsSettings["left_cam"]["image_width"] >> cameraLeftInfo_.cameraIntrisic_.imageCol_;
    fsSettings["left_cam"]["image_height"] >> cameraLeftInfo_.cameraIntrisic_.imageRow_;
    fsSettings["right_cam"]["image_width"] >> cameraRightInfo_.cameraIntrisic_.imageCol_;
    fsSettings["right_cam"]["image_height"] >> cameraRightInfo_.cameraIntrisic_.imageRow_;

    cameraLeftInfo_.cameraIntrisic_.cameraIntrisic_ = cv::Mat(3,3, CV_64F);
    fsSettings["left_cam"]["intrisic_matrix"] >> cameraLeftInfo_.cameraIntrisic_.cameraIntrisic_;

    cameraRightInfo_.cameraIntrisic_.cameraIntrisic_ = cv::Mat(3,3, CV_64F);
    fsSettings["right_cam"]["intrisic_matrix"] >> cameraRightInfo_.cameraIntrisic_.cameraIntrisic_;


    StereoT_.x() = 0.6;
    StereoT_.y() = 0.0;
    StereoT_.z() = 0.0;
    StereoR_ << 1.0,0.0,0.0,
                0.0,1.0,0.0,
                0.0,0.0,1.0;

    cout << "Camera Left  Intrisic Matrix:" << cameraLeftInfo_.cameraIntrisic_.cameraIntrisic_ << endl;
    cout << "Camera Right Intrisic Matrix:" << cameraRightInfo_.cameraIntrisic_.cameraIntrisic_ << endl;
    cout << "Init Success!!" << endl;
    //sleep(1);
}

cv::Point3f ImageProcess::VisualOdom::uv2xyz(cv::Point2f uvLeft, cv::Point2f uvRight)
{
    MatrixXd A(6, 5);
    A.block<3, 3>(0, 0) = -1.0 * Matrix3d::Identity();
    A.block<3, 3>(3, 0) = -1.0 * Matrix3d::Identity();
    MatrixXd l1(3, 1) ;
    l1 << (double)uvLeft.x, (double)uvLeft.y, 1.0;
    MatrixXd l2(3, 1) ;
    l2 << (double)uvRight.x, (double)uvRight.y, 1.0;

    Matrix3d kLeft, kRight;
    cv2eigen(cameraLeftInfo_.cameraIntrisic_.cameraIntrisic_, kLeft);
    cv2eigen(cameraRightInfo_.cameraIntrisic_.cameraIntrisic_, kRight);
    Matrix3d RLeft = Matrix3d::Identity();
    Vector3d TLeft = Vector3d::Zero();
    Matrix3d RRight = Matrix3d::Identity();
    Vector3d TRight = Vector3d::Zero();
    TRight.x() = -0.5;
    TRight.y() = 0;
    TRight.z() = 0;
    Vector3d L1 = (kLeft * RLeft).inverse() * l1;
    Vector3d L2 = (kRight * RRight).inverse() * l2;
    A.block<3, 1> (0,3) = L1;
    A.block<3, 1> (3,4) = L2;
    Vector3d b1, b2;
    b1 =  (kLeft * RLeft).inverse() * kLeft * TLeft;
    b2 =  (kRight * RRight).inverse() * kRight * TRight;
    MatrixXd B(6, 1);
    B << b1, b2;
    MatrixXd Result(5, 1);
    Result = (A.transpose() * A).inverse() * A.transpose() * B;
    Point3f world;
    world.x = Result(0,0);
    world.y = Result(1,0);
    world.z = Result(2,0);
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
    for (int i = 0; i < descriptors_1.rows; i++)
    {
        double dist = match[i].distance;
        if ( dist < min_dist ) min_dist = dist;
        if ( dist > max_dist ) max_dist = dist;
    }

    //当描述子之间的距离大于两倍的最小距离时,即认为匹配有误.但有时候最小距离会非常小,设置一个经验值30作为下限.
    for ( int i = 0; i < descriptors_1.rows; i++ )
    {
        if (match[i].distance <= std::max( 2*min_dist, 30.0 )) {
            matches.push_back (match[i]);
        }
    }
}