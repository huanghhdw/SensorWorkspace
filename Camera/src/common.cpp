#include <common.h>

using namespace  Util;

// 通过欧拉角和平移向量更新ｒ
void CameraPose::UpdateAllWithEulerT()
{
//    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle_[2], Eigen::Vector3f::UnitX()));
//    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle_[1], Eigen::Vector3f::UnitY()));
//    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle_[0], Eigen::Vector3f::UnitZ()));
//    r_ = yawAngle * pitchAngle * rollAngle;   // 右乘绕旋转轴
//    transMatrix_.block<3, 3>(0, 0) = r_;
//    transMatrix_.block<3, 1>(0, 3) = t_;
}

void Util::find_feature_matches ( const cv::Mat& img_1, const cv::Mat& img_2,
                            std::vector<cv::KeyPoint>& keypoints_1,
                            std::vector<cv::KeyPoint>& keypoints_2,
                            std::vector< cv::DMatch >& matches )
{
    //-- 初始化
    cv::Mat descriptors_1, descriptors_2;
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


//************************************
// Description: 根据左右相机中成像坐标求解空间坐标
// Method:    uv2xyz
// FullName:  uv2xyz
// Access:    public
// Parameter: Point2f uvLeft
// Parameter: Point2f uvRight
// Returns:   cv::Point3f
// Author:    小白
// Date:      2017/01/10
// History:
//************************************
cv::Point3f Util::uv2xyz(cv::Point2f uvLeft, cv::Point2f uvRight,
        float leftIntrinsic[3][3], float leftRotation[3][3], float leftTranslation[1][3],
        float rightIntrinsic[3][3], float rightRotation[3][3], float rightTranslation[1][3])
{
    //  [u1]      |X|					  [u2]      |X|
    //Z*|v1| = Ml*|Y|					Z*|v2| = Mr*|Y|
    //  [ 1]      |Z|					  [ 1]      |Z|
    //			  |1|								|1|
    cv::Mat mLeftRotation = cv::Mat(3,3,CV_32F,leftRotation);
    cv::Mat mLeftTranslation = cv::Mat(3,1,CV_32F,leftTranslation);
    cv::Mat mLeftRT = cv::Mat(3,4,CV_32F);//左相机M矩阵
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