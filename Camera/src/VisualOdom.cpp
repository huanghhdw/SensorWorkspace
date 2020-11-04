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
    std::vector<cv::KeyPoint> keypoints_1, keypoints_2;
    cv::Mat descriptors_1, descriptors_2;
    std::vector<cv::DMatch> matches;
    find_feature_matches(currentLeftImg_, currentRightImg_, keypoints_1, keypoints_2,descriptors_1, descriptors_2,matches);
    cout<<"一共找到了"<<matches.size() <<"组匹配点"<<endl;
    //-- 把匹配点转换为vector<Point2f>的形式
    vector<Point2f> leftPoints;
    vector<Point2f> rightPoints;
    std::vector<Point3f> worldPoints;
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

    if (isFirst) {

        return;
    }



    std::vector<Eigen::Vector3f> lastLeft3dPoint;
    std::vector<Eigen::Vector3f> currentLeft3dPoint;
    Eigen::Matrix3f dR;
    Eigen::Vector3f dt;
    ICP(lastLeft3dPoint, currentLeft3dPoint, dR, dt);

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
        ProcessStereoImg(true);
        return;
    }
    currentLeftImg_ = leftImage;
    currentRightImg_ = rightImage;
    ProcessStereoImg(false);
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

//#include <iostream>//输入输出流
//#include <fstream>//文件数据流
//#include <list>//列表
//#include <vector>//容器
//#include <chrono>//计时
//using namespace std;
////opencv
//#include <opencv2/core/core.hpp>
//#include <opencv2/highgui/highgui.hpp>
//#include <opencv2/features2d/features2d.hpp>//二维特征
//#include <opencv2/video/tracking.hpp>//跟踪算法
//
//int main( int argc, char** argv )
//{
//    if ( argc != 2 )
//    {
//        cout<<"用法（TMU数据集）: useLK path_to_dataset"<<endl;
//        return 1;
//    }
//    /*
//     TMU数据集:
//     rgb.txt       记录了RGB图像的采集时间 和对应的文件名
//     depth.txt   记录了深度图像的采集时间 和对应的文件名
//     /rgb           存放 rgb图像  png格式彩色图为八位3通道
//     /depth       存放深度图像  深度图为 16位单通道图像
//     groundtruth.txt 为外部运动捕捉系统采集到的相机位姿  time,tx,ty,tz,qx,qy,qz,qw
//     RGB图像和 深度图像采集独立的 时间不同时，需要对数据进行一次时间上的对齐，
//     时间间隔相差一个阈值认为是同一匹配图
//     可使用 associate.py脚步完成   python associate.py rgb.txt   depth.txt  > associate.txt
//
//     */
//    string path_to_dataset = argv[1];//数据集路径
//    string associate_file = path_to_dataset + "/associate.txt";//匹配好的图像
//
//    ifstream fin( associate_file );//读取文件   ofstream 输出文件流
//    if ( !fin ) //打开文件失败
//    {
//        cerr<<"找不到 associate.txt!"<<endl;
//        return 1;
//    }
//
//    string rgb_file, depth_file, time_rgb, time_depth;//字符串流
//    list<cv::Point2f > keypoints;      // 因为要删除跟踪失败的点，使用list
//    cv::Mat color, depth, last_color;
//
//    for ( int index=0; index<100; index++ )
//    {
//        fin>>time_rgb>>rgb_file>>time_depth>>depth_file;
//        //rgb图像对应时间 rgb图像 深度图像对应时间 深度图像
//        color = cv::imread( path_to_dataset+"/"+rgb_file );//彩色图像
//        depth = cv::imread( path_to_dataset+"/"+depth_file, -1 );//原始图像
//        if (index ==0 )//第一帧图像
//        {
//            // 对第一帧提取FAST特征点
//            vector<cv::KeyPoint> kps;//关键点容器
//            cv::Ptr<cv::FastFeatureDetector> detector = cv::FastFeatureDetector::create();//检测器
//            detector->detect( color, kps );//检测关键点 放入容器内
//            for ( auto kp:kps )
//                keypoints.push_back( kp.pt );//存入 关键点 二维坐标  列表内
//            last_color = color;
//            continue;// index ==0 时 执行到这里 以下不执行
//        }
//
//        if ( color.data==nullptr || depth.data==nullptr )//图像读取错误 跳过
//            continue;
//
//        // 对其他帧用LK跟踪特征点
//        vector<cv::Point2f> next_keypoints; //下一帧关键点
//        vector<cv::Point2f> prev_keypoints; //上一帧关键点
//        for ( auto kp:keypoints )
//            prev_keypoints.push_back(kp);//上一帧关键点
//        vector<unsigned char> status;// 关键点 跟踪状态标志
//        vector<float> error; //信息
//        chrono::steady_clock::time_point t1 = chrono::steady_clock::now();//计时 开始
//        cv::calcOpticalFlowPyrLK( last_color, color, prev_keypoints, next_keypoints, status, error );
//        chrono::steady_clock::time_point t2 = chrono::steady_clock::now();//计时 结束
//        chrono::duration<double> time_used = chrono::duration_cast<chrono::duration<double>>( t2-t1 );
//        cout<<"LK Flow use time："<<time_used.count()<<" seconds."<<endl;
//        // 把跟丢的点删掉
//        int i=0;
//        for ( auto iter=keypoints.begin(); iter!=keypoints.end(); i++)
//        {
//            if ( status[i] == 0 )//状态为0  跟踪 失败
//            {
//                iter = keypoints.erase(iter);// list列表 具有管理表格元素能力  删除跟踪失败的点
//                continue;//跳过执行 后面两步
//            }
//            *iter = next_keypoints[i];// s
//            iter++;// 迭代
//        }
//
//        cout<<"跟踪的关键点数量 tracked keypoints: "<<keypoints.size()<<endl;
//        if (keypoints.size() == 0)
//        {
//            cout<<"所有关键点已丢失 all keypoints are lost."<<endl;
//            break;
//        }
//        // 画出 keypoints
//        cv::Mat img_show = color.clone();//复制图像 开辟新的内存空间
//        for ( auto kp:keypoints )
//            cv::circle(img_show, kp, 10, cv::Scalar(0, 240, 0), 1);//画圆  图像  中心点  半径 颜色 粗细
//        cv::imshow("corners", img_show);//显示
//        cv::waitKey(0);//等待按键按下 遍历下一张图片
//        last_color = color;
//    }
//    return 0;
//}