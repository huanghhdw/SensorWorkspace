//
// Created by huanghh on 2020/10/25.
//

#include "VisualOdom.h"
#include <stdio.h>

using namespace std;

void ImageProcess::VisualOdom::GetCurrentPose(Eigen::Matrix<double, 4, 4> &pose)
{

}

void ImageProcess::VisualOdom::ProcessImage(cv::Mat &leftImage, cv::Mat &rightImage)
{
    usleep(100000);
}

void ImageProcess::VisualOdom::InitCamInfo(std::string camInfoPath)
{
    FILE *fh = fopen(camInfoPath.c_str(),"r");
    if(fh == NULL){
        return;
    }
    fclose(fh);
    cv::FileStorage fsSettings(camInfoPath.c_str(), cv::FileStorage::READ);
    if(!fsSettings.isOpened())
    {
        std::cerr << "ERROR: Wrong path to settings" << std::endl;
    }
    fsSettings["left_cam"]["image_width"] >> cameraIntrisicLeft_.imageCol_;
    fsSettings["left_cam"]["image_height"] >> cameraIntrisicLeft_.imageRow_;
    fsSettings["left_cam"]["projection_parameters"]["fx"] >> cameraIntrisicLeft_.fx_;
    fsSettings["left_cam"]["projection_parameters"]["fy"] >> cameraIntrisicLeft_.fy_;

    fsSettings["right_cam"]["image_width"] >> cameraIntrisicRight_.imageCol_;
    fsSettings["right_cam"]["image_height"] >> cameraIntrisicRight_.imageRow_;
    fsSettings["right_cam"]["projection_parameters"]["fx"] >> cameraIntrisicRight_.fx_;
    fsSettings["right_cam"]["projection_parameters"]["fy"] >> cameraIntrisicRight_.fy_;
}