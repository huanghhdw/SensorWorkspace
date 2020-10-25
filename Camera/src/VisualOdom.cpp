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
    int sizeU, sizeV;
    fsSettings["image_width"] >> sizeU;
    fsSettings["image_height"] >> sizeV;
    double fx, fy;
    fsSettings["projection_parameters"]["fx"] >> fx;
    fsSettings["projection_parameters"]["fy"] >> fy;
    cout << "sizeU: " << sizeU << " sizeV: " << sizeV << endl;
    cout << "fx: " << fx << " fy: " << fy << endl;
    fsSettings.release();
}