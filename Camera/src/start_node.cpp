#include <iostream>
#include <thread>
#include <stdio.h>
#include <list>
#include <unistd.h>
#include <sys/wait.h>
#include <opencv2/opencv.hpp>
#include <string>
#include <atomic>
#include <Eigen/Core>
#include <VisualOdom.h>

using namespace std;
using namespace Eigen;

std::atomic<bool> readyToExit(false);
std::list<cv::Mat> imageLeftList;
std::list<cv::Mat> imageRightList;

void Stop(int)
{
    cout << "Program is soon stop!" << endl;
    readyToExit = true;
}

void ProcessDataThread() {
    ImageProcess::VisualOdom visualOdom;
    visualOdom.InitCamInfo("/home/huanghh/hhh_ws/SensorWorkspace/Camera/config/kitti_odom/cam04-12.yaml");
    string OUTPUT_FOLDER = "/home/huanghh/";
    FILE *outFile;
    outFile = fopen((OUTPUT_FOLDER + "/vio.txt").c_str(), "w");
    if (outFile == NULL)
        printf("Output path dosen't exist: %s\n", OUTPUT_FOLDER.c_str());
    while (!readyToExit) {
        usleep(5000);
        if (outFile != NULL) {
            Eigen::Matrix<double, 4, 4> pose = Eigen::Matrix<double, 4, 4>::Zero();
            if (!imageLeftList.empty() && !imageRightList.empty()) {
                cv::Mat leftImage =  imageLeftList.front();
                cv::Mat rightImage =  imageRightList.front();
                imageLeftList.pop_front();
                imageRightList.pop_front();
                visualOdom.ProcessImage(leftImage, rightImage);
                visualOdom.GetCurrentPose(pose);
                cv::imshow("leftImage", leftImage);
                cv::imshow("rightImage", rightImage);
                cv::waitKey(1);
//            fprintf (outFile, "%f %f %f %f %f %f %f %f %f %f %f %f \n",pose(0,0), pose(0,1), pose(0,2),pose(0,3),
//                     pose(1,0), pose(1,1), pose(1,2),pose(1,3),
//                     pose(2,0), pose(2,1), pose(2,2),pose(2,3));
            }
        }
    }
    if (outFile != NULL)
        fclose(outFile);
}

void PubDataThread()
{
    string sequence = "/home/huanghh/data/08";
    string dataPath = sequence + "/";

    //读取时间戳
    FILE* file;
    file = std::fopen((dataPath + "times.txt").c_str() , "r");
    if(file == NULL){
        printf("cannot find file: %stimes.txt\n", dataPath.c_str());
        return;
    }
    double imageTime;
    vector<double> imageTimeList;
    while ( fscanf(file, "%lf", &imageTime) != EOF)
    {
        imageTimeList.push_back(imageTime);
    }
    std::fclose(file);
    uint32_t processImageNum = 0;

    string leftImagePath, rightImagePath;
    cv::Mat imLeft, imRight;

    //发布图像数据
    while (!readyToExit) {
        if (processImageNum > imageTimeList.size()) {
            break;
        }
        if (imageLeftList.size() == 10) {
            usleep(5000);
            continue;
        }
        printf("\nprocess image %d\n", (int)processImageNum);
        stringstream ss;
        ss << setfill('0') << setw(6) << processImageNum;
        leftImagePath = dataPath + "image_0/" + ss.str() + ".png";
        rightImagePath = dataPath + "image_1/" + ss.str() + ".png";
        imLeft = cv::imread(leftImagePath, CV_LOAD_IMAGE_GRAYSCALE );
        imRight = cv::imread(rightImagePath, CV_LOAD_IMAGE_GRAYSCALE );
        imageLeftList.push_back(imLeft);
        imageRightList.push_back(imRight);
        processImageNum++;
    }
}

int main(int argc, char** argv)
{
    ssignal(SIGINT, Stop);
    thread pubDataThread(PubDataThread);
    thread processDataThread(ProcessDataThread);
    pubDataThread.join();
    processDataThread.join();
	return 0;
}