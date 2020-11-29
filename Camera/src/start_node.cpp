#include <iostream>
#include <string>
#include <thread>
#include <queue>
#include <atomic>
#include <stdio.h>
#include <unistd.h>
#include <sys/wait.h>

#include <opencv2/opencv.hpp>
#include <Eigen/Core>

#include <VisualOdom.h>

using namespace std;
using namespace Eigen;

std::atomic<bool> readyToExit(false);
std::queue<cv::Mat> imageLeftList;
std::queue<cv::Mat> imageRightList;

void Stop(int)
{
    std::cout << "Program is soon stop!" << std::endl;
    readyToExit = true;
}

void ProcessDataThread(string cameraConfigPath) {
    ImageProcess::VisualOdom visualOdom;
    visualOdom.InitCamInfo(cameraConfigPath);
    FILE *outFile;
    outFile = fopen("/home/huanghh/vio.txt", "w");
    if (outFile == NULL)
        printf("Output path dosen't exist\n");
    while (!readyToExit) {
        usleep(5000);
        if (outFile != NULL) {
            Eigen::Matrix<double, 4, 4> pose = Eigen::Matrix<double, 4, 4>::Zero();
            if (!imageLeftList.empty() && !imageRightList.empty()) {
                cv::Mat leftImage =  imageLeftList.front();
                cv::Mat rightImage =  imageRightList.front();
                imageLeftList.pop();
                imageRightList.pop();
                visualOdom.ProcessImage(leftImage, rightImage);
                visualOdom.GetCurrentPose(pose);
                //For Visualization
                int totalRows = leftImage.rows + rightImage.rows;
                cv::Mat mergedPicture(totalRows, leftImage.cols, leftImage.type());
                cv::Mat submat = mergedPicture.rowRange(0, leftImage.rows);
                leftImage.copyTo(submat);
                submat = mergedPicture.rowRange(leftImage.rows, totalRows);
                rightImage.copyTo(submat);
                cv::imshow("Image", mergedPicture);
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

void PubDataThread(string imgDataPath)
{
    string sequence = imgDataPath;
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
        imageLeftList.push(imLeft);
        imageRightList.push(imRight);
        processImageNum++;
    }
}

int main(int argc, char** argv)
{
    ssignal(SIGINT, Stop);
    if (argc != 3) {
        cout << "Please Enter with KITTI Data Path and Camera Config File Path!" << endl;
        cout << "For Example:" << endl;
        cout << "./HSlam /home/huanghh/data/08 /home/huanghh/hhh_ws/SensorWorkspace/Camera/config/camera.yaml" << endl;
        return -1;
    }
    thread pubDataThread(PubDataThread, argv[1]);
    thread processDataThread(ProcessDataThread, argv[2]);
    pubDataThread.join();
    processDataThread.join();
	return 0;
}