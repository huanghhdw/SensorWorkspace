#include <iostream>
#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <cmath>
#include <string>
#include <Eigen/Core>
#include <Eigen/Dense>

using namespace std;
using namespace Eigen;

int main(int argc, char** argv)
{
	string sequence = "/home/huanghh/data/08";
	string dataPath = sequence + "/";

	// load image list
	FILE* file;
	file = std::fopen((dataPath + "times.txt").c_str() , "r");
	if(file == NULL){
	    printf("cannot find file: %stimes.txt\n", dataPath.c_str());
	    return 0;          
	}
	double imageTime;
	vector<double> imageTimeList;
	while ( fscanf(file, "%lf", &imageTime) != EOF)
	{
	    imageTimeList.push_back(imageTime);
	}
	std::fclose(file);
    string OUTPUT_FOLDER = "/home/huanghh/";
	string leftImagePath, rightImagePath;
	cv::Mat imLeft, imRight;
	FILE* outFile;
	outFile = fopen((OUTPUT_FOLDER + "/vio.txt").c_str(),"w");
	if(outFile == NULL)
		printf("Output path dosen't exist: %s\n", OUTPUT_FOLDER.c_str());

	for (size_t i = 0; i < imageTimeList.size(); i++)
	{	
			printf("\nprocess image %d\n", (int)i);
			stringstream ss;
			ss << setfill('0') << setw(6) << i;
			leftImagePath = dataPath + "image_0/" + ss.str() + ".png";
			rightImagePath = dataPath + "image_1/" + ss.str() + ".png";
			imLeft = cv::imread(leftImagePath, CV_LOAD_IMAGE_GRAYSCALE );
			imRight = cv::imread(rightImagePath, CV_LOAD_IMAGE_GRAYSCALE );
			Eigen::Matrix<double, 4, 4> pose = Eigen::Matrix<double, 4, 4>::Zero();
			if(outFile != NULL)
				fprintf (outFile, "%f %f %f %f %f %f %f %f %f %f %f %f \n",pose(0,0), pose(0,1), pose(0,2),pose(0,3),
																	       pose(1,0), pose(1,1), pose(1,2),pose(1,3),
																	       pose(2,0), pose(2,1), pose(2,2),pose(2,3));
			cv::imshow("leftImage", imLeft);
			cv::imshow("rightImage", imRight);
			cv::waitKey(0);
	}
	if(outFile != NULL)
		fclose (outFile);
	return 0;
}