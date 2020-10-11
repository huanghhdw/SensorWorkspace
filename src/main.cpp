#include <fstream>
#include <iostream>
#include <ceres/ceres.h>
#include <glog/logging.h>
#include <opencv2/opencv.hpp>
#include<Eigen/Core>

using namespace std;
using namespace cv;

using ceres::AutoDiffCostFunction;
using ceres::CostFunction;
using ceres::Problem;
using ceres::Solve;
using ceres::Solver;

// A templated cost functor that implements the residual r = 10 -
// x. The method operator() is templated so that we can then use an
// automatic differentiation wrapper around it to generate its
// derivatives.
struct CostFunctor {
    template <typename T>
    bool operator()(const T* const x, T* residual) const {
    residual[0] = 10.0 - x[0];
    return true;
  }
};
    
Point2f xyz2uv(Point3f worldPoint,float intrinsic[3][3],float translation[1][3],float rotation[3][3]);
Point3f uv2xyz(Point2f uvLeft,Point2f uvRight);
 
//图片对数量
int PicNum = 14;
 
//左相机内参数矩阵
float leftIntrinsic[3][3] = {4037.82450,			 0,		947.65449,
									  0,	3969.79038,		455.48718,
									  0,			 0,				1};
//左相机畸变系数
float leftDistortion[1][5] = {0.18962, -4.05566, -0.00510, 0.02895, 0};
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
//右相机畸变系数
float rightDistortion[1][5] = {-0.24195, 5.97763, -0.02057, -0.01429, 0};
//右相机旋转矩阵
float rightRotation[3][3] = {-0.134947,		 0.989568,		-0.050442, 
							  0.752355,		 0.069205,		-0.655113, 
							 -0.644788,		-0.126356,		-0.753845};
//右相机平移向量
float rightTranslation[1][3] = {50.877397, -99.796492, 1507.312197};
 
 
int main()
{
	FLAGS_log_dir = "/home/huanghh/hhh_ws/double_camera/build/bin/log";
	//FLAGS_logtostderr = true;  //设置日志消息是否转到标准输出而不是日志文件
	FLAGS_alsologtostderr = true;  //设置日志消息除了日志文件之外是否去标准输出
	FLAGS_colorlogtostderr = true;  //设置记录到标准输出的颜色消息（如果终端支持）
	// FLAGS_log_prefix = true;  //设置日志前缀是否应该添加到每行输出
	// FLAGS_logbufsecs = 0;  //设置可以缓冲日志的最大秒数，0指实时输出
	// FLAGS_max_log_size = 10;  //设置最大日志文件大小（以MB为单位）
	// FLAGS_stop_logging_if_full_disk = true;  //设置是否在磁盘已满时避免日志记录到磁盘
	google::InitGoogleLogging("double_camera");
	LOG(INFO) << "info test";  //输出一个Info日志
	LOG(WARNING) << "warning test";  //输出一个Warning日志
	LOG(ERROR) << "error test";  //输出一个Error日志
	//LOG(FATAL) << "fatal test";  //输出一个Fatal日志，这是最严重的日志并且输出之后会中止程序
	Mat m = Mat(720,1280,CV_8UC3,Scalar(0, 255, 255));
	imshow("Pure Picture", m);
	cvWaitKey(0);
	//已知空间坐标求成像坐标
	Point3f point(700,220,530);
	cout<<"左相机中坐标："<<endl;
	cout<<xyz2uv(point,leftIntrinsic,leftTranslation,leftRotation)<<endl;
	cout<<"右相机中坐标："<<endl;
	cout<<xyz2uv(point,rightIntrinsic,rightTranslation,rightRotation)<<endl;
 
	//已知左右相机成像坐标求空间坐标
	Point2f l = xyz2uv(point,leftIntrinsic,leftTranslation,leftRotation);
	Point2f r = xyz2uv(point,rightIntrinsic,rightTranslation,rightRotation);
	Point3f worldPoint;
	worldPoint = uv2xyz(l,r);
	cout<<"空间坐标为:"<<endl<<uv2xyz(l,r)<<endl;

	// The variable to solve for with its initial value. It will be
	// mutated in place by the solver.
	double x = 0.5;
	const double initial_x = x;
	// Build the problem.
	Problem problem;
	// Set up the only cost function (also known as residual). This uses
	// auto-differentiation to obtain the derivative (jacobian).
	CostFunction* cost_function =
		new AutoDiffCostFunction<CostFunctor, 1, 1>(new CostFunctor);
	problem.AddResidualBlock(cost_function, nullptr, &x);
	// Run the solver!
	Solver::Options options;
	options.minimizer_progress_to_stdout = true;
	Solver::Summary summary;
	Solve(options, &problem, &summary);
	std::cout << summary.BriefReport() << "\n";
	std::cout << "x : " << initial_x << " -> " << x << "\n";
	google::ShutdownGoogleLogging();
    return 0;
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
Point3f uv2xyz(Point2f uvLeft,Point2f uvRight)
{
	//  [u1]      |X|					  [u2]      |X|
	//Z*|v1| = Ml*|Y|					Z*|v2| = Mr*|Y|
	//  [ 1]      |Z|					  [ 1]      |Z|
	//			  |1|								|1|
	Mat mLeftRotation = Mat(3,3,CV_32F,leftRotation);
	Mat mLeftTranslation = Mat(3,1,CV_32F,leftTranslation);
	Mat mLeftRT = Mat(3,4,CV_32F);//左相机M矩阵
	hconcat(mLeftRotation,mLeftTranslation,mLeftRT);
	Mat mLeftIntrinsic = Mat(3,3,CV_32F,leftIntrinsic);
	Mat mLeftM = mLeftIntrinsic * mLeftRT;
	//cout<<"左相机M矩阵 = "<<endl<<mLeftM<<endl;
 
	Mat mRightRotation = Mat(3,3,CV_32F,rightRotation);
	Mat mRightTranslation = Mat(3,1,CV_32F,rightTranslation);
	Mat mRightRT = Mat(3,4,CV_32F);//右相机M矩阵
	hconcat(mRightRotation,mRightTranslation,mRightRT);
	Mat mRightIntrinsic = Mat(3,3,CV_32F,rightIntrinsic);
	Mat mRightM = mRightIntrinsic * mRightRT;
	//cout<<"右相机M矩阵 = "<<endl<<mRightM<<endl;
 
	//最小二乘法A矩阵
	Mat A = Mat(4,3,CV_32F);
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
	Mat B = Mat(4,1,CV_32F);
	B.at<float>(0,0) = mLeftM.at<float>(0,3) - uvLeft.x * mLeftM.at<float>(2,3);
	B.at<float>(1,0) = mLeftM.at<float>(1,3) - uvLeft.y * mLeftM.at<float>(2,3);
	B.at<float>(2,0) = mRightM.at<float>(0,3) - uvRight.x * mRightM.at<float>(2,3);
	B.at<float>(3,0) = mRightM.at<float>(1,3) - uvRight.y * mRightM.at<float>(2,3);
 
	Mat XYZ = Mat(3,1,CV_32F);
	//采用SVD最小二乘法求解XYZ
	solve(A,B,XYZ,DECOMP_SVD);
 
	//cout<<"空间坐标为 = "<<endl<<XYZ<<endl;
 
	//世界坐标系中坐标
	Point3f world;
	world.x = XYZ.at<float>(0,0);
	world.y = XYZ.at<float>(1,0);
	world.z = XYZ.at<float>(2,0);
 
	return world;
}
 
//************************************
// Description: 将世界坐标系中的点投影到左右相机成像坐标系中
// Method:    xyz2uv
// FullName:  xyz2uv
// Access:    public 
// Parameter: Point3f worldPoint
// Parameter: float intrinsic[3][3]
// Parameter: float translation[1][3]
// Parameter: float rotation[3][3]
// Returns:   cv::Point2f
// Author:    小白
// Date:      2017/01/10
// History:
//************************************
Point2f xyz2uv(Point3f worldPoint,float intrinsic[3][3],float translation[1][3],float rotation[3][3])
{
	//    [fx s x0]							[Xc]		[Xw]		[u]	  1		[Xc]
	//K = |0 fy y0|       TEMP = [R T]		|Yc| = TEMP*|Yw|		| | = —*K *|Yc|
	//    [ 0 0 1 ]							[Zc]		|Zw|		[v]	  Zc	[Zc]
	//													[1 ]
	Point3f c;
	c.x = rotation[0][0]*worldPoint.x + rotation[0][1]*worldPoint.y + rotation[0][2]*worldPoint.z + translation[0][0]*1;
	c.y = rotation[1][0]*worldPoint.x + rotation[1][1]*worldPoint.y + rotation[1][2]*worldPoint.z + translation[0][1]*1;
	c.z = rotation[2][0]*worldPoint.x + rotation[2][1]*worldPoint.y + rotation[2][2]*worldPoint.z + translation[0][2]*1;
 
	Point2f uv;
	uv.x = (intrinsic[0][0]*c.x + intrinsic[0][1]*c.y + intrinsic[0][2]*c.z)/c.z;
	uv.y = (intrinsic[1][0]*c.x + intrinsic[1][1]*c.y + intrinsic[1][2]*c.z)/c.z;
 
	return uv;
}
 