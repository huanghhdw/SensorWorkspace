#include <gtest/gtest.h>
#include <VisualOdom.h>
#include <iostream>

using namespace  Util;
using namespace  std;

TEST(all_test, uv2xyz_Test) //双目通过两个图片的像素坐标u1,v1,u2,v2, 转换到世界坐标系x,y,z
{
    ImageProcess::VisualOdom visualOdom;
    visualOdom.InitCamInfo("/home/huanghh/hhh_ws/SensorWorkspace/Camera/config/camera_for_test.yaml");
    EXPECT_EQ(visualOdom.cameraLeftInfo_.cameraIntrisic_.cameraIntrisic_.at<double>(0,0), 200);
    // K : [200,200,100,100],
    cv::Point3f pWorld(1, 1, 5);
    cv::Point2f uvLeft(140, 140);
    cv::Point2f uvRight(120, 140);
    cv::Point3f pWorldResult = visualOdom.uv2xyz(uvLeft, uvRight);
    cout << "pWorldResult:" << pWorldResult.x << " " << pWorldResult.y << " " << pWorldResult.z << " " << endl;
    EXPECT_EQ(pWorldResult.x, pWorld.x);
    EXPECT_EQ(pWorldResult.y, pWorld.y);
    EXPECT_EQ(pWorldResult.z, pWorld.z);
}

int main(int argc,char **argv){
  testing::InitGoogleTest(&argc,argv);
  return RUN_ALL_TESTS();
}

