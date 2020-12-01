#include <gtest/gtest.h>
#include <VisualOdom.h>

using namespace  Util ;

TEST(all_test, uv2xyz_Test)
{
    ImageProcess::VisualOdom visualOdom;
    visualOdom.InitCamInfo("/home/huanghh/hhh_ws/SensorWorkspace/Camera/config/camera.yaml");
    EXPECT_EQ(visualOdom.cameraLeftInfo_.cameraIntrisic_.cameraIntrisic_.at<double>(0,0), 7.070912000000e+02);
}

int main(int argc,char **argv){
  testing::InitGoogleTest(&argc,argv);
  return RUN_ALL_TESTS();
}

