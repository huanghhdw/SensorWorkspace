#include <gtest/gtest.h>
#include <common.h>

using namespace  Util ;

int add(int a,int b){
    return a+b;
}

TEST(all_test, equal_test)
{
    EXPECT_EQ(add(2,3),5);
}

TEST(all_test, UpdateAllWithEulerT_test)
{
    CameraPose cameraPose;
    cameraPose.eulerAngle_  = Eigen::Vector3d(0, 0, 0);
    cameraPose.t_ = Eigen::Vector3d(0, 0, 0);
    cameraPose.UpdateAllWithEulerT();
    EXPECT_EQ(cameraPose.r_(0,0), 1);
    EXPECT_EQ(cameraPose.r_(0,1), 0);
    EXPECT_EQ(cameraPose.r_(0,2), 0);
    EXPECT_EQ(cameraPose.r_(1,1), 1);
    EXPECT_EQ(cameraPose.r_(2,2), 1);
}

int main(int argc,char **argv){
  testing::InitGoogleTest(&argc,argv);
  return RUN_ALL_TESTS();
}

