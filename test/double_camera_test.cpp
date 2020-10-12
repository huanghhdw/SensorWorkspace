#include <gtest/gtest.h>

int add(int a,int b){
    return a+b;
}

TEST(double_camera_test, equal_test)
{
    EXPECT_EQ(add(2,3),5);
}

