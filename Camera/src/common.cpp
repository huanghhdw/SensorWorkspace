#include <common.h>

using namespace  Util ;

// 通过欧拉角和平移向量更新ｒ
void CameraPose::UpdateAllWithEulerT()
{
    Eigen::AngleAxisd rollAngle(Eigen::AngleAxisd(eulerAngle_[2], Eigen::Vector3d::UnitX()));
    Eigen::AngleAxisd pitchAngle(Eigen::AngleAxisd(eulerAngle_[1], Eigen::Vector3d::UnitY()));
    Eigen::AngleAxisd yawAngle(Eigen::AngleAxisd(eulerAngle_[0], Eigen::Vector3d::UnitZ()));
    r_ = yawAngle * pitchAngle * rollAngle;   // 右乘绕旋转轴
    transMatrix_.block<3, 3>(0, 0) = r_;
    transMatrix_.block<3, 1>(0, 3) = t_;
}