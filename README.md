# 传感器的工作空间

 ## 1.编译及运行
（1）确保以下开源库在系统中安装
 ```` 
  eigen, opencv, gtest
 ````

（2）运行以下命令进行编译：

````shell script
git clone https://github.com/huanghhdw/SensorWorkspace
cd SensorWorkspace
./build.sh
````
运行

````shell script
./build/bin/HSlam /home/huanghh/data/08 /home/huanghh/hhh_ws/SensorWorkspace/Camera/config/camera.yaml
````
  程序第一参数为KIITI数据包的文件路径，第二参数为相机配置文件路径。
  
  
 ## 2.坐标描述
(1)　传感器的坐标 R , t 代表了该传感器在参考坐标系下的旋转矩阵和平移矩阵，描述的是点的变换。即在参考坐标系下的点(或向量)P, 以及在该传感器的坐标下的点（或向量）p, 　则有：
````
	P = R p + t　 
    注意:其中ｔ平移向量代表了该传感器坐标系原点在参考坐标系中的坐标。
 ```` 
 
 
 ## 3. 双目点特征SLAM设计
 
(1) 每一帧左右图像orb进行特点匹配，并进行三角化。

(2) 前一帧左目图像中已经三角化的坐标点与当前帧左目图像进行光流跟踪，得到3D-2D对应关系，进行PNP解算。
 
 
 ## 4. 数据集
（1）KITTI数据集

下载地址：
http://www.cvlibs.net/datasets/kitti/eval_odometry.php

选择 Download odometry data set (grayscale, 22 GB) 即可





