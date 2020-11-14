# 传感器的工作空间

 ## 1.编译及运行
（1）确保以下开源库在系统中安装
 ```` 
  eigen, gtest, glog, ceres, opencv
 ````

（2）在SensorWorkspace目录下运行以下命令进行编译：

````shell script
git clone https://github.com/huanghhdw/SensorWorkspace
cd SensorWorkspace
mkdir build
cd build
cmake ..
make
````
运行

````shell script
cd bin
./Hslam
````
  
 ## 2.坐标描述
(1)　传感器的坐标 R , t 代表了该传感器在参考坐标系下的旋转矩阵和平移矩阵，描述的是点的变换。即在参考坐标系下的点(或向量)P, 以及在该传感器的坐标下的点（或向量）p, 　则有：
````
	P = R p + t　 
    注意:其中ｔ平移向量代表了该传感器坐标系原点在参考坐标系中的坐标。
 ```` 
 
 
 ## 3. 双目点特征SLAM设计思路
 
(1) 每一帧左右图像orb进行特点匹配，并进行三角化。

(2) 前一帧左目图像中已经三角化的坐标点与当前帧左目图像进行光流跟踪，得到3D-2D对应关系，进行PNP解算。
 
 
 ## 4.已实现的函数级功能：
 

 （1）对极几何求R，T
    
 （2）ICP
  
 （3）双目恢复三维坐标点
 
 （4）lsd线匹配
 
 （5）orb特征点的提取和匹配
 
 
 －－－－－－－－
 
 ## 附录：
 ### （1）时间线：
 
 2020.11.04 ICP在slam框架中适配。
 
 2020.11.03 在slam框架中加入双目orb匹配以及立体化特征点。
  
 2020.11.02 初步完成总体增量式定位框架设计。
  
 2020.10.25 调整框架，新增两个线程用于处理数据处理以及数据发布; 创建VO类。
  
 2020.10.24 增加config文件目录，存放相机参数，同时代码增加读取该文件的逻辑
 
 2020.10.23 下载KITTI到ubuntu，测试发布数据图像节点能够正常发布
 
 2020.10.22 编写KITTI读取本地文件的节点，通过gtest测试节点common函数的转换,增加相机信息类
 
 2020.10.13 增加点特征提取，对极几何分解RT,线特征提取LSD方法, 增加数据文件夹
 
 2020.10.12 增加gtest测试功能
  
 2020.10.11 创建工程
 
 ### （2）迭代点：
 
 2020年11月30日完成KITTI数据集有输出结果
 