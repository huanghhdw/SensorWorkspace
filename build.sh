#!/bin/bash


if [ -d "build" ];then
echo "build目录已经存在，重建！！！！！"
rm -rf build
else
echo "重新开始构建！！！！！！！！！！！"
fi

mkdir build
cd build
cmake ..
make
