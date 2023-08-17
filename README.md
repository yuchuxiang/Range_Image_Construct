# Range_Image_Construct
## 代码功能
将机械式激光雷达点云数据转换为距离图和协方差图:  
1、根据扫描点的坐标确定点所在的扫描线;  
2、对点云数据进行插值补全;  
3、根据hash表和滑窗的方法计算点云的协方差;  
4、深度图和协方差图的显示;  

## 使用方法：
'mkdir catkin_ws/src -p  
cd catkin_ws/src  
git clone https://github.com/yuchuxiang/Range_Image_Construct.git  
cd ..  
catkin_make'  

![1692249897040](https://github.com/yuchuxiang/Range_Image_Construct/assets/79077924/5a64e906-a5c3-45e7-b691-0065b62c6a05)

![1692249897034](https://github.com/yuchuxiang/Range_Image_Construct/assets/79077924/a06ab351-c794-4028-9117-c86b701e6453)


