//
// Created by ycx on 23-4-12.
//
//
// Created by yuchuxiang on 23-4-2.
//

#include <iostream>
#include <mutex>
#include <ros/ros.h>
#include <pcl/io/pcd_io.h>
#include <pcl/point_cloud.h>
#include <pcl/point_types.h>
#include <pcl/kdtree/kdtree_flann.h>
#include <pcl/filters/voxel_grid.h>
#include <sensor_msgs/PointCloud2.h>
#include <pcl_conversions/pcl_conversions.h>
#include <opencv2/core/core.hpp>
#include "opencv2/highgui/highgui.hpp"
#include "opencv2/imgproc/imgproc.hpp"
#include <eigen3/Eigen/Eigen>
#include <unordered_map>
#include <thread>
#include "range_seg/adaptive_vox.hpp"

using namespace std;

bool compare(const pair<int, int>& a, const pair<int, int>& b) {
    return a.first < b.first;
}

using pxiel_type = Eigen::Vector2i;

pcl::PointCloud<pcl::PointXYZI>::Ptr temp(new pcl::PointCloud<pcl::PointXYZI>());
pcl::PointCloud<pcl::PointXYZI>::Ptr temp_filter(new pcl::PointCloud<pcl::PointXYZI>());
float fov_up   = 2;
float fov_down = -24.8;
int   image_width  = 1800;
int   image_height = 64;
float max_range = 80;
float min_range = 1;
int   normal_times = 5;
cv::Mat range_image;
cv::Mat filter_image;
cv::Mat covariance_image;
cv::Mat covariance_image_filter_1;
cv::Mat covariance_image_filter_2;
cv::Mat covariance_image_filter_3;
cv::Mat covariance_image_smooth;
cv::Mat covariance_image_gray;
cv::Mat covariance_image_sobel;

unordered_map<int,pcl::PointXYZI>   image_map;
unordered_map<int,Eigen::Vector3f>  cov_map;
std::vector<int>                    loc_gather;
std::vector<std::vector<Eigen::Vector3f>> surf_vec;
std::vector<std::vector<int>>             surf_loc;

std::unordered_map<PIXEL_LOC,std::shared_ptr<QUAD_TREE>> voxel_map;

void build_voxel(std::unordered_map<PIXEL_LOC,std::shared_ptr<QUAD_TREE>>& feat_map, cv::Mat& mat_in, int voxel_width, int voxel_height){
    int width  = mat_in.cols;
    int height = mat_in.rows;

    for (int i = 0; i < height; ++i) {
        for (int j = 0; j < width; ++j) {
            int pixel_raw = i;
            int pixel_col = j;
            Eigen::Vector2i pvec(pixel_raw, pixel_col);
            float loc_rc[2];
            loc_rc[0] = pvec[0]/voxel_height;
            loc_rc[1] = pvec[1]/voxel_width;

            PIXEL_LOC position((int64_t) loc_rc[0], (int64_t) loc_rc[1]);
            auto iter = feat_map.find(position);
            if(iter != feat_map.end())
            {
                iter->second->local_pixel_vec.push_back(pvec);
            }
            else
            {
                std::shared_ptr<QUAD_TREE> ot = std::make_shared<QUAD_TREE>();
                ot->local_pixel_vec.push_back(pvec);
                ot->pixel_center[0] = (0.5+position.x) * voxel_height;
                ot->pixel_center[1] = (0.5+position.y) * voxel_width;
                ot->quater_length_vertical     = voxel_height / 4.0;
                ot->quater_length_horizontal   = voxel_width / 4.0;
                feat_map[position]             = ot;
            }
        }
    }
}




int main(int argc, char** argv) {
    pcl::io::loadPCDFile("/home/ycx/high_way2.pcd",*temp);
    cout<<"size of points:"<<temp->points.size()<<endl;
    std::ofstream ofs("/home/ycx/range.txt");
    std::ofstream ofd("/home/ycx/delete.txt");
    cout<<temp->points.size()<<endl;
    range_image         = cv::Mat::zeros(image_height, image_width, CV_8UC3);
    filter_image        = cv::Mat::zeros(image_height, image_width, CV_8UC3);
    covariance_image    = cv::Mat::zeros(image_height, image_width, CV_8UC3);
    float fov_up_pi    = (fov_up * M_PI)/180;
    float fov_down_pi  = (fov_down * M_PI)/180;
    float fov_whole_pi = fabs(fov_up_pi) + fabs(fov_down_pi);

    auto cur_frame_start = std::chrono::high_resolution_clock::now();
    int size_pointcloud = temp->points.size();

    for (int i = 0; i < size_pointcloud; ++i) {
        pcl::PointXYZI single_point;
        single_point = temp->points[i];
        float range = sqrt(single_point.x * single_point.x + single_point.y * single_point.y + single_point.z * single_point.z);
        single_point.intensity = range;
        int col = round(0.5 *(1- atan2(single_point.y,single_point.x) / M_PI) * image_width);
        int row = round((1 - (asin(single_point.z / range) - fov_down_pi)/fov_whole_pi) * image_height);

        int pixel_value = 200 - round((range - min_range) / (max_range - min_range) * 200);

        if (pixel_value>255){
            pixel_value = 255;
        }
        if (pixel_value<1){
            pixel_value = 1;
        }

        if (col > image_width-1 || col < 0 || row < 0 || row > image_height-1){
            continue;
        }

        range_image.at<cv::Vec3b>(row,col) =  cv::Vec3b(pixel_value,pixel_value,0);

        int loc = row * image_width + col;
        if (!image_map.count(loc)){
            image_map[loc] = single_point;
        }

    }

    std::vector<pxiel_type> pixel_3_group;
    std::vector<pxiel_type> pixel_4_group;
    std::vector<pxiel_type> pixel_8_group;
    std::vector<pxiel_type> pixel_14_group;
    std::vector<pxiel_type> pixel_14_group_;
    std::vector<pxiel_type> pixel_24_group;
    pixel_3_group = {pxiel_type(0,0),pxiel_type(0,1),pxiel_type(1,0)};
    pixel_4_group = {pxiel_type(0,0),pxiel_type(-1,0),pxiel_type(1,0),pxiel_type(0,1),pxiel_type(0,-1)};
    pixel_8_group = {pxiel_type(0,0),pxiel_type(-1,0),pxiel_type(1,0),pxiel_type(0,1),pxiel_type(0,-1),
                     pxiel_type(-1,1),pxiel_type(1,1),pxiel_type(-1,-1),pxiel_type(1,-1)};
    pixel_14_group = {pxiel_type(-1,-2),pxiel_type(-1,-1),pxiel_type(-1,0),pxiel_type(-1,1),pxiel_type(-1,2),
                        pxiel_type(0,-2),pxiel_type(0,-1),pxiel_type(0,0),pxiel_type(0,1),pxiel_type(0,2),
                        pxiel_type(1,-2),pxiel_type(1,-1),pxiel_type(1,0),pxiel_type(1,1),pxiel_type(1,2)};
    pixel_14_group_ = {pxiel_type(-2,-1),pxiel_type(-2,0),pxiel_type(-2,1),
                       pxiel_type(-1,-1),pxiel_type(-1,0),pxiel_type(-1,1),
                       pxiel_type(0,-1),pxiel_type(0,0),pxiel_type(0,1),
                       pxiel_type(1,-1),pxiel_type(1,0),pxiel_type(1,1),
                       pxiel_type(2,-1),pxiel_type(2,0),pxiel_type(2,1)};
    pixel_24_group = {pxiel_type(-2,-2),pxiel_type(-2,-1),pxiel_type(-2,0),pxiel_type(-2,1),pxiel_type(-2,2),
                      pxiel_type(-1,-2),pxiel_type(-1,-1),pxiel_type(-1,0),pxiel_type(-1,1),pxiel_type(-1,2),
                      pxiel_type(0,-2),pxiel_type(0,-1),pxiel_type(0,0),pxiel_type(0,1),pxiel_type(0,2),
                      pxiel_type(1,-2),pxiel_type(1,-1),pxiel_type(1,0),pxiel_type(1,1),pxiel_type(1,2),
                      pxiel_type(2,-2),pxiel_type(2,-1),pxiel_type(2,0),pxiel_type(2,1),pxiel_type(2,2)};

    //点云补全/深度图插值(补点云)
    for (int i = 1; i < image_height-1; ++i) {
        for (int j = 0; j < image_width; ++j) {
            if (range_image.at<cv::Vec3b>(i,j)[0] == 0){
                //计算该像素点的上下俩个像素点
                uchar up_pixel   = range_image.at<cv::Vec3b>(i-1,j)[0];
                uchar down_pixel = range_image.at<cv::Vec3b>(i+1,j)[0];

                int   up_loc     = (i-1)*image_width + j;
                int   down_loc   = (i+1)*image_width + j;
                int   temp_loc   = i * image_width + j;
                if (up_pixel!=0 && down_pixel!=0){
                    float up_range   = image_map[up_loc].intensity;
                    float down_range = image_map[up_loc].intensity;
                    pcl::PointXYZI temp_point;
                    if (fabs(up_range - down_range)/0.5 < 8){
                        Eigen::Vector3f up_p   = image_map[up_loc].getVector3fMap();
                        Eigen::Vector3f down_p = image_map[down_loc].getVector3fMap();
                        Eigen::Vector3f mid_p  = (up_p + down_p)/2;
                        temp_point.getVector3fMap() = mid_p;
                        temp_point.intensity   = sqrt(mid_p.squaredNorm());
                        image_map.insert({temp_loc,temp_point});
                    }
                    range_image.at<cv::Vec3b>(i,j)[0] = (up_pixel + down_pixel)/2;
                    range_image.at<cv::Vec3b>(i,j)[1] = (up_pixel + down_pixel)/2;
                }
            }

        }
    }

    //计算法向量
    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < image_width; ++j) {
            int temp_loc = i * image_width + j;
            if (image_map.count(temp_loc)){
                //获取其八邻域构成一个协方差矩阵，如果点数小于3,则不进行计算
                pcl::PointXYZI temp_point   = image_map[temp_loc];
                Eigen::Vector3f temp_eigen = temp_point.getVector3fMap();
                vector<int> loc_vec;
                pxiel_type temp_pixel(i,j);
                for(auto item:pixel_24_group){
                    auto each_item = item + temp_pixel;
                    int loc_  = each_item[0] * image_width + each_item[1];
                    if (image_map.count(loc_)){
//                        pcl::PointXYZI  find_point   = image_map[loc_];
//                        Eigen::Vector3f find_eigen   = find_point.getVector3fMap();
//                        Eigen::Vector3f dis_eigen    = find_eigen - temp_eigen;
//                        float           dis_         = dis_eigen.squaredNorm();
                        loc_vec.emplace_back(loc_);

                    }
                }
                if (loc_vec.size()>=3){
                    Eigen::Matrix<float, 3, -1> neighbors(3,loc_vec.size());
                    for (int k = 0; k < loc_vec.size(); k++) {
                        neighbors.col(k) = image_map[loc_vec[k]].getArray3fMap();
                    }
//                    Eigen::Vector3f left   = neighbors.col(1) - neighbors.col(0);
//                    Eigen::Vector3f righ   = neighbors.col(2) - neighbors.col(0);
//                    Eigen::Vector3f orient = left.cross(righ).normalized();
//                    cov_map.insert({temp_loc,orient});
                    neighbors.colwise() -= neighbors.rowwise().mean().eval();
                    Eigen::Matrix3f cov  = neighbors * neighbors.transpose() / loc_vec.size();
                    Eigen::Vector3f        orient;
                    Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> saes(cov);
                    orient               = saes.eigenvectors().col(0);
                    cov_map.insert({temp_loc,orient});

                }
            }
        }
    }


#pragma omp parallel for num_threads(4) schedule(guided, 8)
    for (int i = 0; i < image_height; ++i) {
        for (int j = 0; j < image_width; ++j) {
            int temp_loc = i * image_width + j;
            if (cov_map.count(temp_loc)){
                covariance_image.at<cv::Vec3b>(i,j)[0] = 200* abs(cov_map[temp_loc].z());
                covariance_image.at<cv::Vec3b>(i,j)[1] = 200* abs(cov_map[temp_loc].y());
                covariance_image.at<cv::Vec3b>(i,j)[2] = 200* abs(cov_map[temp_loc].x());

//                if (100* abs(cov_map[temp_loc].z()) > 30){
//                    covariance_image.at<cv::Vec3b>(i,j)[0] = 200* abs(cov_map[temp_loc].z());
//                } else{
//                    covariance_image.at<cv::Vec3b>(i,j)[0] = 0;
//                }
//                if (100* abs(cov_map[temp_loc].y()) > 30){
//                    covariance_image.at<cv::Vec3b>(i,j)[1] = 200* abs(cov_map[temp_loc].y());
//                } else{
//                    covariance_image.at<cv::Vec3b>(i,j)[1] = 0;
//                }
//                if (100* abs(cov_map[temp_loc].x()) > 30){
//                    covariance_image.at<cv::Vec3b>(i,j)[2] = 200* abs(cov_map[temp_loc].x());
//                } else{
//                    covariance_image.at<cv::Vec3b>(i,j)[2] = 0;
//                }

            }
        }
    }


//    cv::cvtColor(covariance_image,covariance_image_gray,cv::COLOR_BGR2GRAY);







//    cv::medianBlur(covariance_image,covariance_image_filter_1,3);

//    cv::bilateralFilter(covariance_image_filter_1,covariance_image_filter_2,5,5,5);
//
//    cv::GaussianBlur(covariance_image_filter_1,covariance_image_smooth,cv::Size(3,3),0,0);

//    cv::Sobel(covariance_image_smooth,covariance_image_sobel,);


    auto cur_frame_end = std::chrono::high_resolution_clock::now();
    auto cur_frame_time = std::chrono::duration_cast<std::chrono::milliseconds>(cur_frame_end - cur_frame_start).count();
    std::cout << "时间:  " << cur_frame_time << "ms" << std::endl;


    cv::imshow("range_image",range_image);
    cv::imshow("covariance_image",covariance_image);
//    cv::imshow("covariance_image_gray",covariance_image_gray);

//    cv::imshow("covariance_image_filter_1",covariance_image_filter_1);
//    cv::imshow("covariance_image_filter_2",covariance_image_filter_2);
//    cv::imshow("covariance_image_smooth",covariance_image_smooth);
    cv::waitKey();


    return 0;
}




//    build_voxel(voxel_map,range_image,12,12);
//
//    cout<<"voxel size: "<<voxel_map.size()<<endl;
//
//    for(auto iter=voxel_map.begin(); iter!=voxel_map.end(); ++iter)
//    {
//        iter->second->recut_pixel_voxel(0,  image_map, image_width, iter->second->plane_pixel_vec, iter->second->plane_loc_vec);
//        surf_vec.emplace_back(iter->second->plane_pixel_vec);
//        surf_loc.emplace_back(iter->second->plane_loc_vec);
//    }
//    cout<<"surf_vec: "<<surf_vec.size()<<endl;
//    cout<<"surf_loc: "<<surf_loc.size()<<endl;
//
//    for (int i = 0; i < surf_vec.size(); ++i) {
//        std::vector<Eigen::Vector3f>    single_voxel = surf_vec[i];
//        std::vector<int>                single_loc   = surf_loc[i];
//        for (int j = 0; j < single_voxel.size(); ++j) {
//            Eigen::Vector3f p = single_voxel[j];
//            pcl::PointXYZI point;
//            point.getVector3fMap() = p;
//            temp_filter->push_back(point);
//            loc_gather.emplace_back(single_loc[j]);
//        }
//    }
//
//    cout<<"temp_filter: "<<temp_filter->points.size()<<endl;
//    cout<<"loc_gather: "<<loc_gather.size()<<endl;
//    pcl::io::savePCDFile("/home/ycx/plane.pcd",*temp_filter);
//
//    for (int k = 0; k < loc_gather.size(); ++k) {
//        int temp_loc = loc_gather[k];
//
//        int i = temp_loc/image_width;
//        int j = temp_loc%image_width;
//
//        filter_image.at<cv::Vec3b>(i, j)[0] = range_image.at<cv::Vec3b>(i,j)[0];
//        filter_image.at<cv::Vec3b>(i, j)[1] = range_image.at<cv::Vec3b>(i,j)[1];
//        filter_image.at<cv::Vec3b>(i, j)[2] = range_image.at<cv::Vec3b>(i,j)[2];
//
//    }