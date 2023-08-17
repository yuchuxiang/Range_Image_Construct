//
// Created by ycx on 23-4-12.
//

#ifndef RANGE_SEG_ADAPTIVE_VOX_HPP
#define RANGE_SEG_ADAPTIVE_VOX_HPP

#include "omp.h"
#include <Eigen/Core>
#include <Eigen/Dense>
#include <Eigen/StdVector>
#include <execution>
#include <openssl/md5.h>
#include <string>
#include <unordered_map>
#include <cstdlib>
#include <ctime>
#include <iostream>
#include <type_traits>
#include <unordered_map>

using namespace std;

int MIN_PS = 7;

class PIXEL_LOC
{
public:
    int64_t x, y;//x是行, y是列

    PIXEL_LOC(int64_t vx=0, int64_t vy=0): x(vx), y(vy){}

    bool operator== (const PIXEL_LOC &other) const
    {
        return (x==other.x && y==other.y);
    }

};

namespace std
{
    template<>
    struct hash<PIXEL_LOC>
    {
        size_t operator() (const PIXEL_LOC &s) const
        {
            using std::size_t; using std::hash;
            return ((hash<int64_t>()(s.x) ^ (hash<int64_t>()(s.y) << 1)) >> 1);
        }
    };
}

class QUAD_TREE
{
public:
    std::vector<Eigen::Vector2i> local_pixel_vec;
    std::vector<Eigen::Vector3f> plane_pixel_vec;
    vector<int>                  plane_loc_vec;
    vector<int>                  loc_vec;
    std::shared_ptr<QUAD_TREE>   leaves[4];
    float       pixel_center[2];
    int         octo_state;
    float       quater_length_horizontal;
    float       quater_length_vertical;
    float       feat_eigen_ratio_1;
    float       feat_eigen_ratio_2;


    QUAD_TREE(){//构造函数
        for(int i=0; i<4; i++) {
            leaves[i] = nullptr;
        }
        octo_state = 0;
        local_pixel_vec.clear();
        plane_pixel_vec.clear();
        feat_eigen_ratio_1 = 0;
        feat_eigen_ratio_2 = 0;
        loc_vec.clear();
        plane_loc_vec.clear();
    }

    void calculate_cov(std::unordered_map<int,pcl::PointXYZI>& cloud_map, int image_width){
        //首先获取当前体素中所有的loc,计算这些loc对应点云构成的的协方差矩阵
        Eigen::Matrix<float, 3, -1> neighbors(3,local_pixel_vec.size());
        for (int k = 0; k < local_pixel_vec.size(); k++) {
            int temp_loc = local_pixel_vec[k].x() * image_width + local_pixel_vec[k].y();
            if (cloud_map.count(temp_loc)){
                loc_vec.emplace_back(temp_loc);
            }
        }
        if (loc_vec.size() > MIN_PS){
            Eigen::Matrix<float, 3, -1> neighbors(3,loc_vec.size());
            for (int i = 0; i < loc_vec.size(); i++) {
                neighbors.col(i) = cloud_map[loc_vec[i]].getVector3fMap();
            }
            neighbors.colwise() -= neighbors.rowwise().mean().eval();
            Eigen::Matrix3f cov  = neighbors * neighbors.transpose() / loc_vec.size();
            Eigen::Vector3f        orient;
            Eigen::SelfAdjointEigenSolver<Eigen::Matrix3f> saes(cov);
            orient               = saes.eigenvectors().col(0);
            feat_eigen_ratio_1   = saes.eigenvalues()[2] / saes.eigenvalues()[0];
            feat_eigen_ratio_2   = saes.eigenvalues()[1] / saes.eigenvalues()[0];

            if(isnan(feat_eigen_ratio_1) || isnan(feat_eigen_ratio_2))
            {
                feat_eigen_ratio_1 = -1;
            }

        } else{
            feat_eigen_ratio_1 = -1;
        }

    }

    void recut_pixel_voxel(int layer, std::unordered_map<int,pcl::PointXYZI>& cloud_map, int image_width, std::vector<Eigen::Vector3f>& plane_vec, vector<int>& location_vec){
        if (local_pixel_vec.size() < MIN_PS) {
            return;
        }

        if (layer >= 3) {
            octo_state = 0;
            return;
        }

        calculate_cov(cloud_map,image_width);

        if (feat_eigen_ratio_1 >= 9 && feat_eigen_ratio_2 >= 7 )
        {
            for (int i = 0; i < loc_vec.size(); ++i) {
                plane_vec.emplace_back(cloud_map[loc_vec[i]].getVector3fMap());
                location_vec.emplace_back(loc_vec[i]);
            }
            return;
        }else
        {
            octo_state = 1;
            for (int i = 0; i < local_pixel_vec.size(); ++i) {
                int xy[2] = {0, 0};
                if (local_pixel_vec[i].x() >= pixel_center[0]) {
                    xy[0] = 1;
                }
                if (local_pixel_vec[i].y() >= pixel_center[1]) {
                    xy[1] = 1;
                }
                int leafnum = 2 * xy[0] + xy[1];
                if (leaves[leafnum] == nullptr) {
                    leaves[leafnum] = std::make_shared<QUAD_TREE>();
                    leaves[leafnum]->pixel_center[0] = pixel_center[0] + (2 * xy[0] - 1) * quater_length_vertical;
                    leaves[leafnum]->pixel_center[1] = pixel_center[1] + (2 * xy[1] - 1) * quater_length_horizontal;
                    leaves[leafnum]->quater_length_vertical = quater_length_vertical / 2;
                    leaves[leafnum]->quater_length_horizontal = quater_length_horizontal / 2;
                }
                leaves[leafnum]->local_pixel_vec.push_back(local_pixel_vec[i]);
            }
            layer++;
            for (int i = 0; i < 4; ++i) {
                if (leaves[i] != nullptr) {
                    leaves[i]->recut_pixel_voxel(layer,cloud_map,image_width,plane_vec,location_vec);
                }
            }
        }
    }
};





#endif //RANGE_SEG_ADAPTIVE_VOX_HPP
