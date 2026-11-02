#pragma once

#include <algorithm>
#include <chrono>
#include <cmath>
#include <cstring>
#include <fstream>
#include <iostream>
#include <limits>
#include <queue>
#include <random>
#include <sstream>
#include <string>
#include <unordered_map>
#include <vector>
#include <time.h>

#include <pcl/point_types.h>
#include <pcl/features/normal_3d.h>
#include <pcl/kdtree/kdtree_flann.h>

#include <igl/voxel_grid.h>
#include <igl/fast_winding_number.h>
#include <igl/octree.h>
#include <igl/copyleft/cgal/point_areas.h>

#include "geometrycentral/vector3.h"
#include "TurboDWG.h"

using namespace geometrycentral;

extern "C" void  GenerateKDTreeForGWN(std::vector<float>&, std::vector<float>&, std::vector<float>&,
    std::vector<float>&, std::vector<float>&, std::vector<float>&,
    std::vector<float>&, std::vector<float>&,
    bool, std::unordered_map<Vector3, int>&);

extern "C" void CudaKNNWithKDTreeforGWN(
    std::vector<float>&, std::vector<float>&, std::vector<float>&,
    std::vector<float>&, std::vector<float>&, std::vector<float>&,
    std::vector<float>&, std::vector<float>&,
    std::vector<std::vector<int>>&,
    int);

namespace test3D
{
    Eigen::MatrixXd Ps, Ns;

    Eigen::VectorXd As;

    Eigen::MatrixXi Point_Neighbours;
    std::vector<float> Bound_Upper_KDT;
    std::vector<float> Bound_Lower_KDT;

    std::vector<std::vector<int > > O_PI;
    Eigen::MatrixXi O_CH;
    Eigen::MatrixXd O_CN;
    Eigen::VectorXd O_W;

    Eigen::MatrixXd O_CM;
    Eigen::VectorXd O_R;
    Eigen::MatrixXd O_EC;

    std::pair<std::vector<Vector3>, std::vector<Vector3> > ReadPointandNormals(std::string filename, bool pre_process)
    {
        double scale = 0;

        std::vector<Vector3> tmpps, tmpNormals;
        std::ifstream in;
        in.open(filename);
        int cnt = 0;
        for (std::string line; std::getline(in, line);) {
            std::istringstream lineStream(line);
            double x, y, z, nx, ny, nz;
            lineStream >> x >> y >> z >> nx >> ny >> nz;
            tmpps.push_back(Vector3{ x, y, z });
            tmpNormals.push_back(Vector3{ nx, ny, nz });
            cnt++;
        }
        std::cout << "point cloud cnt = " << cnt << "\n";
        in.close();

        while (true)
        {
            float min_x = std::numeric_limits<float>::max();
            float max_x = std::numeric_limits<float>::lowest();
            float min_y = min_x, max_y = max_x;
            float min_z = min_x, max_z = max_x;

            for (const auto& p : tmpps)
            {
                if (p.x < min_x) min_x = p.x;
                if (p.x > max_x) max_x = p.x;
                if (p.y < min_y) min_y = p.y;
                if (p.y > max_y) max_y = p.y;
                if (p.z < min_z) min_z = p.z;
                if (p.z > max_z) max_z = p.z;
            }

            float center_x = (min_x + max_x) / 2.0f;
            float center_y = (min_y + max_y) / 2.0f;
            float center_z = (min_z + max_z) / 2.0f;

            scale = std::max({ max_x - min_x, max_y - min_y, max_z - min_z });

            float scale_factor = sqrt(1.0f);
            for (auto& p : tmpps)
            {
                p.x = ((p.x - center_x) / scale) * scale_factor;
                p.y = ((p.y - center_y) / scale) * scale_factor;
                p.z = ((p.z - center_z) / scale) * scale_factor;
            }

            std::cout << "scale = " << scale << "}\n";
            std::cout << "{" << center_x << ", " << center_y << ", " << center_z << "}\n";

            if (!pre_process)
                break;

            std::vector<float> point_x, point_y, point_z;
            point_x.resize(tmpps.size()); point_y.resize(tmpps.size()); point_z.resize(tmpps.size());
            std::vector<float> normal_x, normal_y, normal_z;
            normal_x.resize(tmpps.size()); normal_y.resize(tmpps.size()); normal_z.resize(tmpps.size());
            for (int i = 0; i < tmpps.size(); i++)
            {
                point_x[i] = tmpps[i].x; point_y[i] = tmpps[i].y; point_z[i] = tmpps[i].z;
                normal_x[i] = tmpNormals[i].x; normal_y[i] = tmpNormals[i].y; normal_z[i] = tmpNormals[i].z;
            }

            float dist_threshold = 3.0f;
            std::vector<float> out_point_x, out_point_y, out_point_z;
            std::vector<float> out_normal_x, out_normal_y, out_normal_z;
            TurboDWG turboDWG;
            turboDWG.PreProcessOutliers(dist_threshold,
                point_x, point_y, point_z,
                normal_x, normal_y, normal_z,
                out_point_x, out_point_y, out_point_z,
                out_normal_x, out_normal_y, out_normal_z);

            tmpps.resize(out_point_x.size());
           tmpNormals.resize(out_point_x.size());

            for (int i = 0; i < tmpps.size(); i++)
            {
                tmpps[i].x = out_point_x[i]; tmpps[i].y = out_point_y[i]; tmpps[i].z = out_point_z[i];
                tmpNormals[i].x = out_normal_x[i]; tmpNormals[i].y = out_normal_y[i]; tmpNormals[i].z = out_normal_z[i];
            }

            pre_process = false;
        }

        std::cout << "register init point cloud \n";

        return std::make_pair(tmpps, tmpNormals);
    }

    void DataprePointCloud(std::string filepath, int max_depth, int input_normal, int com_As, bool pre_process, clock_t& start, std::chrono::high_resolution_clock::time_point& start_time_s)
    {
        auto pair_ps_and_ns = ReadPointandNormals(filepath, pre_process);

        start = clock();
        start_time_s = std::chrono::high_resolution_clock::now();

        int totPoints = pair_ps_and_ns.first.size();

        std::vector<Vector3> Vec3_Ns_ORI;
        Vec3_Ns_ORI = pair_ps_and_ns.second;

        Ps.resize(totPoints, 3);
        Ns.resize(totPoints, 3);

        std::vector<float> PointsPos_X_KDT;
        std::vector<float> PointsPos_Y_KDT;
        std::vector<float> PointsPos_Z_KDT;

        PointsPos_X_KDT.resize(totPoints);
        PointsPos_Y_KDT.resize(totPoints);
        PointsPos_Z_KDT.resize(totPoints);

        std::vector<float> init_points_pos_x;
        std::vector<float> init_points_pos_y;
        std::vector<float> init_points_pos_z;
        init_points_pos_x.resize(totPoints);
        init_points_pos_y.resize(totPoints);
        init_points_pos_z.resize(totPoints);

        for (int i = 0; i < totPoints; i++) {
            init_points_pos_x[i] = pair_ps_and_ns.first[i].x;
            init_points_pos_y[i] = pair_ps_and_ns.first[i].y;
            init_points_pos_z[i] = pair_ps_and_ns.first[i].z;
        }

        Bound_Upper_KDT.resize(3);
        Bound_Lower_KDT.resize(3);

        std::unordered_map<geometrycentral::Vector3, int> points_indices_map;
        GenerateKDTreeForGWN(init_points_pos_x, init_points_pos_y, init_points_pos_z,
            PointsPos_X_KDT, PointsPos_Y_KDT, PointsPos_Z_KDT,
            Bound_Upper_KDT, Bound_Lower_KDT,
            true,
            points_indices_map);

        for (int i = 0; i < totPoints; i++)
        {
            pair_ps_and_ns.first[i].x = PointsPos_X_KDT[i];
            pair_ps_and_ns.first[i].y = PointsPos_Y_KDT[i];
            pair_ps_and_ns.first[i].z = PointsPos_Z_KDT[i];
        }

        std::ofstream outFile("E:/dwg++/mesh/input_data.xyz");
        for (int i = 0; i < totPoints; i++)
        {
            int ori_index = points_indices_map[pair_ps_and_ns.first[i]];

            outFile << pair_ps_and_ns.first[i].x << " " << pair_ps_and_ns.first[i].y << " " << pair_ps_and_ns.first[i].z << " " <<
                Vec3_Ns_ORI[ori_index].x << " " << Vec3_Ns_ORI[ori_index].y << " " << Vec3_Ns_ORI[ori_index].z << "\n";
        }

        outFile.close();

        if (com_As && input_normal != 1 && input_normal != 3)
        {
            std::cout << "PCA Start \n";
            pcl::PointCloud<pcl::PointXYZ>::Ptr cloud(new pcl::PointCloud<pcl::PointXYZ>);

            for (int i = 0; i < totPoints; i++)
            {
                pcl::PointXYZ pclPoint;
                pclPoint.x = PointsPos_X_KDT[i];
                pclPoint.y = PointsPos_Y_KDT[i];
                pclPoint.z = PointsPos_Z_KDT[i];
                cloud->points.push_back(pclPoint);
            }

            pcl::NormalEstimation<pcl::PointXYZ, pcl::Normal> ne;
            ne.setInputCloud(cloud);

            pcl::search::KdTree<pcl::PointXYZ>::Ptr tree(new pcl::search::KdTree<pcl::PointXYZ>);
            ne.setSearchMethod(tree);

            pcl::PointCloud<pcl::Normal>::Ptr cloud_normals(new pcl::PointCloud<pcl::Normal>);

            ne.setKSearch(10);

            Eigen::Vector4f centroid;

            centroid[0] = (Bound_Upper_KDT[0] + Bound_Lower_KDT[0]) / 2.0f;
            centroid[1] = (Bound_Upper_KDT[1] + Bound_Lower_KDT[1]) / 2.0f;
            centroid[2] = (Bound_Upper_KDT[2] + Bound_Lower_KDT[2]) / 2.0f;

            ne.setViewPoint(centroid[0], centroid[1], centroid[2]);
            std::cout << "center: " << centroid[0] << " " << centroid[1] << " " << centroid[2] << std::endl;

            ne.compute(*cloud_normals);

            for (int i = 0; i < totPoints; i++)
            {
                pair_ps_and_ns.second[i].x = cloud_normals->points[i].normal_x;
                pair_ps_and_ns.second[i].y = cloud_normals->points[i].normal_y;
                pair_ps_and_ns.second[i].z = cloud_normals->points[i].normal_z;
            }

            std::cout << "PCA End \n";
        }
        else
        {
            for (int i = 0; i < totPoints; i++)
            {
                int ori_index = points_indices_map[pair_ps_and_ns.first[i]];
                pair_ps_and_ns.second[i] = Vec3_Ns_ORI[ori_index];
            }
        }

        Ps.resize(totPoints, 3);
        Ns.resize(totPoints, 3);

        for (int i = 0; i < totPoints; i++)
        {
            Ps(i, 0) = pair_ps_and_ns.first[i].x;
            Ps(i, 1) = pair_ps_and_ns.first[i].y;
            Ps(i, 2) = pair_ps_and_ns.first[i].z;

            Ns(i, 0) = pair_ps_and_ns.second[i].x;
            Ns(i, 1) = pair_ps_and_ns.second[i].y;
            Ns(i, 2) = pair_ps_and_ns.second[i].z;
        }

        std::cout << "Ps, Ns read \n";

        std::cout << "Build Tree" << std::endl;
        igl::octreeDWG(Ps, O_PI, O_CH, O_CN, O_W, max_depth);
        std::cout << "Build Done" << std::endl;

        if (com_As)
        {
            int knn_k = 64;
            std::vector<std::vector<int>> results_pos_index; results_pos_index.resize(totPoints);
            for (int i = 0; i < totPoints; i++)
            {
                std::vector<int> c_result_pos_index; c_result_pos_index.resize(knn_k);
                results_pos_index[i] = c_result_pos_index;
            }

            CudaKNNWithKDTreeforGWN(
                PointsPos_X_KDT, PointsPos_Y_KDT, PointsPos_Z_KDT,
                PointsPos_X_KDT, PointsPos_Y_KDT, PointsPos_Z_KDT,
                Bound_Upper_KDT, Bound_Lower_KDT,
                results_pos_index,
                knn_k
            );

            int rows = results_pos_index.size();
            int cols = results_pos_index[0].size();

            Point_Neighbours.resize(rows, cols);
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    Point_Neighbours(i, j) = results_pos_index[i][j];
                }
            }
        }

        std::cout << "DataprePointCloud \n";
    }

    Eigen::VectorXd GetRandomNs(int totPoints)
    {

        unsigned int seed = 20260816;
        std::mt19937 gen(seed);
        std::uniform_real_distribution<> dis(0, PI * 2.0);

        Eigen::VectorXd uvs(totPoints * 2);

        for (int i = 0; i < totPoints; i++)
        {
            double u = dis(gen), v = dis(gen);

            uvs(i * 2 + 0) = u;
            uvs(i * 2 + 1) = v;
        }
        return uvs;
    }

    inline void GetRadiiandAs(const Eigen::MatrixXd& P,
        Eigen::MatrixXd& leaves_Ns,
        Eigen::VectorXd& radii,
        Eigen::VectorXd& leaves_As,
        Eigen::MatrixXi& out_point_neigh,
        std::vector<float> bound_upper,
        std::vector<float> bound_lower,
        int knn_k = 2,
        bool com_As = false)
    {

        std::vector<float> points_pos_x;
        std::vector<float> points_pos_y;
        std::vector<float> points_pos_z;
        points_pos_x.resize(P.rows());
        points_pos_y.resize(P.rows());
        points_pos_z.resize(P.rows());

        for (int i = 0; i < P.rows(); i++)
        {
            points_pos_x[i] = P(i, 0);
            points_pos_y[i] = P(i, 1);
            points_pos_z[i] = P(i, 2);
        }

        std::vector<std::vector<int > > knn_mid_Ps;
        knn_mid_Ps.resize(P.rows());
        for (int i = 0; i < knn_mid_Ps.size(); i++)
        {
            std::vector<int> c_vec;
            c_vec.resize(knn_k);
            knn_mid_Ps[i] = c_vec;
        }

        CudaKNNWithKDTreeforGWN(
            points_pos_x, points_pos_y, points_pos_z,
            points_pos_x, points_pos_y, points_pos_z,
            bound_upper, bound_lower,
            knn_mid_Ps,
            knn_k
        );

        radii.resize(P.rows());
        int kw = knn_k;
        radii.setZero();
        for (int i = 0; i < P.rows(); i++)
        {
            double tmp_r = 0;
            for (int j = 0; j < knn_k; j++)
            {

                int pIdx = knn_mid_Ps[i][j];

                tmp_r += (P.row(i) - P.row(pIdx)).squaredNorm();
            }
            tmp_r /= (1.0 * kw);
            tmp_r = std::sqrt(tmp_r);
            tmp_r = std::min(tmp_r, 0.015);
            tmp_r = std::max(tmp_r, 0.0015);
            radii(i) = tmp_r;
        }

        if (com_As)
        {
            std::cout << "Start As computation" << std::endl;
            int knn_size = 20;
            knn_mid_Ps.clear();
            knn_mid_Ps.resize(P.rows());
            for (int i = 0; i < knn_mid_Ps.size(); i++)
            {
                std::vector<int> c_vec;
                c_vec.resize(knn_size);
                knn_mid_Ps[i] = c_vec;
            }

            CudaKNNWithKDTreeforGWN(
                points_pos_x, points_pos_y, points_pos_z,
                points_pos_x, points_pos_y, points_pos_z,
                bound_upper, bound_lower,
                knn_mid_Ps,
                knn_size
            );

            int rows = knn_mid_Ps.size();
            int cols = knn_mid_Ps[0].size();

            out_point_neigh.resize(rows, cols);
            for (int i = 0; i < rows; ++i)
            {
                for (int j = 0; j < cols; ++j)
                {
                    out_point_neigh(i, j) = knn_mid_Ps[i][j];
                }
            }

            igl::copyleft::cgal::point_areas(P, out_point_neigh, leaves_Ns, leaves_As);

            std::cout << "End As computation" << std::endl;
        }
    }

    void GetOctreeLeaves(const std::vector<std::vector<int> >& point_indices,
        const Eigen::MatrixXi& CH,
        const Eigen::MatrixXd& CN,
        Eigen::MatrixXd& leaves,
        Eigen::VectorXi& leaves_idx,
        double lowest_points = 15)
    {
        typedef Eigen::Matrix<double, 1, 3> RowVec3;
        std::vector<RowVec3> centers;
        std::vector<int> _leaves_idx;
        int nowidx = 0;
        std::queue<int> q;
        q.push(nowidx);
        while (q.size() > 0)
        {
            int idx = q.front();
            q.pop();
            if (CH(idx, 0) == -1)
            {
                if (point_indices[idx].size() > lowest_points)
                {
                    centers.emplace_back(CN.row(idx));
                    _leaves_idx.emplace_back(idx);
                }
                continue;
            }
            for (int child = 0; child < 8; child++)
            {
                int child_index = CH(idx, child);
                if ((child_index != -1) && (point_indices[child_index].size() > 0))
                {
                    q.push(child_index);
                }
            }
        }
        leaves.resize(centers.size(), 3);
        for (int i = 0; i < centers.size(); i++)
        {
            leaves.row(i) = centers[i];
        }
        leaves_idx.resize(_leaves_idx.size());
        for (int i = 0; i < _leaves_idx.size(); i++)
        {
            leaves_idx(i) = _leaves_idx[i];
        }
    }

    void SmoothNormalstoOctreeLeaves(const Eigen::MatrixXd& leaves_center,
        const Eigen::VectorXi& leaves_idx,
        const std::vector<std::vector<int> >& points_indices,
        const Eigen::MatrixXd& octree_W,
        const Eigen::MatrixXd& Ps,
        const Eigen::MatrixXd& Ns,
        Eigen::MatrixXd& leaves_Ns)
    {
        auto gaussProx = [=](double x) ->double {
            double x2 = x * x;
            return std::exp(-x2 / 4.0);
        };
        typedef Eigen::Matrix<double, 1, 3> RowVec3;
        RowVec3 _; _.setConstant(0);
        std::cout << "_ = " << _ << "\n";
        std::cout << "_(0) = " << _(0) << "\n";
        std::cout << "_.rows = " << _.rows() << ", _cols = " << _.cols() << "\n";
        leaves_Ns.resize(leaves_idx.size(), 3);
        std::cout << "O_W.rows = " << octree_W.rows() << ", O_W.cols = " << O_W.cols() << "\n";
        std::cout << "leaves_center.rows = " << leaves_center.rows() << "\n";
        std::cout << "leaves_idx.rows = " << leaves_idx.rows() << "\n";

#pragma omp parallel for
        for (int i = 0; i < leaves_idx.size(); i++) {
            int O_id = leaves_idx(i);
            RowVec3 O_C = leaves_center.row(i);

            double O_W = octree_W(O_id);

            RowVec3 O_N;
            O_N.setConstant(0);

            for (int j = 0; j < points_indices[O_id].size(); j++) {
                int P_id = points_indices[O_id][j];
                RowVec3 nowP = Ps.row(P_id);
                RowVec3 nowN = Ns.row(P_id);
                for (int k = 0; k < 3; k++) {
                    O_N(k) += gaussProx((nowP(k) - O_C(k)) / O_W) * nowN(k);
                }
            }
            leaves_Ns.row(i) = O_N;
        }
    }

    inline void TurboIterative(clock_t start_o, std::chrono::high_resolution_clock::time_point start_time_s, std::string out_mesh_path, std::string out_mesh_full_path, int input_normal, int grid_reso, int final_reso, int kw, int final_kw, int normal_kw, int max_depth,
        float sigma, int min_num, bool use_screened, bool use_radii, int com_As, float As_threshold, float offset_1, float offset_2, int use_leaves_generate, int total_iter, int test_iter, int output_mode,
        float learning_rate_Ai, float learning_rate_Ai_2, float learning_rate_Ci, int step_iteration, float diric_scale, int diric_reso, float ai_reg, float ai_reg_2, float ci_reg_c, float ci_reg_a,
        float ai_rad, float ci_rad, float weight_scale,
        int before_ci_max_loop,
        int ci_max_loop_iteration,
        int max_loop,
        float ai_target_start,
        float ai_target_mid,

        float ci_target,
        float ai_smooth_rad, bool ai_adjust,

        float ci_in_thres, float ci_out_thres,
        float ci_final_in_thres, float ci_final_out_thres,
        int ci_level,
        float ci_var_thres,
        float ai_var_thres_0,
        float ai_var_thres_1,
        int DWG_extract_mode,
        int DWG_initial_mode,
        float inside_x,
        float inside_y,
        float inside_z)
    {

        int totPoints = Ps.rows();

        Eigen::MatrixXd Eigen_Ns(totPoints, 3);

        std::cout << "[Debug] grid reso: " << grid_reso << std::endl;
        std::cout << "[Debug] max depth: " << max_depth << std::endl;
        std::cout << "[Debug] normal kw: " << normal_kw << std::endl;
        std::cout << "[Debug] use radii: " << use_radii << std::endl;
        std::cout << "[Debug] offset 1: " << offset_1 << std::endl;
        std::cout << "[Debug] offset 2: " << offset_2 << std::endl;

        Eigen::VectorXd xs = GetRandomNs(totPoints);

        bool isOrignNormal = false;
        if (input_normal == 1 || (com_As && com_As != 111))
            isOrignNormal = true;

        int minPointsOnLeaves = min_num;

        if (isOrignNormal)
        {
            for (int i = 0; i < totPoints; i++)
            {
                double ns_norm = sqrt(Ns(i, 0) * Ns(i, 0) + Ns(i, 1) * Ns(i, 1) + Ns(i, 2) * Ns(i, 2));
                Ns(i, 0) = Ns(i, 0) / ns_norm;
                Ns(i, 1) = Ns(i, 1) / ns_norm;
                Ns(i, 2) = Ns(i, 2) / ns_norm;

                Eigen_Ns(i, 0) = Ns(i, 0);
                Eigen_Ns(i, 1) = Ns(i, 1);
                Eigen_Ns(i, 2) = Ns(i, 2);
            }
        }
        else
        {
            std::cout << "begin generate random normals \n";
            for (int i = 0; i < totPoints; i++)
            {
                double u = xs(i * 2 + 0);
                double v = xs(i * 2 + 1);
                Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };

                Eigen_Ns(i, 0) = nowN.x;
                Eigen_Ns(i, 1) = nowN.y;
                Eigen_Ns(i, 2) = nowN.z;
            }
            std::cout << "generate normals end \n";
        }

        Eigen::VectorXd leaves_As;
        Eigen::VectorXi leaves_idx;
        Eigen::MatrixXd igl_octree_leaves;

        GetOctreeLeaves(O_PI, O_CH, O_CN, igl_octree_leaves, leaves_idx, minPointsOnLeaves);
        std::cout << "igl_octree_leaves.size = " << igl_octree_leaves.rows() << "\n";

        Eigen::MatrixXd leaves_Ns;
        SmoothNormalstoOctreeLeaves(igl_octree_leaves, leaves_idx, O_PI, O_W, Ps, Eigen_Ns, leaves_Ns);
        std::cout << "smooth normals to Octree leaves \n";

        leaves_Ns = leaves_Ns.normalized();

        std::cout << "Eigen_Ns shape = " << Eigen_Ns.rows() << ", " << Eigen_Ns.cols() << "\n";
        std::cout << "Ps shape = " << Ps.rows() << ", " << Ps.cols() << "\n";
        std::cout << "igl_octree_leaves.size = " << igl_octree_leaves.rows() << "\n";

        std::vector<float> init_points_pos_x;
        std::vector<float> init_points_pos_y;
        std::vector<float> init_points_pos_z;
        init_points_pos_x.resize(igl_octree_leaves.rows());
        init_points_pos_y.resize(igl_octree_leaves.rows());
        init_points_pos_z.resize(igl_octree_leaves.rows());

        for (int i = 0; i < igl_octree_leaves.rows(); i++)
        {
            init_points_pos_x[i] = igl_octree_leaves(i, 0);
            init_points_pos_y[i] = igl_octree_leaves(i, 1);
            init_points_pos_z[i] = igl_octree_leaves(i, 2);
        }

        std::vector<float> sort_octree_leaves_x;
        std::vector<float> sort_octree_leaves_y;
        std::vector<float> sort_octree_leaves_z;
        sort_octree_leaves_x.resize(igl_octree_leaves.rows());
        sort_octree_leaves_y.resize(igl_octree_leaves.rows());
        sort_octree_leaves_z.resize(igl_octree_leaves.rows());

        std::vector<float> bound_upper; bound_upper.resize(3);
        std::vector<float> bound_lower; bound_lower.resize(3);
        std::unordered_map<geometrycentral::Vector3, int> points_indices_map;
        GenerateKDTreeForGWN(init_points_pos_x, init_points_pos_y, init_points_pos_z,
            sort_octree_leaves_x, sort_octree_leaves_y, sort_octree_leaves_z,
            bound_upper, bound_lower,
            true,
            points_indices_map);

        init_points_pos_x.clear();
        init_points_pos_y.clear();
        init_points_pos_z.clear();

        Eigen::MatrixXd sort_octree_leaves; sort_octree_leaves.resizeLike(igl_octree_leaves);
        for (int i = 0; i < igl_octree_leaves.rows(); i++)
        {
            sort_octree_leaves(i, 0) = sort_octree_leaves_x[i];
            sort_octree_leaves(i, 1) = sort_octree_leaves_y[i];
            sort_octree_leaves(i, 2) = sort_octree_leaves_z[i];
        }
        sort_octree_leaves_x.clear();
        sort_octree_leaves_y.clear();
        sort_octree_leaves_z.clear();

        Eigen::MatrixXd sort_leaves_Ns; sort_leaves_Ns.resizeLike(leaves_Ns);
        for (int i = 0; i < igl_octree_leaves.rows(); i++)
        {
            geometrycentral::Vector3 c_pos;
            c_pos[0] = sort_octree_leaves(i, 0);
            c_pos[1] = sort_octree_leaves(i, 1);
            c_pos[2] = sort_octree_leaves(i, 2);
            int ori_index = points_indices_map[c_pos];
            sort_leaves_Ns(i, 0) = leaves_Ns(ori_index, 0);
            sort_leaves_Ns(i, 1) = leaves_Ns(ori_index, 1);
            sort_leaves_Ns(i, 2) = leaves_Ns(ori_index, 2);

            double ns_norm = sqrt(sort_leaves_Ns(i, 0) * sort_leaves_Ns(i, 0) + sort_leaves_Ns(i, 1) * sort_leaves_Ns(i, 1) + sort_leaves_Ns(i, 2) * sort_leaves_Ns(i, 2));
            sort_leaves_Ns(i, 0) = sort_leaves_Ns(i, 0) / ns_norm;
            sort_leaves_Ns(i, 1) = sort_leaves_Ns(i, 1) / ns_norm;
            sort_leaves_Ns(i, 2) = sort_leaves_Ns(i, 2) / ns_norm;
        }

        // Build octree 2
        std::vector<std::vector<int > > O_PI_2;
        Eigen::MatrixXi O_CH_2;
        Eigen::MatrixXd O_CN_2;
        Eigen::VectorXd O_W_2;

        std::cout << "Build Tree 2" << std::endl;
        igl::octreeDWG(sort_octree_leaves, O_PI_2, O_CH_2, O_CN_2, O_W_2, max_depth);
        std::cout << "Build Tree 2 Done" << std::endl;

        Eigen::MatrixXd O_CM_2;
        Eigen::VectorXd O_R_2;
        Eigen::MatrixXd O_EC_2;

        std::cout << "igl_octree_leaves.size = " << igl_octree_leaves.rows() << "\n";

        Eigen::VectorXd leaves_radii;
        leaves_As.resizeLike(leaves_idx);
        leaves_As.setConstant(2e-6);

        Eigen::MatrixXi point_neighs;
        GetRadiiandAs(sort_octree_leaves, sort_leaves_Ns, leaves_radii, leaves_As, point_neighs, bound_upper, bound_lower, kw, com_As);

        Eigen::MatrixXf octree_leaves_f; octree_leaves_f.resizeLike(igl_octree_leaves);
        Eigen::MatrixXf leaves_Ns_f; leaves_Ns_f.resizeLike(leaves_Ns);
        Eigen::VectorXf leaves_As_f; leaves_As_f.resizeLike(leaves_As);
        Eigen::VectorXf leaves_radii_f; leaves_radii_f.resizeLike(leaves_radii);
        for (int i = 0; i < octree_leaves_f.rows(); i++)
        {
            octree_leaves_f(i, 0) = (float)sort_octree_leaves(i, 0);
            octree_leaves_f(i, 1) = (float)sort_octree_leaves(i, 1);
            octree_leaves_f(i, 2) = (float)sort_octree_leaves(i, 2);

            leaves_Ns_f(i, 0) = (float)sort_leaves_Ns(i, 0);
            leaves_Ns_f(i, 1) = (float)sort_leaves_Ns(i, 1);
            leaves_Ns_f(i, 2) = (float)sort_leaves_Ns(i, 2);

            leaves_As_f[i] = (float)leaves_As[i];
            leaves_radii_f[i] = (float)leaves_radii[i];
        }
        igl_octree_leaves.resize(0, 0);

        if (input_normal == 2 && com_As)
        {

            Eigen::VectorXd xs = GetRandomNs(leaves_Ns_f.rows());

            for (int i = 0; i < leaves_Ns_f.rows(); i++)
            {
                float u = xs(i * 2 + 0);
                float v = xs(i * 2 + 1);
                Vector3 nowN{ std::sin(u) * std::cos(v), std::sin(u) * std::sin(v), std::cos(u) };

                leaves_Ns_f(i, 0) = (float)nowN.x;
                leaves_Ns_f(i, 1) = (float)nowN.y;
                leaves_Ns_f(i, 2) = (float)nowN.z;
            }
        }

        Eigen::MatrixXf Ps_f; Ps_f.resizeLike(Ps);
        for (int i = 0; i < Ps.rows(); i++)
        {
            Ps_f(i, 0) = (float)Ps(i, 0);
            Ps_f(i, 1) = (float)Ps(i, 1);
            Ps_f(i, 2) = (float)Ps(i, 2);
        }

        int s = grid_reso;
        Eigen::MatrixXf grid_points_f;
        Eigen::RowVector3i res;
        {
            igl::voxel_grid(Ps_f, offset_1, s, 1, grid_points_f, res);
        }
        std::cout << "res_x = " << res(0) << ", res_y = " << res(1) << ", res_z = " << res(2) << "\n";
        std::cout << "GV shape = " << grid_points_f.rows() << ", " << grid_points_f.cols() << "\n";

        std::cout << "FWN Pre" << std::endl;
        igl::fast_winding_number(sort_octree_leaves, sort_leaves_Ns, leaves_As, O_PI_2, O_CH_2, 0, O_CM_2, O_R_2, O_EC_2);
        std::cout << "FWN Pre Done" << std::endl;

        sort_octree_leaves.resize(0, 0);
        sort_leaves_Ns.resize(0, 0);
        leaves_As.resize(0);

        TurboDWG turbodwg(octree_leaves_f, leaves_Ns_f, leaves_As_f, leaves_radii_f, grid_points_f,
            res(0), res(1), res(2),
            bound_upper, bound_lower);

        clock_t iter_time;
        Eigen::VectorXf As_out; As_out.resize(0);
        Eigen::VectorXf Cs_out; Cs_out.resize(0);
        if (com_As == 2)
        {
            As_out.resize(octree_leaves_f.rows());
            iter_time = turbodwg.TurboScreenedDWGAsOptLaunch(O_PI_2, O_CH_2, O_CM_2, O_R_2, O_EC_2, 2.3, sigma, total_iter, use_screened, use_radii, As_threshold, input_normal, start_o, out_mesh_path, test_iter, 5e-7, 100, As_out);
        }
        else if (com_As == 3)
        {
            As_out.resize(octree_leaves_f.rows());
            iter_time = turbodwg.TurboScreenedDWGAsCsOptLaunch(O_PI_2, O_CH_2, O_CM_2, O_R_2, O_EC_2, 2.3, sigma, total_iter, use_screened, use_radii, As_threshold, input_normal, start_o, out_mesh_path, test_iter, 1e-7, 100, 1e-4, 15000, As_out);
        }
        if (com_As == 4 || com_As == 5)
        {
            As_out.resize(octree_leaves_f.rows());
            Cs_out.resize(octree_leaves_f.rows());
            iter_time = turbodwg.TurboScreenedDWGOptLaunch(O_PI_2, O_CH_2, O_CM_2, O_R_2, O_EC_2, 2.3, sigma, total_iter, use_screened, use_radii, As_threshold, input_normal, start_o, out_mesh_path, test_iter,
                                                           learning_rate_Ai, learning_rate_Ai_2, learning_rate_Ci, step_iteration, diric_scale, diric_reso, ai_reg, ai_reg_2, ci_reg_c, ci_reg_a,
                                                           ai_rad, ci_rad, weight_scale,
                                                            before_ci_max_loop,
                                                            ci_max_loop_iteration,
                                                            max_loop,
                                                            ai_target_start,
                                                            ai_target_mid,

                                                            ci_target,
                                                            ai_smooth_rad,
                                                            ai_adjust,

                                                            ci_in_thres, ci_out_thres,
                                                            ci_final_in_thres, ci_final_out_thres,
                                                            ci_level,
                                                            ci_var_thres,
                                                            ai_var_thres_0,
                                                            ai_var_thres_1,
                                                            DWG_extract_mode,
                                                            DWG_initial_mode,
                                                            inside_x,
                                                            inside_y,
                                                            inside_z,
                                                            As_out, Cs_out);
        }
        else if(com_As == 0|| com_As == 1)
        {
            iter_time = turbodwg.TurboScreenedDWGLaunch(O_PI_2, O_CH_2, O_CM_2, O_R_2, O_EC_2, 2.3, sigma, total_iter, use_screened, use_radii, As_threshold, input_normal, start_o, out_mesh_path, test_iter);
        }

        if (use_leaves_generate == 1)
        {
            s = final_reso;
            Eigen::MatrixXf mesh_GV;
            igl::voxel_grid(octree_leaves_f, 0.01, s, 1, mesh_GV, res);
            turbodwg.GenerateMesh(mesh_GV, res(0), res(1), res(2), 2.3, sigma, final_kw, use_radii, out_mesh_full_path, iter_time, start_o);
        }
        else if (use_leaves_generate == 0 && !com_As)
        {
            std::cout << "FWN Pre" << std::endl;

            O_PI_2.clear();
            O_CH_2.resize(0, 0);
            O_CM_2.resize(0, 0);
            O_CN_2.resize(0, 0);
            O_EC_2.resize(0, 0);
            O_R_2.resize(0);

            Eigen::VectorXd points_As;
            points_As.resize(Ps.rows());
            if (com_As)
            {
                for (int i = 0; i < Ps.rows(); i++)
                {
                    points_As[i] = As[i];
                }
            }
            else
            {
                points_As.setConstant(2e-6);
            }
            igl::fast_winding_number(Ps, Eigen_Ns, points_As, O_PI, O_CH, 0, O_CM, O_R, O_EC);
            std::cout << "FWN Pre Done" << std::endl;

            octree_leaves_f.resize(0, 0);
            leaves_As_f.resize(0);

            Eigen::MatrixXf Ns_f; Ns_f.resizeLike(Eigen_Ns);
            Eigen::VectorXf As_f; As_f.resizeLike(points_As);
            Eigen::VectorXf Cs_f; Cs_f.resizeLike(points_As);
            for (int i = 0; i < Ps.rows(); i++)
            {
                Ns_f(i, 0) = (float)Eigen_Ns(i, 0);
                Ns_f(i, 1) = (float)Eigen_Ns(i, 1);
                Ns_f(i, 2) = (float)Eigen_Ns(i, 2);

                As_f[i] = (float)points_As[i];
                Cs_f[i] = 1.0f;
            }

            s = final_reso;
            Eigen::MatrixXf mesh_GV;
            igl::voxel_grid(Ps_f, offset_2, s, 1, mesh_GV, res);
            turbodwg.GenerateMesh(Ps_f, As_f, Cs_f, mesh_GV,
                O_PI, O_CH, O_CM, O_R, O_EC,
                res(0), res(1), res(2),
                Bound_Upper_KDT, Bound_Lower_KDT,
                2.3, sigma, final_kw, normal_kw, use_screened, use_radii, As_threshold, out_mesh_full_path, iter_time, start_o, output_mode, final_reso);
        }
        else if (use_leaves_generate == 0 && com_As)
        {
            O_PI_2.clear();
            O_CH_2.resize(0, 0);
            O_CM_2.resize(0, 0);
            O_CN_2.resize(0, 0);
            O_EC_2.resize(0, 0);
            O_R_2.resize(0);

            Eigen::VectorXd points_As;
            points_As.resize(Ps.rows());
            points_As.setConstant(2e-6);

            octree_leaves_f.resize(0, 0);
            leaves_As_f.resize(0);

            Eigen::MatrixXf Ns_f; Ns_f.resizeLike(Eigen_Ns);
            for (int i = 0; i < Ps.rows(); i++)
            {
                Ns_f(i, 0) = (float)Eigen_Ns(i, 0);
                Ns_f(i, 1) = (float)Eigen_Ns(i, 1);
                Ns_f(i, 2) = (float)Eigen_Ns(i, 2);
            }

            Eigen::VectorXf As_f;
            Eigen::VectorXf Cs_f;
            if (com_As == 2 || com_As == 5)
            {
                As_f.resize(As_out.size());
                Cs_f.resize(Cs_out.size());

                for(int i = 0; i < As_out.rows(); i++)
                    As_f[i] = As_out[i];

                for (int i = 0; i < As_out.rows(); i++)
                    Cs_f[i] = Cs_out[i];
            }
            else
            {
                As_f.resize(Ps.rows());
                for (int i = 0; i < Ps.rows(); i++)
                    As_f[i] = (float)points_As[i];

                Cs_f.resize(Ps.rows());
                for (int i = 0; i < Ps.rows(); i++)
                    Cs_f[i] = 1.0f;
            }

            s = final_reso;
            Eigen::MatrixXf mesh_GV;
            igl::voxel_grid(Ps_f, offset_2, s, 1, mesh_GV, res);

            bool As_opt = false;
            if (com_As == 2 || com_As == 5)
                As_opt = true;
            else
                As_opt = false;

            turbodwg.PreGeneration(Ps_f, As_f, Cs_f, mesh_GV,
                res(0), res(1), res(2),
                Bound_Upper_KDT, Bound_Lower_KDT,
                normal_kw,
                As_opt,
                Eigen_Ns);

            if (com_As == 1  )
            {
                igl::copyleft::cgal::point_areas(
                    Ps, Point_Neighbours, Eigen_Ns, points_As);

                turbodwg.UpdateAs(points_As);
            }

            std::cout << "FWN Pre" << std::endl;
            igl::fast_winding_number(Ps, Eigen_Ns, points_As, O_PI, O_CH, 0, O_CM, O_R, O_EC);
            std::cout << "FWN Pre Done" << std::endl;

            if (com_As)
            {
                if (com_As == 2)
                {
                    turbodwg.MeshAsOptGeneration(O_PI, O_CH, O_CM, O_R, O_EC,
                        res(0), res(1), res(2), 2.3, sigma, final_kw, use_screened,
                        use_radii, As_threshold, out_mesh_full_path, start_o, start_time_s, output_mode, final_reso, 5e-7, 100);
                }
                else
                {
                    turbodwg.MeshGeneration(O_PI, O_CH, O_CM, O_R, O_EC, res(0), res(1), res(2), 2.3, sigma, final_kw, use_screened, use_radii, As_threshold, out_mesh_full_path, start_o, start_time_s, output_mode, final_reso, weight_scale);
                }
            }
        }
    }
}
