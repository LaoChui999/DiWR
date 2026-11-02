#pragma once

#include <thrust/reduce.h>
#include <thrust/device_ptr.h>
#include <thrust/extrema.h>
#include <thrust/device_vector.h>
#include <thrust/transform_reduce.h>
#include <thrust/functional.h>
#include <thrust/sort.h>
#include <thrust/inner_product.h>
#include <thrust/iterator/zip_iterator.h>
#include <thrust/tuple.h>

#include <unordered_map>
#include <chrono>
#include <ctime>
#include <set>
#include <string>

#include "Turbo_GWN_l.h"
#include "MarchingCubeCuda.h"
#include "TurboNormalInit.h"
#include "RMSProp.h"

void GenerateKDTreeForDWG(
	float3* d_points,
	float3* d_normals,
	int points_size,
	float3* bound_upper, float3* bound_lower);

void CudaKNNWithKDTreeforDWG(
	float3* d_points,
	int numPoints,
	float3* d_queries,
	int numQueries,
	float3* bound_upper, float3* bound_lower,
	int* d_results,
	int k);

class TurboDWG
{
public:
	TurboDWG(Eigen::MatrixXf& octree_leaves, Eigen::MatrixXf& leaves_Ns, Eigen::VectorXf& leaves_As, Eigen::VectorXf radii, Eigen::MatrixXf& grid_points, int res_x, int res_y, int res_z, std::vector<float> bound_upper_kdtree, std::vector<float> bound_lower_kdtree);
	TurboDWG() = default;
	void PreTurboDWG(int point_cloud_size, Eigen::VectorXf& leaves_As, Eigen::VectorXf& leaves_Cs, Eigen::MatrixXf& grid_points, int res_x, int res_y, int res_z);
	clock_t TurboScreenedDWGLaunch(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC, float beta, float sigma, int total_iters, bool use_screened, bool use_radii, float As_threshold, int input_normal, clock_t start_o, std::string mesh_path, int test_iter);
	clock_t TurboScreenedDWGAsOptLaunch(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC, float beta, float sigma, int total_iters, bool use_screened, bool use_radii, float As_threshold, int input_normal, clock_t start_o, std::string mesh_path, int test_iter, float learning_rate, int total_As_opt_iteration, Eigen::VectorXf& As_out);
	clock_t TurboScreenedDWGOptLaunch(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC, float beta, float sigma, int total_iters, bool use_screened, bool use_radii, float As_threshold, int input_normal, clock_t start_o, std::string mesh_path, int test_iter,
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
		float inside_z,
		Eigen::VectorXf& As_out, Eigen::VectorXf& Cs_out);
	clock_t TurboScreenedDWGAsCsOptLaunch(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC, float beta, float sigma, int total_iters, bool use_screened, bool use_radii, float As_threshold, int input_normal, clock_t start_o, std::string mesh_path, int test_iter, float learning_rate_a, int total_As_opt_iteration, float learning_rate_c, int total_Cs_opt_iteration, Eigen::VectorXf& As_out);
	void UpdateRadii(int k_size);
	void UpdateAs(Eigen::VectorXd& leaves_As);

	void CiComputation(float mean_wn);

	void PreGeneration(Eigen::MatrixXf& cloud_points, Eigen::VectorXf& points_As, Eigen::VectorXf& points_Cs, Eigen::MatrixXf& grid_points,
		int res_x, int res_y, int res_z,
		std::vector<float> bound_upper_kdtree, std::vector<float> bound_lower_kdtree,
		int k_normal_size,
		bool As_opt,
		Eigen::MatrixXd& out_normal);
	void GenerateMesh(Eigen::MatrixXf& grid_points, int res_x, int res_y, int res_z, float beta, float sigma, int k_size, bool use_radii, std::string mesh_full_path, clock_t strat_time, clock_t strat);
	void GenerateMesh(Eigen::MatrixXf& cloud_points, Eigen::VectorXf& leaves_As, Eigen::VectorXf& leaves_Cs, Eigen::MatrixXf& grid_points,
		std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC,
		int res_x, int res_y, int res_z,
		std::vector<float> bound_upper_kdtree, std::vector<float> bound_lower_kdtree,
		float beta, float sigma, int k_size, int k_normal_size, bool use_screened, bool use_radii, float As_threshold, std::string mesh_full_path, clock_t strat_time, clock_t strat, int output_mode, int sample_reso);
	void MeshGeneration(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC,
		int res_x, int res_y, int res_z,
		float beta, float sigma, int k_size, bool use_screened, bool use_radii, float As_threshold, std::string mesh_full_path, clock_t start, std::chrono::high_resolution_clock::time_point start_time_s, int output_mode, int sample_reso, float weight_scale);
	void MeshAsOptGeneration(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC,
		int res_x, int res_y, int res_z,
		float beta, float sigma, int k_size, bool use_screened, bool use_radii, float As_threshold, std::string mesh_full_path, clock_t start, std::chrono::high_resolution_clock::time_point start_time_s, int output_mode, int sample_reso,
		float learning_rate, int total_As_opt_iteration);
	void GWNAsOptimization(float beta, float sigma, bool use_screened, bool use_radii, float learning_rate, int total_iteration, TurboLibiGWN* turbogwn);

	void sGWNOptimization(float beta, float sigma, bool use_screened, bool use_radii,   float learning_rate_Ai, float learning_rate_Ai_2, float learning_rate_Ci,
		int total_iteration, TurboLibiGWN* turbogwn, std::string mesh_path,
		int loop, int ci_max_loop,
		float diric_scale, int diric_reso,
		float ai_target_start, float ai_target_mid,
		float ai_reg, float ai_reg_2,
		float ci_target, float ci_reg_c, float ci_reg_a,
		float ai_rad, float ci_rad,
		float ai_smooth_rad, bool ai_adjust,

		float ci_in_thres, float ci_out_thres,
		float ci_final_in_thres, float ci_final_out_thres,
		int ci_level,
		float ci_var_thres,
		int loss_mode,
		float weight_scale,
		float inside_x, float inside_y, float inside_z);
	void ComputeVoxelWeight(float threshold, float voxel_length, float* distances, float* weight, float size);

	void PreProcessOutliers(float dist_threshold, std::vector<float>& point_x, std::vector<float>& point_y, std::vector<float>& point_z,
		std::vector<float>& normal_x, std::vector<float>& normal_y, std::vector<float>& normal_z,
		std::vector<float>& out_point_x, std::vector<float>& out_point_y, std::vector<float>& out_point_z,
		std::vector<float>& out_normal_x, std::vector<float>& out_normal_y, std::vector<float>& out_normal_z);
protected:
	float3* m_Point_Cloud_GPU = 0;
	float3* m_Grid_Points_GPU = 0;

	float* m_PC_WN_GPU = 0;
	float* m_GP_WN_GPU = 0;

	float3* m_Normals_GPU = 0;

	float* m_As_GPU = 0;
	float* m_ini_As_GPU = 0;
	float* m_ini_constant_As_GPU = 0;

	float* m_Cs_GPU = 0;
	float* m_last_Cs_GPU = 0;

	float* m_nUi_GPU = 0;
	float* m_nVi_GPU = 0;

	float* m_Radii_GPU = 0;

	int m_Point_Cloud_size = 0;
	int m_Grid_Points_size = 0;

	int m_Grid_Res_X = 0;
	int m_Grid_Res_Y = 0;
	int m_Grid_Res_Z = 0;

	float3* m_Bound_Upper_KDT_GPU = 0;
	float3* m_Bound_Lower_KDT_GPU = 0;

	TurboLibiGWN* m_GWN_GPU = 0;

	bool m_Ci_Opted = false;

	void OutlierFilter(float* As, float threshold, int point_num);

	void InitialCi(float scale, float radius2, float* gwn, std::string mesh_path, int loop, int ci_level,

		float ci_in_thres, float ci_out_thres);
	void FinalizeCi(float* gwn, std::string mesh_path,

		float ci_in_thres, float ci_out_thres,
		int set_loop, int top_m, int point_size );
	void AdjustAi(float scale, float radius2, float smooth_radius, bool ai_adjust, int loop, int loss_mode);

	void GetPlaneField(int sample_size, int axis, float beta, float sigma, bool use_radii, int loop, std::string mesh_path);
    int SetSamplings(float3* uni_grid_samples_gpu, float* distances, std::vector<bool>& uni_grid_samples_is_near, int sample_grid_size, float dist_thres, float beta, float sigma, bool use_radii, int loop);
	int SetMode(float* last_Ai, float ai_var_thres_0, float ai_var_thres_1, int c_loop, int c_total_loop, int after_ci_loop, int before_ci_max_loop, int max_loop, int loss_mode);
};
