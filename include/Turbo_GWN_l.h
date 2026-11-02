#pragma once
#include <cuda.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>

#include <fstream>

#include <Eigen/Core>

#include "geometrycentral/vector3.h"

struct OctreeNodeLibiGPU
{
	int* Children = nullptr;
	int Father = -1;
	int IndexInFather = -1;
	unsigned int* Point_Cloud_Indices = nullptr;
	unsigned int Point_Cloud_Num = 0;

	float As_Total = 0.0;
	float* As_Grad_x = nullptr;
	float* As_Grad_y = nullptr;
	float* As_Grad_z = nullptr;

	float* Cs_Grad_x = nullptr;
	float* Cs_Grad_y = nullptr;
	float* Cs_Grad_z = nullptr;

	int* Query_Buffer = nullptr;
	int* Query_Index = nullptr;
	bool Large_Node = false;

	float3 Mass_Center;
	float3 Mass_Normal;
	float Max_R;
	float Mass_Screened;
};

class TurboLibiGWN
{
public:

	TurboLibiGWN() = default;
	void PreLibiGWN(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC);

	void TurboLibiGWNLaunch(float3* query_points,
		float3* point_cloud,
		int query_size,
		float3* i_Normals,
		float* i_As,
		float& beta,
		float* out_gwn);

	void TurboLibiScreenedGWNLaunch(float3* query_points,
		float3* point_cloud,
		int query_size,
		float3* i_Normals,
		float* i_As,
		float* i_radii,
		float& beta,
		float& sigma,
		float* out_gwn,
		bool use_radii);

	void TurboGWNGradientLaunch(float3* query_points,
		float3* point_cloud,
		int query_size,
		float3* i_Normals,
		float* i_As,
		float* i_Cs,
		float* i_radii,
		float& beta,
		float& sigma,
		float3* out_gwn,
		bool use_radii,
		bool use_screened);

	void TurboGWNGradientBackwardLaunch(float3* query_points,
		float3* point_cloud,
		int query_size,
		float3* i_Normals,
		float* i_As,
		float* i_Cs,
		float* i_radii,

		float3* gwn_grad,

		float* Ai_grad,
		float* Ci_grad,
		float* grad_scale_coff,
		float grad_volume_scale,
		float& beta,
		float& sigma,
		bool use_radii,
		bool use_screened,
		bool com_Ai_grad,
		bool com_Ci_grad,
		bool com_normal_grad,
		int normal_mode);

	void TurboGWNGradientAsPatchBackwardLaunch(
		float3* query_points,
		float3* point_cloud,
		float3* i_Normals,
		float* i_Cs,
		float3* gwn_grad,
		float* Ai_grad,
		float* grad_scale_coff,
		float grad_volume_scale,
		int query_size);

	void TurboGWNGradientCsPatchBackwardLaunch(
		float3* query_points,
		float3* point_cloud,
		float3* i_Normals,
		float* i_As,
		float3* gwn_grad,
		float* Ci_grad,
		float* grad_scale_coff,
		float grad_volume_scale,
		int query_size);

	void TurboGWNAsPatchBackwardLaunch(
		float3* query_points,
		float3* point_cloud,
		float3* i_Normals,
		float* i_Cs,
		float* gwn,
		float* Ai_grad,
		float target_gwn,
		float grad_scale,
		int batch_size);

	void TurboGWNCsPatchBackwardLaunch(
		float3 * query_points,
		float3 * point_cloud,
		float3 * i_Normals,
		float* i_As,
		float* gwn,
		float* Ai_grad,
		float target_gwn,
		float grad_scale,
		int batch_size);

	void TurboLibiScreenedGWNCsLaunch(float3* query_points,
		float3* point_cloud,
		int query_size,
		float3* i_Normals,
		float* i_As,
		float* i_Cs,
		float* i_radii,
		float& beta,
		float& sigma,
		float* out_gwn,
		bool use_radii);

	void UpdateMassNormals(float3* i_Normals, float* i_As);

	void UpdateGWNInfo(float3* i_Normals, float3* point_cloud, float* i_As, float* i_radii);
	void UpdateGWNAllInfo(float3* i_Normals, float3* point_cloud, float* i_As, float* i_Cs);
	void UpdateGWNInfoAsGrad(float3* i_Normals, float3* point_cloud, float* i_As);
	void UpdateGWNInfoAsCsGrad(float3* i_Normals, float3* point_cloud, float* i_As, float* i_Cs);
	void EarseMemory();

	void AllocateQueryMem(int query_size, int point_size);

	void TurboLibiScreenedGWNAsCsBackwardLaunch(float3* query_points,
		float3* point_cloud,
		int query_size,
		float3* i_Normals,
		float* i_As,
		float* i_Cs,
		float* i_radii,
		float& beta,
		float& sigma,
		bool use_radii,
		bool use_screened,
		float* target_gwn,
		float* grad_scale_Ai,
		float grad_scale_Ci,
		float* out_gwn,
		float* Ai_grad,
		float* Ci_grad,
		bool com_Ai,
		bool com_Ci,
		bool patch);

protected:
	OctreeNodeLibiGPU* m_Octree_GPU = 0;
	OctreeNodeLibiGPU* m_Octree_CPU = 0;
	int m_Node_Size = 0;

	int* m_Large_Node_Index = 0;
	int* m_Large_Node_Array = 0;
	int* m_Large_Node_Point_Index = 0;
	int* m_Large_Node_Point_Local_Index = 0;
	int m_Large_Node_Size = 0;
	int m_Node_Array_Size = 0;
	int m_Allocate_Size = 0;

	int* m_Point_Grad_Index = 0;
	float* m_Point_Gradient = 0;

};
