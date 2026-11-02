# include "TurboDWG.h"

#define gpuErrchk(ans) { gpuAssert((ans), #ans, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* command, const char* file, int line, bool abort = true)
{

	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

TurboDWG::TurboDWG(Eigen::MatrixXf& octree_leaves, Eigen::MatrixXf& leaves_Ns, Eigen::VectorXf& leaves_As, Eigen::VectorXf radii, Eigen::MatrixXf& grid_points, int res_x, int res_y, int res_z, std::vector<float> bound_upper_kdtree, std::vector<float> bound_lower_kdtree)
{
	int point_cloud_size = octree_leaves.rows();
	cudaMallocManaged((char**)&this->m_Point_Cloud_GPU, point_cloud_size * sizeof(*this->m_Point_Cloud_GPU));
	std::unique_ptr<float3[]> point_cloud_cpu = std::make_unique<float3[]>(size_t(point_cloud_size));
	for (int i = 0; i < point_cloud_size; i++)
	{
		point_cloud_cpu[i] = make_float3(octree_leaves(i, 0), octree_leaves(i, 1), octree_leaves(i, 2));
	}
	gpuErrchk(cudaMemcpy(this->m_Point_Cloud_GPU, point_cloud_cpu.get(), point_cloud_size * sizeof(float3), cudaMemcpyHostToDevice));
	this->m_Point_Cloud_size = point_cloud_size;

	int leaves_Ns_size = leaves_Ns.rows();
	cudaMallocManaged((char**)&this->m_Normals_GPU, leaves_Ns_size * sizeof(*this->m_Normals_GPU));
	std::unique_ptr<float3[]> leaves_Ns_cpu = std::make_unique<float3[]>(size_t(leaves_Ns_size));
	for (int i = 0; i < leaves_Ns_size; i++)
	{
		leaves_Ns_cpu[i] = make_float3(leaves_Ns(i, 0), leaves_Ns(i, 1), leaves_Ns(i, 2));
	}
	gpuErrchk(cudaMemcpy(this->m_Normals_GPU, leaves_Ns_cpu.get(), leaves_Ns_size * sizeof(float3), cudaMemcpyHostToDevice));

	int As_size = leaves_As.size();

	cudaMallocManaged((char**)&this->m_As_GPU, As_size * sizeof(*this->m_As_GPU));
	cudaMallocManaged((char**)&this->m_ini_As_GPU, As_size * sizeof(*this->m_ini_As_GPU));
	cudaMallocManaged((char**)&this->m_ini_constant_As_GPU, As_size * sizeof(*this->m_ini_constant_As_GPU));
	std::unique_ptr<float[]> As_cpu = std::make_unique<float[]>(size_t(As_size));
	for (int i = 0; i < As_size; i++)
	{
		As_cpu[i] = leaves_As[i];
	}
	gpuErrchk(cudaMemcpy(this->m_As_GPU, As_cpu.get(), As_size * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(this->m_ini_As_GPU, As_cpu.get(), As_size * sizeof(float), cudaMemcpyHostToDevice));
	gpuErrchk(cudaMemcpy(this->m_ini_constant_As_GPU, As_cpu.get(), As_size * sizeof(float), cudaMemcpyHostToDevice));

	int Cs_size = leaves_As.size();
	cudaMallocManaged((char**)&this->m_Cs_GPU, Cs_size * sizeof(*this->m_Cs_GPU));
	cudaMallocManaged((char**)&this->m_last_Cs_GPU, Cs_size * sizeof(*this->m_last_Cs_GPU));
	for (int i = 0; i < Cs_size; i++)
	{
		this->m_Cs_GPU[i] = 1.0f;
	}

	int nui_size = leaves_As.size();
	cudaMallocManaged((char**)&this->m_nUi_GPU, nui_size * sizeof(*this->m_nUi_GPU));
	int nvi_size = leaves_As.size();
	cudaMallocManaged((char**)&this->m_nVi_GPU, nvi_size * sizeof(*this->m_nVi_GPU));

	int radii_size = radii.size();

	cudaMallocManaged((char**)&this->m_Radii_GPU, radii_size * sizeof(*this->m_Radii_GPU));
	std::unique_ptr<float[]> radii_cpu = std::make_unique<float[]>(size_t(radii_size));
	for (int i = 0; i < radii_size; i++)
	{
		radii_cpu[i] = radii[i];
	}
	gpuErrchk(cudaMemcpy(this->m_Radii_GPU, radii_cpu.get(), radii_size * sizeof(float), cudaMemcpyHostToDevice));

	int grid_points_size = grid_points.rows();
	cudaMallocManaged((char**)&this->m_Grid_Points_GPU, grid_points_size * sizeof(*this->m_Grid_Points_GPU));
	std::unique_ptr<float3[]> grid_points_cpu = std::make_unique<float3[]>(size_t(grid_points_size));
	for (int i = 0; i < grid_points_size; i++)
	{
		grid_points_cpu[i] = make_float3(grid_points(i, 0), grid_points(i, 1), grid_points(i, 2));
	}
	gpuErrchk(cudaMemcpy(this->m_Grid_Points_GPU, grid_points_cpu.get(), grid_points_size * sizeof(float3), cudaMemcpyHostToDevice));
	this->m_Grid_Points_size = grid_points_size;

	this->m_Grid_Res_X = res_x;
	this->m_Grid_Res_Y = res_y;
	this->m_Grid_Res_Z = res_z;

	cudaMallocManaged((char**)&this->m_PC_WN_GPU, point_cloud_size * sizeof(*this->m_PC_WN_GPU));
	cudaMallocManaged((char**)&this->m_GP_WN_GPU, grid_points_size * sizeof(*this->m_GP_WN_GPU));

	cudaMallocManaged((char**)&this->m_Bound_Upper_KDT_GPU, sizeof(*this->m_Bound_Upper_KDT_GPU));
	this->m_Bound_Upper_KDT_GPU[0].x = bound_upper_kdtree[0];
	this->m_Bound_Upper_KDT_GPU[0].y = bound_upper_kdtree[1];
	this->m_Bound_Upper_KDT_GPU[0].z = bound_upper_kdtree[2];
	cudaMallocManaged((char**)&this->m_Bound_Lower_KDT_GPU, sizeof(*this->m_Bound_Lower_KDT_GPU));
	this->m_Bound_Lower_KDT_GPU[0].x = bound_lower_kdtree[0];
	this->m_Bound_Lower_KDT_GPU[0].y = bound_lower_kdtree[1];
	this->m_Bound_Lower_KDT_GPU[0].z = bound_lower_kdtree[2];

	std::cout << "Res X: " << this->m_Grid_Res_X << " " << "Res Y: " << this->m_Grid_Res_Y << " " << "Res Z: " << this->m_Grid_Res_Z << std::endl;
	std::cout << "GPU Point Cloud Size: " << this->m_Point_Cloud_size << std::endl;
}

void TurboDWG::PreTurboDWG(int point_cloud_size, Eigen::VectorXf& leaves_As, Eigen::VectorXf& leaves_Cs, Eigen::MatrixXf& grid_points, int res_x, int res_y, int res_z)
{
	int As_size = leaves_As.size();
	cudaMallocManaged((char**)&this->m_As_GPU, As_size * sizeof(*this->m_As_GPU));
	std::unique_ptr<float[]> As_cpu = std::make_unique<float[]>(size_t(As_size));
	for (int i = 0; i < As_size; i++)
	{
		As_cpu[i] = leaves_As[i];
	}
	gpuErrchk(cudaMemcpy(this->m_As_GPU, As_cpu.get(), As_size * sizeof(float), cudaMemcpyHostToDevice));

	int Cs_size = leaves_Cs.size();
	cudaMallocManaged((char**)&this->m_Cs_GPU, Cs_size * sizeof(*this->m_Cs_GPU));
	std::unique_ptr<float[]> Cs_cpu = std::make_unique<float[]>(size_t(Cs_size));
	for (int i = 0; i < Cs_size; i++)
	{
		Cs_cpu[i] = leaves_Cs[i];
	}
	gpuErrchk(cudaMemcpy(this->m_Cs_GPU, Cs_cpu.get(), Cs_size * sizeof(float), cudaMemcpyHostToDevice));

	int radii_size = point_cloud_size;
	cudaMallocManaged((char**)&this->m_Radii_GPU, radii_size * sizeof(*this->m_Radii_GPU));

	int grid_points_size = grid_points.rows();
	cudaMallocManaged((char**)&this->m_Grid_Points_GPU, grid_points_size * sizeof(*this->m_Grid_Points_GPU));
	std::unique_ptr<float3[]> grid_points_cpu = std::make_unique<float3[]>(size_t(grid_points_size));
	for (int i = 0; i < grid_points_size; i++)
	{
		grid_points_cpu[i] = make_float3(grid_points(i, 0), grid_points(i, 1), grid_points(i, 2));
	}
	gpuErrchk(cudaMemcpy(this->m_Grid_Points_GPU, grid_points_cpu.get(), grid_points_size * sizeof(float3), cudaMemcpyHostToDevice));
	this->m_Grid_Points_size = grid_points_size;

	this->m_Grid_Res_X = res_x;
	this->m_Grid_Res_Y = res_y;
	this->m_Grid_Res_Z = res_z;

	cudaMallocManaged((char**)&this->m_PC_WN_GPU, point_cloud_size * sizeof(*this->m_PC_WN_GPU));
	cudaMallocManaged((char**)&this->m_GP_WN_GPU, grid_points_size * sizeof(*this->m_GP_WN_GPU));

	std::cout << "GPU Point Cloud Size: " << point_cloud_size << std::endl;
	std::cout << "Res X: " << this->m_Grid_Res_X << " " << "Res Y: " << this->m_Grid_Res_Y << " " << "Res Z: " << this->m_Grid_Res_Z << std::endl;
}

__global__
void ResetZero(float3* __restrict__ vector,
	int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		vector[tid] = make_float3(0.0f, 0.0f, 0.0f);
	}
}

__global__
void LoadZeroData(float3* __restrict__ zero_points,
	float3* __restrict__ nonzero_points,
	int* __restrict__ zero_points_index,
	int* __restrict__ nonzero_points_index,
	float3* __restrict__ nonzero_points_normal,
	float* __restrict__ tmp_lengths,
	float3* __restrict__ point_pos,
	float3* __restrict__ point_normal,
	int* c_zero_index,
	int* c_nonzero_index,
	int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		if (tmp_lengths[tid] < 1e-10)
		{
			int c_ato_index = atomicAdd(c_zero_index, 1);

			zero_points[c_ato_index] = point_pos[tid];
			zero_points_index[c_ato_index] = tid;
		}
		else
		{
			int c_ato_index = atomicAdd(c_nonzero_index, 1);

			nonzero_points[c_ato_index] = point_pos[tid];
			nonzero_points_index[c_ato_index] = tid;
			nonzero_points_normal[c_ato_index] = point_normal[tid];
		}
	}
}

__global__
void UpdateNormals(float3* __restrict__ prev_normals,
	int* __restrict__ knn_indices,
	float3* __restrict__ normals,
	int knn_size, int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		for (int j = 0; j < knn_size; j++)
		{
			int idx = knn_indices[tid * knn_size + j];

			atomicAdd(&(prev_normals[idx].x), normals[tid].x);
			atomicAdd(&(prev_normals[idx].y), normals[tid].y);
			atomicAdd(&(prev_normals[idx].z), normals[tid].z);
		}
	}
}

__global__
void LoadNormals(float3* __restrict__ prev_normals,
	int* __restrict__ knn_indices,
	float3* __restrict__ normals,
	int knn_size, int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		float3 sum_normal = make_float3(0.0f, 0.0f, 0.0f);

		for (int j = 0; j < knn_size; j++)
		{
			int idx = knn_indices[tid * knn_size + j];
			sum_normal.x += prev_normals[idx].x;
			sum_normal.y += prev_normals[idx].y;
			sum_normal.z += prev_normals[idx].z;
		}
		sum_normal = make_float3(sum_normal.x / ((float)knn_size), sum_normal.y / ((float)knn_size), sum_normal.z / ((float)knn_size));

		float length = __fsqrt_rn(sum_normal.x * sum_normal.x +
			sum_normal.y * sum_normal.y +
			sum_normal.z * sum_normal.z);

		float normal_scale = __frcp_rn(length + 1e-8);

		normals[tid] = make_float3(sum_normal.x * normal_scale,
			sum_normal.y * normal_scale,
			sum_normal.z * normal_scale);
	}
}

__global__
void LoadWeight(float* __restrict__ prev_weights,
	int* __restrict__ knn_indices,
	float* __restrict__ weights,
	int knn_size, int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		float sum_weight = 0.0f;

		for (int j = 0; j < knn_size; j++)
		{
			int idx = knn_indices[tid * knn_size + j];
			sum_weight += prev_weights[idx];
		}
		sum_weight = sum_weight / ((float)knn_size);

		weights[tid] = sum_weight;
	}
}

__global__
void NormalizeNormals(float3* __restrict__ tmp_normals, float3* __restrict__ ori_normals, float* __restrict__ lengths, int* __restrict__ zero_size, int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		float length = __fsqrt_rn(tmp_normals[tid].x * tmp_normals[tid].x +
			tmp_normals[tid].y * tmp_normals[tid].y +
			tmp_normals[tid].z * tmp_normals[tid].z);
		lengths[tid] = length;

		float normal_scale = __frcp_rn(length + 1e-8);

		if (length < 1e-10)
		{
			atomicAdd(zero_size, 1);
		}
		else
		{
			ori_normals[tid] = make_float3(tmp_normals[tid].x * normal_scale,
				tmp_normals[tid].y * normal_scale,
				tmp_normals[tid].z * normal_scale);
		}
	}
}

__global__
void SimpleUpdateNonzeroNormals(
	int* __restrict__ knn_indices,
	int* __restrict__ zero_points_index,
	float3* __restrict__ nonzero_points_normal,
	float3* __restrict__ ori_normals,
	int k_size,
	int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		float3 sum_normal = make_float3(0.0f, 0.0f, 0.0f);

		for (int i = 0; i < k_size; i++)
		{
			int c_index = knn_indices[tid * k_size + i];
			sum_normal.x += nonzero_points_normal[c_index].x;
			sum_normal.y += nonzero_points_normal[c_index].y;
			sum_normal.z += nonzero_points_normal[c_index].z;

		}

		float length = __fsqrt_rn(sum_normal.x * sum_normal.x +
			sum_normal.y * sum_normal.y +
			sum_normal.z * sum_normal.z);

		float normal_scale = __frcp_rn(length + 1e-8);

		int ori_index = zero_points_index[tid];
		ori_normals[ori_index] = make_float3(sum_normal.x * normal_scale,
			sum_normal.y * normal_scale,
			sum_normal.z * normal_scale);

	}
}

__global__
void SimpleUpdateRadii(
	int* __restrict__ knn_indices,
	float3* __restrict__ points,
	float* __restrict__ radii,
	int k_size,
	int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		float radii_value = 0;
		for (int i = 0; i < k_size; i++)
		{
			float3 point_diff;
			int c_index = knn_indices[tid * k_size + i];
			point_diff.x += (points[tid].x - points[c_index].x);
			point_diff.y += (points[tid].y - points[c_index].y);
			point_diff.z += (points[tid].z - points[c_index].z);

			radii_value += (point_diff.x * point_diff.x + point_diff.y * point_diff.y + point_diff.z * point_diff.z);
		}

		radii_value /= ((float)k_size);
		radii_value = __fsqrt_rn(radii_value);
		radii_value = (radii_value < 0.0015f) ? 0.0015f : radii_value;
		radii_value = (radii_value > 0.015f) ? 0.015f : radii_value;

		radii[tid] = radii_value;
	}
}

__global__
void AsFiltertoZero(
	float* __restrict__ As,
	int* __restrict__ As_indices,
	int filter_num)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < filter_num)
	{
		int c_index = As_indices[tid];
		As[c_index] = 1e-9;
	}
}

__global__
void AsRegKernal(float* __restrict__ out_As_grad, float grad_scale, int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		atomicAdd(&(out_As_grad[tid]), grad_scale);
	}
}

__global__
void CiRegKernal(float* __restrict__ Ai, float* __restrict__ out_Ci_grad, float grad_scale, int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{

		float c_grad = grad_scale * Ai[tid];
		atomicAdd(&(out_Ci_grad[tid]), c_grad);
	}
}

__global__
void CiCrossRegKernal(float* __restrict__ Ci, float* __restrict__ out_Ci_grad, float grad_scale, int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{

		float c_grad = 0.0f;
		if (Ci[tid] < 0 || Ci[tid] >1)
			c_grad = grad_scale * (2.0f * Ci[tid] - 1.0f);
		else
			c_grad = grad_scale * (1.0f - 2.0f * Ci[tid]);

		atomicAdd(&(out_Ci_grad[tid]), c_grad);
	}
}

__global__
void AsClipKernel(float* __restrict__ data, float* __restrict__ ini_data, float* __restrict__ Ci, int com_size)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < com_size)
	{
		float prev_value = data[tid];
		data[tid] = fminf(fmaxf(prev_value, ini_data[tid] / 100.0f), ini_data[tid] * 1000.0f);

	}
}

__global__
void CiClipKernel(float* __restrict__ data, int com_size, bool clip)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	if (tid < com_size)
	{
		float prev_value = data[tid];

		data[tid] = fmaxf(prev_value, 0.01f);

		if (clip)
		{
			data[tid] = fminf(fmaxf(prev_value, 0.01f), 0.99f);
		}
	}
}

__global__
void ComPtoMDist(float3* __restrict__ points,
	float3* __restrict__ mesh_points,
	int* __restrict__ knn_indices,
	float* __restrict__ distances,
	int knn_size, int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		float c_sum_distance = 0.0f;

		for (int j = 0; j < knn_size; j++)
		{
			float3 c_vec;

			int idx = knn_indices[tid * knn_size + j];
			c_vec.x = points[tid].x - mesh_points[idx].x;
			c_vec.y = points[tid].y - mesh_points[idx].y;
			c_vec.z = points[tid].z - mesh_points[idx].z;

			float c_dist = __fsqrt_rn(c_vec.x * c_vec.x +
				c_vec.y * c_vec.y +
				c_vec.z * c_vec.z);

			c_sum_distance += c_dist;
		}

		distances[tid] = c_sum_distance / ((float)knn_size);
	}
}

__global__
void ComPtoMMinDist(float3* __restrict__ points,
	float3* __restrict__ mesh_points,
	float* __restrict__ Ci,
	int* __restrict__ knn_indices,
	float* __restrict__ distances,
	int knn_size, int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		float min_distance = 99999999.0f;

		for (int j = 0; j < knn_size; j++)
		{
			float3 c_vec;

			int idx = knn_indices[tid * knn_size + j];
			c_vec.x = points[tid].x - mesh_points[idx].x;
			c_vec.y = points[tid].y - mesh_points[idx].y;
			c_vec.z = points[tid].z - mesh_points[idx].z;

			float c_dist = __fsqrt_rn(c_vec.x * c_vec.x +
				c_vec.y * c_vec.y +
				c_vec.z * c_vec.z);

			if (Ci[idx] >= 0.8f)
			{
				min_distance = fminf(min_distance, c_dist);

			}
		}

		distances[tid] = min_distance;

	}
}

__global__
void BaseDistClassifyPoint(float irregural_value,
	float* __restrict__ distances,
	int* __restrict__ regural_indices,
	int* __restrict__ irregural_indices,
	int* __restrict__ regural_count,
	int* __restrict__ irregural_count,
	bool* __restrict__ indictor,
	int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		if (distances[tid] > irregural_value)
		{
			int idx = atomicAdd(irregural_count, 1);
			irregural_indices[idx] = tid;
			indictor[tid] = false;
		}
		else
		{
			int idx = atomicAdd(regural_count, 1);
			regural_indices[idx] = tid;
			indictor[tid] = true;
		}
	}
}

__global__
void MapDisttoCi(float* __restrict__ distances, float* __restrict__ points_Ci, float min_dist, float max_dist, int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		if (distances[tid] >= max_dist)
		{
			points_Ci[tid] = 0.0f;//maybe
		}
		else if (distances[tid] <= min_dist)
		{
			points_Ci[tid] = 1.0f;//maybe
		}
		else
		{
			points_Ci[tid] = 1.0f - ((distances[tid] - min_dist) / (max_dist - min_dist));
		}
	}
}

__global__
void BlurCi(float* __restrict__ point_Ci,
	float* __restrict__ new_Ci,
	int* __restrict__ knn_indices,
	int knn_size, int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		float new_c_Ci = 0.0f;
		for (int j = 0; j < knn_size; j++)
		{
			int idx = knn_indices[tid * knn_size + j];
			new_c_Ci += point_Ci[idx];
		}

		new_c_Ci /= ((float)knn_size);
		new_Ci[tid] = new_c_Ci;
	}
}

__global__
void CorrectIrregularNormal(float3* __restrict__ normals,
	float* __restrict__ point_Ci,
	int* __restrict__ knn_indices,
	int* __restrict__ irregular_indices,
	bool* __restrict__ indictor,
	int count_threshold,
	int knn_size, int com_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < com_size)
	{
		int c_index = irregular_indices[tid];

		int regular_count = 0;
		float c_Ci_sum = 0.0f;
		for (int j = 0; j < knn_size; j++)
		{
			int idx = knn_indices[c_index * knn_size + j];

			c_Ci_sum += point_Ci[idx];

			if (indictor[idx])
				regular_count++;
		}

		if (regular_count >= count_threshold)
		{
			float3 c_vec = make_float3(0.0f, 0.0f, 0.0f);
			for (int j = 0; j < knn_size; j++)
			{
				int idx = knn_indices[c_index * knn_size + j];

				c_vec.x += (point_Ci[idx] / c_Ci_sum) * normals[idx].x;
				c_vec.y += (point_Ci[idx] / c_Ci_sum) * normals[idx].y;
				c_vec.z += (point_Ci[idx] / c_Ci_sum) * normals[idx].z;
			}

			normals[c_index].x = c_vec.x;
			normals[c_index].y = c_vec.y;
			normals[c_index].z = c_vec.z;
		}
	}
}

__global__
void comP2Pdistance(float3* __restrict__ points, float* __restrict__ distances , int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) return;

	float3 pi = points[tid];

	float avg_distance = 0.0f;
	for (int j = 0; j < N; ++j)
	{
		float3 pj = points[j];
		float dx = pi.x - pj.x;
		float dy = pi.y - pj.y;
		float dz = pi.z - pj.z;

		float c_distance = sqrtf(dx * dx + dy * dy + dz * dz);
		avg_distance += c_distance;
	}
	avg_distance /= float(N);

	distances[tid] = avg_distance;

}

struct sqdiff_op
{
	__host__ __device__
		float operator()(const thrust::tuple<float, float>& t) const
	{
		float d = fabs(thrust::get<1>(t) - thrust::get<0>(t)) / fabs(thrust::get<0>(t));
		return d;
	}
};

clock_t TurboDWG::TurboScreenedDWGLaunch(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC, float beta, float sigma_in, int total_iters, bool use_screened, bool use_radii, float As_threshold, int input_normal, clock_t start_o, std::string mesh_path, int test_iter)
{
	cudaMallocManaged((char**)&this->m_GWN_GPU, sizeof(*this->m_GWN_GPU));
	this->m_GWN_GPU->PreLibiGWN(O_PI, O_CH, O_CM, O_R, O_EC);

	float* winding_sum = 0;
	cudaMallocManaged((char**)&winding_sum, sizeof(*winding_sum));

	float3* mid_points = 0;
	float3* mid_normals = 0;

	int mesh_buffer_size = std::max((this->m_Grid_Res_X * this->m_Grid_Res_Y * this->m_Grid_Res_Z) / 2, 1000000);
	float3* mid_points_buffer = 0;
	float3* mid_normals_buffer = 0;
	cudaMallocManaged((char**)&mid_points_buffer, mesh_buffer_size * sizeof(*mid_points_buffer));
	cudaMallocManaged((char**)&mid_normals_buffer, mesh_buffer_size * sizeof(*mid_normals_buffer));

	int* mid_points_knn_gpu = 0;
	int* zero_points_knn_gpu = 0;

	float3* tmp_normals = 0;
	cudaMallocManaged((char**)&tmp_normals, this->m_Point_Cloud_size * sizeof(*tmp_normals));
	float* tmp_lengths = 0;
	cudaMallocManaged((char**)&tmp_lengths, this->m_Point_Cloud_size * sizeof(*tmp_lengths));
	int* zero_size = 0;
	cudaMallocManaged((char**)&zero_size, sizeof(*zero_size));

	int* c_zero_index = 0;
	cudaMallocManaged((char**)&c_zero_index, sizeof(*c_zero_index));
	int* c_nonzero_index = 0;
	cudaMallocManaged((char**)&c_nonzero_index, sizeof(*c_nonzero_index));

	float3* zero_points = 0;
	float3* nonzero_points = 0;
	int* zero_points_index = 0;
	int* nonzero_points_index = 0;
	float3* nonzero_points_normal = 0;

	float3* nonzero_points_upper = 0;
	float3* nonzero_points_lower = 0;
	cudaMallocManaged((char**)&nonzero_points_upper, sizeof(*nonzero_points_upper));
	cudaMallocManaged((char**)&nonzero_points_lower, sizeof(*nonzero_points_lower));

	float3* tmp_zero_normals = 0;

	clock_t end_o = clock();
	std::cout << "Prepare Time = " << (end_o - start_o) / CLOCKS_PER_SEC << std::endl;

	if (input_normal == 3)
	{
		TurboNormalInitialization turbo_normal_init;
		turbo_normal_init.GaussMapInit(this->m_Point_Cloud_GPU, this->m_Normals_GPU, this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU, this->m_Point_Cloud_size);

		std::cout << "Gauss Map Normal Initialized" << std::endl;
	}

	std::ofstream outFile_norms0(mesh_path + "test_Ai0.txt");
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		outFile_norms0 << this->m_As_GPU[i] << "\n";
	}
	outFile_norms0.close();

	OutlierFilter(this->m_As_GPU, As_threshold, this->m_Point_Cloud_size);

	std::ofstream outFile_norms(mesh_path + "test_Ai.txt");
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		outFile_norms << this->m_As_GPU[i] << "\n";
	}
	outFile_norms.close();

	std::ofstream outFile_points(mesh_path + "test_points.txt");
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		outFile_points << this->m_Point_Cloud_GPU[i].x << " " << this->m_Point_Cloud_GPU[i].y << " " << this->m_Point_Cloud_GPU[i].z << " " << "\n";
	}
	outFile_points.close();

	clock_t start_iter = clock();
	int c_iter = 0;
	while (c_iter < total_iters)
	{
		clock_t start, finish;
		start = clock();

		std::cout << "iter = " << c_iter << std::endl;

		clock_t start_gwn, finish_gwn;
		start_gwn = clock();

		float sigma = sigma_in;

		this->m_GWN_GPU->UpdateGWNInfo(this->m_Normals_GPU, this->m_Point_Cloud_GPU, this->m_As_GPU, this->m_Radii_GPU);

		if (use_screened)
		{
			this->m_GWN_GPU->TurboLibiScreenedGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);
			this->m_GWN_GPU->TurboLibiScreenedGWNLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Radii_GPU, beta, sigma, this->m_GP_WN_GPU, use_radii);
		}
		else
		{
			this->m_GWN_GPU->TurboLibiGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_PC_WN_GPU);
			this->m_GWN_GPU->TurboLibiGWNLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_GP_WN_GPU);

		}

		finish_gwn = clock();
		std::cout << "GWN time = " << (finish_gwn - start_gwn) / CLOCKS_PER_SEC << std::endl;

		thrust::device_ptr<float> pc_wn_thrust_ptr(this->m_PC_WN_GPU);
		float pc_wn_sum = thrust::reduce(pc_wn_thrust_ptr, pc_wn_thrust_ptr + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
		float mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);

		std::cout << "W_mean = " << mean_wn << std::endl;

		if (c_iter % 3 == 0 && c_iter != 0)
		{
			mean_wn *= 0.3;
		}

		TurboMC turbomc;
		std::string mesh_full_path = "nothing";
		int num_tri = turbomc.TurboMarchingCubes(this->m_GP_WN_GPU, this->m_Grid_Points_GPU, this->m_Grid_Res_X, this->m_Grid_Res_Y, this->m_Grid_Res_Z, mean_wn, c_iter, mesh_path, mesh_full_path, mid_points_buffer, mid_normals_buffer, test_iter);
		cudaMallocManaged((char**)&mid_points, num_tri * sizeof(*mid_points));
		cudaMallocManaged((char**)&mid_normals, num_tri * sizeof(*mid_normals));
		cudaMemcpy(mid_points, mid_points_buffer, num_tri * sizeof(*mid_points_buffer), cudaMemcpyDeviceToDevice);
		cudaMemcpy(mid_normals, mid_normals_buffer, num_tri * sizeof(*mid_normals_buffer), cudaMemcpyDeviceToDevice);

		int k_size = 15;
		cudaMallocManaged((void**)&mid_points_knn_gpu, num_tri * k_size * sizeof(*mid_points_knn_gpu));
		CudaKNNWithKDTreeforDWG(this->m_Point_Cloud_GPU,
			this->m_Point_Cloud_size,
			mid_points,
			num_tri,
			this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU,
			mid_points_knn_gpu,
			k_size
		);

		int threadsPerBlock_normalize = 128;
		int blocksPerGrid_normalize = (this->m_Point_Cloud_size + threadsPerBlock_normalize - 1) / threadsPerBlock_normalize;
		ResetZero << < blocksPerGrid_normalize, threadsPerBlock_normalize >> > (tmp_normals, this->m_Point_Cloud_size);
		cudaDeviceSynchronize();

		int threadsPerBlock = 128;
		int blocksPerGrid = (num_tri + threadsPerBlock - 1) / threadsPerBlock;
		UpdateNormals << < blocksPerGrid, threadsPerBlock >> > (tmp_normals, mid_points_knn_gpu, mid_normals, k_size, num_tri);
		cudaDeviceSynchronize();

		zero_size[0] = 0;
		NormalizeNormals << < blocksPerGrid_normalize, threadsPerBlock_normalize >> > (tmp_normals, this->m_Normals_GPU, tmp_lengths, zero_size, this->m_Point_Cloud_size);
		cudaDeviceSynchronize();

		int zero_k_size = 10;
		if (zero_size[0] > zero_k_size)
		{
			int nonzero_size = this->m_Point_Cloud_size - zero_size[0];
			cudaMallocManaged((char**)&zero_points, zero_size[0] * sizeof(*zero_points));
			cudaMallocManaged((char**)&nonzero_points, nonzero_size * sizeof(*nonzero_points));
			cudaMallocManaged((char**)&zero_points_index, zero_size[0] * sizeof(*zero_points_index));
			cudaMallocManaged((char**)&nonzero_points_index, nonzero_size * sizeof(*nonzero_points_index));
			cudaMallocManaged((char**)&nonzero_points_normal, nonzero_size * sizeof(*nonzero_points_normal));

			c_zero_index[0] = 0;
			c_nonzero_index[0] = 0;
			LoadZeroData << < blocksPerGrid_normalize, threadsPerBlock_normalize >> > (zero_points, nonzero_points, zero_points_index, nonzero_points_index, nonzero_points_normal, tmp_lengths,
				this->m_Point_Cloud_GPU, this->m_Normals_GPU, c_zero_index, c_nonzero_index, this->m_Point_Cloud_size);
			cudaDeviceSynchronize();

			GenerateKDTreeForDWG(nonzero_points, nonzero_points_normal, nonzero_size, nonzero_points_upper, nonzero_points_lower);

			cudaMallocManaged((void**)&zero_points_knn_gpu, zero_size[0] * zero_k_size * sizeof(*zero_points_knn_gpu));
			CudaKNNWithKDTreeforDWG(nonzero_points,
				nonzero_size,
				zero_points,
				zero_size[0],
				nonzero_points_upper, nonzero_points_lower,
				zero_points_knn_gpu,
				zero_k_size
			);

			std::cout << "zero_size[0] = " << zero_size[0] << std::endl;
			cudaMallocManaged((void**)&tmp_zero_normals, zero_size[0] * sizeof(*tmp_zero_normals));

			int threadsPerBlock_zero = 128;
			int blocksPerGrid_zero = (zero_size[0] + threadsPerBlock - 1) / threadsPerBlock;
			SimpleUpdateNonzeroNormals << < blocksPerGrid_zero, threadsPerBlock_zero >> > (zero_points_knn_gpu, zero_points_index, nonzero_points_normal, this->m_Normals_GPU, zero_k_size, zero_size[0]);
			cudaDeviceSynchronize();

			std::cout << "number of zero normals = " << zero_size[0] << std::endl;

			cudaFree(zero_points_knn_gpu);
			zero_points_knn_gpu = 0;
			cudaFree(zero_points);
			zero_points = 0;
			cudaFree(nonzero_points);
			nonzero_points = 0;
			cudaFree(zero_points_index);
			zero_points_index = 0;
			cudaFree(nonzero_points_index);
			nonzero_points_index = 0;
			cudaFree(nonzero_points_normal);
			nonzero_points_normal = 0;
			cudaFree(tmp_zero_normals);
			tmp_zero_normals = 0;
		}

		cudaFree(mid_points);
		mid_points = 0;
		cudaFree(mid_normals);
		mid_normals = 0;
		cudaFree(mid_points_knn_gpu);
		mid_points_knn_gpu = 0;

		c_iter++;

		finish = clock();
		std::cout << "iter time = " << (finish - start) / CLOCKS_PER_SEC << std::endl;
		std::cout << std::endl;
	}

	cudaFree(winding_sum); winding_sum = 0;
	cudaFree(tmp_normals); tmp_normals = 0;
	cudaFree(tmp_lengths); tmp_lengths = 0;
	cudaFree(zero_size); zero_size = 0;
	cudaFree(nonzero_points_upper); nonzero_points_upper = 0;
	cudaFree(nonzero_points_lower); nonzero_points_lower = 0;
	cudaFree(this->m_Grid_Points_GPU); this->m_Grid_Points_GPU = 0;
	cudaFree(this->m_GP_WN_GPU); this->m_GP_WN_GPU = 0;

	clock_t end_iter = clock();

	return end_iter - start_iter;
}

clock_t TurboDWG::TurboScreenedDWGAsOptLaunch(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC, float beta, float sigma_in, int total_iters, bool use_screened, bool use_radii, float As_threshold, int input_normal, clock_t start_o, std::string mesh_path, int test_iter, float learning_rate, int total_As_opt_iteration, Eigen::VectorXf& As_out)
{
	int loop_times = 1;
	int loop_iter = ceil((float)total_iters / (float)(loop_times + 1));

	cudaMallocManaged((char**)&this->m_GWN_GPU, sizeof(*this->m_GWN_GPU));
	this->m_GWN_GPU->PreLibiGWN(O_PI, O_CH, O_CM, O_R, O_EC);

	float* winding_sum = 0;
	cudaMallocManaged((char**)&winding_sum, sizeof(*winding_sum));

	float3* mid_points = 0;
	float3* mid_normals = 0;

	int mesh_buffer_size = std::max((this->m_Grid_Res_X * this->m_Grid_Res_Y * this->m_Grid_Res_Z) / 2, 1000000);
	float3* mid_points_buffer = 0;
	float3* mid_normals_buffer = 0;
	cudaMallocManaged((char**)&mid_points_buffer, mesh_buffer_size * sizeof(*mid_points_buffer));
	cudaMallocManaged((char**)&mid_normals_buffer, mesh_buffer_size * sizeof(*mid_normals_buffer));

	int* mid_points_knn_gpu = 0;
	int* zero_points_knn_gpu = 0;

	float3* tmp_normals = 0;
	cudaMallocManaged((char**)&tmp_normals, this->m_Point_Cloud_size * sizeof(*tmp_normals));
	float* tmp_lengths = 0;
	cudaMallocManaged((char**)&tmp_lengths, this->m_Point_Cloud_size * sizeof(*tmp_lengths));
	int* zero_size = 0;
	cudaMallocManaged((char**)&zero_size, sizeof(*zero_size));

	int* c_zero_index = 0;
	cudaMallocManaged((char**)&c_zero_index, sizeof(*c_zero_index));
	int* c_nonzero_index = 0;
	cudaMallocManaged((char**)&c_nonzero_index, sizeof(*c_nonzero_index));

	float3* zero_points = 0;
	float3* nonzero_points = 0;
	int* zero_points_index = 0;
	int* nonzero_points_index = 0;
	float3* nonzero_points_normal = 0;

	float3* nonzero_points_upper = 0;
	float3* nonzero_points_lower = 0;
	cudaMallocManaged((char**)&nonzero_points_upper, sizeof(*nonzero_points_upper));
	cudaMallocManaged((char**)&nonzero_points_lower, sizeof(*nonzero_points_lower));

	float3* tmp_zero_normals = 0;

	clock_t end_o = clock();
	std::cout << "Prepare Time = " << (end_o - start_o) / CLOCKS_PER_SEC << std::endl;

	if (input_normal == 3)
	{
		TurboNormalInitialization turbo_normal_init;
		turbo_normal_init.GaussMapInit(this->m_Point_Cloud_GPU, this->m_Normals_GPU, this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU, this->m_Point_Cloud_size);

		std::cout << "Gauss Map Normal Initialized" << std::endl;
	}

	OutlierFilter(this->m_As_GPU, As_threshold, this->m_Point_Cloud_size);

	clock_t start_iter = clock();
	int c_iter = 0;
	while (c_iter < total_iters)
	{
		clock_t start, finish;
		start = clock();

		std::cout << "iter = " << c_iter << std::endl;

		clock_t start_gwn, finish_gwn;
		start_gwn = clock();

		float sigma = sigma_in;
		if (c_iter < 4) {
			sigma = 20.0f; // 20.0

		}
		if (c_iter > 0 && c_iter % 7 == 0) {
			sigma = 10.0f; // 10.0
		}

		this->m_GWN_GPU->UpdateMassNormals(this->m_Normals_GPU, this->m_As_GPU);

		if (use_screened)
		{
			this->m_GWN_GPU->TurboLibiScreenedGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);
			this->m_GWN_GPU->TurboLibiScreenedGWNLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Radii_GPU, beta, sigma, this->m_GP_WN_GPU, use_radii);
		}
		else
		{
			this->m_GWN_GPU->TurboLibiGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_PC_WN_GPU);
			this->m_GWN_GPU->TurboLibiGWNLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_GP_WN_GPU);
		}

		finish_gwn = clock();
		std::cout << "GWN time = " << (finish_gwn - start_gwn) / CLOCKS_PER_SEC << std::endl;

		thrust::device_ptr<float> pc_wn_thrust_ptr(this->m_PC_WN_GPU);
		float pc_wn_sum = thrust::reduce(pc_wn_thrust_ptr, pc_wn_thrust_ptr + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
		float mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);

		std::cout << "W_mean = " << mean_wn << std::endl;

		if (c_iter % 5 == 0 && c_iter != 0)
		{
			mean_wn *= 0.5;
		}

		TurboMC turbomc;
		std::string mesh_full_path = "nothing";
		int num_tri = turbomc.TurboMarchingCubes(this->m_GP_WN_GPU, this->m_Grid_Points_GPU, this->m_Grid_Res_X, this->m_Grid_Res_Y, this->m_Grid_Res_Z, mean_wn, c_iter, mesh_path, mesh_full_path, mid_points_buffer, mid_normals_buffer, test_iter);
		cudaMallocManaged((char**)&mid_points, num_tri * sizeof(*mid_points));
		cudaMallocManaged((char**)&mid_normals, num_tri * sizeof(*mid_normals));
		cudaMemcpy(mid_points, mid_points_buffer, num_tri * sizeof(*mid_points_buffer), cudaMemcpyDeviceToDevice);
		cudaMemcpy(mid_normals, mid_normals_buffer, num_tri * sizeof(*mid_normals_buffer), cudaMemcpyDeviceToDevice);

		int k_size = 15;
		cudaMallocManaged((void**)&mid_points_knn_gpu, num_tri * k_size * sizeof(*mid_points_knn_gpu));
		CudaKNNWithKDTreeforDWG(this->m_Point_Cloud_GPU,
			this->m_Point_Cloud_size,
			mid_points,
			num_tri,
			this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU,
			mid_points_knn_gpu,
			k_size
		);

		int threadsPerBlock_normalize = 128;
		int blocksPerGrid_normalize = (this->m_Point_Cloud_size + threadsPerBlock_normalize - 1) / threadsPerBlock_normalize;
		ResetZero << < blocksPerGrid_normalize, threadsPerBlock_normalize >> > (tmp_normals, this->m_Point_Cloud_size);
		cudaDeviceSynchronize();

		int threadsPerBlock = 128;
		int blocksPerGrid = (num_tri + threadsPerBlock - 1) / threadsPerBlock;
		UpdateNormals << < blocksPerGrid, threadsPerBlock >> > (tmp_normals, mid_points_knn_gpu, mid_normals, k_size, num_tri);
		cudaDeviceSynchronize();

		zero_size[0] = 0;
		NormalizeNormals << < blocksPerGrid_normalize, threadsPerBlock_normalize >> > (tmp_normals, this->m_Normals_GPU, tmp_lengths, zero_size, this->m_Point_Cloud_size);
		cudaDeviceSynchronize();

		int zero_k_size = 10;
		if (zero_size[0] > zero_k_size)
		{
			int nonzero_size = this->m_Point_Cloud_size - zero_size[0];
			cudaMallocManaged((char**)&zero_points, zero_size[0] * sizeof(*zero_points));
			cudaMallocManaged((char**)&nonzero_points, nonzero_size * sizeof(*nonzero_points));
			cudaMallocManaged((char**)&zero_points_index, zero_size[0] * sizeof(*zero_points_index));
			cudaMallocManaged((char**)&nonzero_points_index, nonzero_size * sizeof(*nonzero_points_index));
			cudaMallocManaged((char**)&nonzero_points_normal, nonzero_size * sizeof(*nonzero_points_normal));

			c_zero_index[0] = 0;
			c_nonzero_index[0] = 0;
			LoadZeroData << < blocksPerGrid_normalize, threadsPerBlock_normalize >> > (zero_points, nonzero_points, zero_points_index, nonzero_points_index, nonzero_points_normal, tmp_lengths,
				this->m_Point_Cloud_GPU, this->m_Normals_GPU, c_zero_index, c_nonzero_index, this->m_Point_Cloud_size);
			cudaDeviceSynchronize();

			GenerateKDTreeForDWG(nonzero_points, nonzero_points_normal, nonzero_size, nonzero_points_upper, nonzero_points_lower);

			cudaMallocManaged((void**)&zero_points_knn_gpu, zero_size[0] * zero_k_size * sizeof(*zero_points_knn_gpu));
			CudaKNNWithKDTreeforDWG(nonzero_points,
				nonzero_size,
				zero_points,
				zero_size[0],
				nonzero_points_upper, nonzero_points_lower,
				zero_points_knn_gpu,
				zero_k_size
			);

			std::cout << "zero_size[0] = " << zero_size[0] << std::endl;
			cudaMallocManaged((void**)&tmp_zero_normals, zero_size[0] * sizeof(*tmp_zero_normals));

			int threadsPerBlock_zero = 128;
			int blocksPerGrid_zero = (zero_size[0] + threadsPerBlock - 1) / threadsPerBlock;
			SimpleUpdateNonzeroNormals << < blocksPerGrid_zero, threadsPerBlock_zero >> > (zero_points_knn_gpu, zero_points_index, nonzero_points_normal, this->m_Normals_GPU, zero_k_size, zero_size[0]);
			cudaDeviceSynchronize();

			std::cout << "number of zero normals = " << zero_size[0] << std::endl;

			cudaFree(zero_points_knn_gpu);
			zero_points_knn_gpu = 0;
			cudaFree(zero_points);
			zero_points = 0;
			cudaFree(nonzero_points);
			nonzero_points = 0;
			cudaFree(zero_points_index);
			zero_points_index = 0;
			cudaFree(nonzero_points_index);
			nonzero_points_index = 0;
			cudaFree(nonzero_points_normal);
			nonzero_points_normal = 0;
			cudaFree(tmp_zero_normals);
			tmp_zero_normals = 0;
		}

		cudaFree(mid_points);
		mid_points = 0;
		cudaFree(mid_normals);
		mid_normals = 0;
		cudaFree(mid_points_knn_gpu);
		mid_points_knn_gpu = 0;

		c_iter++;

		finish = clock();
		std::cout << "iter time = " << (finish - start) / CLOCKS_PER_SEC << std::endl;
		std::cout << std::endl;

		if ((c_iter % loop_iter) == 0 && loop_times > 0)
		{
			this->GWNAsOptimization(beta, sigma_in, use_screened, use_radii, learning_rate, total_As_opt_iteration, this->m_GWN_GPU);
			loop_times -= 1;
		}
	}

	for (int i = 0; i < this->m_Point_Cloud_size; i++)
		As_out[i] = this->m_As_GPU[i];

	cudaFree(winding_sum); winding_sum = 0;
	cudaFree(tmp_normals); tmp_normals = 0;
	cudaFree(tmp_lengths); tmp_lengths = 0;
	cudaFree(zero_size); zero_size = 0;
	cudaFree(nonzero_points_upper); nonzero_points_upper = 0;
	cudaFree(nonzero_points_lower); nonzero_points_lower = 0;
	cudaFree(this->m_Grid_Points_GPU); this->m_Grid_Points_GPU = 0;
	cudaFree(this->m_GP_WN_GPU); this->m_GP_WN_GPU = 0;

	clock_t end_iter = clock();

	return end_iter - start_iter;
}

clock_t TurboDWG::TurboScreenedDWGOptLaunch(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC, float beta, float sigma_in, int total_iters, bool use_screened, bool use_radii, float As_threshold, int input_normal, clock_t start_o, std::string mesh_path, int test_iter,
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
	Eigen::VectorXf& As_out, Eigen::VectorXf& Cs_out)
{
	float avg_ini_Ai = 0.0f;
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		avg_ini_Ai += this->m_As_GPU[i];
	}

	avg_ini_Ai /= float(this->m_Point_Cloud_size);

	if (DWG_initial_mode == 0)
	{
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
			this->m_As_GPU[i] = avg_ini_Ai;
	}

	for (int i = 0; i < this->m_Point_Cloud_size; i++)
		this->m_Cs_GPU[i] = 1.0f;

	std::cout << "avg_ini_Ai: " << avg_ini_Ai << std::endl;
	std::ofstream outFile_norms(mesh_path + "test_Ai.txt");
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		outFile_norms << this->m_As_GPU[i] << "\n";
	}
	outFile_norms.close();

	cudaMallocManaged((char**)&this->m_GWN_GPU, sizeof(*this->m_GWN_GPU));
	this->m_GWN_GPU->PreLibiGWN(O_PI, O_CH, O_CM, O_R, O_EC);

	float* winding_sum = 0;
	cudaMallocManaged((char**)&winding_sum, sizeof(*winding_sum));

	float3* mid_points = 0;
	float3* mid_normals = 0;

	int mesh_buffer_size = std::max((this->m_Grid_Res_X * this->m_Grid_Res_Y * this->m_Grid_Res_Z) / 2, 1000000);
	float3* mid_points_buffer = 0;
	float3* mid_normals_buffer = 0;
	cudaMallocManaged((char**)&mid_points_buffer, mesh_buffer_size * sizeof(*mid_points_buffer));
	cudaMallocManaged((char**)&mid_normals_buffer, mesh_buffer_size * sizeof(*mid_normals_buffer));

	int* mid_points_knn_gpu = 0;
	int* zero_points_knn_gpu = 0;

	float3* tmp_normals = 0;
	cudaMallocManaged((char**)&tmp_normals, this->m_Point_Cloud_size * sizeof(*tmp_normals));
	float* tmp_lengths = 0;
	cudaMallocManaged((char**)&tmp_lengths, this->m_Point_Cloud_size * sizeof(*tmp_lengths));
	int* zero_size = 0;
	cudaMallocManaged((char**)&zero_size, sizeof(*zero_size));

	int* c_zero_index = 0;
	cudaMallocManaged((char**)&c_zero_index, sizeof(*c_zero_index));
	int* c_nonzero_index = 0;
	cudaMallocManaged((char**)&c_nonzero_index, sizeof(*c_nonzero_index));

	float3* zero_points = 0;
	float3* nonzero_points = 0;
	int* zero_points_index = 0;
	int* nonzero_points_index = 0;
	float3* nonzero_points_normal = 0;

	float* nonzero_Ci = 0;

	float3* nonzero_points_upper = 0;
	float3* nonzero_points_lower = 0;
	cudaMallocManaged((char**)&nonzero_points_upper, sizeof(*nonzero_points_upper));
	cudaMallocManaged((char**)&nonzero_points_lower, sizeof(*nonzero_points_lower));

	float3* tmp_zero_normals = 0;

	clock_t end_o = clock();
	std::cout << "Prepare Time = " << (end_o - start_o) / CLOCKS_PER_SEC << std::endl;

	if (input_normal == 3)
	{
		TurboNormalInitialization turbo_normal_init;
		turbo_normal_init.GaussMapInit(this->m_Point_Cloud_GPU, this->m_Normals_GPU, this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU, this->m_Point_Cloud_size);

		std::cout << "Gauss Map Normal Initialized" << std::endl;
	}

	OutlierFilter(this->m_As_GPU, As_threshold, this->m_Point_Cloud_size);

	clock_t start_iter = clock();
	int c_iter = 0;

	bool ini_ai = false;
	int loop_times = 0;
	int loss_mode = 0;
	float* last_Ai = 0;
	int c_total_loop = 0;
	int c_after_ci_opt_loop = 0;
	bool ci_opt_count_reset = false;
	cudaMallocManaged((char**)&last_Ai, this->m_Point_Cloud_size * sizeof(*last_Ai));
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		last_Ai[i] = this->m_As_GPU[i];
	}

	std::ofstream ofss(mesh_path + std::to_string(-1) + "_" + std::to_string(5000) + "_test.ply");

	std::ofstream outFile_points(mesh_path + "test_points.txt");
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		outFile_points << this->m_Point_Cloud_GPU[i].x << " " << this->m_Point_Cloud_GPU[i].y << " " << this->m_Point_Cloud_GPU[i].z << "\n";
	}
	outFile_points.close();

	std::ofstream outFile_norms66(mesh_path + "test_normals.txt");
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		outFile_norms66 << this->m_Normals_GPU[i].x << " " << this->m_Normals_GPU[i].y << " " << this->m_Normals_GPU[i].z << "\n";
	}
	outFile_norms66.close();

	ofss << "ply\n";
	ofss << "format ascii 1.0\n";
	ofss << "element vertex " << this->m_Point_Cloud_size << "\n";
	ofss << "property float x\n";
	ofss << "property float y\n";
	ofss << "property float z\n";
	ofss << "property float nx\n";
	ofss << "property float ny\n";
	ofss << "property float nz\n";
	ofss << "property float quality\n";
	ofss << "end_header\n";

	float max_q = 0.0f;
	float min_q = FLT_MAX;
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		if (m_Cs_GPU[i] * m_As_GPU[i] > max_q)
			max_q = m_Cs_GPU[i] * m_As_GPU[i];
		if (m_Cs_GPU[i] * m_As_GPU[i] < min_q)
			min_q = m_Cs_GPU[i] * m_As_GPU[i];
	}

	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		float c_q = ((m_Cs_GPU[i] * m_As_GPU[i] - min_q) / (max_q - min_q)) * weight_scale + 0.001f;

		ofss << this->m_Point_Cloud_GPU[i].x << " " << this->m_Point_Cloud_GPU[i].y << " " << this->m_Point_Cloud_GPU[i].z << " "
			<< this->m_Normals_GPU[i].x << " " << this->m_Normals_GPU[i].y << " " << this->m_Normals_GPU[i].z << " "
			<< c_q << "\n";
	}
	ofss.close();

	while (c_iter < total_iters)
	{
		clock_t start, finish;
		start = clock();

		std::cout << "iter = " << c_iter << std::endl;

		clock_t start_gwn, finish_gwn;
		start_gwn = clock();

		float sigma = sigma_in;

		this->m_GWN_GPU->UpdateGWNAllInfo(this->m_Normals_GPU, this->m_Point_Cloud_GPU, this->m_As_GPU, this->m_Cs_GPU);

		if (use_screened)
		{
			this->m_GWN_GPU->TurboLibiScreenedGWNCsLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);
			this->m_GWN_GPU->TurboLibiScreenedGWNCsLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma, this->m_GP_WN_GPU, use_radii);
		}
		else
		{
			this->m_GWN_GPU->TurboLibiGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_PC_WN_GPU);
			this->m_GWN_GPU->TurboLibiGWNLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_GP_WN_GPU);
		}

		finish_gwn = clock();
		std::cout << "GWN time = " << (finish_gwn - start_gwn) / CLOCKS_PER_SEC << std::endl;

		thrust::device_ptr<float> pc_wn_thrust_ptr(this->m_PC_WN_GPU);
		float pc_wn_sum = thrust::reduce(pc_wn_thrust_ptr, pc_wn_thrust_ptr + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
		float mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);

		std::cout << "W_mean = " << mean_wn << std::endl;

		if (DWG_extract_mode == 2 || DWG_extract_mode == 3)
		{
			int wn_time = 2;

			if (loop_times % wn_time == 0)
			{
				if (((c_iter + 1) % wn_time == 0 && c_iter != 0))
				{
					mean_wn *= 0.35;//0.35
				}

				if (DWG_extract_mode == 2)
				{
					if (wn_time <= 2)
					{
						if (c_iter % 29 > 25 && c_iter % 29 < 28)
						{
							mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);
						}
					}
				}
			}
			else
			{
				if ((c_iter % wn_time == 0 && c_iter != 0))
				{
					mean_wn *= 0.35;
				}

				if (DWG_extract_mode == 2)
				{
					if (wn_time <= 2)
					{
						if (c_iter % 29 > 25 && c_iter % 29 < 28)
						{
							mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);
						}
					}
				}
			}
		}

		if (DWG_extract_mode == 1)
		{

			{
				int wn_time = 5;

				if (c_iter % wn_time == 0 && c_iter != 0 || (c_iter - 27) % 29 == 0)
				{
					mean_wn *= 0.5;//0.35
				}
			}
		}

		std::cout << "W_mean_1 = " << mean_wn << std::endl;

		TurboMC turbomc;
		std::string mesh_full_path = "nothing";

		int num_tri = turbomc.TurboMarchingCubes(this->m_GP_WN_GPU, this->m_Grid_Points_GPU, this->m_Grid_Res_X, this->m_Grid_Res_Y, this->m_Grid_Res_Z, mean_wn, c_iter, mesh_path, mesh_full_path, mid_points_buffer, mid_normals_buffer, test_iter);
		cudaMallocManaged((char**)&mid_points, num_tri * sizeof(*mid_points));
		cudaMallocManaged((char**)&mid_normals, num_tri * sizeof(*mid_normals));
		cudaMemcpy(mid_points, mid_points_buffer, num_tri * sizeof(*mid_points_buffer), cudaMemcpyDeviceToDevice);
		cudaMemcpy(mid_normals, mid_normals_buffer, num_tri * sizeof(*mid_normals_buffer), cudaMemcpyDeviceToDevice);

		int k_size = 15;
		cudaMallocManaged((void**)&mid_points_knn_gpu, num_tri * k_size * sizeof(*mid_points_knn_gpu));
		CudaKNNWithKDTreeforDWG(this->m_Point_Cloud_GPU,
			this->m_Point_Cloud_size,
			mid_points,
			num_tri,
			this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU,
			mid_points_knn_gpu,
			k_size
		);

		int threadsPerBlock_normalize = 128;
		int blocksPerGrid_normalize = (this->m_Point_Cloud_size + threadsPerBlock_normalize - 1) / threadsPerBlock_normalize;
		ResetZero << < blocksPerGrid_normalize, threadsPerBlock_normalize >> > (tmp_normals, this->m_Point_Cloud_size);
		cudaDeviceSynchronize();

		int threadsPerBlock = 128;
		int blocksPerGrid = (num_tri + threadsPerBlock - 1) / threadsPerBlock;
		UpdateNormals << < blocksPerGrid, threadsPerBlock >> > (tmp_normals, mid_points_knn_gpu, mid_normals, k_size, num_tri);
		cudaDeviceSynchronize();

		zero_size[0] = 0;
		NormalizeNormals << < blocksPerGrid_normalize, threadsPerBlock_normalize >> > (tmp_normals, this->m_Normals_GPU, tmp_lengths, zero_size, this->m_Point_Cloud_size);
		cudaDeviceSynchronize();

		int zero_k_size = 10;
		if (zero_size[0] > zero_k_size)
		{
			int nonzero_size = this->m_Point_Cloud_size - zero_size[0];
			cudaMallocManaged((char**)&zero_points, zero_size[0] * sizeof(*zero_points));
			cudaMallocManaged((char**)&nonzero_points, nonzero_size * sizeof(*nonzero_points));
			cudaMallocManaged((char**)&zero_points_index, zero_size[0] * sizeof(*zero_points_index));
			cudaMallocManaged((char**)&nonzero_points_index, nonzero_size * sizeof(*nonzero_points_index));
			cudaMallocManaged((char**)&nonzero_points_normal, nonzero_size * sizeof(*nonzero_points_normal));
			cudaMallocManaged((char**)&nonzero_Ci, nonzero_size * sizeof(*nonzero_Ci));

			c_zero_index[0] = 0;
			c_nonzero_index[0] = 0;
			LoadZeroData << < blocksPerGrid_normalize, threadsPerBlock_normalize >> > (zero_points, nonzero_points, zero_points_index, nonzero_points_index, nonzero_points_normal, tmp_lengths,
				this->m_Point_Cloud_GPU, this->m_Normals_GPU, c_zero_index, c_nonzero_index, this->m_Point_Cloud_size);
			cudaDeviceSynchronize();

			GenerateKDTreeForDWG(nonzero_points, nonzero_points_normal, nonzero_size, nonzero_points_upper, nonzero_points_lower);

			cudaMallocManaged((void**)&zero_points_knn_gpu, zero_size[0] * zero_k_size * sizeof(*zero_points_knn_gpu));
			CudaKNNWithKDTreeforDWG(nonzero_points,
				nonzero_size,
				zero_points,
				zero_size[0],
				nonzero_points_upper, nonzero_points_lower,
				zero_points_knn_gpu,
				zero_k_size
			);

			std::cout << "zero_size[0] = " << zero_size[0] << std::endl;
			cudaMallocManaged((void**)&tmp_zero_normals, zero_size[0] * sizeof(*tmp_zero_normals));

			int threadsPerBlock_zero = 128;
			int blocksPerGrid_zero = (zero_size[0] + threadsPerBlock - 1) / threadsPerBlock;
			SimpleUpdateNonzeroNormals << < blocksPerGrid_zero, threadsPerBlock_zero >> > (zero_points_knn_gpu, zero_points_index, nonzero_points_normal, this->m_Normals_GPU, zero_k_size, zero_size[0]);

			cudaDeviceSynchronize();

			std::cout << "number of zero normals = " << zero_size[0] << std::endl;

			cudaFree(zero_points_knn_gpu);
			zero_points_knn_gpu = 0;
			cudaFree(zero_points);
			zero_points = 0;
			cudaFree(nonzero_points);
			nonzero_points = 0;
			cudaFree(zero_points_index);
			zero_points_index = 0;
			cudaFree(nonzero_points_index);
			nonzero_points_index = 0;
			cudaFree(nonzero_points_normal);
			nonzero_points_normal = 0;
			cudaFree(tmp_zero_normals);
			tmp_zero_normals = 0;
			cudaFree(nonzero_Ci);
			nonzero_Ci = 0;
		}

		cudaFree(mid_points);
		mid_points = 0;
		cudaFree(mid_normals);
		mid_normals = 0;
		cudaFree(mid_points_knn_gpu);
		mid_points_knn_gpu = 0;

		c_iter++;

		finish = clock();
		std::cout << "iter time = " << (finish - start) / CLOCKS_PER_SEC << std::endl;
		std::cout << std::endl;

		if ((c_iter + 1) % 49 == 0)//((c_iter + 1) >= 3)
		{
			if (!ini_ai)
			{
				if (DWG_initial_mode == 1)
				{
					for (int i = 0; i < this->m_Point_Cloud_size; i++)
						this->m_As_GPU[i] = avg_ini_Ai;
				}

				ini_ai = true;

				for (int i = 0; i < this->m_Point_Cloud_size; i++)
				{
					last_Ai[i] = this->m_As_GPU[i];
				}
			}

			std::cout << "loss_mode: " << loss_mode << std::endl;
			this->sGWNOptimization(beta, sigma_in, use_screened, use_radii,   learning_rate_Ai, learning_rate_Ai_2, learning_rate_Ci, step_iteration, this->m_GWN_GPU, mesh_path,
				loop_times, ci_max_loop_iteration, diric_scale, diric_reso, ai_target_start, ai_target_mid,   ai_reg, ai_reg_2, ci_target, ci_reg_c, ci_reg_a, ai_rad, ci_rad,
				ai_smooth_rad, ai_adjust,

				ci_in_thres, ci_out_thres,
				ci_final_in_thres, ci_final_out_thres,
				ci_level,
				ci_var_thres,
				loss_mode,
				weight_scale,
				inside_x, inside_y, inside_z);

			if (loss_mode == 4)
				break;

			c_after_ci_opt_loop++;
			if (loss_mode == 1)
			{
				ci_opt_count_reset = true;
			}
			if (ci_opt_count_reset)
			{
				c_after_ci_opt_loop = 0;
				ci_opt_count_reset = false;
			}

			loss_mode = SetMode(last_Ai, ai_var_thres_0, ai_var_thres_1, loop_times, c_total_loop, c_after_ci_opt_loop, before_ci_max_loop, max_loop, loss_mode);

			if (loss_mode == 1)
			{
				c_total_loop++;
			}

			loop_times++;
		}
	}

	for (int i = 0; i < this->m_Point_Cloud_size; i++)
		As_out[i] = this->m_As_GPU[i];
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
		Cs_out[i] = this->m_Cs_GPU[i];

	cudaFree(winding_sum); winding_sum = 0;
	cudaFree(tmp_normals); tmp_normals = 0;
	cudaFree(tmp_lengths); tmp_lengths = 0;
	cudaFree(zero_size); zero_size = 0;
	cudaFree(nonzero_points_upper); nonzero_points_upper = 0;
	cudaFree(nonzero_points_lower); nonzero_points_lower = 0;
	cudaFree(this->m_Grid_Points_GPU); this->m_Grid_Points_GPU = 0;
	cudaFree(this->m_GP_WN_GPU); this->m_GP_WN_GPU = 0;

	clock_t end_iter = clock();

	return end_iter - start_iter;
}

clock_t TurboDWG::TurboScreenedDWGAsCsOptLaunch(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC,
	float beta, float sigma_in, int total_iters, bool use_screened, bool use_radii, float As_threshold, int input_normal, clock_t start_o, std::string mesh_path, int test_iter,
	float learning_rate_a, int total_As_opt_iteration, float learning_rate_c, int total_Cs_opt_iteration, Eigen::VectorXf& As_out)
{
	std::ofstream outFile_p("E:/dwg++/mesh/217/points.txt");
	outFile_p << this->m_Point_Cloud_size << std::endl;
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		outFile_p << this->m_Point_Cloud_GPU[i].x << " " << this->m_Point_Cloud_GPU[i].y << " " << this->m_Point_Cloud_GPU[i].z << "\n";
	}
	outFile_p.close();

	int loop_times = 10;
	int loop_iter = ceil((float)total_iters / (float)(loop_times + 1));

	int loop_times_2 = 2;
	int loop_iter_2 = ceil((float)total_iters / (float)(loop_times_2 + 1));

	cudaMallocManaged((char**)&this->m_GWN_GPU, sizeof(*this->m_GWN_GPU));
	this->m_GWN_GPU->PreLibiGWN(O_PI, O_CH, O_CM, O_R, O_EC);

	float* winding_sum = 0;
	cudaMallocManaged((char**)&winding_sum, sizeof(*winding_sum));

	float3* mid_points = 0;
	float3* mid_normals = 0;

	int mesh_buffer_size = std::max((this->m_Grid_Res_X * this->m_Grid_Res_Y * this->m_Grid_Res_Z) / 2, 1000000);
	float3* mid_points_buffer = 0;
	float3* mid_normals_buffer = 0;
	cudaMallocManaged((char**)&mid_points_buffer, mesh_buffer_size * sizeof(*mid_points_buffer));
	cudaMallocManaged((char**)&mid_normals_buffer, mesh_buffer_size * sizeof(*mid_normals_buffer));

	int* mid_points_knn_gpu = 0;
	int* zero_points_knn_gpu = 0;

	float3* tmp_normals = 0;
	cudaMallocManaged((char**)&tmp_normals, this->m_Point_Cloud_size * sizeof(*tmp_normals));
	float* tmp_lengths = 0;
	cudaMallocManaged((char**)&tmp_lengths, this->m_Point_Cloud_size * sizeof(*tmp_lengths));
	int* zero_size = 0;
	cudaMallocManaged((char**)&zero_size, sizeof(*zero_size));

	int* c_zero_index = 0;
	cudaMallocManaged((char**)&c_zero_index, sizeof(*c_zero_index));
	int* c_nonzero_index = 0;
	cudaMallocManaged((char**)&c_nonzero_index, sizeof(*c_nonzero_index));

	float3* zero_points = 0;
	float3* nonzero_points = 0;
	int* zero_points_index = 0;
	int* nonzero_points_index = 0;
	float3* nonzero_points_normal = 0;

	float3* nonzero_points_upper = 0;
	float3* nonzero_points_lower = 0;
	cudaMallocManaged((char**)&nonzero_points_upper, sizeof(*nonzero_points_upper));
	cudaMallocManaged((char**)&nonzero_points_lower, sizeof(*nonzero_points_lower));

	float3* tmp_zero_normals = 0;

	clock_t end_o = clock();
	std::cout << "Prepare Time = " << (end_o - start_o) / CLOCKS_PER_SEC << std::endl;

	if (input_normal == 3)
	{
		TurboNormalInitialization turbo_normal_init;
		turbo_normal_init.GaussMapInit(this->m_Point_Cloud_GPU, this->m_Normals_GPU, this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU, this->m_Point_Cloud_size);

		std::cout << "Gauss Map Normal Initialized" << std::endl;
	}

	OutlierFilter(this->m_As_GPU, As_threshold, this->m_Point_Cloud_size);

	clock_t start_iter = clock();
	int c_iter = 0;
	while (c_iter < total_iters)
	{
		clock_t start, finish;
		start = clock();

		std::cout << "iter = " << c_iter << std::endl;

		clock_t start_gwn, finish_gwn;
		start_gwn = clock();

		float sigma = sigma_in;
		if (c_iter < 4) {
			sigma = 20.0f; // 20.0

		}
		if (c_iter > 0 && c_iter % 7 == 0) {
			sigma = 10.0f; // 10.0
		}

		this->m_GWN_GPU->UpdateGWNAllInfo(this->m_Normals_GPU, this->m_Point_Cloud_GPU, this->m_As_GPU, this->m_Cs_GPU);

		if (use_screened)
		{
			this->m_GWN_GPU->TurboLibiScreenedGWNCsLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);
			this->m_GWN_GPU->TurboLibiScreenedGWNCsLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma, this->m_GP_WN_GPU, use_radii);
		}

		finish_gwn = clock();
		std::cout << "GWN time = " << (finish_gwn - start_gwn) / CLOCKS_PER_SEC << std::endl;

		thrust::device_ptr<float> pc_wn_thrust_ptr(this->m_PC_WN_GPU);
		float pc_wn_sum = thrust::reduce(pc_wn_thrust_ptr, pc_wn_thrust_ptr + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
		float mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);

		std::cout << "W_mean = " << mean_wn << std::endl;

		float c_mean_wn = mean_wn;
		if (c_iter % 5 == 0 && c_iter != 0)
		{
			c_mean_wn *= 0.5;
		}

		TurboMC turbomc;
		std::string mesh_full_path = "nothing";
		int num_tri = turbomc.TurboMarchingCubes(this->m_GP_WN_GPU, this->m_Grid_Points_GPU, this->m_Grid_Res_X, this->m_Grid_Res_Y, this->m_Grid_Res_Z, c_mean_wn, c_iter, mesh_path, mesh_full_path, mid_points_buffer, mid_normals_buffer, test_iter);
		cudaMallocManaged((char**)&mid_points, num_tri * sizeof(*mid_points));
		cudaMallocManaged((char**)&mid_normals, num_tri * sizeof(*mid_normals));
		cudaMemcpy(mid_points, mid_points_buffer, num_tri * sizeof(*mid_points_buffer), cudaMemcpyDeviceToDevice);
		cudaMemcpy(mid_normals, mid_normals_buffer, num_tri * sizeof(*mid_normals_buffer), cudaMemcpyDeviceToDevice);

		int k_size = 15;
		cudaMallocManaged((void**)&mid_points_knn_gpu, num_tri * k_size * sizeof(*mid_points_knn_gpu));
		CudaKNNWithKDTreeforDWG(this->m_Point_Cloud_GPU,
			this->m_Point_Cloud_size,
			mid_points,
			num_tri,
			this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU,
			mid_points_knn_gpu,
			k_size
		);

		int threadsPerBlock_normalize = 128;
		int blocksPerGrid_normalize = (this->m_Point_Cloud_size + threadsPerBlock_normalize - 1) / threadsPerBlock_normalize;
		ResetZero << < blocksPerGrid_normalize, threadsPerBlock_normalize >> > (tmp_normals, this->m_Point_Cloud_size);
		cudaDeviceSynchronize();

		int threadsPerBlock = 128;
		int blocksPerGrid = (num_tri + threadsPerBlock - 1) / threadsPerBlock;
		UpdateNormals << < blocksPerGrid, threadsPerBlock >> > (tmp_normals, mid_points_knn_gpu, mid_normals, k_size, num_tri);
		cudaDeviceSynchronize();

		zero_size[0] = 0;
		NormalizeNormals << < blocksPerGrid_normalize, threadsPerBlock_normalize >> > (tmp_normals, this->m_Normals_GPU, tmp_lengths, zero_size, this->m_Point_Cloud_size);
		cudaDeviceSynchronize();

		int zero_k_size = 10;
		if (zero_size[0] > zero_k_size)
		{
			int nonzero_size = this->m_Point_Cloud_size - zero_size[0];
			cudaMallocManaged((char**)&zero_points, zero_size[0] * sizeof(*zero_points));
			cudaMallocManaged((char**)&nonzero_points, nonzero_size * sizeof(*nonzero_points));
			cudaMallocManaged((char**)&zero_points_index, zero_size[0] * sizeof(*zero_points_index));
			cudaMallocManaged((char**)&nonzero_points_index, nonzero_size * sizeof(*nonzero_points_index));
			cudaMallocManaged((char**)&nonzero_points_normal, nonzero_size * sizeof(*nonzero_points_normal));

			c_zero_index[0] = 0;
			c_nonzero_index[0] = 0;
			LoadZeroData << < blocksPerGrid_normalize, threadsPerBlock_normalize >> > (zero_points, nonzero_points, zero_points_index, nonzero_points_index, nonzero_points_normal, tmp_lengths,
				this->m_Point_Cloud_GPU, this->m_Normals_GPU, c_zero_index, c_nonzero_index, this->m_Point_Cloud_size);
			cudaDeviceSynchronize();

			GenerateKDTreeForDWG(nonzero_points, nonzero_points_normal, nonzero_size, nonzero_points_upper, nonzero_points_lower);

			cudaMallocManaged((void**)&zero_points_knn_gpu, zero_size[0] * zero_k_size * sizeof(*zero_points_knn_gpu));
			CudaKNNWithKDTreeforDWG(nonzero_points,
				nonzero_size,
				zero_points,
				zero_size[0],
				nonzero_points_upper, nonzero_points_lower,
				zero_points_knn_gpu,
				zero_k_size
			);

			std::cout << "zero_size[0] = " << zero_size[0] << std::endl;
			cudaMallocManaged((void**)&tmp_zero_normals, zero_size[0] * sizeof(*tmp_zero_normals));

			int threadsPerBlock_zero = 128;
			int blocksPerGrid_zero = (zero_size[0] + threadsPerBlock - 1) / threadsPerBlock;
			SimpleUpdateNonzeroNormals << < blocksPerGrid_zero, threadsPerBlock_zero >> > (zero_points_knn_gpu, zero_points_index, nonzero_points_normal, this->m_Normals_GPU, zero_k_size, zero_size[0]);
			cudaDeviceSynchronize();

			std::cout << "number of zero normals = " << zero_size[0] << std::endl;

			cudaFree(zero_points_knn_gpu);
			zero_points_knn_gpu = 0;
			cudaFree(zero_points);
			zero_points = 0;
			cudaFree(nonzero_points);
			nonzero_points = 0;
			cudaFree(zero_points_index);
			zero_points_index = 0;
			cudaFree(nonzero_points_index);
			nonzero_points_index = 0;
			cudaFree(nonzero_points_normal);
			nonzero_points_normal = 0;
			cudaFree(tmp_zero_normals);
			tmp_zero_normals = 0;
		}

		cudaFree(mid_points);
		mid_points = 0;
		cudaFree(mid_normals);
		mid_normals = 0;
		cudaFree(mid_points_knn_gpu);
		mid_points_knn_gpu = 0;

		c_iter++;

		finish = clock();
		std::cout << "iter time = " << (finish - start) / CLOCKS_PER_SEC << std::endl;
		std::cout << std::endl;

		if (c_iter >= 30 && (c_iter - 1) % 5 == 0)
		{

			CiComputation(mean_wn);
			loop_times -= 1;
		}

	}

	for (int i = 0; i < this->m_Point_Cloud_size; i++)
		As_out[i] = this->m_As_GPU[i];

	cudaFree(winding_sum); winding_sum = 0;
	cudaFree(tmp_normals); tmp_normals = 0;
	cudaFree(tmp_lengths); tmp_lengths = 0;
	cudaFree(zero_size); zero_size = 0;
	cudaFree(nonzero_points_upper); nonzero_points_upper = 0;
	cudaFree(nonzero_points_lower); nonzero_points_lower = 0;
	cudaFree(this->m_Grid_Points_GPU); this->m_Grid_Points_GPU = 0;
	cudaFree(this->m_GP_WN_GPU); this->m_GP_WN_GPU = 0;

	clock_t end_iter = clock();

	return end_iter - start_iter;
}

void TurboDWG::UpdateRadii(int k_size)
{
	int* knn_points = 0;
	cudaMallocManaged((void**)&knn_points, this->m_Point_Cloud_size * k_size * sizeof(*knn_points));
	CudaKNNWithKDTreeforDWG(this->m_Point_Cloud_GPU,
		this->m_Point_Cloud_size,
		this->m_Point_Cloud_GPU,
		this->m_Point_Cloud_size,
		this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU,
		knn_points,
		k_size
	);

	int threadsPerBlock = 128;
	int blocksPerGrid = (this->m_Point_Cloud_size + threadsPerBlock - 1) / threadsPerBlock;
	SimpleUpdateRadii << < blocksPerGrid, threadsPerBlock >> > (knn_points, this->m_Point_Cloud_GPU, this->m_Radii_GPU, k_size, this->m_Point_Cloud_size);
	cudaDeviceSynchronize();
}

void TurboDWG::UpdateAs(Eigen::VectorXd& leaves_As)
{
	std::unique_ptr<float[]> h_As = std::make_unique<float[]>(size_t(this->m_Point_Cloud_size));
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		h_As[i] = (float)leaves_As[i];
	}
	gpuErrchk(cudaMemcpy(this->m_As_GPU, h_As.get(), this->m_Point_Cloud_size * sizeof(float), cudaMemcpyHostToDevice));
}

struct squared_diff
{
	float mean;
	squared_diff(float _mean) : mean(_mean) {}
	__host__ __device__ float operator()(const float& x) const
	{
		return (x - mean) * (x - mean);
	}
};

void TurboDWG::CiComputation(float mean_wn)
{
	float3* mid_points = 0;
	float3* mid_normals = 0;

	int mesh_buffer_size = std::max((this->m_Grid_Res_X * this->m_Grid_Res_Y * this->m_Grid_Res_Z) / 2, 1000000);
	float3* mid_points_buffer = 0;
	float3* mid_normals_buffer = 0;
	cudaMallocManaged((char**)&mid_points_buffer, mesh_buffer_size * sizeof(*mid_points_buffer));
	cudaMallocManaged((char**)&mid_normals_buffer, mesh_buffer_size * sizeof(*mid_normals_buffer));

	TurboMC turbomc;
	std::string mesh_full_path = "nothing";
	int num_tri = turbomc.TurboMarchingCubes(this->m_GP_WN_GPU, this->m_Grid_Points_GPU, this->m_Grid_Res_X, this->m_Grid_Res_Y, this->m_Grid_Res_Z, mean_wn, 0, mesh_full_path, mesh_full_path, mid_points_buffer, mid_normals_buffer, 999999);
	cudaMallocManaged((char**)&mid_points, num_tri * sizeof(*mid_points));
	cudaMallocManaged((char**)&mid_normals, num_tri * sizeof(*mid_normals));
	cudaMemcpy(mid_points, mid_points_buffer, num_tri * sizeof(*mid_points_buffer), cudaMemcpyDeviceToDevice);
	cudaMemcpy(mid_normals, mid_normals_buffer, num_tri * sizeof(*mid_normals_buffer), cudaMemcpyDeviceToDevice);

	float3* mid_points_upper = 0;
	float3* mid_points_lower = 0;
	cudaMallocManaged((char**)&mid_points_upper, sizeof(*mid_points_upper));
	cudaMallocManaged((char**)&mid_points_lower, sizeof(*mid_points_lower));

	GenerateKDTreeForDWG(mid_points, mid_normals, num_tri, mid_points_upper, mid_points_lower);

	int k_size = 15;
	int* mid_points_knn_gpu = 0;
	cudaMallocManaged((void**)&mid_points_knn_gpu, this->m_Point_Cloud_size * k_size * sizeof(*mid_points_knn_gpu));
	CudaKNNWithKDTreeforDWG(mid_points,
		num_tri,
		this->m_Point_Cloud_GPU,
		this->m_Point_Cloud_size,
		mid_points_upper, mid_points_lower,
		mid_points_knn_gpu,
		k_size
	);

	float* distances_gpu = 0;
	cudaMallocManaged((void**)&distances_gpu, this->m_Point_Cloud_size * sizeof(*distances_gpu));

	int threadsPerBlock = 128;
	int blocksPerGrid = (this->m_Point_Cloud_size + threadsPerBlock - 1) / threadsPerBlock;
	ComPtoMDist << < blocksPerGrid, threadsPerBlock >> > (this->m_Point_Cloud_GPU, mid_points, mid_points_knn_gpu, distances_gpu, k_size, this->m_Point_Cloud_size);
	cudaDeviceSynchronize();

	thrust::device_vector<float> d_distances(distances_gpu, distances_gpu + this->m_Point_Cloud_size);

	float distance_sum = thrust::reduce(d_distances.begin(), d_distances.end(), 0.0f, thrust::plus<float>());
	float mean_distance = distance_sum / ((float)this->m_Point_Cloud_size);

	thrust::device_vector<float> d_squared_diffs(this->m_Point_Cloud_size);
	thrust::transform(d_distances.begin(), d_distances.end(), d_squared_diffs.begin(), squared_diff(mean_distance));

	float squared_sum_distance = thrust::reduce(d_squared_diffs.begin(), d_squared_diffs.end(), 0.0f, thrust::plus<float>());
	float std_dev_distance = sqrtf(squared_sum_distance / ((float)this->m_Point_Cloud_size));

	float irregular_t = 2.0f;//maybe important coff
	float irregular_value = squared_sum_distance + irregular_t * std_dev_distance;

	int* regular_indices_buffer = 0;
	cudaMallocManaged((void**)&regular_indices_buffer, this->m_Point_Cloud_size * sizeof(*regular_indices_buffer));
	int* irregular_indices_buffer = 0;
	cudaMallocManaged((void**)&irregular_indices_buffer, this->m_Point_Cloud_size * sizeof(*irregular_indices_buffer));
	int* regular_count_gpu = 0;
	cudaMallocManaged((void**)&regular_count_gpu, sizeof(*regular_count_gpu));
	regular_count_gpu[0] = 0;
	int* irregular_count_gpu = 0;
	cudaMallocManaged((void**)&irregular_count_gpu, sizeof(*irregular_count_gpu));
	irregular_count_gpu[0] = 0;

	bool* indictor = 0;
	cudaMallocManaged((void**)&indictor, this->m_Point_Cloud_size * sizeof(*indictor));

	BaseDistClassifyPoint << < blocksPerGrid, threadsPerBlock >> > (irregular_value, distances_gpu,
		regular_indices_buffer,
		irregular_indices_buffer,
		regular_count_gpu,
		irregular_count_gpu,
		indictor,
		this->m_Point_Cloud_size);
	cudaDeviceSynchronize();

	int regural_count, irregural_count;
	cudaMemcpy(&regural_count, regular_count_gpu, sizeof(int), cudaMemcpyDeviceToHost);
	cudaMemcpy(&irregural_count, irregular_count_gpu, sizeof(int), cudaMemcpyDeviceToHost);

	int* regular_indices = 0;
	cudaMallocManaged((void**)&regular_indices, regural_count * sizeof(*regular_indices));
	int* irregular_indices = 0;
	cudaMallocManaged((void**)&irregular_indices, irregural_count * sizeof(*irregular_indices));
	cudaMemcpy(regular_indices, regular_indices_buffer, regural_count * sizeof(*regular_indices_buffer), cudaMemcpyDeviceToDevice);
	cudaMemcpy(irregular_indices, irregular_indices_buffer, irregural_count * sizeof(*irregular_indices_buffer), cudaMemcpyDeviceToDevice);

	thrust::device_vector<float> d_sorted_distances = d_distances;
	thrust::sort(d_sorted_distances.begin(), d_sorted_distances.end());

	float threshold_sort = 0.1;//maybe
	float inv_threshold_sort = 1.0f - threshold_sort;
	int idx_thres = static_cast<int>(threshold_sort * this->m_Point_Cloud_size);
	int idx_inv_thres = static_cast<int>(inv_threshold_sort * this->m_Point_Cloud_size);

	float val_thres = d_sorted_distances[idx_thres];
	float val_inv_thres = d_sorted_distances[idx_inv_thres];

	MapDisttoCi << < blocksPerGrid, threadsPerBlock >> > (distances_gpu, this->m_Cs_GPU, val_thres, val_inv_thres, this->m_Point_Cloud_size);
	cudaDeviceSynchronize();

	int blur_k_size = 30;//maybe
	int* blur_Ci_knn_gpu = 0;
	cudaMallocManaged((void**)&blur_Ci_knn_gpu, this->m_Point_Cloud_size * blur_k_size * sizeof(*blur_Ci_knn_gpu));
	CudaKNNWithKDTreeforDWG(this->m_Point_Cloud_GPU,
		this->m_Point_Cloud_size,
		this->m_Point_Cloud_GPU,
		this->m_Point_Cloud_size,
		this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU,
		blur_Ci_knn_gpu,
		blur_k_size
	);

	float* blur_Ci = 0;
	cudaMallocManaged((void**)&blur_Ci, this->m_Point_Cloud_size * sizeof(*blur_Ci));
	BlurCi << < blocksPerGrid, threadsPerBlock >> > (this->m_Cs_GPU, blur_Ci, blur_Ci_knn_gpu, blur_k_size, this->m_Point_Cloud_size);
	cudaDeviceSynchronize();

	cudaMemcpy(this->m_Cs_GPU, blur_Ci, this->m_Point_Cloud_size * sizeof(float), cudaMemcpyDeviceToDevice);

	float count_t = 0.333f;//maybe
	int count_threshold = static_cast<int>(blur_k_size * count_t);
	CorrectIrregularNormal << < blocksPerGrid, threadsPerBlock >> > (m_Normals_GPU, this->m_Cs_GPU, blur_Ci_knn_gpu, irregular_indices, indictor, count_threshold, blur_k_size, irregural_count);
	cudaDeviceSynchronize();

	cudaFree(mid_points); mid_points = 0;
	cudaFree(mid_normals); mid_normals = 0;
	cudaFree(mid_points_buffer); mid_points_buffer = 0;
	cudaFree(mid_normals_buffer); mid_normals_buffer = 0;
	cudaFree(mid_points_upper); mid_points_upper = 0;
	cudaFree(mid_points_lower); mid_points_lower = 0;
	cudaFree(mid_points_knn_gpu); mid_points_knn_gpu = 0;
	cudaFree(distances_gpu); distances_gpu = 0;
	cudaFree(regular_indices_buffer);  regular_indices_buffer = 0;
	cudaFree(irregular_indices_buffer); irregular_indices_buffer = 0;
	cudaFree(regular_count_gpu); regular_count_gpu = 0;
	cudaFree(irregular_count_gpu); irregular_count_gpu = 0;
	cudaFree(indictor); indictor = 0;
	cudaFree(regular_indices); regular_indices = 0;
	cudaFree(irregular_indices); irregular_indices = 0;
	cudaFree(blur_Ci_knn_gpu); blur_Ci_knn_gpu = 0;
	cudaFree(blur_Ci); blur_Ci = 0;

	std::ofstream outFile_Cs("E:/dwg++/mesh/217/Cs.txt", std::ios::trunc);
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		outFile_Cs << this->m_Cs_GPU[i] << "\n";
	}
	outFile_Cs.close();

	std::cout << "Cs Opt finished!" << std::endl;
}

void TurboDWG::PreGeneration(Eigen::MatrixXf& cloud_points, Eigen::VectorXf& points_As, Eigen::VectorXf& points_Cs, Eigen::MatrixXf& grid_points, int res_x, int res_y, int res_z, std::vector<float> bound_upper_kdtree, std::vector<float> bound_lower_kdtree, int k_normal_size, bool As_opt, Eigen::MatrixXd& out_normal)
{
	cudaFree(this->m_PC_WN_GPU); this->m_PC_WN_GPU = 0;
	cudaFree(this->m_As_GPU); this->m_As_GPU = 0;
	cudaFree(this->m_Cs_GPU); this->m_Cs_GPU = 0;
	cudaFree(this->m_Radii_GPU); this->m_Radii_GPU = 0;

	this->m_GWN_GPU->EarseMemory();

	PreTurboDWG(cloud_points.rows(), points_As, points_Cs, grid_points, res_x, res_y, res_z);

	float3* cloud_points_gpu = 0;
	int cloud_points_size = cloud_points.rows();
	cudaMallocManaged((char**)&cloud_points_gpu, cloud_points_size * sizeof(*cloud_points_gpu));
	std::unique_ptr<float3[]> cloud_points_cpu = std::make_unique<float3[]>(size_t(cloud_points_size));
	for (int i = 0; i < cloud_points_size; i++)
	{
		cloud_points_cpu[i] = make_float3(cloud_points(i, 0), cloud_points(i, 1), cloud_points(i, 2));
	}
	gpuErrchk(cudaMemcpy(cloud_points_gpu, cloud_points_cpu.get(), cloud_points_size * sizeof(float3), cudaMemcpyHostToDevice));

	int* knn_normal_points = 0;

	cudaMallocManaged((void**)&knn_normal_points, cloud_points_size * k_normal_size * sizeof(*knn_normal_points));
	CudaKNNWithKDTreeforDWG(this->m_Point_Cloud_GPU,
		this->m_Point_Cloud_size,
		cloud_points_gpu,
		cloud_points_size,
		this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU,
		knn_normal_points,
		k_normal_size
	);

	int* knn_ci_points = 0;
	int k_size = 1;
	cudaMallocManaged((void**)&knn_ci_points, cloud_points_size * k_size * sizeof(*knn_ci_points));
	CudaKNNWithKDTreeforDWG(this->m_Point_Cloud_GPU,
		this->m_Point_Cloud_size,
		cloud_points_gpu,
		cloud_points_size,
		this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU,
		knn_ci_points,
		k_size
	);

	cudaFree(this->m_Point_Cloud_GPU); this->m_Point_Cloud_GPU = 0;

	float3* points_normal_gpu = 0;
	cudaMallocManaged((char**)&points_normal_gpu, cloud_points_size * sizeof(*points_normal_gpu));

	int threadsPerBlock = 128;
	int blocksPerGrid = (cloud_points_size + threadsPerBlock - 1) / threadsPerBlock;
	LoadNormals << < blocksPerGrid, threadsPerBlock >> > (this->m_Normals_GPU, knn_normal_points, points_normal_gpu, k_normal_size, cloud_points_size);
	cudaDeviceSynchronize();

	if (As_opt)
	{
		float* points_As_gpu = 0;
		cudaMallocManaged((char**)&points_As_gpu, cloud_points_size * sizeof(*points_As_gpu));
		LoadWeight << < blocksPerGrid, threadsPerBlock >> > (this->m_As_GPU, knn_normal_points, points_As_gpu, k_normal_size, cloud_points_size);
		cudaDeviceSynchronize();

		cudaFree(this->m_As_GPU); this->m_As_GPU = 0;
		this->m_As_GPU = points_As_gpu;

		float* points_Cs_gpu = 0;
		cudaMallocManaged((char**)&points_Cs_gpu, cloud_points_size * sizeof(*points_Cs_gpu));
		LoadWeight << < blocksPerGrid, threadsPerBlock >> > (this->m_Cs_GPU, knn_ci_points, points_Cs_gpu, k_size, cloud_points_size);
		cudaDeviceSynchronize();

		cudaFree(this->m_Cs_GPU); this->m_Cs_GPU = 0;
		this->m_Cs_GPU = points_Cs_gpu;
	}

	cudaFree(this->m_Normals_GPU); this->m_Normals_GPU = 0;

	this->m_Bound_Upper_KDT_GPU[0].x = bound_upper_kdtree[0];
	this->m_Bound_Upper_KDT_GPU[0].y = bound_upper_kdtree[1];
	this->m_Bound_Upper_KDT_GPU[0].z = bound_upper_kdtree[2];

	this->m_Bound_Lower_KDT_GPU[0].x = bound_lower_kdtree[0];
	this->m_Bound_Lower_KDT_GPU[0].y = bound_lower_kdtree[1];
	this->m_Bound_Lower_KDT_GPU[0].z = bound_lower_kdtree[2];

	this->m_Point_Cloud_GPU = cloud_points_gpu;
	this->m_Point_Cloud_size = cloud_points_size;
	this->m_Normals_GPU = points_normal_gpu;

	std::unique_ptr<float3[]> h_normal = std::make_unique<float3[]>(size_t(this->m_Point_Cloud_size));
	cudaMemcpy(h_normal.get(), this->m_Normals_GPU, this->m_Point_Cloud_size * sizeof(float3), cudaMemcpyDeviceToHost);
	for (int i = 0; i < this->m_Point_Cloud_size; ++i)
	{
		out_normal(i, 0) = (double)h_normal[i].x;
		out_normal(i, 1) = (double)h_normal[i].y;
		out_normal(i, 2) = (double)h_normal[i].z;
	}
}

void TurboDWG::GenerateMesh(Eigen::MatrixXf& grid_points, int res_x, int res_y, int res_z, float beta, float sigma, int k_size, bool use_radii, std::string mesh_full_path, clock_t iter_time, clock_t start)
{
	clock_t start_time = clock();
	float3* grid_points_gpu = 0;
	int grid_points_size = grid_points.rows();
	cudaMallocManaged((char**)&grid_points_gpu, grid_points_size * sizeof(*grid_points_gpu));
	std::unique_ptr<float3[]> grid_points_cpu = std::make_unique<float3[]>(size_t(grid_points_size));
	for (int i = 0; i < grid_points_size; i++)
	{
		grid_points_cpu[i] = make_float3(grid_points(i, 0), grid_points(i, 1), grid_points(i, 2));
	}
	gpuErrchk(cudaMemcpy(grid_points_gpu, grid_points_cpu.get(), grid_points_size * sizeof(float3), cudaMemcpyHostToDevice));

	float* grid_point_gwn = 0;
	cudaMallocManaged((char**)&grid_point_gwn, grid_points_size * sizeof(*grid_point_gwn));

	UpdateRadii(k_size);
	this->m_GWN_GPU->UpdateMassNormals(this->m_Normals_GPU, this->m_As_GPU);
	this->m_GWN_GPU->TurboLibiScreenedGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);
	this->m_GWN_GPU->TurboLibiScreenedGWNLaunch(grid_points_gpu, this->m_Point_Cloud_GPU, grid_points_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Radii_GPU, beta, sigma, grid_point_gwn, use_radii);

	thrust::device_ptr<float> pc_wn_thrust_ptr(this->m_PC_WN_GPU);
	float pc_wn_sum = thrust::reduce(pc_wn_thrust_ptr, pc_wn_thrust_ptr + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
	float mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);

	int mesh_buffer_size = std::max((res_x * res_y * res_z) / 2, 1000000);
	float3* mid_points_buffer = 0;
	float3* mid_normals_buffer = 0;
	cudaMallocManaged((char**)&mid_points_buffer, mesh_buffer_size * sizeof(*mid_points_buffer));
	cudaMallocManaged((char**)&mid_normals_buffer, mesh_buffer_size * sizeof(*mid_normals_buffer));

	TurboMC turbomc;
	std::string mesh_path = "nothing";
	int num_tri = turbomc.TurboMarchingCubes(grid_point_gwn, grid_points_gpu, res_x, res_y, res_z, mean_wn, 123456, mesh_path, mesh_full_path, mid_points_buffer, mid_normals_buffer, 999999);

	clock_t end_time = clock();
	std::cout << "Iter Time = " << (end_time - start_time + iter_time) / CLOCKS_PER_SEC << std::endl;
	std::cout << "Total Time = " << (end_time - start) / CLOCKS_PER_SEC << std::endl;
}

void TurboDWG::GenerateMesh(Eigen::MatrixXf& cloud_points, Eigen::VectorXf& points_As, Eigen::VectorXf& leaves_Cs, Eigen::MatrixXf& grid_points,
	std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC,
	int res_x, int res_y, int res_z,
	std::vector<float> bound_upper_kdtree, std::vector<float> bound_lower_kdtree,
	float beta, float sigma, int k_size, int k_normal_size, bool use_screened, bool use_radii, float As_threshold, std::string mesh_full_path, clock_t iter_time, clock_t start, int output_mode, int sample_reso)
{

	clock_t start_time = clock();

	cudaFree(this->m_PC_WN_GPU); this->m_PC_WN_GPU = 0;
	cudaFree(this->m_As_GPU); this->m_As_GPU = 0;
	cudaFree(this->m_Radii_GPU); this->m_Radii_GPU = 0;

	this->m_GWN_GPU->EarseMemory();

	PreTurboDWG(cloud_points.rows(), points_As, leaves_Cs, grid_points, res_x, res_y, res_z);

	float3* cloud_points_gpu = 0;
	int cloud_points_size = cloud_points.rows();
	cudaMallocManaged((char**)&cloud_points_gpu, cloud_points_size * sizeof(*cloud_points_gpu));
	std::unique_ptr<float3[]> cloud_points_cpu = std::make_unique<float3[]>(size_t(cloud_points_size));
	for (int i = 0; i < cloud_points_size; i++)
	{
		cloud_points_cpu[i] = make_float3(cloud_points(i, 0), cloud_points(i, 1), cloud_points(i, 2));
	}
	gpuErrchk(cudaMemcpy(cloud_points_gpu, cloud_points_cpu.get(), cloud_points_size * sizeof(float3), cudaMemcpyHostToDevice));

	int* knn_normal_points = 0;

	cudaMallocManaged((void**)&knn_normal_points, cloud_points_size * k_normal_size * sizeof(*knn_normal_points));
	CudaKNNWithKDTreeforDWG(this->m_Point_Cloud_GPU,
		this->m_Point_Cloud_size,
		cloud_points_gpu,
		cloud_points_size,
		this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU,
		knn_normal_points,
		k_normal_size
	);

	cudaFree(this->m_Point_Cloud_GPU); this->m_Point_Cloud_GPU = 0;
	float3* points_normal_gpu = 0;
	cudaMallocManaged((char**)&points_normal_gpu, cloud_points_size * sizeof(*points_normal_gpu));

	int threadsPerBlock = 128;
	int blocksPerGrid = (cloud_points_size + threadsPerBlock - 1) / threadsPerBlock;
	LoadNormals << < blocksPerGrid, threadsPerBlock >> > (this->m_Normals_GPU, knn_normal_points, points_normal_gpu, k_normal_size, cloud_points_size);
	cudaDeviceSynchronize();

	cudaFree(this->m_Normals_GPU); this->m_Normals_GPU = 0;

	this->m_Bound_Upper_KDT_GPU[0].x = bound_upper_kdtree[0];
	this->m_Bound_Upper_KDT_GPU[0].y = bound_upper_kdtree[1];
	this->m_Bound_Upper_KDT_GPU[0].z = bound_upper_kdtree[2];

	this->m_Bound_Lower_KDT_GPU[0].x = bound_lower_kdtree[0];
	this->m_Bound_Lower_KDT_GPU[0].y = bound_lower_kdtree[1];
	this->m_Bound_Lower_KDT_GPU[0].z = bound_lower_kdtree[2];

	this->m_Point_Cloud_GPU = cloud_points_gpu;
	this->m_Point_Cloud_size = cloud_points_size;
	this->m_Normals_GPU = points_normal_gpu;

	TurboLibiGWN turbogwn;
	turbogwn.PreLibiGWN(O_PI, O_CH, O_CM, O_R, O_EC);

	UpdateRadii(k_size);

	turbogwn.UpdateMassNormals(this->m_Normals_GPU, this->m_As_GPU);

	OutlierFilter(this->m_As_GPU, As_threshold, this->m_Point_Cloud_size);

	std::cout << "Start GWN Computation" << std::endl;
	if (use_screened)
	{
		turbogwn.TurboLibiScreenedGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);
		turbogwn.TurboLibiScreenedGWNLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Radii_GPU, beta, sigma, this->m_GP_WN_GPU, use_radii);

	}
	else
	{
		turbogwn.TurboLibiGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_PC_WN_GPU);
		turbogwn.TurboLibiGWNLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_GP_WN_GPU);

	}

	thrust::device_ptr<float> pc_wn_thrust_ptr(this->m_PC_WN_GPU);
	float pc_wn_sum = thrust::reduce(pc_wn_thrust_ptr, pc_wn_thrust_ptr + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
	float mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);
	std::cout << "mean_wn: " << mean_wn << std::endl;

	int mesh_buffer_size = std::max((res_x * res_y * res_z) / 2, 1000000);
	float3* mid_points_buffer = 0;
	float3* mid_normals_buffer = 0;
	cudaMallocManaged((char**)&mid_points_buffer, mesh_buffer_size * sizeof(*mid_points_buffer));
	cudaMallocManaged((char**)&mid_normals_buffer, mesh_buffer_size * sizeof(*mid_normals_buffer));

	if (output_mode == 1 || output_mode == 3 || output_mode == 4)
	{
		std::string mesh_path = "nothing";
		TurboMC turbomc;
		int num_tri = turbomc.TurboMarchingCubes(this->m_GP_WN_GPU, this->m_Grid_Points_GPU, res_x, res_y, res_z, mean_wn, 123456, mesh_path, mesh_full_path, mid_points_buffer, mid_normals_buffer, 999999);

		clock_t end_time = clock();
		std::cout << "Iter Time = " << iter_time / CLOCKS_PER_SEC << std::endl;
		std::cout << "Total Time = " << (end_time - start) / CLOCKS_PER_SEC << std::endl;
	}

	if (output_mode == 2 || output_mode == 3)
	{
		if (output_mode == 2)
		{
			clock_t end_time = clock();
			std::cout << "Iter Time = " << iter_time / CLOCKS_PER_SEC << std::endl;
			std::cout << "Total Time = " << (end_time - start) / CLOCKS_PER_SEC << std::endl;
		}

		std::cout << "Start save" << std::endl;

		std::ofstream outFile(mesh_full_path + ".xyz");
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			outFile << this->m_Point_Cloud_GPU[i].x << " " << this->m_Point_Cloud_GPU[i].y << " " << this->m_Point_Cloud_GPU[i].z << " " <<
				this->m_Normals_GPU[i].x << " " << this->m_Normals_GPU[i].y << " " << this->m_Normals_GPU[i].z << "\n";
		}

		outFile.close();
		std::cout << "Finish save" << std::endl;
	}

}

void TurboDWG::MeshGeneration(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC, int res_x, int res_y, int res_z, float beta, float sigma, int k_size, bool use_screened, bool use_radii, float As_threshold, std::string mesh_full_path, clock_t start, std::chrono::high_resolution_clock::time_point start_time_s, int output_mode, int sample_reso, float weight_scale)
{

	TurboLibiGWN turbogwn;
	turbogwn.PreLibiGWN(O_PI, O_CH, O_CM, O_R, O_EC);

	turbogwn.UpdateGWNAllInfo(this->m_Normals_GPU, this->m_Point_Cloud_GPU, this->m_As_GPU, this->m_Cs_GPU);

	UpdateRadii(k_size);

	OutlierFilter(this->m_As_GPU, As_threshold, this->m_Point_Cloud_size);

	std::cout << "Start GWN Computation" << std::endl;
	if (use_screened)
	{
		turbogwn.TurboLibiScreenedGWNCsLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);
		turbogwn.TurboLibiScreenedGWNCsLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma, this->m_GP_WN_GPU, use_radii);

	}
	else
	{
		turbogwn.TurboLibiGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_PC_WN_GPU);
		turbogwn.TurboLibiGWNLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_GP_WN_GPU);

	}

	thrust::device_ptr<float> pc_wn_thrust_ptr(this->m_PC_WN_GPU);
	float pc_wn_sum = thrust::reduce(pc_wn_thrust_ptr, pc_wn_thrust_ptr + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
	float mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);
	std::cout << "mean_wn: " << mean_wn << std::endl;

	int mesh_buffer_size = std::max((res_x * res_y * res_z) / 2, 1000000);
	float3* c_mid_points_buffer = 0;
	float3* c_mid_normals_buffer = 0;
	cudaMallocManaged((char**)&c_mid_points_buffer, mesh_buffer_size * sizeof(*c_mid_points_buffer));
	cudaMallocManaged((char**)&c_mid_normals_buffer, mesh_buffer_size * sizeof(*c_mid_normals_buffer));

	if (output_mode == 1 || output_mode == 3 || output_mode == 4)
	{
		std::string mesh_path = "nothing";
		TurboMC turbomc;
		int num_tri = turbomc.TurboMarchingCubes(this->m_GP_WN_GPU, this->m_Grid_Points_GPU, res_x, res_y, res_z, mean_wn, 123456, mesh_path, mesh_full_path, c_mid_points_buffer, c_mid_normals_buffer, 999999);

		clock_t end_time = clock();
		std::chrono::high_resolution_clock::time_point end_time_s = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_s - start_time_s);

		std::cout << "Total Time = " << (end_time - start) / CLOCKS_PER_SEC << std::endl;
		std::cout << "Total Time (Wall time) = " << elapsed_time.count() << " s" << std::endl;

	}

	if (output_mode == 2 || output_mode == 3)
	{
		if (output_mode == 2)
		{
			clock_t end_time = clock();
			std::chrono::high_resolution_clock::time_point end_time_s = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_s - start_time_s);

			std::cout << "Total Time = " << (end_time - start) / CLOCKS_PER_SEC << std::endl;
			std::cout << "Total Time (Wall time) = " << elapsed_time.count() << " s" << std::endl;
		}

		std::cout << "Start save" << std::endl;

		std::ofstream outFile(mesh_full_path + ".xyz");
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			outFile << this->m_Point_Cloud_GPU[i].x << " " << this->m_Point_Cloud_GPU[i].y << " " << this->m_Point_Cloud_GPU[i].z << " " <<
				this->m_Normals_GPU[i].x << " " << this->m_Normals_GPU[i].y << " " << this->m_Normals_GPU[i].z << "\n";
		}

		outFile.close();

		std::ofstream outFile_As(mesh_full_path + "_As.txt");
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			outFile_As << this->m_As_GPU[i] << "\n";
		}
		outFile_As.close();

		std::ofstream outFile_Cs(mesh_full_path + "_Cs.txt");
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			outFile_Cs << this->m_Cs_GPU[i] << "\n";
		}
		outFile_Cs.close();

		std::ofstream outFile_AsCs(mesh_full_path + "_AsCs.txt");
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			outFile_AsCs << this->m_As_GPU[i] * this->m_Cs_GPU[i] << "\n";
		}
		outFile_AsCs.close();

		std::ofstream outFile_points(mesh_full_path + "_points.txt");
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			outFile_points << this->m_Point_Cloud_GPU[i].x << " " << this->m_Point_Cloud_GPU[i].y << " " << this->m_Point_Cloud_GPU[i].z << "\n";
		}
		outFile_points.close();

		std::ofstream outFile_normals(mesh_full_path + "_normals.txt");
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			outFile_normals << this->m_Normals_GPU[i].x << " " << this->m_Normals_GPU[i].y << " " << this->m_Normals_GPU[i].z << "\n";
		}
		outFile_normals.close();

		std::ofstream ofs(mesh_full_path + "_opt_points.ply");

		ofs << "ply\n";
		ofs << "format ascii 1.0\n";
		ofs << "element vertex " << this->m_Point_Cloud_size << "\n";
		ofs << "property float x\n";
		ofs << "property float y\n";
		ofs << "property float z\n";
		ofs << "property float nx\n";
		ofs << "property float ny\n";
		ofs << "property float nz\n";
		ofs << "property float quality\n";
		ofs << "end_header\n";

		float max_q = 0.0f;
		float min_q = FLT_MAX;
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			if (m_Cs_GPU[i] * m_As_GPU[i] > max_q)
				max_q = m_Cs_GPU[i] * m_As_GPU[i];
			if (m_Cs_GPU[i] * m_As_GPU[i] < min_q)
				min_q = m_Cs_GPU[i] * m_As_GPU[i];
		}

		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			float c_q = ((m_Cs_GPU[i] * m_As_GPU[i] - min_q) / (max_q - min_q)) * weight_scale + 0.001f;

			ofs << this->m_Point_Cloud_GPU[i].x << " " << this->m_Point_Cloud_GPU[i].y << " " << this->m_Point_Cloud_GPU[i].z << " "
				<< this->m_Normals_GPU[i].x << " " << this->m_Normals_GPU[i].y << " " << this->m_Normals_GPU[i].z << " "
				<< c_q << "\n";
		}

		ofs.close();

		std::cout << "Finish save" << std::endl;
	}

}

void TurboDWG::MeshAsOptGeneration(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC, int res_x, int res_y, int res_z, float beta, float sigma, int k_size, bool use_screened, bool use_radii, float As_threshold, std::string mesh_full_path, clock_t start, std::chrono::high_resolution_clock::time_point start_time_s, int output_mode, int sample_reso, float learning_rate, int total_As_opt_iteration)
{

	float3* uni_grid_samples_gpu = 0;

	int sample_grid_size = sample_reso * sample_reso * sample_reso;
	cudaMallocManaged((char**)&uni_grid_samples_gpu, sample_grid_size * sizeof(*uni_grid_samples_gpu));
	std::unique_ptr<float3[]> grid_samples_cpu = std::make_unique<float3[]>(size_t(sample_grid_size));

	int grid_count = 0;
	float step = 4.0f / (sample_reso - 1);
	for (int i = 0; i < sample_reso; ++i)
	{
		for (int j = 0; j < sample_reso; ++j)
		{
			for (int k = 0; k < sample_reso; ++k)
			{
				float x = -2.0f + i * step;
				float y = -2.0f + j * step;
				float z = -2.0f + k * step;

				grid_samples_cpu[grid_count++] = make_float3(x, y, z);
			}
		}
	}
	gpuErrchk(cudaMemcpy(uni_grid_samples_gpu, grid_samples_cpu.get(), sample_grid_size * sizeof(float3), cudaMemcpyHostToDevice));

	float* uni_grid_samples_wn_gpu = 0;
	cudaMallocManaged((char**)&uni_grid_samples_wn_gpu, sample_grid_size * sizeof(*uni_grid_samples_wn_gpu));

	TurboLibiGWN turbogwn;
	turbogwn.PreLibiGWN(O_PI, O_CH, O_CM, O_R, O_EC);

	turbogwn.UpdateGWNInfo(this->m_Normals_GPU, this->m_Point_Cloud_GPU, this->m_As_GPU, this->m_Radii_GPU);

	UpdateRadii(k_size);

	std::cout << "Start GWN Computation" << std::endl;
	if (use_screened)
	{
		turbogwn.TurboLibiScreenedGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);
		turbogwn.TurboLibiScreenedGWNLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Radii_GPU, beta, sigma, this->m_GP_WN_GPU, use_radii);
		turbogwn.TurboLibiScreenedGWNLaunch(uni_grid_samples_gpu, this->m_Point_Cloud_GPU, sample_grid_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Radii_GPU, beta, sigma, uni_grid_samples_wn_gpu, use_radii);
	}
	else
	{
		turbogwn.TurboLibiGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_PC_WN_GPU);
		turbogwn.TurboLibiGWNLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_GP_WN_GPU);
		turbogwn.TurboLibiGWNLaunch(uni_grid_samples_gpu, this->m_Point_Cloud_GPU, sample_grid_size, this->m_Normals_GPU, this->m_As_GPU, beta, uni_grid_samples_wn_gpu);
	}

	thrust::device_ptr<float> pc_wn_thrust_ptr(this->m_PC_WN_GPU);
	float pc_wn_sum = thrust::reduce(pc_wn_thrust_ptr, pc_wn_thrust_ptr + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
	float mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);

	int mesh_buffer_size = std::max((res_x * res_y * res_z) / 2, 1000000);
	float3* c_mid_points_buffer = 0;
	float3* c_mid_normals_buffer = 0;
	cudaMallocManaged((char**)&c_mid_points_buffer, mesh_buffer_size * sizeof(*c_mid_points_buffer));
	cudaMallocManaged((char**)&c_mid_normals_buffer, mesh_buffer_size * sizeof(*c_mid_normals_buffer));

	if (output_mode == 1 || output_mode == 3 || output_mode == 4)
	{
		std::string mesh_path = "nothing";
		TurboMC turbomc;
		int num_tri = turbomc.TurboMarchingCubes(this->m_GP_WN_GPU, this->m_Grid_Points_GPU, res_x, res_y, res_z, mean_wn, 123456, mesh_path, mesh_full_path, c_mid_points_buffer, c_mid_normals_buffer, 999999);

		clock_t end_time = clock();
		std::chrono::high_resolution_clock::time_point end_time_s = std::chrono::high_resolution_clock::now();
		std::chrono::duration<double> elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_s - start_time_s);

		std::cout << "Total Time = " << (end_time - start) / CLOCKS_PER_SEC << std::endl;
		std::cout << "Total Time (Wall time) = " << elapsed_time.count() << " s" << std::endl;
	}

	if (output_mode == 2 || output_mode == 3)
	{
		if (output_mode == 2)
		{
			clock_t end_time = clock();
			std::chrono::high_resolution_clock::time_point end_time_s = std::chrono::high_resolution_clock::now();
			std::chrono::duration<double> elapsed_time = std::chrono::duration_cast<std::chrono::duration<double>>(end_time_s - start_time_s);

			std::cout << "Total Time = " << (end_time - start) / CLOCKS_PER_SEC << std::endl;
			std::cout << "Total Time (Wall time) = " << elapsed_time.count() << " s" << std::endl;
		}

		std::cout << "Start save" << std::endl;

		std::ofstream outFile(mesh_full_path + ".xyz");
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			outFile << this->m_Point_Cloud_GPU[i].x << " " << this->m_Point_Cloud_GPU[i].y << " " << this->m_Point_Cloud_GPU[i].z << " " <<
				this->m_Normals_GPU[i].x << " " << this->m_Normals_GPU[i].y << " " << this->m_Normals_GPU[i].z << "\n";
		}

		outFile.close();

		std::ofstream outFile_As(mesh_full_path + "_As.txt");
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			outFile_As << this->m_As_GPU[i] << "\n";
		}
		outFile_As.close();

		std::cout << "Finish save" << std::endl;
	}

	if (output_mode == 4)
	{
		std::cout << "Start save GWN" << std::endl;

		std::ofstream outFile(mesh_full_path + "_grid_gwn.txt");
		outFile << sample_grid_size << std::endl;
		for (int i = 0; i < sample_grid_size; i++)
		{
			outFile << uni_grid_samples_wn_gpu[i] << "\n";
		}
		outFile.close();

		std::ofstream outFile_p(mesh_full_path + "_point_gwn.txt");
		outFile_p << this->m_Point_Cloud_size << std::endl;
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			outFile_p << this->m_Point_Cloud_GPU[i].x << " " << this->m_Point_Cloud_GPU[i].y << " " << this->m_Point_Cloud_GPU[i].z << " " <<
				this->m_PC_WN_GPU[i] << "\n";
		}
		outFile_p.close();

		std::cout << "Finish save GWN" << std::endl;
	}
}

void TurboDWG::GWNAsOptimization(float beta, float sigma, bool use_screened, bool use_radii, float learning_rate, int total_iteration, TurboLibiGWN* turbogwn)
{
	float* As_grad = 0;
	cudaMallocManaged((char**)&As_grad, this->m_Point_Cloud_size * sizeof(*As_grad));
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
		As_grad[i] = 0.0f;

	float* As_rms = 0;
	cudaMallocManaged((char**)&As_rms, this->m_Point_Cloud_size * sizeof(*As_rms));
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
		As_rms[i] = 0.0f;

	RMSPropOpt rms_prop;

	std::ofstream file;
	file.open("E:/dwg++/mesh/As_loss.txt", std::ios::app);

	thrust::device_ptr<float> ini_As_thrust_ptr(this->m_ini_As_GPU);
	float ini_As_sum = thrust::reduce(ini_As_thrust_ptr, ini_As_thrust_ptr + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());

	thrust::device_ptr<float> As_thrust_ptr(this->m_As_GPU);
	int threadsPerBlock = 128;
	int blocksPerGrid = (this->m_Point_Cloud_size + threadsPerBlock - 1) / threadsPerBlock;

	std::cout << "Start As optimization" << std::endl;
	for (int c_iter = 0; c_iter < total_iteration; c_iter++)
	{
		turbogwn->UpdateGWNInfo(this->m_Normals_GPU, this->m_Point_Cloud_GPU, this->m_As_GPU, this->m_Radii_GPU);
		turbogwn->UpdateGWNInfoAsGrad(this->m_Normals_GPU, this->m_Point_Cloud_GPU, this->m_As_GPU);

		if (use_screened)
			turbogwn->TurboLibiScreenedGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);
		else
			turbogwn->TurboLibiGWNLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, beta, this->m_PC_WN_GPU);

		thrust::device_ptr<float> pc_wn_thrust_ptr(this->m_PC_WN_GPU);
		float pc_wn_sum = thrust::reduce(pc_wn_thrust_ptr, pc_wn_thrust_ptr + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
		float mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);

		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			file << As_grad[i] << "\n";
		}

		float As_sum = thrust::reduce(As_thrust_ptr, As_thrust_ptr + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
		float grad_As_scale = 2e-1f;
		grad_As_scale = ((As_sum - ini_As_sum) > 0.0f) ? grad_As_scale : -grad_As_scale;

		AsRegKernal << < blocksPerGrid, threadsPerBlock >> > (As_grad, grad_As_scale, this->m_Point_Cloud_size);
		cudaDeviceSynchronize();

		float loss_sum = 0;
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			loss_sum += ((this->m_PC_WN_GPU[i] - mean_wn) * (this->m_PC_WN_GPU[i] - mean_wn));
		}
		float loss = loss_sum / ((float)this->m_Point_Cloud_size);

		std::cout << "Iter: " << c_iter << std::endl;
		std::cout << "Winding number loss: " << loss << std::endl;

		if (file.is_open())
		{

		}

		rms_prop.RMSPropUpdate(this->m_As_GPU, As_rms, As_grad, 0.9f, learning_rate, 1e-8, -1e-9, this->m_Point_Cloud_size);

	}

	file.close();

	cudaFree(As_grad); As_grad = 0;
	cudaFree(As_rms); As_rms = 0;
}

__global__ void computeScaleLoss(float3* gwn_grad, float* scale, float* loss, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;

	float grad_magnitude = gwn_grad[idx].x * gwn_grad[idx].x + gwn_grad[idx].y * gwn_grad[idx].y + gwn_grad[idx].z * gwn_grad[idx].z;
	atomicAdd(loss, scale[idx] * grad_magnitude);
}

__global__ void computeDiriLoss(float3* gwn_grad, float* volume_scale, float volume, float* loss, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;

	float grad_magnitude = gwn_grad[idx].x * gwn_grad[idx].x + gwn_grad[idx].y * gwn_grad[idx].y + gwn_grad[idx].z * gwn_grad[idx].z;
	float c_loss = volume_scale[idx] * volume * grad_magnitude;
	atomicAdd(loss, c_loss);
}

__global__ void computeScaleTargetLoss(float* gwn, float scale, float* loss, float target, int size)
{
	int idx = blockIdx.x * blockDim.x + threadIdx.x;
	if (idx >= size) return;

	float idx_loss = scale * (gwn[idx] - target) * (gwn[idx] - target);
	atomicAdd(loss, idx_loss);
}

__global__ void cartesian2SphericalKernel(float3* xyz, float* u, float* v, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) return;

	float xi = xyz[tid].x;
	float yi = xyz[tid].y;
	float zi = xyz[tid].z;

	u[tid] = atan2f(yi, xi);
	v[tid] = acosf(zi);
}

__global__
void countNeighborsSimple(float3* __restrict__ points, int* __restrict__ counts, float* __restrict__ Ci, float radius2, float C_thres, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	float3 pi = points[i];
	int    cnt = 0;

	for (int j = 0; j < N; ++j)
	{
		float3 pj = points[j];
		float dx = pi.x - pj.x;
		float dy = pi.y - pj.y;
		float dz = pi.z - pj.z;
		if (dx * dx + dy * dy + dz * dz <= radius2)
		{
			if (Ci[j] > C_thres)
			{
				++cnt;
			}
		}
	}

	counts[i] = cnt;
}

__global__
void smoothAi(float3* __restrict__ points, float* __restrict__ Ai, float* __restrict__ Ci, float* __restrict__ new_Ai, float radius2, int N)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid >= N) return;

	float3 pi = points[tid];
	int    cnt = 0;
	float  avg_Ai = 0.0f;
	float weight_para = 50.0f;

	for (int j = 0; j < N; ++j)
	{
		float3 pj = points[j];
		float dx = pi.x - pj.x;
		float dy = pi.y - pj.y;
		float dz = pi.z - pj.z;
		float dist_square = dx * dx + dy * dy + dz * dz;
		if (dist_square <= radius2)
		{
			float c_weigh = expf(-dist_square / weight_para);
			if (Ci[j] > 0.8f)
			{
				avg_Ai += c_weigh * Ai[j];
				cnt++;
			}
		}
	}

	new_Ai[tid] = Ai[tid];
	if (Ci[tid] > 0.8f)
	{
		if (cnt > 0)
			new_Ai[tid] = avg_Ai / float(cnt);
	}
}

__global__
void countNeighborsWeightSimple(float3* __restrict__ points, float* __restrict__ Ai, float* __restrict__ Ci, int* counts, float* weights, float radius2, int N)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= N) return;

	float3 pi = points[i];
	float weight = 0.0f;

	for (int j = 0; j < N; ++j)
	{
		float3 pj = points[j];
		float dx = pi.x - pj.x;
		float dy = pi.y - pj.y;
		float dz = pi.z - pj.z;
		if (dx * dx + dy * dy + dz * dz <= radius2)
		{
			weight += (Ai[j] * Ci[j]);
		}
	}

	weights[i] = weight;
}

void TurboDWG::InitialCi(float scale, float radius, float* gwn, std::string mesh_path, int loop, int ci_level,

	float ci_in_thres, float ci_out_thres)
{
	int* counts = 0;
	cudaMallocManaged((void**)&counts, this->m_Point_Cloud_size * sizeof(*counts));
	int threadsPerBlock = 128;
	int blocksPerGrid = (this->m_Point_Cloud_size + threadsPerBlock - 1) / threadsPerBlock;
	float radius2 = radius * radius * scale;//0.0016//0.0032
	countNeighborsSimple << < blocksPerGrid, threadsPerBlock >> > (this->m_Point_Cloud_GPU, counts, this->m_Cs_GPU, radius2, 0.0f, this->m_Point_Cloud_size);
	cudaDeviceSynchronize();

	float* weights = 0;
	cudaMallocManaged((void**)&weights, this->m_Point_Cloud_size * sizeof(*weights));
	countNeighborsWeightSimple << < blocksPerGrid, threadsPerBlock >> > (this->m_Point_Cloud_GPU, this->m_As_GPU, this->m_Cs_GPU, counts, weights, radius2, this->m_Point_Cloud_size);
	cudaDeviceSynchronize();

	std::vector<int> labels; labels.resize(this->m_Point_Cloud_size);

	float mean_gwn = 0.0f;
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		mean_gwn += gwn[i];
	}
	mean_gwn /= float(this->m_Point_Cloud_size);
	std::cout << "mean_gwn: " << mean_gwn << std::endl;

	if (mean_gwn < 0.0f)
	{
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			gwn[i] = -gwn[i];
		}
		mean_gwn = -mean_gwn;
	}

	std::ofstream file_Ci(mesh_path + "test_gwn.txt");
	if (file_Ci.is_open())
	{
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
			file_Ci << gwn[i] << "\n";
	}
	file_Ci.close();

	float* dist_gwn = 0;
	cudaMallocManaged((void**)&dist_gwn, this->m_Point_Cloud_size * sizeof(*dist_gwn));
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		dist_gwn[i] = fabs(mean_gwn - gwn[i]);
	}
	std::ofstream outFile_ds(mesh_path + std::to_string(loop) + "_" + "test_dist_gwn.txt");
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		outFile_ds << dist_gwn[i] << "\n";
	}
	outFile_ds.close();

	thrust::device_vector<float> d_weights(gwn, gwn + this->m_Point_Cloud_size);
	thrust::device_vector<float> sorted_weights = d_weights;

	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		weights[i] = gwn[i];
	}

	if (mean_gwn > 0)
	{
		thrust::sort(sorted_weights.begin(), sorted_weights.end() );
		int idx1 = static_cast<int>((1.0f - 0.9f) * this->m_Point_Cloud_size);
		int idx2 = 0;
		float c1 = mean_gwn - ci_out_thres;
		float c2 = sorted_weights[idx2];

		if (ci_out_thres > 0.0f)
		{
			for (int iter = 0; iter < 10000; ++iter)
			{
				bool changed = false;

				for (size_t i = 0; i < this->m_Point_Cloud_size; ++i)
				{
					float d1 = fabs(weights[i] - c1);
					float d2 = fabs(weights[i] - c2);
					int new_label = (d1 < d2) ? 0 : 1;
					if (labels[i] != new_label)
					{
						labels[i] = new_label;
						changed = true;
					}
				}

				if (!changed) break;

				std::vector<float> cluster1, cluster2;
				for (size_t i = 0; i < this->m_Point_Cloud_size; ++i)
				{
					if (labels[i] == 0)
						cluster1.push_back(weights[i]);
					else
						cluster2.push_back(weights[i]);
				}

				if (!cluster2.empty())
				{
					float sum = 0.0f;
					for (float x : cluster2)
						sum += x;
					c2 = sum / ((float)cluster2.size());
				}
			}

			std::cout << "c1: " << c1 << " c2: " << c2 << std::endl;

			float min_thres = c2;
			for (int i = 0; i < this->m_Point_Cloud_size; i++)
			{
				if (weights[i] < min_thres)
					this->m_Cs_GPU[i] = 0.01f;

				else
				{
					this->m_Cs_GPU[i] = 0.99f;
				}
			}
		}

		thrust::device_vector<float> d_Ai(counts, counts + this->m_Point_Cloud_size);
		thrust::device_vector<float> sorted_Ai = d_Ai;
		thrust::sort(sorted_Ai.begin(), sorted_Ai.end());

		float dmin = float(sorted_Ai[0]);
		float dmax = float(sorted_Ai[this->m_Point_Cloud_size - 1]);
		float range = dmax - dmin;

		int min_level = ci_level;
		int max_level = ci_level;
		float level_ratio = float(max_level) / float(min_level);
		int level_times = ceil(log2(level_ratio)) + 1;

		std::vector<std::vector<float>> levels_ci;
		levels_ci.resize(level_times);
		for (int t = 0; t < level_times; t++)
		{
			levels_ci[t].resize(this->m_Point_Cloud_size);
		}

		thrust::sort(sorted_weights.begin(), sorted_weights.end(), thrust::greater<float>());
		idx1 = static_cast<int>((1.0f - 0.9f) * this->m_Point_Cloud_size);
		idx2 = 0;
		c1 = mean_gwn + ci_in_thres;
		c2 = sorted_weights[idx2];

		if (ci_in_thres > 0.0f)
		{
			for (int iter = 0; iter < 10000; ++iter)
			{
				bool changed = false;

				for (size_t i = 0; i < this->m_Point_Cloud_size; ++i)
				{
					float d1 = fabs(weights[i] - c1);
					float d2 = fabs(weights[i] - c2);
					int new_label = (d1 < d2) ? 0 : 1;
					if (labels[i] != new_label)
					{
						labels[i] = new_label;
						changed = true;
					}
				}

				if (!changed) break;

				std::vector<float> cluster1, cluster2;
				for (size_t i = 0; i < this->m_Point_Cloud_size; ++i)
				{
					if (labels[i] == 0)
						cluster1.push_back(weights[i]);
					else
						cluster2.push_back(weights[i]);
				}

				if (!cluster2.empty())
				{
					float sum = 0.0f;
					for (float x : cluster2)
						sum += x;
					c2 = sum / ((float)cluster2.size());
				}
			}

			std::cout << "c1: " << c1 << " c2: " << c2 << std::endl;

			float max_thres = c2;
			for (int i = 0; i < this->m_Point_Cloud_size; i++)
			{
				if (weights[i] > max_thres)
					this->m_Cs_GPU[i] = 0.01f;

			}
		}

        std::ofstream outFile_bi(mesh_path + std::to_string(loop) + "_" + "ci_bimeans.txt");
        for (int i = 0; i < this->m_Point_Cloud_size; i++)
        {
            outFile_bi << this->m_Cs_GPU[i] << "\n";
        }
        outFile_bi.close();

		bool inside_bin = false;
		bool outside_bin = false;
		if (ci_in_thres > 0.0f)
			inside_bin = true;
		if (ci_out_thres > 0.0f)
			outside_bin = true;

		float bin_thres = 0.1f;

		for (int t = 0; t < level_times; t++)
		{
			int level = (t == 0) ? min_level : min_level * t * 2;

			float interval = (range > 0.0f) ? (range / float(level)) : 1.0f;

			std::vector<float> sumCi(level, 0.0f);
			std::vector<int>   cnt(level, 0);
			std::vector<int>   Ci_level; Ci_level.resize(this->m_Point_Cloud_size);

			for (int i = 0; i < this->m_Point_Cloud_size; ++i)
			{
				int idx = (range > 0) ? static_cast<int>((float(counts[i]) - dmin) / interval) : 0;
				if (idx < 0) idx = 0;
				if (idx >= level) idx = level - 1;

				Ci_level[i] = idx;
				sumCi[idx] += this->m_Cs_GPU[i];
				cnt[idx] += 1;
			}

			std::vector<float> avgCi(level, 0.0f);
			for (int k = 0; k < level; ++k)
			{
				if (cnt[k] > 0)
				{
					avgCi[k] = sumCi[k] / cnt[k];
				}
			}

			for (int i = 0; i < this->m_Point_Cloud_size; ++i)
			{

				levels_ci[t][i] = avgCi[Ci_level[i]];
			}
		}

		for (int i = 0; i < this->m_Point_Cloud_size; ++i)
		{
			if (outside_bin)
			{
				if (weights[i] < mean_gwn - bin_thres)
				{
					float new_ci = 0.0f;
					for (int t = 0; t < level_times; t++)
					{
						new_ci += levels_ci[t][i];
					}

					new_ci /= float(level_times);

					this->m_Cs_GPU[i] = new_ci;
				}
			}

			if (inside_bin)
			{
				if (weights[i] > mean_gwn + bin_thres)
				{
					float new_ci = 0.0f;
					for (int t = 0; t < level_times; t++)
					{
						new_ci += levels_ci[t][i];
					}

					new_ci /= float(level_times);

					this->m_Cs_GPU[i] = new_ci;
				}
			}
		}

	}

	cudaFree(counts); counts = 0;
	cudaFree(weights); weights = 0;
}

struct CCDelta
{
	float c1;
	float c2;
};

void TurboDWG::FinalizeCi(float* gwn, std::string mesh_path,

	float ci_in_thres, float ci_out_thres,
	int set_loop, int top_m, int point_size )
{
	float mean_gwn = 0.0f;
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		mean_gwn += gwn[i];
	}
	mean_gwn /= float(this->m_Point_Cloud_size);

	if (mean_gwn < 0.0f)
	{
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			gwn[i] = -gwn[i];
		}
		mean_gwn = -mean_gwn;
	}
	std::cout << "mean_gwn: " << mean_gwn << std::endl;

	std::ofstream file_Ci(mesh_path + "test_gwn.txt");
	if (file_Ci.is_open())
	{
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
			file_Ci << gwn[i] << "\n";
	}
	file_Ci.close();

	float final_in_gwn_thres = 0.0;
	float final_out_gwn_thres = 0.0;

	thrust::device_vector<float> d_gwn(gwn, gwn + this->m_Point_Cloud_size);
	thrust::device_vector<float> sorted_gwn = d_gwn;

	for (int c_mode = 0; c_mode <= 1; c_mode++)
	{
		if (ci_out_thres <= 0 && c_mode == 0)
		{
			final_out_gwn_thres = -999999.0f;
			continue;
		}
		if (ci_in_thres <= 0 && c_mode == 1)
		{
			final_in_gwn_thres = 999999.0f;
			continue;
		}

		std::vector<int> labels; labels.resize(point_size);

		std::vector<CCDelta> c2_log;
		c2_log.resize(set_loop);
		for (int j = 0; j < set_loop; j++)
		{

			if (c_mode == 0)
				thrust::sort(sorted_gwn.begin(), sorted_gwn.end() );
			else
				thrust::sort(sorted_gwn.begin(), sorted_gwn.end(), thrust::greater<float>());

			int idx2 = 0;
			float c1 = mean_gwn;
			float c2 = sorted_gwn[idx2];

			if (c_mode == 0)
			{
				c1 = mean_gwn - ci_out_thres;
			}
			if (c_mode == 1)
			{
				c1 = mean_gwn + ci_in_thres;
			}

			for (int iter = 0; iter < 10000; ++iter)
			{
				bool changed = false;

				for (size_t i = 0; i < point_size; ++i)
				{
					float d1 = fabs(gwn[i] - c1);
					float d2 = fabs(gwn[i] - c2);
					int new_label = (d1 < d2) ? 0 : 1;
					if (labels[i] != new_label)
					{
						labels[i] = new_label;
						changed = true;
					}
				}

				if (!changed) break;

				std::vector<float> cluster1, cluster2;
				for (size_t i = 0; i < point_size; ++i)
				{
					if (labels[i] == 0)
						cluster1.push_back(gwn[i]);
					else
						cluster2.push_back(gwn[i]);
				}

				if (!cluster2.empty())
				{
					float sum = 0.0f;
					for (float x : cluster2)
						sum += x;
					c2 = sum / ((float)cluster2.size());
				}
			}

			std::cout << "c1: " << c1 << " c2: " << c2 << std::endl;

			c2_log[j].c1 = c1;
			c2_log[j].c2 = c2;
		}

		for (int j = 0; j < top_m; j++)
		{
			float c2 = c2_log[j].c2;
			if (c_mode == 0)
			{
				final_out_gwn_thres += c2;
			}
			else
			{
				final_in_gwn_thres += c2;
			}

			std::cout << "select " << "c1: " << c2_log[j].c1 << " c2: " << c2 << std::endl;
		}
	}
	final_in_gwn_thres /= float(top_m);
	final_out_gwn_thres /= float(top_m);

	std::cout << "final_in_gwn_thres: " << final_in_gwn_thres << " final_out_gwn_thres: " << final_out_gwn_thres << std::endl;

	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		if (gwn[i] < final_out_gwn_thres)
		{
			this->m_Cs_GPU[i] = 0.01f;
		}
		else if (gwn[i] > final_in_gwn_thres)
		{
			this->m_Cs_GPU[i] = 0.01f;
		}
		else if (this->m_Cs_GPU[i] < 0.5f)
		{
			this->m_Cs_GPU[i] = 0.01f;
		}
		else if (this->m_Cs_GPU[i] >= 0.5f)
		{
			this->m_Cs_GPU[i] = 0.99f;
		}
	}
}

void TurboDWG::AdjustAi(float scale, float radius, float smooth_radius, bool ai_adjust, int loop, int loss_mode)
{
	int* counts = 0;
	cudaMallocManaged((void**)&counts, this->m_Point_Cloud_size * sizeof(*counts));
	int threadsPerBlock = 128;
	int blocksPerGrid = (this->m_Point_Cloud_size + threadsPerBlock - 1) / threadsPerBlock;
	float radius2 = radius * radius * scale;//0.0016
	countNeighborsSimple << < blocksPerGrid, threadsPerBlock >> > (this->m_Point_Cloud_GPU, counts, this->m_Cs_GPU, radius2, 0.0f, this->m_Point_Cloud_size);
	cudaDeviceSynchronize();

	thrust::device_vector<float> d_counts(counts, counts + this->m_Point_Cloud_size);
	thrust::device_vector<float> sorted_counts = d_counts;
	thrust::sort(sorted_counts.begin(), sorted_counts.end());

	float dmin = float(sorted_counts[0]);
	float dmax = float(sorted_counts[this->m_Point_Cloud_size - 1]);
	float range = dmax - dmin;

	int min_level = 256;
	int max_level = 256;

	if (this->m_Ci_Opted)
	{
		min_level = 32;
		max_level = 32;
	}

	float level_ratio = float(max_level) / float(min_level);
	int level_times = ceil(log2(level_ratio)) + 1;

	std::cout << "level_times " << level_times << std::endl;
	std::vector<std::vector<float>> levels_ai;
	levels_ai.resize(level_times);
	for (int t = 0; t < level_times; t++)
	{
		levels_ai[t].resize(this->m_Point_Cloud_size);
	}

	for (int t = 0; t < level_times; t++)
	{
		int level = (t == 0) ? min_level : min_level * t * 2;

		float interval = (range > 0.0f) ? (range / float(level)) : 1.0f;

		std::vector<float> sumAi(level, 0.0f);
		std::vector<int>   cnt(level, 0);
		std::vector<int>   Ai_level; Ai_level.resize(this->m_Point_Cloud_size);

		for (int i = 0; i < this->m_Point_Cloud_size; ++i)
		{
			int idx = (range > 0) ? static_cast<int>((float(counts[i]) - dmin) / interval) : 0;
			if (idx < 0) idx = 0;
			if (idx >= level) idx = level - 1;

			Ai_level[i] = idx;

			sumAi[idx] += this->m_As_GPU[i];
			cnt[idx] += 1;
		}

		std::vector<float> avgAi(level, 0.0f);
		for (int k = 0; k < level; ++k)
		{
			if (cnt[k] > 0)
			{
				avgAi[k] = sumAi[k] / cnt[k];
			}
		}

		for (int i = 0; i < this->m_Point_Cloud_size; ++i)
		{
			levels_ai[t][i] = avgAi[Ai_level[i]];
		}
	}

	if (ai_adjust)
	{

			for (int i = 0; i < this->m_Point_Cloud_size; ++i)
			{
				float new_ai = 0.0f;
				for (int t = 0; t < level_times; t++)
				{
					new_ai += levels_ai[t][i];
				}

				new_ai /= float(level_times);

				this->m_As_GPU[i] = new_ai;
			}

	}

	float* new_Ai = 0;
	cudaMallocManaged((void**)&new_Ai, this->m_Point_Cloud_size * sizeof(*new_Ai));

	if (loop >= 0  )
	{
		float radius2;
		if (loop == 0)
		{
			radius2 = smooth_radius * smooth_radius;
		}
		else if (loop > 0 &&  loss_mode == 2)
		{
			radius2 = smooth_radius * smooth_radius;
		}
		else
		{
			radius2 = smooth_radius * smooth_radius;
		}

		smoothAi << < blocksPerGrid, threadsPerBlock >> > (this->m_Point_Cloud_GPU, this->m_As_GPU, this->m_Cs_GPU, new_Ai, radius2, this->m_Point_Cloud_size);
		cudaDeviceSynchronize();

		for (int i = 0; i < this->m_Point_Cloud_size; ++i)
		{
			this->m_As_GPU[i] = new_Ai[i];
		}
	}

	cudaFree(counts); counts = 0;
	cudaFree(new_Ai); new_Ai = 0;
}

void TurboDWG::GetPlaneField(int sample_reso, int axis, float beta, float sigma, bool use_radii, int loop, std::string mesh_path)
{
	float extend_range = 0.25;

	float plane_x_up = this->m_Bound_Upper_KDT_GPU[0].x + extend_range;
	float plane_x_low = this->m_Bound_Lower_KDT_GPU[0].x - extend_range;

	float plane_y_up = this->m_Bound_Upper_KDT_GPU[0].y + extend_range;
	float plane_y_low = this->m_Bound_Lower_KDT_GPU[0].y - extend_range;

	float plane_z_up = this->m_Bound_Upper_KDT_GPU[0].z + extend_range;
	float plane_z_low = this->m_Bound_Lower_KDT_GPU[0].z - extend_range;

	float plane_x_len = plane_x_up - plane_x_low;
	float plane_y_len = plane_y_up - plane_y_low;
	float plane_z_len = plane_z_up - plane_z_low;

	float plane_max_len;

	float plane_a_len, plane_b_len;
	float plane_a_s, plane_b_s;
	float plane_pos;
	if (axis == 0)
	{
		plane_max_len = fmaxf(plane_y_len, plane_z_len);
		plane_a_len = plane_y_len;
		plane_b_len = plane_z_len;

		plane_pos = 0.05f;

		plane_a_s = plane_y_low;
		plane_b_s = plane_z_low;
	}
	else if (axis == 1)
	{
		plane_max_len = fmaxf(plane_x_len, plane_z_len);
		plane_a_len = plane_x_len;
		plane_b_len = plane_z_len;
		plane_pos = (this->m_Bound_Upper_KDT_GPU[0].y + this->m_Bound_Lower_KDT_GPU[0].y) / 2.0f;

		plane_a_s = plane_x_low;
		plane_b_s = plane_z_low;
	}
	else if (axis == 2)
	{
		plane_max_len = fmaxf(plane_x_len, plane_y_len);
		plane_a_len = plane_x_len;
		plane_b_len = plane_y_len;
		plane_pos = (this->m_Bound_Upper_KDT_GPU[0].z + this->m_Bound_Lower_KDT_GPU[0].z) / 2.0f;

		plane_a_s = plane_x_low;
		plane_b_s = plane_y_low;
	}

	float grid_len = plane_max_len / ((float)sample_reso);
	int grid_a_size = ceilf(plane_a_len / grid_len);
	int grid_b_size = ceilf(plane_b_len / grid_len);
	int sample_grid_size = grid_a_size * grid_b_size;

	float3* uni_grid_samples_gpu = 0;
	cudaMallocManaged((char**)&uni_grid_samples_gpu, sample_grid_size * sizeof(*uni_grid_samples_gpu));
	std::unique_ptr<float3[]> grid_samples_cpu = std::make_unique<float3[]>(size_t(sample_grid_size));
	int grid_count = 0;
	for (int j = 0; j < grid_a_size; ++j)
	{
		for (int k = 0; k < grid_b_size; ++k)
		{
			float a = plane_a_s + j * grid_len;
			float b = plane_b_s + k * grid_len;

			if (axis == 0)
			{
				grid_samples_cpu[grid_count++] = make_float3(plane_pos, a, b);
			}
			else if (axis == 1)
			{
				grid_samples_cpu[grid_count++] = make_float3(a, plane_pos, b);
			}
			else if (axis == 2)
			{
				grid_samples_cpu[grid_count++] = make_float3(a, b, plane_pos);
			}
		}
	}
	gpuErrchk(cudaMemcpy(uni_grid_samples_gpu, grid_samples_cpu.get(), sample_grid_size * sizeof(float3), cudaMemcpyHostToDevice));

	float* grid_samples_gwn = 0;
	cudaMallocManaged((char**)&grid_samples_gwn, sample_grid_size * sizeof(*grid_samples_gwn));
	this->m_GWN_GPU->TurboLibiScreenedGWNCsLaunch(uni_grid_samples_gpu, this->m_Point_Cloud_GPU, sample_grid_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma, grid_samples_gwn, use_radii);

	this->m_GWN_GPU->TurboLibiScreenedGWNCsLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);

	thrust::device_ptr<float> pc_wn_thrust_ptr(this->m_PC_WN_GPU);
	float pc_wn_sum = thrust::reduce(pc_wn_thrust_ptr, pc_wn_thrust_ptr + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
	float mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);

	std::ofstream outFile(mesh_path + std::to_string(loop) + "_plane_field.txt");
	outFile << grid_a_size << " " << grid_b_size << "\n";
	for (int i = 0; i < sample_grid_size; i++)
	{
		if (mean_wn < 0)
			grid_samples_gwn[i] = -grid_samples_gwn[i];

		outFile << uni_grid_samples_gpu[i].x << " " << uni_grid_samples_gpu[i].y << " " << uni_grid_samples_gpu[i].z << " " << grid_samples_gwn[i] << "\n";
	}
	outFile.close();

	cudaFree(uni_grid_samples_gpu);
	cudaFree(grid_samples_gwn);
}

int TurboDWG::SetSamplings(float3* uni_grid_samples_gpu, float* distances, std::vector<bool>& uni_grid_samples_is_near, int sample_grid_size, float dist_thres, float beta, float sigma, bool use_radii, int loop)
{
	this->m_GWN_GPU->TurboLibiScreenedGWNCsLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);
	thrust::device_ptr<float> pc_wn_thrust_ptr_ori(this->m_PC_WN_GPU);
	float pc_wn_sum = thrust::reduce(pc_wn_thrust_ptr_ori, pc_wn_thrust_ptr_ori + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
	float mean_wn = pc_wn_sum / ((float)this->m_Point_Cloud_size);

	TurboMC turbomc;
	std::string mesh_full_path = "nothing";

	int mesh_buffer_size = std::max((this->m_Grid_Res_X * this->m_Grid_Res_Y * this->m_Grid_Res_Z) / 2, 1000000);
	float3* mid_points_buffer = 0;
	float3* mid_normals_buffer = 0;
	cudaMallocManaged((char**)&mid_points_buffer, mesh_buffer_size * sizeof(*mid_points_buffer));
	cudaMallocManaged((char**)&mid_normals_buffer, mesh_buffer_size * sizeof(*mid_normals_buffer));

	int num_tri = turbomc.TurboMarchingCubes(this->m_GP_WN_GPU, this->m_Grid_Points_GPU, this->m_Grid_Res_X, this->m_Grid_Res_Y, this->m_Grid_Res_Z, mean_wn, 888, mesh_full_path, mesh_full_path, mid_points_buffer, mid_normals_buffer, 999999);

	float3* mid_points = 0;
	float3* mid_normals = 0;
	cudaMallocManaged((char**)&mid_points, num_tri * sizeof(*mid_points));
	cudaMallocManaged((char**)&mid_normals, num_tri * sizeof(*mid_normals));
	cudaMemcpy(mid_points, mid_points_buffer, num_tri * sizeof(*mid_points_buffer), cudaMemcpyDeviceToDevice);
	cudaMemcpy(mid_normals, mid_normals_buffer, num_tri * sizeof(*mid_normals_buffer), cudaMemcpyDeviceToDevice);

	float3* points_upper = 0;
	float3* points_lower = 0;
	cudaMallocManaged((char**)&points_upper, sizeof(*points_upper));
	cudaMallocManaged((char**)&points_lower, sizeof(*points_lower));
	GenerateKDTreeForDWG(mid_points, mid_normals, num_tri, points_upper, points_lower);

	int k_size = 15;
	int* sample_knn_gpu = 0;
	cudaMallocManaged((void**)&sample_knn_gpu, sample_grid_size * k_size * sizeof(*sample_knn_gpu));

	CudaKNNWithKDTreeforDWG(this->m_Point_Cloud_GPU,
		this->m_Point_Cloud_size,
		uni_grid_samples_gpu,
		sample_grid_size,
		this->m_Bound_Upper_KDT_GPU, this->m_Bound_Lower_KDT_GPU,
		sample_knn_gpu,
		k_size
	);

	int threadsPerBlock = 128;
	const int blocksPerGrid_gp = (sample_grid_size + threadsPerBlock - 1) / threadsPerBlock;

	ComPtoMMinDist << < blocksPerGrid_gp, threadsPerBlock >> > (uni_grid_samples_gpu, this->m_Point_Cloud_GPU, this->m_Cs_GPU, sample_knn_gpu, distances, k_size, sample_grid_size);
	cudaDeviceSynchronize();

	int num_near_point = 0;
	for (int i = 0; i < sample_grid_size; i++)
	{
		if (distances[i] < dist_thres)
		{
			uni_grid_samples_is_near[i] = true;
			num_near_point += 1;
		}
	}

	int sample_points_size = sample_grid_size - num_near_point;
	std::cout << "sample_points_size: " << sample_points_size << std::endl;

	cudaFree(mid_points_buffer); mid_points_buffer = 0;
	cudaFree(mid_normals_buffer); mid_normals_buffer = 0;
	cudaFree(mid_points); mid_points = 0;
	cudaFree(mid_normals);  mid_normals = 0;
	cudaFree(points_upper); points_upper = 0;
	cudaFree(points_lower); points_lower = 0;
	cudaFree(sample_knn_gpu); sample_knn_gpu = 0;

	return sample_points_size;
}

int TurboDWG::SetMode(float* last_Ai, float ai_var_thres_0, float ai_var_thres_1, int c_loop, int c_total_loop, int after_ci_loop, int before_ci_max_loop, int max_loop, int loss_mode)
{
	int new_loss_mode = loss_mode;

	thrust::device_ptr<const float> last_Ai_ptr(last_Ai), Ai_ptr(this->m_As_GPU);

	typedef thrust::tuple<thrust::device_ptr<const float>, thrust::device_ptr<const float>> FloatTuple;
	typedef thrust::zip_iterator<FloatTuple> ZipIter;

	ZipIter begin = thrust::make_zip_iterator(thrust::make_tuple(last_Ai_ptr, Ai_ptr));
	ZipIter end = begin + this->m_Point_Cloud_size;
	float vary_sum = thrust::transform_reduce(begin, end, sqdiff_op(), 0.0f, thrust::plus<float>());
	float vary_ratio = vary_sum / float(this->m_Point_Cloud_size);
	std::cout << "Ai vary ratio = " << vary_ratio << std::endl;

	float vary_max = thrust::transform_reduce(begin, end, sqdiff_op(), 0.0f, thrust::maximum<float>());
	std::cout << "Ai vary ratio max = " << vary_max << std::endl;

	if (loss_mode == 1)
	{
		thrust::device_ptr<const float> last_Ci_ptr(this->m_last_Cs_GPU), Ci_ptr(this->m_Cs_GPU);

		ZipIter begin = thrust::make_zip_iterator(thrust::make_tuple(last_Ci_ptr, Ci_ptr));
		ZipIter end = begin + this->m_Point_Cloud_size;
		float vary_sum = thrust::transform_reduce(begin, end, sqdiff_op(), 0.0f, thrust::plus<float>());
		float vary_ratio = vary_sum / float(this->m_Point_Cloud_size);
		std::cout << "last Ci vary ratio = " << vary_ratio << std::endl;

	}

	if (
        (loss_mode == 0 && vary_ratio <= ai_var_thres_0) ||
        (loss_mode == 2 && vary_ratio <= ai_var_thres_0) ||
        (loss_mode == 2 && after_ci_loop == before_ci_max_loop) ||
		(loss_mode == 3 && after_ci_loop == before_ci_max_loop) ||
		(loss_mode == 3 && vary_ratio <= ai_var_thres_1))
	{
		new_loss_mode = 1;

		for (int i = 0; i < this->m_Point_Cloud_size; i++)
		{
			this->m_last_Cs_GPU[i] = this->m_Cs_GPU[i];
		}
	}
	else if (loss_mode == 1)
	{
		new_loss_mode = 0;
	}

	if ((c_total_loop == max_loop) && (new_loss_mode == 1))
	{
		new_loss_mode = 4;
	}

	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		last_Ai[i] = this->m_As_GPU[i];
	}

	return new_loss_mode;
}

void ComRealBBox(float3* points, float* Ci, float dist_threshold, int counts_threshold, float3& lower, float3& upper, int point_size)
{
	float* distances = 0;
	cudaMallocManaged((char**)&distances, point_size * sizeof(*distances));

	int threadsPerBlock = 128;
	int blocksPerGrid = (point_size + threadsPerBlock - 1) / threadsPerBlock;
	comP2Pdistance << < blocksPerGrid, threadsPerBlock >> > (points, distances,   point_size);
	cudaDeviceSynchronize();

	float mean_dist = 0.0f;
	float variance = 0.0f;
	for (int i = 0; i < point_size; i++)
	{
		mean_dist += distances[i];
		variance += distances[i] * distances[i];
	}
	mean_dist /= float(point_size);
	variance /= float(point_size);
	variance -= (mean_dist * mean_dist);
	float stddev = sqrtf(variance);

	int* counts = 0;
	cudaMallocManaged((void**)&counts, point_size * sizeof(*counts));
	float radius2 = 0.04f * 0.04f;
	countNeighborsSimple << < blocksPerGrid, threadsPerBlock >> > (points, counts, Ci, radius2, 0.0f, point_size);
	cudaDeviceSynchronize();

	lower = make_float3(9999999999.0f, 9999999999.0f, 9999999999.0f);
	upper = make_float3(-9999999999.0f, -9999999999.0f, -9999999999.0f);

	for (int i = 0; i < point_size; i++)
	{

		if ((distances[i] > mean_dist + dist_threshold * stddev) && counts[i] <= counts_threshold)
		{
			continue;
		}
		else
		{
			if (points[i].x < lower.x)
			{
				lower.x = points[i].x;
			}
			if (points[i].y < lower.y)
			{
				lower.y = points[i].y;
			}
			if (points[i].z < lower.z)
			{
				lower.z = points[i].z;
			}

			if (points[i].x > upper.x)
			{
				upper.x = points[i].x;
			}
			if (points[i].y > upper.y)
			{
				upper.y = points[i].y;
			}
			if (points[i].z > upper.z)
			{
				upper.z = points[i].z;
			}
		}
	}

	cudaFree(distances); distances = 0;
	cudaFree(counts); counts = 0;
}

void TurboDWG::sGWNOptimization(float beta, float sigma, bool use_screened, bool use_radii,   float learning_rate_Ai, float learning_rate_Ai_2, float learning_rate_Ci,
	int loop_iteration, TurboLibiGWN* turbogwn, std::string mesh_path,
	int loop, int ci_max_loop_iteration,
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
	float inside_x, float inside_y, float inside_z)
{
	std::cout << "diric_scale, " << diric_scale << " diric_reso," << diric_reso << " ai_target_mid," << ai_target_mid << " ci_var_thres," << ci_var_thres << std::endl;

	float* Ai_grad = 0;
	cudaMallocManaged((char**)&Ai_grad, this->m_Point_Cloud_size * sizeof(*Ai_grad));
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
		Ai_grad[i] = 0.0f;

	float* Ai_rms = 0;
	cudaMallocManaged((char**)&Ai_rms, this->m_Point_Cloud_size * sizeof(*Ai_rms));
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
		Ai_rms[i] = 0.0f;

	float* Ci_grad = 0;
	cudaMallocManaged((char**)&Ci_grad, this->m_Point_Cloud_size * sizeof(*Ci_grad));
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
		Ci_grad[i] = 0.0f;

	float* Ci_rms = 0;
	cudaMallocManaged((char**)&Ci_rms, this->m_Point_Cloud_size * sizeof(*Ci_rms));
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
		Ci_rms[i] = 0.0f;

	RMSPropOpt rms_prop;

	int threadsPerBlock = 128;

	int mesh_buffer_size = std::max((this->m_Grid_Res_X * this->m_Grid_Res_Y * this->m_Grid_Res_Z) / 2, 1000000);
	float3* c_mid_points_buffer = 0;
	float3* c_mid_normals_buffer = 0;
	cudaMallocManaged((char**)&c_mid_points_buffer, mesh_buffer_size * sizeof(*c_mid_points_buffer));
	cudaMallocManaged((char**)&c_mid_normals_buffer, mesh_buffer_size * sizeof(*c_mid_normals_buffer));

	std::ofstream outFile_points(mesh_path + "test_points.txt");
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		outFile_points << this->m_Point_Cloud_GPU[i].x << " " << this->m_Point_Cloud_GPU[i].y << " " << this->m_Point_Cloud_GPU[i].z << "\n";
	}
	outFile_points.close();

	std::ofstream outFile_norms(mesh_path + "test_normals.txt");
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		outFile_norms << this->m_Normals_GPU[i].x << " " << this->m_Normals_GPU[i].y << " " << this->m_Normals_GPU[i].z << "\n";
	}
	outFile_norms.close();

	float3 lower;
	float3 upper;
	float bbox_dist_threshold = 2.0f;
	int counts_threshold = 1;
	ComRealBBox(this->m_Point_Cloud_GPU, this->m_Cs_GPU, bbox_dist_threshold, counts_threshold, lower, upper, this->m_Point_Cloud_size);
	std::cout << "lower: " << lower.x << " " << lower.y << " " << lower.z << std::endl;
	std::cout << "upper: " << upper.x << " " << upper.y << " " << upper.z << std::endl;

	int sample_reso = diric_reso;

	float model_x_len = upper.x - lower.x;
	float model_y_len = upper.y - lower.y;
	float model_z_len = upper.z - lower.z;

	float box_x_len = model_x_len + 0.1f;
	float box_y_len = model_y_len + 0.1f;
	float box_z_len = model_z_len + 0.1f;
	float box_max_len = fmaxf(box_x_len, box_y_len);
	box_max_len = fmaxf(box_max_len, box_z_len);
	float voxel_len = box_max_len / ((float)sample_reso);
	int voxel_x_size = ceilf(box_x_len / voxel_len);
	int voxel_y_size = ceilf(box_y_len / voxel_len);
	int voxel_z_size = ceilf(box_z_len / voxel_len);
	int sample_grid_size = voxel_x_size * voxel_y_size * voxel_z_size;
	std::cout << "voxel_x_size: " << voxel_x_size << " voxel_y_size: " << voxel_y_size << " voxel_z_size: " << voxel_z_size << std::endl;
	float grad_volume_scale = voxel_len * voxel_len * voxel_len;

	float3* uni_grid_samples_gpu = 0;
	cudaMallocManaged((char**)&uni_grid_samples_gpu, sample_grid_size * sizeof(*uni_grid_samples_gpu));

	float box_x_s = lower.x - 0.5f * (box_x_len - model_x_len);
	float box_y_s = lower.y - 0.5f * (box_y_len - model_y_len);
	float box_z_s = lower.z - 0.5f * (box_z_len - model_z_len);
	std::unique_ptr<float3[]> grid_samples_cpu = std::make_unique<float3[]>(size_t(sample_grid_size));
	int grid_count = 0;
	for (int i = 0; i < voxel_x_size; ++i)
	{
		for (int j = 0; j < voxel_y_size; ++j)
		{
			for (int k = 0; k < voxel_z_size; ++k)
			{
				float x = box_x_s + i * voxel_len;
				float y = box_y_s + j * voxel_len;
				float z = box_z_s + k * voxel_len;

				grid_samples_cpu[grid_count++] = make_float3(x, y, z);
			}
		}
	}
	gpuErrchk(cudaMemcpy(uni_grid_samples_gpu, grid_samples_cpu.get(), sample_grid_size * sizeof(float3), cudaMemcpyHostToDevice));

	float model_max_len = fmaxf(model_x_len, model_y_len);
	model_max_len = fmaxf(model_max_len, model_z_len);
	float model_min_len = fminf(model_x_len, model_y_len);
	model_min_len = fminf(model_min_len, model_z_len);

	const int blocksPerGrid = (this->m_Point_Cloud_size + threadsPerBlock - 1) / threadsPerBlock;

	float dist_thres = 3.0e-2f  ;
	std::vector<bool> uni_grid_samples_is_near;
	uni_grid_samples_is_near.resize(sample_grid_size);
	float* distances_gpu = 0;
	cudaMallocManaged((void**)&distances_gpu, sample_grid_size * sizeof(*distances_gpu));
	int sample_points_size = SetSamplings(uni_grid_samples_gpu, distances_gpu, uni_grid_samples_is_near, sample_grid_size, dist_thres, beta, sigma, use_radii, loop);

	float3* sample_points_gpu = 0;
	cudaMallocManaged((char**)&sample_points_gpu, sample_points_size * sizeof(*sample_points_gpu));

	float* sampling_distances = 0;
	cudaMallocManaged((char**)&sampling_distances, sample_points_size * sizeof(*sampling_distances));

	int sample_count = 0;
	for (int i = 0; i < sample_grid_size; i++)
	{
		if (!uni_grid_samples_is_near[i])
		{
			sample_points_gpu[sample_count].x = uni_grid_samples_gpu[i].x;
			sample_points_gpu[sample_count].y = uni_grid_samples_gpu[i].y;
			sample_points_gpu[sample_count].z = uni_grid_samples_gpu[i].z;

			sampling_distances[sample_count] = distances_gpu[i];

			sample_count++;
		}
	}

	std::cout << "voxel_len " << voxel_len << std::endl;
	float* grad_dirichlet_scale = 0;
	cudaMallocManaged((char**)&grad_dirichlet_scale, sample_points_size * sizeof(*grad_dirichlet_scale));
	ComputeVoxelWeight(dist_thres, voxel_len, sampling_distances, grad_dirichlet_scale, sample_points_size);

	float* grad_dirichlet_scale_a = 0;
	cudaMallocManaged((char**)&grad_dirichlet_scale_a, sample_points_size * sizeof(*grad_dirichlet_scale_a));
	for (int i = 0; i < sample_points_size; i++)
		grad_dirichlet_scale_a[i] = grad_dirichlet_scale[i] * diric_scale;

	float* grad_dirichlet_scale_c = 0;
	cudaMallocManaged((char**)&grad_dirichlet_scale_c, sample_points_size * sizeof(*grad_dirichlet_scale_c));
	for (int i = 0; i < sample_points_size; i++)
		grad_dirichlet_scale_c[i] = grad_dirichlet_scale[i] * diric_scale;

	float3* gwn_grad_sample = 0;
	cudaMallocManaged((char**)&gwn_grad_sample, sample_points_size * sizeof(*gwn_grad_sample));

	turbogwn->UpdateGWNAllInfo(this->m_Normals_GPU, this->m_Point_Cloud_GPU, this->m_As_GPU, this->m_Cs_GPU);

	turbogwn->TurboLibiScreenedGWNCsLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);
	thrust::device_ptr<float> pc_wn_thrust_ptr_ori(this->m_PC_WN_GPU);
	float pc_wn_sum_ori = thrust::reduce(pc_wn_thrust_ptr_ori, pc_wn_thrust_ptr_ori + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
	float mean_wn_ori = pc_wn_sum_ori / ((float)this->m_Point_Cloud_size);

	float3* inside_points = 0;
	int inside_size = 1;
	cudaMallocManaged((char**)&inside_points, inside_size * sizeof(*inside_points));
	inside_points[0].x = inside_x; inside_points[0].y = inside_y; inside_points[0].z = inside_z;

	float3* outside_points = 0;
	int outside_size = 1;
	cudaMallocManaged((char**)&outside_points, outside_size * sizeof(*outside_points));
	outside_points[0].x = this->m_Bound_Upper_KDT_GPU[0].x + model_max_len * 0.05f;
	outside_points[0].y = this->m_Bound_Upper_KDT_GPU[0].y + model_max_len * 0.05f;
	outside_points[0].z = this->m_Bound_Upper_KDT_GPU[0].z + model_max_len * 0.05f;

	float* inside_value = 0;
	cudaMallocManaged((char**)&inside_value, inside_size * sizeof(*inside_value));
	inside_value[0] = (mean_wn_ori < 0.0f) ? -1.0f : 1.0f;
	float* outside_value;
	cudaMallocManaged((char**)&outside_value, outside_size * sizeof(*outside_value));
	for (int i = 0; i < outside_size; i++)
		outside_value[i] = 0.0f;

	float* inside_gwn = 0;
	cudaMallocManaged((char**)&inside_gwn, inside_size * sizeof(*inside_gwn));
	float* outside_gwn = 0;
	cudaMallocManaged((char**)&outside_gwn, outside_size * sizeof(*outside_gwn));

	turbogwn->TurboLibiScreenedGWNCsLaunch(this->m_Grid_Points_GPU, this->m_Point_Cloud_GPU, this->m_Grid_Points_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma, this->m_GP_WN_GPU, use_radii);
	std::string mesh_full_path = "nothing";
	TurboMC turbomc;
	int num_tri = turbomc.TurboMarchingCubes(this->m_GP_WN_GPU, this->m_Grid_Points_GPU, this->m_Grid_Res_X, this->m_Grid_Res_Y, this->m_Grid_Res_Z, mean_wn_ori, 5000, mesh_path, mesh_full_path, c_mid_points_buffer, c_mid_normals_buffer, 10);

	float* last_Ci = 0;

	float* dirichlet_loss = 0;
	cudaMallocManaged((char**)&dirichlet_loss, sizeof(*dirichlet_loss));
	float* target_loss = 0;
	cudaMallocManaged((char**)&target_loss, sizeof(*target_loss));

	int normal_mode = 1;
	if (normal_mode == 1)
	{
		cartesian2SphericalKernel << <blocksPerGrid, threadsPerBlock >> > (this->m_Normals_GPU, this->m_nUi_GPU, this->m_nVi_GPU, this->m_Point_Cloud_size);
		cudaDeviceSynchronize();
	}

	bool opt_Ai = false;
	bool opt_Ci = false;

	float sigma_0 = 0.0f;

	float learning_rate_Ai_c = learning_rate_Ai;
	if (loop >= 2)
	{
		if (learning_rate_Ai_2 < 0.0f)
			learning_rate_Ai_c = learning_rate_Ai * 0.1f;
		else
			learning_rate_Ai_c = learning_rate_Ai_2;
	}

	thrust::device_ptr<float> ci_ptr = thrust::device_pointer_cast(this->m_Cs_GPU);
	thrust::device_ptr<float> ai_ptr = thrust::device_pointer_cast(this->m_As_GPU);

	std::ofstream file(mesh_path + std::to_string(loop) + "_" + std::to_string(5000) + "_test_Ai.txt");
	if (file.is_open())
	{
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
			file << m_As_GPU[i] << "\n";
	}
	file.close();

	std::ofstream fileC(mesh_path + std::to_string(loop) + "_" + std::to_string(5000) + "_test_Ci.txt");
	if (fileC.is_open())
	{
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
			fileC << m_Cs_GPU[i] << "\n";
	}
	fileC.close();

	std::ofstream fileAC(mesh_path + std::to_string(loop) + "_" + std::to_string(5000) + "_test_AiCi.txt");
	if (fileAC.is_open())
	{
		for (int i = 0; i < this->m_Point_Cloud_size; i++)
			fileAC << m_Cs_GPU[i] * m_As_GPU[i] << "\n";
	}
	fileAC.close();

	if (loss_mode == 4)
	{

		int set_loop = 1;
		int top_m = 1;
		FinalizeCi(this->m_PC_WN_GPU, mesh_path, ci_final_in_thres, ci_final_out_thres, set_loop, top_m, this->m_Point_Cloud_size);

		std::ofstream fileC(mesh_path + std::to_string(loop) + "_" + std::to_string(5111) + "_test_Ci.txt");
		if (fileC.is_open())
		{
			for (int i = 0; i < this->m_Point_Cloud_size; i++)
				fileC << m_Cs_GPU[i] << "\n";
		}
		fileC.close();
	}

	std::ofstream ofss(mesh_path + std::to_string(loop) + "_" + std::to_string(5000) + "_test.ply");

	ofss << "ply\n";
	ofss << "format ascii 1.0\n";
	ofss << "element vertex " << this->m_Point_Cloud_size << "\n";
	ofss << "property float x\n";
	ofss << "property float y\n";
	ofss << "property float z\n";
	ofss << "property float nx\n";
	ofss << "property float ny\n";
	ofss << "property float nz\n";
	ofss << "property float quality\n";
	ofss << "end_header\n";

	float max_q = 0.0f;
	float min_q = FLT_MAX;
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		if (m_Cs_GPU[i] * m_As_GPU[i] > max_q)
			max_q = m_Cs_GPU[i] * m_As_GPU[i];
		if (m_Cs_GPU[i] * m_As_GPU[i] < min_q)
			min_q = m_Cs_GPU[i] * m_As_GPU[i];
	}

	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		float c_q = ((m_Cs_GPU[i] * m_As_GPU[i] - min_q) / (max_q - min_q)) * weight_scale + 0.001f;

		ofss << this->m_Point_Cloud_GPU[i].x << " " << this->m_Point_Cloud_GPU[i].y << " " << this->m_Point_Cloud_GPU[i].z << " "
			<< this->m_Normals_GPU[i].x << " " << this->m_Normals_GPU[i].y << " " << this->m_Normals_GPU[i].z << " "
			<< c_q << "\n";
	}
	ofss.close();

	if (loss_mode == 4)
	{
		cudaFree(Ai_grad); Ai_grad = 0;
		cudaFree(Ai_rms); Ai_rms = 0;
		cudaFree(Ci_grad); Ai_grad = 0;
		cudaFree(Ci_rms); Ai_rms = 0;
		cudaFree(c_mid_points_buffer); c_mid_points_buffer = 0;
		cudaFree(c_mid_normals_buffer); c_mid_normals_buffer = 0;
		cudaFree(uni_grid_samples_gpu); uni_grid_samples_gpu = 0;

		cudaFree(distances_gpu); distances_gpu = 0;
		cudaFree(sample_points_gpu); sample_points_gpu = 0;
		cudaFree(sampling_distances); sampling_distances = 0;
		cudaFree(grad_dirichlet_scale); grad_dirichlet_scale = 0;
		cudaFree(grad_dirichlet_scale_a); grad_dirichlet_scale_a = 0;
		cudaFree(grad_dirichlet_scale_c); grad_dirichlet_scale_c = 0;
		cudaFree(gwn_grad_sample); gwn_grad_sample = 0;
		cudaFree(inside_points); inside_points = 0;
		cudaFree(outside_points); outside_points = 0;
		cudaFree(inside_value); inside_value = 0;
		cudaFree(outside_value); outside_value = 0;
		cudaFree(inside_gwn); inside_gwn = 0;
		cudaFree(outside_gwn); outside_gwn = 0;
		cudaFree(dirichlet_loss); dirichlet_loss = 0;

		return;
	}

	float ini_total_area_sum;
	float c_sum = 0.0f;
	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		c_sum += this->m_As_GPU[i] * this->m_Cs_GPU[i];
	}

	ini_total_area_sum = c_sum  ;

	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		this->m_ini_As_GPU[i] = ini_total_area_sum / float(this->m_Point_Cloud_size);
	}

	if (loss_mode == 1)
	{
		InitialCi(1.0f, ci_rad, this->m_PC_WN_GPU, mesh_path, loop, ci_level, ci_in_thres, ci_out_thres);

		std::ofstream fileC(mesh_path + std::to_string(loop) + "_" + std::to_string(5111) + "_test_Ci.txt");
		if (fileC.is_open())
		{
			for (int i = 0; i < this->m_Point_Cloud_size; i++)
				fileC << m_Cs_GPU[i] << "\n";
		}
		fileC.close();

		this->m_Ci_Opted = true;
	}

	turbogwn->AllocateQueryMem(sample_points_size, this->m_Point_Cloud_size);

	float* point_target_gwn = 0;
	cudaMallocManaged((char**)&point_target_gwn, this->m_Point_Cloud_size * sizeof(*point_target_gwn));
	float* grad_Ai_target_scale = 0;
	cudaMallocManaged((char**)&grad_Ai_target_scale, this->m_Point_Cloud_size * sizeof(*grad_Ai_target_scale));

	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{
		point_target_gwn[i] = (mean_wn_ori > 0.0f) ? 0.5f : -0.5f;
	}

	for (int i = 0; i < this->m_Point_Cloud_size; i++)
	{

		if (loss_mode == 0)
			grad_Ai_target_scale[i] = ai_target_start * diric_scale;//3.0//5.0
		else if (loss_mode == 2 || loss_mode == 3)
			grad_Ai_target_scale[i] = ai_target_mid * diric_scale;//0.2
	}

	GetPlaneField(1024, 0, beta, sigma, use_radii, loop, mesh_path);

	std::ofstream file_ld(mesh_path + std::to_string(loop) + "_ld_loss.txt", std::ios::out | std::ios::app);
	std::ofstream file_la(mesh_path + std::to_string(loop) + "_la_loss.txt", std::ios::out | std::ios::app);
	std::ofstream file_lo(mesh_path + std::to_string(loop) + "_lo_loss.txt", std::ios::out | std::ios::app);
	std::ofstream file_li(mesh_path + std::to_string(loop) + "_li_loss.txt", std::ios::out | std::ios::app);
	std::ofstream file_ls(mesh_path + std::to_string(loop) + "_ls_loss.txt", std::ios::out | std::ios::app);

	std::ofstream outFile_sams(mesh_path + std::to_string(loop) + "_test_sams.txt");
	for (int i = 0; i < sample_points_size; i++)
	{
		outFile_sams << sample_points_gpu[i].x << " " << sample_points_gpu[i].y << " " << sample_points_gpu[i].z << "\n";
	}
	outFile_sams.close();

	int c_loop_iteration = (loss_mode == 1) ? ci_max_loop_iteration : loop_iteration;

	std::cout << "Start optimization" << std::endl;
	for (int c_iter = 0; c_iter < c_loop_iteration; c_iter++)
	{

		if (loss_mode != 1)
		{
			opt_Ci = false;
			opt_Ai = true;

			if ((c_iter + 1) % 30 == 0)
			{
				AdjustAi(1.0f, ai_rad, ai_smooth_rad, ai_adjust, loop, loss_mode);
			}
		}
		else
		{
			opt_Ci = true;
			opt_Ai = false;

			if (c_iter == 0)
			{
				cudaMallocManaged((char**)&last_Ci, this->m_Point_Cloud_size * sizeof(*last_Ci));
				for (int i = 0; i < this->m_Point_Cloud_size; i++)
				{
					last_Ci[i] = this->m_Cs_GPU[i];
				}
			}

			if ((c_iter + 1) % 30 == 0)
			{
				thrust::device_ptr<const float> last_Ci_ptr(last_Ci), ci_ptr(this->m_Cs_GPU);

				typedef thrust::tuple<thrust::device_ptr<const float>, thrust::device_ptr<const float>> FloatTuple;
				typedef thrust::zip_iterator<FloatTuple> ZipIter;

				ZipIter begin = thrust::make_zip_iterator(thrust::make_tuple(last_Ci_ptr, ci_ptr));
				ZipIter end = begin + this->m_Point_Cloud_size;
				float vary_sum = thrust::transform_reduce(begin, end, sqdiff_op(), 0.0f, thrust::plus<float>());
				float vary_ratio = vary_sum / float(this->m_Point_Cloud_size);
				std::cout << "Ci vary ratio = " << vary_ratio << std::endl;

				float vary_max = thrust::transform_reduce(begin, end, sqdiff_op(), 0.0f, thrust::maximum<float>());
				std::cout << "Ci vary ratio max = " << vary_max << std::endl;

				if (vary_ratio <= ci_var_thres)
				{
					for (int i = 0; i < this->m_Point_Cloud_size; i++)
					{
						this->m_Cs_GPU[i] = fminf(fmaxf(this->m_Cs_GPU[i], 0.01f), 0.99f);
					}

					break;
				}

				for (int i = 0; i < this->m_Point_Cloud_size; i++)
				{
					last_Ci[i] = this->m_Cs_GPU[i];
				}
			}
		}

		std::cout << "ini_total_area_sum: " << ini_total_area_sum << std::endl;

		turbogwn->UpdateGWNAllInfo(this->m_Normals_GPU, this->m_Point_Cloud_GPU, this->m_As_GPU, this->m_Cs_GPU);

		turbogwn->TurboLibiScreenedGWNCsLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma, this->m_PC_WN_GPU, use_radii);
		float pc_wn_sum_ori = thrust::reduce(pc_wn_thrust_ptr_ori, pc_wn_thrust_ptr_ori + this->m_Point_Cloud_size, 0.0f, thrust::plus<float>());
		float mean_wn_ori = pc_wn_sum_ori / ((float)this->m_Point_Cloud_size);

		turbogwn->TurboLibiScreenedGWNCsLaunch(inside_points, this->m_Point_Cloud_GPU, inside_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma_0, inside_gwn, use_radii);
		turbogwn->TurboLibiScreenedGWNCsLaunch(outside_points, this->m_Point_Cloud_GPU, outside_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU, beta, sigma_0, outside_gwn, use_radii);

		turbogwn->TurboGWNGradientLaunch(sample_points_gpu, this->m_Point_Cloud_GPU, sample_points_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU,
			beta, sigma_0, gwn_grad_sample, use_radii, use_screened);

		if (opt_Ai)
		{
			dirichlet_loss[0] = 0.0f;
			const int blocksPerGrid_sp = (sample_points_size + threadsPerBlock - 1) / threadsPerBlock;

			computeDiriLoss << < blocksPerGrid_sp, threadsPerBlock >> > (gwn_grad_sample, grad_dirichlet_scale, grad_volume_scale, dirichlet_loss, sample_points_size);
			cudaDeviceSynchronize();

			std::cout << "Iter: " << c_iter << " Dirichlet loss: " << dirichlet_loss[0] << std::endl;
			file_ld << dirichlet_loss[0] << std::endl;

			turbogwn->UpdateGWNInfoAsCsGrad(this->m_Normals_GPU, this->m_Point_Cloud_GPU, this->m_As_GPU, this->m_Cs_GPU);

			if (loop >= 0)
			{
				turbogwn->TurboLibiScreenedGWNAsCsBackwardLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU,
					beta, sigma, use_radii, use_screened, point_target_gwn, grad_Ai_target_scale, 0.0f, this->m_PC_WN_GPU, Ai_grad, Ci_grad, true, false, true);

				float grad_scale = grad_Ai_target_scale[0] * (1.0f / ((float)this->m_Point_Cloud_size));
				turbogwn->TurboGWNAsPatchBackwardLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Normals_GPU, this->m_Cs_GPU, this->m_PC_WN_GPU, Ai_grad, point_target_gwn[0], grad_scale, 5000000);

				target_loss[0] = 0.0f;
				float grad_scale_com = 1.0f / ((float)this->m_Point_Cloud_size);
				computeScaleTargetLoss << < blocksPerGrid, threadsPerBlock >> > (this->m_PC_WN_GPU, grad_scale_com, target_loss, point_target_gwn[0], this->m_Point_Cloud_size);
				cudaDeviceSynchronize();

				std::cout << "Iter: " << c_iter << " Target loss: " << target_loss[0] << std::endl;
				file_ls << target_loss[0] << std::endl;
			}

			std::cout << "inside_gwn:" << inside_gwn[0] << std::endl;
			std::cout << "outside_gwn:" << outside_gwn[0] << std::endl;

			if (loop >= 0)
			{

				turbogwn->TurboGWNGradientBackwardLaunch(sample_points_gpu, this->m_Point_Cloud_GPU, sample_points_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU,
					gwn_grad_sample,   Ai_grad, Ci_grad, grad_dirichlet_scale_a, grad_volume_scale, beta, sigma_0, use_radii, use_screened, true, false, false, normal_mode);

				turbogwn->TurboGWNGradientAsPatchBackwardLaunch(sample_points_gpu, this->m_Point_Cloud_GPU, this->m_Normals_GPU, this->m_Cs_GPU, gwn_grad_sample, Ai_grad, grad_dirichlet_scale_a, grad_volume_scale, sample_points_size);

			}

			float c_total_area_sum = thrust::inner_product(ci_ptr, ci_ptr + this->m_Point_Cloud_size, ai_ptr, 0.0f);

			if (loop >= 0)
			{
				if (ini_total_area_sum != c_total_area_sum)
				{
					float grad_As_reg_scale;
					if (!this->m_Ci_Opted)
					{

						grad_As_reg_scale = ai_reg * diric_scale;
					}
					else
					{
						grad_As_reg_scale = ai_reg_2 * diric_scale;
					}

					grad_As_reg_scale = ((c_total_area_sum - ini_total_area_sum) > 0.0f) ? grad_As_reg_scale : -grad_As_reg_scale;

					CiRegKernal << < blocksPerGrid, threadsPerBlock >> > (this->m_Cs_GPU, Ai_grad, grad_As_reg_scale, this->m_Point_Cloud_size);
					cudaDeviceSynchronize();
				}
			}

			file_la << fabs(c_total_area_sum - ini_total_area_sum) << std::endl;
			file_li << (inside_gwn[0] - inside_value[0]) * (inside_gwn[0] - inside_value[0]) << std::endl;
			file_lo << (outside_gwn[0] - outside_value[0]) * (outside_gwn[0] - outside_value[0]) << std::endl;
		}

		if (opt_Ci)
		{
			dirichlet_loss[0] = 0.0f;
			const int blocksPerGrid_sp = (sample_points_size + threadsPerBlock - 1) / threadsPerBlock;
			computeScaleLoss << < blocksPerGrid_sp, threadsPerBlock >> > (gwn_grad_sample, grad_dirichlet_scale, dirichlet_loss, sample_points_size);
			cudaDeviceSynchronize();

			float c_loss = dirichlet_loss[0] / ((float)sample_points_size);
			std::cout << "Iter: " << c_iter << " Dirichlet loss: " << c_loss << std::endl;

			turbogwn->UpdateGWNInfoAsCsGrad(this->m_Normals_GPU, this->m_Point_Cloud_GPU, this->m_As_GPU, this->m_Cs_GPU);

			float grad_Ci_target_scale = ci_target * diric_scale;
			turbogwn->TurboLibiScreenedGWNAsCsBackwardLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Point_Cloud_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU,
				beta, sigma, use_radii, use_screened, point_target_gwn, grad_Ai_target_scale, grad_Ci_target_scale, this->m_PC_WN_GPU, Ai_grad, Ci_grad, false, true, true);

			float grad_scale = grad_Ci_target_scale * (1.0f / ((float)this->m_Point_Cloud_size));
			turbogwn->TurboGWNCsPatchBackwardLaunch(this->m_Point_Cloud_GPU, this->m_Point_Cloud_GPU, this->m_Normals_GPU, this->m_As_GPU, this->m_PC_WN_GPU, Ci_grad, point_target_gwn[0], grad_scale, 5000000);

			std::cout << "inside_gwn:" << inside_gwn[0] << std::endl;
			std::cout << "outside_gwn:" << outside_gwn[0] << std::endl;

			turbogwn->TurboGWNGradientBackwardLaunch(sample_points_gpu, this->m_Point_Cloud_GPU, sample_points_size, this->m_Normals_GPU, this->m_As_GPU, this->m_Cs_GPU, this->m_Radii_GPU,
				gwn_grad_sample,   Ai_grad, Ci_grad, grad_dirichlet_scale_c, grad_volume_scale, beta, sigma_0, use_radii, use_screened, false, true, false, normal_mode);

			turbogwn->TurboGWNGradientCsPatchBackwardLaunch(sample_points_gpu, this->m_Point_Cloud_GPU, this->m_Normals_GPU, this->m_As_GPU, gwn_grad_sample, Ci_grad, grad_dirichlet_scale_c, grad_volume_scale, sample_points_size);

			int threadsPerBlock = 128;
			int blocksPerGrid = (this->m_Point_Cloud_size + threadsPerBlock - 1) / threadsPerBlock;

			float grad_Ci_reg_scale = ci_reg_c * diric_scale;
			CiCrossRegKernal << < blocksPerGrid, threadsPerBlock >> > (this->m_Cs_GPU, Ci_grad, grad_Ci_reg_scale, this->m_Point_Cloud_size);
			cudaDeviceSynchronize();

			float grad_Ci_reg_scale_area = ci_reg_a * diric_scale;
			float c_total_area_sum = thrust::inner_product(ci_ptr, ci_ptr + this->m_Point_Cloud_size, ai_ptr, 0.0f);
			if (ini_total_area_sum != c_total_area_sum)
			{
				int threadsPerBlock = 128;
				int blocksPerGrid = (this->m_Point_Cloud_size + threadsPerBlock - 1) / threadsPerBlock;
				grad_Ci_reg_scale_area = ((c_total_area_sum - ini_total_area_sum) > 0.0f) ? grad_Ci_reg_scale_area : -grad_Ci_reg_scale_area;
				CiRegKernal << < blocksPerGrid, threadsPerBlock >> > (this->m_As_GPU, Ci_grad, grad_Ci_reg_scale_area, this->m_Point_Cloud_size);
				cudaDeviceSynchronize();
			}
		}

		if (opt_Ai)
		{
			rms_prop.RMSPropUpdate(this->m_As_GPU, Ai_rms, Ai_grad, 0.9f, learning_rate_Ai_c, 1e-8, -1e-9, this->m_Point_Cloud_size);

			AsClipKernel << < blocksPerGrid, threadsPerBlock >> > (this->m_As_GPU, this->m_ini_As_GPU, this->m_Cs_GPU, this->m_Point_Cloud_size);
			cudaDeviceSynchronize();
		}

		if (opt_Ci)
		{
			rms_prop.RMSPropUpdate(this->m_Cs_GPU, Ci_rms, Ci_grad, 0.9f, learning_rate_Ci, 1e-8, -1e-9, this->m_Point_Cloud_size);

			CiClipKernel << < blocksPerGrid, threadsPerBlock >> > (this->m_Cs_GPU, this->m_Point_Cloud_size, ((c_iter + 1) % 30 == 0));
			cudaDeviceSynchronize();
		}

	}

	cudaFree(Ai_grad); Ai_grad = 0;
	cudaFree(Ai_rms); Ai_rms = 0;
	cudaFree(Ci_grad); Ai_grad = 0;
	cudaFree(Ci_rms); Ai_rms = 0;
	cudaFree(c_mid_points_buffer); c_mid_points_buffer = 0;
	cudaFree(c_mid_normals_buffer); c_mid_normals_buffer = 0;
	cudaFree(uni_grid_samples_gpu); uni_grid_samples_gpu = 0;

	cudaFree(distances_gpu); distances_gpu = 0;
	cudaFree(sample_points_gpu); sample_points_gpu = 0;
	cudaFree(sampling_distances); sampling_distances = 0;
	cudaFree(grad_dirichlet_scale); grad_dirichlet_scale = 0;
	cudaFree(grad_dirichlet_scale_a); grad_dirichlet_scale_a = 0;
	cudaFree(grad_dirichlet_scale_c); grad_dirichlet_scale_c = 0;
	cudaFree(gwn_grad_sample); gwn_grad_sample = 0;
	cudaFree(inside_points); inside_points = 0;
	cudaFree(outside_points); outside_points = 0;
	cudaFree(inside_value); inside_value = 0;
	cudaFree(outside_value); outside_value = 0;
	cudaFree(inside_gwn); inside_gwn = 0;
	cudaFree(outside_gwn); outside_gwn = 0;

	cudaFree(dirichlet_loss); dirichlet_loss = 0;
}

void TurboDWG::ComputeVoxelWeight(float threshold, float voxel_length, float* distances, float* weight, float size)
{
	float scale = 1.0f / voxel_length;
	float radius = threshold * scale;

	float va_s = sqrtf(radius * radius - 0.5f);
	float va_00 = 1.0f / 12.0f - radius * radius;
	float va_01 = atanf(0.5f * va_s / (0.5f - radius * radius));
	float va_1 = va_s / 3.0f;
	float va_20 = -va_00;
	float va_21 = atanf(0.5f / va_s);
	float va_30 = 4.0f / 3.0f * radius * radius * radius;
	float va_31 = atanf(0.25f / (radius * va_s));
	float va = va_00 * va_01 + va_1 + va_20 * va_21 - va_30 * va_31;

	for (int i = 0; i < size; i++)
	{
		float distance_p2s = scale * distances[i] - radius;

		weight[i] = 1.0f;
	}
}

void TurboDWG::PreProcessOutliers(float dist_threshold, std::vector<float>& point_x, std::vector<float>& point_y, std::vector<float>& point_z,
	std::vector<float>& normal_x, std::vector<float>& normal_y, std::vector<float>& normal_z,
	std::vector<float>& out_point_x, std::vector<float>& out_point_y, std::vector<float>& out_point_z,
	std::vector<float>& out_normal_x, std::vector<float>& out_normal_y, std::vector<float>& out_normal_z)
{
	int point_size = point_x.size();
	float3* points = 0;
	cudaMallocManaged((char**)&points, point_size * sizeof(*points));
	for (int i = 0; i < point_size; i++)
	{
		points[i].x = point_x[i]; points[i].y = point_y[i]; points[i].z = point_z[i];
	}

	float* distances = 0;
	cudaMallocManaged((char**)&distances, point_size * sizeof(*distances));

	float* valid = 0;
	cudaMallocManaged((char**)&valid, point_size * sizeof(*valid));

	int threadsPerBlock = 128;
	int blocksPerGrid = (point_size + threadsPerBlock - 1) / threadsPerBlock;
	comP2Pdistance << < blocksPerGrid, threadsPerBlock >> > (points, distances,   point_size);
	cudaDeviceSynchronize();

	float mean_dist = 0.0f;
	float variance = 0.0f;
	for (int i = 0; i < point_size; i++)
	{
		mean_dist += distances[i];
		variance += distances[i] * distances[i];
	}
	mean_dist /= float(point_size);
	variance /= float(point_size);
	variance -= (mean_dist * mean_dist);
	float stddev = sqrtf(variance);

	for (int i = 0; i < point_size; i++)
	{

		if (distances[i] < mean_dist + dist_threshold * stddev)
		{
			out_point_x.push_back(point_x[i]);
			out_point_y.push_back(point_y[i]);
			out_point_z.push_back(point_z[i]);

			out_normal_x.push_back(normal_x[i]);
			out_normal_y.push_back(normal_y[i]);
			out_normal_z.push_back(normal_z[i]);

			valid[i] = 1.0f;
		}
		else
			valid[i] = 0.0f;
	}

	std::ofstream file_Ci("E:/dwg++/mesh/0522/test_dist.txt");
	if (file_Ci.is_open())
	{
		for (int i = 0; i < point_size; i++)
			file_Ci << valid[i] << "\n";
	}
	file_Ci.close();

	std::ofstream file_pi("E:/dwg++/mesh/0522/test_points.txt");
	if (file_pi.is_open())
	{
		for (int i = 0; i < point_size; i++)
			file_pi << points[i].x << " " << points[i].y << " " << points[i].z << "\n";
	}
	file_pi.close();

	std::ofstream file_ni("E:/dwg++/mesh/0522/0_test_normals.txt");
	if (file_ni.is_open())
	{
		for (int i = 0; i < point_size; i++)
			file_ni << normal_x[i] << " " << normal_y[i] << " " << normal_z[i] << "\n";
	}
	file_ni.close();

	cudaFree(points); points = 0;
	cudaFree(distances); distances = 0;
	cudaFree(valid); valid = 0;
}

void TurboDWG::OutlierFilter(float* As, float threshold, int point_num)
{
	thrust::device_vector<int> d_indices(point_num);
	thrust::sequence(d_indices.begin(), d_indices.end());

	thrust::device_vector<float> d_data(As, As + point_num);

	thrust::sort_by_key(d_data.begin(), d_data.end(), d_indices.begin(), thrust::greater<float>());

	int threshold_max_index = ceil((float)point_num * threshold);

	int threadsPerBlock = 128;
	int blocksPerGrid = (threshold_max_index + threadsPerBlock - 1) / threadsPerBlock;

	int* raw_indices = thrust::raw_pointer_cast(d_indices.data());

	AsFiltertoZero << < blocksPerGrid, threadsPerBlock >> > (As, raw_indices, threshold_max_index);
	cudaDeviceSynchronize();
}
