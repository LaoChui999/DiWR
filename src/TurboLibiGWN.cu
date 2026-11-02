#include "Turbo_GWN_l.h"

#define gpuErrchk(ans) { gpuAssert((ans), #ans, __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* command, const char* file, int line, bool abort = true)
{

	if (code != cudaSuccess)
	{
		fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
		if (abort) exit(code);
	}
}

inline dim3 computeNbBlocks(unsigned int nbThreads, unsigned int nbThreadsPerBlock) {

	dim3 nbBlocks = ceil(((float)(nbThreads)+(float)nbThreadsPerBlock - 1.0f) / (float)nbThreadsPerBlock);
	return nbBlocks;
}

void TurboLibiGWN::PreLibiGWN(std::vector<std::vector<int>>& O_PI, Eigen::MatrixXi& O_CH, Eigen::MatrixXd& O_CM, Eigen::VectorXd& O_R, Eigen::MatrixXd& O_EC)
{
	gpuErrchk(cudaPeekAtLastError());

	unsigned int node_num = O_R.rows();

	std::cout << "node_num: " << node_num << std::endl;

	this->m_Node_Size = node_num;
	cudaMallocManaged((char**)&this->m_Octree_GPU, node_num * sizeof(*this->m_Octree_GPU));

	OctreeNodeLibiGPU* octree_cpu = new OctreeNodeLibiGPU[node_num];

	int child_num = 8;

	uint32_t num_indices_to_copy = 0;
	std::vector<unsigned int> whose_indices_to_copy;

	for (int i = 0; i < O_R.rows(); i++)
	{
		octree_cpu[i].Mass_Center.x = O_CM(i, 0);
		octree_cpu[i].Mass_Center.y = O_CM(i, 1);
		octree_cpu[i].Mass_Center.z = O_CM(i, 2);

		octree_cpu[i].Mass_Normal.x = O_EC(i, 0);
		octree_cpu[i].Mass_Normal.y = O_EC(i, 1);
		octree_cpu[i].Mass_Normal.z = O_EC(i, 2);

		octree_cpu[i].Max_R = O_R[i];
	}

	octree_cpu[0].Father = -1;
	for (int i = 0; i < O_CH.rows(); i++)
	{
		for (int j = 0; j < child_num; j++)
		{
			int c_ci = O_CH(i, j);

			if (c_ci != -1)
			{
				octree_cpu[c_ci].Father = i;
				octree_cpu[c_ci].IndexInFather = j;
			}
		}

		octree_cpu[i].Point_Cloud_Num = O_PI[i].size();

		if (octree_cpu[i].Point_Cloud_Num != 0)
		{
			whose_indices_to_copy.push_back(i);
			num_indices_to_copy += octree_cpu[i].Point_Cloud_Num;
		}
	}

	if (num_indices_to_copy > 0)
	{
		unsigned int* point_indices = nullptr;
		cudaMallocManaged((char**)&point_indices, num_indices_to_copy * sizeof(*point_indices));

		uint32_t offset = 0;
		for (unsigned int i : whose_indices_to_copy)
		{
			octree_cpu[i].Point_Cloud_Indices = point_indices + offset;
			offset += octree_cpu[i].Point_Cloud_Num;
		}

		std::unique_ptr<unsigned int[]> idxCPU = std::make_unique<unsigned int[]>(num_indices_to_copy * sizeof(unsigned int));
		unsigned int* idx = idxCPU.get();
		for (unsigned int i : whose_indices_to_copy)
		{
			for (unsigned int j = 0; j < octree_cpu[i].Point_Cloud_Num; j++)
				*(idx++) = O_PI[i][j];
		}
		gpuErrchk(cudaMemcpy(point_indices, idxCPU.get(), num_indices_to_copy * sizeof(unsigned int), cudaMemcpyHostToDevice));

		float* as_grad_device_x = nullptr;
		cudaMallocManaged((char**)&as_grad_device_x, num_indices_to_copy * sizeof(*as_grad_device_x));

		offset = 0;
		for (unsigned int i : whose_indices_to_copy)
		{
			octree_cpu[i].As_Grad_x = as_grad_device_x + offset;
			offset += octree_cpu[i].Point_Cloud_Num;
		}

		std::unique_ptr<float[]> as_grad_host_x = std::make_unique<float[]>(num_indices_to_copy * sizeof(float));
		float* as_grad_x = as_grad_host_x.get();
		for (unsigned int i : whose_indices_to_copy)
		{
			for (unsigned int j = 0; j < octree_cpu[i].Point_Cloud_Num; j++)
				*(as_grad_x++) = 0.0f;
		}
		gpuErrchk(cudaMemcpy(as_grad_device_x, as_grad_host_x.get(), num_indices_to_copy * sizeof(float), cudaMemcpyHostToDevice));

		float* as_grad_device_y = nullptr;
		cudaMallocManaged((char**)&as_grad_device_y, num_indices_to_copy * sizeof(*as_grad_device_y));

		offset = 0;
		for (unsigned int i : whose_indices_to_copy)
		{
			octree_cpu[i].As_Grad_y = as_grad_device_y + offset;
			offset += octree_cpu[i].Point_Cloud_Num;
		}

		std::unique_ptr<float[]> as_grad_host_y = std::make_unique<float[]>(num_indices_to_copy * sizeof(float));
		float* as_grad_y = as_grad_host_y.get();
		for (unsigned int i : whose_indices_to_copy)
		{
			for (unsigned int j = 0; j < octree_cpu[i].Point_Cloud_Num; j++)
				*(as_grad_y++) = 0.0f;
		}
		gpuErrchk(cudaMemcpy(as_grad_device_y, as_grad_host_y.get(), num_indices_to_copy * sizeof(float), cudaMemcpyHostToDevice));

		float* as_grad_device_z = nullptr;
		cudaMallocManaged((char**)&as_grad_device_z, num_indices_to_copy * sizeof(*as_grad_device_z));

		offset = 0;
		for (unsigned int i : whose_indices_to_copy)
		{
			octree_cpu[i].As_Grad_z = as_grad_device_z + offset;
			offset += octree_cpu[i].Point_Cloud_Num;
		}

		std::unique_ptr<float[]> as_grad_host_z = std::make_unique<float[]>(num_indices_to_copy * sizeof(float));
		float* as_grad_z = as_grad_host_z.get();
		for (unsigned int i : whose_indices_to_copy)
		{
			for (unsigned int j = 0; j < octree_cpu[i].Point_Cloud_Num; j++)
				*(as_grad_z++) = 0.0f;
		}
		gpuErrchk(cudaMemcpy(as_grad_device_z, as_grad_host_z.get(), num_indices_to_copy * sizeof(float), cudaMemcpyHostToDevice));

		float* cs_grad_device_x = nullptr;
		cudaMallocManaged((char**)&cs_grad_device_x, num_indices_to_copy * sizeof(*cs_grad_device_x));

		offset = 0;
		for (unsigned int i : whose_indices_to_copy)
		{
			octree_cpu[i].Cs_Grad_x = cs_grad_device_x + offset;
			offset += octree_cpu[i].Point_Cloud_Num;
		}

		std::unique_ptr<float[]> cs_grad_host_x = std::make_unique<float[]>(num_indices_to_copy * sizeof(float));
		float* cs_grad_x = cs_grad_host_x.get();
		for (unsigned int i : whose_indices_to_copy)
		{
			for (unsigned int j = 0; j < octree_cpu[i].Point_Cloud_Num; j++)
				*(cs_grad_x++) = 0.0f;
		}
		gpuErrchk(cudaMemcpy(cs_grad_device_x, cs_grad_host_x.get(), num_indices_to_copy * sizeof(float), cudaMemcpyHostToDevice));

		float* cs_grad_device_y = nullptr;
		cudaMallocManaged((char**)&cs_grad_device_y, num_indices_to_copy * sizeof(*cs_grad_device_y));

		offset = 0;
		for (unsigned int i : whose_indices_to_copy)
		{
			octree_cpu[i].Cs_Grad_y = cs_grad_device_y + offset;
			offset += octree_cpu[i].Point_Cloud_Num;
		}

		std::unique_ptr<float[]> cs_grad_host_y = std::make_unique<float[]>(num_indices_to_copy * sizeof(float));
		float* cs_grad_y = cs_grad_host_y.get();
		for (unsigned int i : whose_indices_to_copy)
		{
			for (unsigned int j = 0; j < octree_cpu[i].Point_Cloud_Num; j++)
				*(cs_grad_y++) = 0.0f;
		}
		gpuErrchk(cudaMemcpy(cs_grad_device_y, cs_grad_host_y.get(), num_indices_to_copy * sizeof(float), cudaMemcpyHostToDevice));

		float* cs_grad_device_z = nullptr;
		cudaMallocManaged((char**)&cs_grad_device_z, num_indices_to_copy * sizeof(*cs_grad_device_z));

		offset = 0;
		for (unsigned int i : whose_indices_to_copy)
		{
			octree_cpu[i].Cs_Grad_z = cs_grad_device_z + offset;
			offset += octree_cpu[i].Point_Cloud_Num;
		}

		std::unique_ptr<float[]> cs_grad_host_z = std::make_unique<float[]>(num_indices_to_copy * sizeof(float));
		float* cs_grad_z = cs_grad_host_z.get();
		for (unsigned int i : whose_indices_to_copy)
		{
			for (unsigned int j = 0; j < octree_cpu[i].Point_Cloud_Num; j++)
				*(cs_grad_z++) = 0.0f;
		}
		gpuErrchk(cudaMemcpy(cs_grad_device_z, cs_grad_host_z.get(), num_indices_to_copy * sizeof(float), cudaMemcpyHostToDevice));

	}

	int* node_indices = nullptr;
	int total_node_num = ((int)O_CH.rows()) * child_num;
	cudaMallocManaged((char**)&node_indices, total_node_num * sizeof(*node_indices));

	uint32_t offset = 0;
	for (int i = 0; i < O_CH.rows(); i++)
	{
		octree_cpu[i].Children = node_indices + offset;
		offset += child_num;
	}

	std::unique_ptr<int[]> idxCPU = std::make_unique<int[]>(total_node_num * sizeof(int));
	int* idx = idxCPU.get();
	for (int i = 0; i < O_CH.rows(); i++)
	{
		for (int j = 0; j < 8; j++)
			*(idx++) = O_CH(i, j);
	}
	gpuErrchk(cudaMemcpy(node_indices, idxCPU.get(), total_node_num * sizeof(int), cudaMemcpyHostToDevice));

	gpuErrchk(cudaMemcpy(this->m_Octree_GPU, octree_cpu, node_num * sizeof(OctreeNodeLibiGPU), cudaMemcpyHostToDevice));

	this->m_Octree_CPU = octree_cpu;
}

__device__
float GWNLibiPStoQ(OctreeNodeLibiGPU* root_node, float3& q, float3* i_points, float3* i_Normals, float* i_As, float beta)
{
	float gwn = 0.0;
	bool continue_traversal = true;

	int iter_num = 0;
	int points_num = 0;

	int traversal_index = 0;

	int current_node_index = 0;

	float PI_4 = 4.0f * 3.14159265358979323;

	while (continue_traversal)
	{
		if (root_node[current_node_index].Children[0] == -1)
		{
			for (int i_point_in_cn = 0; i_point_in_cn < root_node[current_node_index].Point_Cloud_Num; i_point_in_cn++)
			{
				int p_idx = root_node[current_node_index].Point_Cloud_Indices[i_point_in_cn];
				float3& p = i_points[p_idx];
				float3& N = i_Normals[p_idx];
				float _A = i_As[p_idx];

				float3 _R;
				_R.x = p.x - q.x; _R.y = p.y - q.y; _R.z = p.z - q.z;
				float _R_norm = sqrtf(_R.x * _R.x + _R.y * _R.y + _R.z * _R.z);
				float dot_RN = _R.x * N.x + _R.y * N.y + _R.z * N.z;
				dot_RN *= _A;
				dot_RN /= (PI_4 * _R_norm * _R_norm * _R_norm + 1e-7);
				gwn += dot_RN;

				points_num++;
			}

			while (true)
			{
				bool is_last_child = true;

				if (root_node[current_node_index].IndexInFather == 7)
					is_last_child = true;
				else if (root_node[current_node_index].Father == -1)
				{
					continue_traversal = false;
					break;
				}
				else
				{
					traversal_index = root_node[current_node_index].IndexInFather + 1;
					current_node_index = root_node[current_node_index].Father;
					break;
				}
				if (root_node[current_node_index].Father != -1 && is_last_child)
				{
					current_node_index = root_node[current_node_index].Father;

					if (root_node[current_node_index].Father == -1)
					{
						continue_traversal = false;
						break;
					}
				}
				else
				{
					break;
				}
			}

		}
		else
		{

			bool descend = false;
			for (int c = traversal_index; c < 8; c++)
			{
				int cn_child = root_node[current_node_index].Children[c];
				if (root_node[cn_child].Point_Cloud_Num > 0)
				{
					float3& _massCenter = root_node[cn_child].Mass_Center;
					float _max_R = root_node[cn_child].Max_R;
					float3 MCToq;
					MCToq.x = _massCenter.x - q.x; MCToq.y = _massCenter.y - q.y; MCToq.z = _massCenter.z - q.z;
					float mctop_norm = sqrtf(MCToq.x * MCToq.x + MCToq.y * MCToq.y + MCToq.z * MCToq.z);

					if (mctop_norm > beta * _max_R)
					{
						if (root_node[cn_child].Children[0] == -1)
						{
							for (int i_point_in_cnc = 0; i_point_in_cnc < root_node[cn_child].Point_Cloud_Num; i_point_in_cnc++)
							{
								int p_idx = root_node[cn_child].Point_Cloud_Indices[i_point_in_cnc];
								float3& p = i_points[p_idx];
								float3& N = i_Normals[p_idx];
								float _A = i_As[p_idx];

								float3 _R;
								_R.x = p.x - q.x; _R.y = p.y - q.y; _R.z = p.z - q.z;
								float _R_norm = sqrtf(_R.x * _R.x + _R.y * _R.y + _R.z * _R.z);
								float dot_RN = _R.x * N.x + _R.y * N.y + _R.z * N.z;
								dot_RN *= _A;
								dot_RN /= (PI_4 * _R_norm * _R_norm * _R_norm + 1e-7);
								gwn += dot_RN;

								points_num++;
							}
						}
						else
						{
							float3& _massNormal = root_node[cn_child].Mass_Normal;

							float dot_RN = MCToq.x * _massNormal.x + MCToq.y * _massNormal.y + MCToq.z * _massNormal.z;

							dot_RN /= (PI_4 * mctop_norm * mctop_norm * mctop_norm + 1e-7);
							gwn += dot_RN;

							points_num++;
						}
					}
					else
					{
						current_node_index = cn_child;
						descend = true;
						break;
					}
				}
			}

			traversal_index = 0;

			if (!descend)
			{
				while (true)
				{
					bool is_last_child = true;

					if (root_node[current_node_index].IndexInFather == 7)
						is_last_child = true;
					else if (root_node[current_node_index].Father == -1)
					{
						continue_traversal = false;
						break;
					}
					else
					{
						traversal_index = root_node[current_node_index].IndexInFather + 1;
						current_node_index = root_node[current_node_index].Father;
						break;
					}
					if (root_node[current_node_index].Father != -1 && is_last_child)
					{
						current_node_index = root_node[current_node_index].Father;

						if (root_node[current_node_index].Father == -1)
						{
							continue_traversal = false;
							break;
						}
					}
					else
					{
						break;
					}
				}
			}
		}

		iter_num++;
	}

	return gwn;
}

__device__
float ScreenedGWNLibiPStoQ(OctreeNodeLibiGPU* __restrict__ root_node, float3& q, float3* __restrict__ i_points, float3* __restrict__ i_Normals, float* __restrict__ i_As, float* __restrict__ i_radii, float beta, float sigma, bool use_radii)
{
	float gwn = 0.0;
	bool continue_traversal = true;

	int iter_num = 0;
	int points_num = 0;

	int traversal_index = 0;

	int current_node_index = 0;

	float PI_4 = 4.0f * 3.14159265358979323;

	while (continue_traversal)
	{
		if (root_node[current_node_index].Children[0] == -1)
		{

			for (int i_point_in_cn = 0; i_point_in_cn < root_node[current_node_index].Point_Cloud_Num; i_point_in_cn++)
			{
				int p_idx = root_node[current_node_index].Point_Cloud_Indices[i_point_in_cn];
				float3& p = i_points[p_idx];
				float3& N = i_Normals[p_idx];
				float _A = i_As[p_idx];

				float3 _R;
				_R.x = p.x - q.x; _R.y = p.y - q.y; _R.z = p.z - q.z;
				float _R_norm = sqrtf(_R.x * _R.x + _R.y * _R.y + _R.z * _R.z);

				if (_R_norm > 1e-7)
				{
					float dot_RN = _R.x * N.x + _R.y * N.y + _R.z * N.z;
					dot_RN *= _A;

					float screened = (use_radii) ? i_radii[p_idx] : 0.0015f;
					bool sign = (dot_RN > 0.0f) ? true : false;

					if (screened > _R_norm)
					{
						float up = (screened * sigma + 1.0f) * dot_RN;
						float down = expf(screened * sigma) * (PI_4 * screened * screened * screened) + 1e-7;
						float res = up / down;

						gwn += res;

					}
					else
					{
						float up = (_R_norm * sigma + 1.0f) * dot_RN;
						float down = expf(_R_norm * sigma) * (PI_4 * _R_norm * _R_norm * _R_norm) + 1e-7;
						float res = up / down;

						gwn += res;

					}
				}

				points_num++;
			}

			while (true)
			{
				bool is_last_child = true;

				if (root_node[current_node_index].IndexInFather == 7)
					is_last_child = true;
				else if (root_node[current_node_index].Father == -1)
				{
					continue_traversal = false;
					break;
				}
				else
				{
					traversal_index = root_node[current_node_index].IndexInFather + 1;
					current_node_index = root_node[current_node_index].Father;
					break;
				}
				if (root_node[current_node_index].Father != -1 && is_last_child)
				{
					current_node_index = root_node[current_node_index].Father;

					if (root_node[current_node_index].Father == -1)
					{
						continue_traversal = false;
						break;
					}
				}
				else
				{
					break;
				}
			}

		}
		else
		{

			bool descend = false;
			for (int c = traversal_index; c < 8; c++)
			{
				int cn_child = root_node[current_node_index].Children[c];
				if (root_node[cn_child].Point_Cloud_Num > 0)
				{
					float3& _massCenter = root_node[cn_child].Mass_Center;
					float _max_R = root_node[cn_child].Max_R;
					float3 MCToq;
					MCToq.x = _massCenter.x - q.x; MCToq.y = _massCenter.y - q.y; MCToq.z = _massCenter.z - q.z;
					float mctop_norm = sqrtf(MCToq.x * MCToq.x + MCToq.y * MCToq.y + MCToq.z * MCToq.z);

					if (mctop_norm > beta * _max_R)
					{
						if (root_node[cn_child].Children[0] == -1)
						{
							for (int i_point_in_cnc = 0; i_point_in_cnc < root_node[cn_child].Point_Cloud_Num; i_point_in_cnc++)
							{
								int p_idx = root_node[cn_child].Point_Cloud_Indices[i_point_in_cnc];
								float3& p = i_points[p_idx];
								float3& N = i_Normals[p_idx];
								float _A = i_As[p_idx];

								float3 _R;
								_R.x = p.x - q.x; _R.y = p.y - q.y; _R.z = p.z - q.z;
								float _R_norm = sqrtf(_R.x * _R.x + _R.y * _R.y + _R.z * _R.z);

								if (_R_norm > 1e-7)
								{
									float dot_RN = _R.x * N.x + _R.y * N.y + _R.z * N.z;
									dot_RN *= _A;

									float screened = (use_radii) ? i_radii[p_idx] : 0.0015f;
									bool sign = (dot_RN > 0.0f) ? true : false;

									if (screened > _R_norm)
									{
										float up = (screened * sigma + 1.0f) * dot_RN;
										float down = expf(screened * sigma) * (PI_4 * screened * screened * screened) + 1e-7;
										float res = up / down;

										gwn += res;

									}
									else
									{
										float up = (_R_norm * sigma + 1.0f) * dot_RN;
										float down = expf(_R_norm * sigma) * (PI_4 * _R_norm * _R_norm * _R_norm) + 1e-7;
										float res = up / down;

										gwn += res;

									}
								}

								points_num++;
							}
						}
						else
						{
							float3& _massNormal = root_node[cn_child].Mass_Normal;

							float _R_norm = mctop_norm;
							if (_R_norm > 1e-7)
							{
								float dot_RN = MCToq.x * _massNormal.x + MCToq.y * _massNormal.y + MCToq.z * _massNormal.z;

								float screened = 0.0015f;

								bool sign = (dot_RN > 0.0f) ? true : false;

								if (screened > _R_norm)
								{
									float up = (screened * sigma + 1.0f) * dot_RN;
									float down = expf(screened * sigma) * (PI_4 * screened * screened * screened) + 1e-7;
									float res = up / down;

									gwn += res;

								}
								else
								{
									float up = (_R_norm * sigma + 1.0f) * dot_RN;
									float down = expf(_R_norm * sigma) * (PI_4 * _R_norm * _R_norm * _R_norm) + 1e-7;
									float res = up / down;

									gwn += res;

								}
							}

							points_num++;
						}
					}
					else
					{
						current_node_index = cn_child;
						descend = true;
						break;
					}
				}
			}

			traversal_index = 0;

			if (!descend)
			{
				while (true)
				{
					bool is_last_child = true;

					if (root_node[current_node_index].IndexInFather == 7)
						is_last_child = true;
					else if (root_node[current_node_index].Father == -1)
					{
						continue_traversal = false;
						break;
					}
					else
					{
						traversal_index = root_node[current_node_index].IndexInFather + 1;
						current_node_index = root_node[current_node_index].Father;
						break;
					}
					if (root_node[current_node_index].Father != -1 && is_last_child)
					{
						current_node_index = root_node[current_node_index].Father;

						if (root_node[current_node_index].Father == -1)
						{
							continue_traversal = false;
							break;
						}
					}
					else
					{
						break;
					}
				}
			}
		}

		iter_num++;
	}

	return gwn;
}

__device__
float ScreenedGWNCsLibiPStoQ(OctreeNodeLibiGPU* __restrict__ root_node, float3& q, float3* __restrict__ i_points, float3* __restrict__ i_Normals, float* __restrict__ i_As, float* __restrict__ i_radii, float* __restrict__ i_Cs, float beta, float sigma, bool use_radii)
{
	float gwn = 0.0;
	bool continue_traversal = true;

	int iter_num = 0;
	int points_num = 0;

	int traversal_index = 0;

	int current_node_index = 0;

	float PI_4 = 4.0f * 3.14159265358979323;

	while (continue_traversal)
	{
		if (root_node[current_node_index].Children[0] == -1)
		{

			for (int i_point_in_cn = 0; i_point_in_cn < root_node[current_node_index].Point_Cloud_Num; i_point_in_cn++)
			{
				int p_idx = root_node[current_node_index].Point_Cloud_Indices[i_point_in_cn];
				float3& p = i_points[p_idx];
				float3& N = i_Normals[p_idx];
				float _A = i_As[p_idx];
				float _C = i_Cs[p_idx];

				float3 _R;
				_R.x = p.x - q.x; _R.y = p.y - q.y; _R.z = p.z - q.z;
				float _R_norm = sqrtf(_R.x * _R.x + _R.y * _R.y + _R.z * _R.z);

				if (_R_norm > 1e-10)
				{
					float dot_RN = _R.x * N.x + _R.y * N.y + _R.z * N.z;
					dot_RN *= (_A * _C);

					float screened = (use_radii) ? i_radii[p_idx] : 0.0015f;
					bool sign = (dot_RN > 0.0f) ? true : false;

					if (screened > _R_norm)
					{
						float up = (screened * sigma + 1.0f) * dot_RN;
						float down = expf(screened * sigma) * (PI_4 * screened * screened * screened) + 1e-7;
						float res = up / down;

						gwn += res;
					}
					else
					{
						float up = (_R_norm * sigma + 1.0f) * dot_RN;
						float down = expf(_R_norm * sigma) * (PI_4 * _R_norm * _R_norm * _R_norm) + 1e-7;
						float res = up / down;

						gwn += res;
					}
				}

				points_num++;
			}

			while (true)
			{
				bool is_last_child = true;

				if (root_node[current_node_index].IndexInFather == 7)
					is_last_child = true;
				else if (root_node[current_node_index].Father == -1)
				{
					continue_traversal = false;
					break;
				}
				else
				{
					traversal_index = root_node[current_node_index].IndexInFather + 1;
					current_node_index = root_node[current_node_index].Father;
					break;
				}
				if (root_node[current_node_index].Father != -1 && is_last_child)
				{
					current_node_index = root_node[current_node_index].Father;

					if (root_node[current_node_index].Father == -1)
					{
						continue_traversal = false;
						break;
					}
				}
				else
				{
					break;
				}
			}

		}
		else
		{

			bool descend = false;
			for (int c = traversal_index; c < 8; c++)
			{
				int cn_child = root_node[current_node_index].Children[c];
				if (root_node[cn_child].Point_Cloud_Num > 0)
				{
					float3& _massCenter = root_node[cn_child].Mass_Center;
					float _max_R = root_node[cn_child].Max_R;
					float3 MCToq;
					MCToq.x = _massCenter.x - q.x; MCToq.y = _massCenter.y - q.y; MCToq.z = _massCenter.z - q.z;
					float mctop_norm = sqrtf(MCToq.x * MCToq.x + MCToq.y * MCToq.y + MCToq.z * MCToq.z);

					if (mctop_norm > beta * _max_R)
					{
						if (root_node[cn_child].Children[0] == -1)
						{
							for (int i_point_in_cnc = 0; i_point_in_cnc < root_node[cn_child].Point_Cloud_Num; i_point_in_cnc++)
							{
								int p_idx = root_node[cn_child].Point_Cloud_Indices[i_point_in_cnc];
								float3& p = i_points[p_idx];
								float3& N = i_Normals[p_idx];
								float _A = i_As[p_idx];
								float _C = i_Cs[p_idx];

								float3 _R;
								_R.x = p.x - q.x; _R.y = p.y - q.y; _R.z = p.z - q.z;
								float _R_norm = sqrtf(_R.x * _R.x + _R.y * _R.y + _R.z * _R.z);

								if (_R_norm > 1e-10)
								{
									float dot_RN = _R.x * N.x + _R.y * N.y + _R.z * N.z;
									dot_RN *= (_A * _C);

									float screened = (use_radii) ? i_radii[p_idx] : 0.0015f;
									bool sign = (dot_RN > 0.0f) ? true : false;

									if (screened > _R_norm)
									{
										float up = (screened * sigma + 1.0f) * dot_RN;
										float down = expf(screened * sigma) * (PI_4 * screened * screened * screened) + 1e-7;
										float res = up / down;

										gwn += res;
									}
									else
									{
										float up = (_R_norm * sigma + 1.0f) * dot_RN;
										float down = expf(_R_norm * sigma) * (PI_4 * _R_norm * _R_norm * _R_norm) + 1e-7;
										float res = up / down;

										gwn += res;
									}
								}

								points_num++;
							}
						}
						else
						{
							float3& _massNormal = root_node[cn_child].Mass_Normal;

							float _R_norm = mctop_norm;
							if (_R_norm > 1e-10)
							{
								float dot_RN = MCToq.x * _massNormal.x + MCToq.y * _massNormal.y + MCToq.z * _massNormal.z;

								float screened = 0.0015f;
								bool sign = (dot_RN > 0.0f) ? true : false;

								if (screened > _R_norm)
								{
									float up = (screened * sigma + 1.0f) * dot_RN;
									float down = expf(screened * sigma) * (PI_4 * screened * screened * screened) + 1e-7;
									float res = up / down;

									gwn += res;
								}
								else
								{
									float up = (_R_norm * sigma + 1.0f) * dot_RN;
									float down = expf(_R_norm * sigma) * (PI_4 * _R_norm * _R_norm * _R_norm) + 1e-7;
									float res = up / down;

									gwn += res;
								}
							}

							points_num++;
						}
					}
					else
					{
						current_node_index = cn_child;
						descend = true;
						break;
					}
				}
			}

			traversal_index = 0;

			if (!descend)
			{
				while (true)
				{
					bool is_last_child = true;

					if (root_node[current_node_index].IndexInFather == 7)
						is_last_child = true;
					else if (root_node[current_node_index].Father == -1)
					{
						continue_traversal = false;
						break;
					}
					else
					{
						traversal_index = root_node[current_node_index].IndexInFather + 1;
						current_node_index = root_node[current_node_index].Father;
						break;
					}
					if (root_node[current_node_index].Father != -1 && is_last_child)
					{
						current_node_index = root_node[current_node_index].Father;

						if (root_node[current_node_index].Father == -1)
						{
							continue_traversal = false;
							break;
						}
					}
					else
					{
						break;
					}
				}
			}
		}

		iter_num++;
	}

	return gwn;
}

__global__
void TurboLibiGWNKernal(int query_size, OctreeNodeLibiGPU* root_node, float3* q, float3* i_points, float3* i_Normals, float* i_As, float beta, float* gwn_results)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int tid_i = tid; tid_i < query_size; tid_i += stride)
	{
		gwn_results[tid_i] = GWNLibiPStoQ(root_node, q[tid_i], i_points, i_Normals, i_As, beta);

	}
}

__global__
void TurboLibiScreenedGWNKernal(int query_size, OctreeNodeLibiGPU* root_node, float3* q, float3* i_points, float3* i_Normals, float* i_As, float* i_radii, float beta, float sigma, float* gwn_results, bool use_radii)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int tid_i = tid; tid_i < query_size; tid_i += stride)
	{
		gwn_results[tid_i] = ScreenedGWNLibiPStoQ(root_node, q[tid_i], i_points, i_Normals, i_As, i_radii, beta, sigma, use_radii);

	}
}

__global__
void TurboLibiScreenedGWNCsKernal(int query_size, OctreeNodeLibiGPU* root_node, float3* q, float3* i_points, float3* i_Normals, float* i_As, float* i_radii, float* i_Cs, float beta, float sigma, float* gwn_results, bool use_radii)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int tid_i = tid; tid_i < query_size; tid_i += stride)
	{
		gwn_results[tid_i] = ScreenedGWNCsLibiPStoQ(root_node, q[tid_i], i_points, i_Normals, i_As, i_radii, i_Cs, beta, sigma, use_radii);
	}
}

void TurboLibiGWN::TurboLibiGWNLaunch(float3* query_points, float3* point_cloud, int query_size, float3* i_Normals, float* i_As, float& beta, float* out_gwn)
{
	int blockSize = 128;
	dim3 numBlocks = computeNbBlocks((unsigned int)query_size, blockSize);

	TurboLibiGWNKernal << <numBlocks, blockSize >> > (query_size, this->m_Octree_GPU, query_points, point_cloud, i_Normals, i_As, beta, out_gwn);

	cudaDeviceSynchronize();

	std::cout << "GWN Done" << std::endl;
}

__device__
float3 CalMassNormal(OctreeNodeLibiGPU* node, float3* i_Normals, float* i_As)
{
	float3 new_mass_normal;
	new_mass_normal.x = 0.0;
	new_mass_normal.y = 0.0;
	new_mass_normal.z = 0.0;
	for (int i = 0; i < node->Point_Cloud_Num; i++)
	{
		int curr_point_index = node->Point_Cloud_Indices[i];
		new_mass_normal.x += i_As[curr_point_index] * i_Normals[curr_point_index].x;
		new_mass_normal.y += i_As[curr_point_index] * i_Normals[curr_point_index].y;
		new_mass_normal.z += i_As[curr_point_index] * i_Normals[curr_point_index].z;
	}

	return new_mass_normal;
}

__global__
void UpdateMassNormalsKernal(int nodes_size, OctreeNodeLibiGPU* root_node, float3* i_Normals, float* i_As)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int tid_i = tid; tid_i < nodes_size; tid_i += stride)
	{
		root_node[tid_i].Mass_Normal = CalMassNormal(root_node + tid_i, i_Normals, i_As);
	}
}

__global__
void UpdateGWNInfoKernal(int nodes_size, OctreeNodeLibiGPU* root_node, float3* i_Normals, float3* point_cloud, float* i_As, float* i_radii)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int tid_i = tid; tid_i < nodes_size; tid_i += stride)
	{
		OctreeNodeLibiGPU* node = root_node + tid_i;

		float3 new_mass_normal;
		new_mass_normal.x = 0.0;
		new_mass_normal.y = 0.0;
		new_mass_normal.z = 0.0;

		float3 new_mass_center;
		new_mass_center.x = 0.0;
		new_mass_center.y = 0.0;
		new_mass_center.z = 0.0;

		float areatotal = 0.0;

		float new_screened = 0.0f;

		for (int i = 0; i < node->Point_Cloud_Num; i++)
		{
			int curr_point_index = node->Point_Cloud_Indices[i];
			new_mass_normal.x += i_As[curr_point_index] * i_Normals[curr_point_index].x;
			new_mass_normal.y += i_As[curr_point_index] * i_Normals[curr_point_index].y;
			new_mass_normal.z += i_As[curr_point_index] * i_Normals[curr_point_index].z;

			areatotal += i_As[curr_point_index];
			new_mass_center.x += i_As[curr_point_index] * point_cloud[curr_point_index].x;
			new_mass_center.y += i_As[curr_point_index] * point_cloud[curr_point_index].y;
			new_mass_center.z += i_As[curr_point_index] * point_cloud[curr_point_index].z;

			new_screened += i_As[curr_point_index] * i_radii[curr_point_index];

			node->As_Grad_x[i] = 0.0f;
			node->As_Grad_y[i] = 0.0f;
			node->As_Grad_z[i] = 0.0f;
		}

		new_mass_center.x = new_mass_center.x / areatotal;
		new_mass_center.y = new_mass_center.y / areatotal;
		new_mass_center.z = new_mass_center.z / areatotal;

		new_screened /= areatotal;

		node->As_Total = areatotal;

		float c_norm = 0;
		float max_norm = 0;
		for (int i = 0; i < node->Point_Cloud_Num; i++)
		{
			int curr_point_index = node->Point_Cloud_Indices[i];
			float3 point_diff;
			point_diff.x = point_cloud[curr_point_index].x - new_mass_center.x;
			point_diff.y = point_cloud[curr_point_index].y - new_mass_center.y;
			point_diff.z = point_cloud[curr_point_index].z - new_mass_center.z;

			c_norm = __fsqrt_rn(point_diff.x * point_diff.x +
				point_diff.y * point_diff.y +
				point_diff.z * point_diff.z);

			if (c_norm > max_norm)
			{
				max_norm = c_norm;
			}
		}

		root_node[tid_i].Mass_Normal = new_mass_normal;
		root_node[tid_i].Mass_Center = new_mass_center;
		root_node[tid_i].Max_R = max_norm;
		root_node[tid_i].Mass_Screened = new_screened;
	}
}

__global__
void UpdateGWNInfoAllKernal(int nodes_size, OctreeNodeLibiGPU* root_node, float3* i_Normals, float3* point_cloud, float* i_As, float* i_Cs)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int tid_i = tid; tid_i < nodes_size; tid_i += stride)
	{
		OctreeNodeLibiGPU* node = root_node + tid_i;

		float3 new_mass_normal;
		new_mass_normal.x = 0.0;
		new_mass_normal.y = 0.0;
		new_mass_normal.z = 0.0;

		float3 new_mass_center;
		new_mass_center.x = 0.0;
		new_mass_center.y = 0.0;
		new_mass_center.z = 0.0;

		float areatotal = 0.0;

		for (int i = 0; i < node->Point_Cloud_Num; i++)
		{
			int curr_point_index = node->Point_Cloud_Indices[i];
			new_mass_normal.x += i_As[curr_point_index] * i_Normals[curr_point_index].x * i_Cs[curr_point_index];
			new_mass_normal.y += i_As[curr_point_index] * i_Normals[curr_point_index].y * i_Cs[curr_point_index];
			new_mass_normal.z += i_As[curr_point_index] * i_Normals[curr_point_index].z * i_Cs[curr_point_index];

			areatotal += (i_As[curr_point_index] * i_Cs[curr_point_index]);
			new_mass_center.x += i_As[curr_point_index] * point_cloud[curr_point_index].x * i_Cs[curr_point_index];
			new_mass_center.y += i_As[curr_point_index] * point_cloud[curr_point_index].y * i_Cs[curr_point_index];
			new_mass_center.z += i_As[curr_point_index] * point_cloud[curr_point_index].z * i_Cs[curr_point_index];

			node->As_Grad_x[i] = 0.0f;
			node->As_Grad_y[i] = 0.0f;
			node->As_Grad_z[i] = 0.0f;

			node->Cs_Grad_x[i] = 0.0f;
			node->Cs_Grad_y[i] = 0.0f;
			node->Cs_Grad_z[i] = 0.0f;
		}

		new_mass_center.x = new_mass_center.x / areatotal;
		new_mass_center.y = new_mass_center.y / areatotal;
		new_mass_center.z = new_mass_center.z / areatotal;

		node->As_Total = areatotal;

		float c_norm = 0;
		float max_norm = 0;
		for (int i = 0; i < node->Point_Cloud_Num; i++)
		{
			int curr_point_index = node->Point_Cloud_Indices[i];
			float3 point_diff;
			point_diff.x = point_cloud[curr_point_index].x - new_mass_center.x;
			point_diff.y = point_cloud[curr_point_index].y - new_mass_center.y;
			point_diff.z = point_cloud[curr_point_index].z - new_mass_center.z;

			c_norm = __fsqrt_rn(point_diff.x * point_diff.x +
				point_diff.y * point_diff.y +
				point_diff.z * point_diff.z);

			if (c_norm > max_norm)
			{
				max_norm = c_norm;
			}
		}

		root_node[tid_i].Mass_Normal = new_mass_normal;
		root_node[tid_i].Mass_Center = new_mass_center;
		root_node[tid_i].Max_R = max_norm;
	}
}

__global__
void UpdateGWNInfoAsGradKernal(int nodes_size, OctreeNodeLibiGPU* root_node, float3* i_Normals, float3* point_cloud, float* i_As)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int tid_i = tid; tid_i < nodes_size; tid_i += stride)
	{
		OctreeNodeLibiGPU* node = root_node + tid_i;

		for (int i = 0; i < node->Point_Cloud_Num; i++)
		{
			int curr_point_index = node->Point_Cloud_Indices[i];

			float c_As_total = node->As_Total;
			float c_As_total_sqr = node->As_Total * node->As_Total + 1e-7;

			float c_As_grad_mc_x = (point_cloud[curr_point_index].x * c_As_total - node->Mass_Center.x * c_As_total) / c_As_total_sqr;

			float c_As_grad_mc_y = (point_cloud[curr_point_index].y * c_As_total - node->Mass_Center.y * c_As_total) / c_As_total_sqr;

			float c_As_grad_mc_z = (point_cloud[curr_point_index].z * c_As_total - node->Mass_Center.z * c_As_total) / c_As_total_sqr;

			node->As_Grad_x[i] = c_As_grad_mc_x;
			node->As_Grad_y[i] = c_As_grad_mc_y;
			node->As_Grad_z[i] = c_As_grad_mc_z;
		}
	}
}

__global__
void UpdateGWNInfoAsCsGradKernal(int nodes_size, OctreeNodeLibiGPU* root_node, float3* i_Normals, float3* point_cloud, float* i_As, float* i_Cs)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int tid_i = tid; tid_i < nodes_size; tid_i += stride)
	{
		OctreeNodeLibiGPU* node = root_node + tid_i;

		for (int i = 0; i < node->Point_Cloud_Num; i++)
		{
			int curr_point_index = node->Point_Cloud_Indices[i];

			float c_As_total = node->As_Total;
			float c_As_total_sqr = node->As_Total * node->As_Total + 1e-8;

			float c_As_grad_mc_x = (point_cloud[curr_point_index].x * i_Cs[curr_point_index] * c_As_total - node->Mass_Center.x * c_As_total * i_Cs[curr_point_index]) / c_As_total_sqr;
			float c_Cs_grad_mc_x = (point_cloud[curr_point_index].x * i_As[curr_point_index] * c_As_total - node->Mass_Center.x * c_As_total * i_As[curr_point_index]) / c_As_total_sqr;

			float c_As_grad_mc_y = (point_cloud[curr_point_index].y * i_Cs[curr_point_index] * c_As_total - node->Mass_Center.y * c_As_total * i_Cs[curr_point_index]) / c_As_total_sqr;
			float c_Cs_grad_mc_y = (point_cloud[curr_point_index].y * i_As[curr_point_index] * c_As_total - node->Mass_Center.y * c_As_total * i_As[curr_point_index]) / c_As_total_sqr;

			float c_As_grad_mc_z = (point_cloud[curr_point_index].z * i_Cs[curr_point_index] * c_As_total - node->Mass_Center.z * c_As_total * i_Cs[curr_point_index]) / c_As_total_sqr;
			float c_Cs_grad_mc_z = (point_cloud[curr_point_index].z * i_As[curr_point_index] * c_As_total - node->Mass_Center.z * c_As_total * i_As[curr_point_index]) / c_As_total_sqr;

			node->As_Grad_x[i] = c_As_grad_mc_x;
			node->As_Grad_y[i] = c_As_grad_mc_y;
			node->As_Grad_z[i] = c_As_grad_mc_z;

			node->Cs_Grad_x[i] = c_Cs_grad_mc_x;
			node->Cs_Grad_y[i] = c_Cs_grad_mc_y;
			node->Cs_Grad_z[i] = c_Cs_grad_mc_z;
		}
	}
}

void TurboLibiGWN::UpdateMassNormals(float3* i_Normals, float* i_As)
{
	int blockSize = 128;
	dim3 numBlocks = computeNbBlocks((unsigned int)this->m_Node_Size, blockSize);

	UpdateMassNormalsKernal << <numBlocks, blockSize >> > (this->m_Node_Size, this->m_Octree_GPU, i_Normals, i_As);

	cudaDeviceSynchronize();
}

void TurboLibiGWN::UpdateGWNInfo(float3* i_Normals, float3* point_cloud, float* i_As, float* i_radii)
{
	int blockSize = 128;
	dim3 numBlocks = computeNbBlocks((unsigned int)this->m_Node_Size, blockSize);

	UpdateGWNInfoKernal << <numBlocks, blockSize >> > (this->m_Node_Size, this->m_Octree_GPU, i_Normals, point_cloud, i_As, i_radii);

	cudaDeviceSynchronize();
}

void TurboLibiGWN::UpdateGWNAllInfo(float3* i_Normals, float3* point_cloud, float* i_As, float* i_Cs)
{
	int blockSize = 128;
	dim3 numBlocks = computeNbBlocks((unsigned int)this->m_Node_Size, blockSize);

	UpdateGWNInfoAllKernal << <numBlocks, blockSize >> > (this->m_Node_Size, this->m_Octree_GPU, i_Normals, point_cloud, i_As, i_Cs);

	cudaDeviceSynchronize();
}

void TurboLibiGWN::UpdateGWNInfoAsGrad(float3* i_Normals, float3* point_cloud, float* i_As)
{
	int blockSize = 128;
	dim3 numBlocks = computeNbBlocks((unsigned int)this->m_Node_Size, blockSize);

	UpdateGWNInfoAsGradKernal << <numBlocks, blockSize >> > (this->m_Node_Size, this->m_Octree_GPU, i_Normals, point_cloud, i_As);

	cudaDeviceSynchronize();
}

void TurboLibiGWN::UpdateGWNInfoAsCsGrad(float3* i_Normals, float3* point_cloud, float* i_As, float* i_Cs)
{
	int blockSize = 128;
	dim3 numBlocks = computeNbBlocks((unsigned int)this->m_Node_Size, blockSize);

	UpdateGWNInfoAsCsGradKernal << <numBlocks, blockSize >> > (this->m_Node_Size, this->m_Octree_GPU, i_Normals, point_cloud, i_As, i_Cs);

	cudaDeviceSynchronize();
}

void TurboLibiGWN::EarseMemory()
{
	for (int i = 0; i < m_Node_Size; i++)
	{
		if (m_Octree_CPU[i].Point_Cloud_Num > 0)
		{
			gpuErrchk(cudaFree(m_Octree_CPU[i].Point_Cloud_Indices));
			break;
		}
	}

	gpuErrchk(cudaFree(m_Octree_CPU[0].Children));

	delete[] m_Octree_CPU;
	m_Octree_CPU = nullptr;
	gpuErrchk(cudaFree(m_Octree_GPU));
	m_Octree_GPU = nullptr;
}

void TurboLibiGWN::AllocateQueryMem(int query_size, int point_size)
{

	int allocate_size = (query_size > point_size) ? query_size : point_size;

	bool need_reallocate = allocate_size > this->m_Allocate_Size;
	bool initialze = (this->m_Allocate_Size == 0) ? true: false;
	if (need_reallocate || initialze)
	{
		std::cout << "Reallocating GPU buffers: " << this->m_Allocate_Size
			<< " -> " << allocate_size << std::endl;
		this->m_Allocate_Size = allocate_size;
	}
	else
	{
		std::cout << "Keep previous GPU buffer allocation (size "
			<< this->m_Allocate_Size << ")" << std::endl;

		return;
	}

	std::vector<int> large_node_index;
	large_node_index.reserve(100000000);

	int total_num = 0;
	int threshold = 1000;
	for (int i = 0; i < this->m_Node_Size; i++)
	{
		if (this->m_Octree_GPU[i].Point_Cloud_Num >= threshold)
		{
			if (need_reallocate || initialze)
			{
				if (this->m_Octree_GPU[i].Query_Buffer)
					cudaFree(this->m_Octree_GPU[i].Query_Buffer);
				if (this->m_Octree_GPU[i].Query_Index)
					cudaFree(this->m_Octree_GPU[i].Query_Index);

				cudaMallocManaged((char**)&this->m_Octree_GPU[i].Query_Buffer, allocate_size * sizeof(*this->m_Octree_GPU[i].Query_Buffer));
				std::unique_ptr<int[]> i_query_cpu = std::make_unique<int[]>(size_t(allocate_size));
				for (int k = 0; k < allocate_size; k++)
				{
					i_query_cpu[k] = -1;
				}
				gpuErrchk(cudaMemcpy(this->m_Octree_GPU[i].Query_Buffer, i_query_cpu.get(), allocate_size * sizeof(int), cudaMemcpyHostToDevice));

				cudaMallocManaged((char**)&this->m_Octree_GPU[i].Query_Index, sizeof(*this->m_Octree_GPU[i].Query_Index));
				this->m_Octree_GPU[i].Query_Index[0] = 0;
			}

			if (initialze)
			{
				this->m_Octree_GPU[i].Large_Node = true;

				large_node_index.push_back(i);
				this->m_Node_Array_Size += this->m_Octree_GPU[i].Point_Cloud_Num;
			}

			total_num++;
		}
	}

	if (initialze)
	{
		this->m_Large_Node_Size = large_node_index.size();
		cudaMallocManaged((char**)&this->m_Large_Node_Index, this->m_Large_Node_Size * sizeof(*this->m_Large_Node_Index));
		cudaMallocManaged((char**)&this->m_Large_Node_Array, this->m_Node_Array_Size * sizeof(*this->m_Large_Node_Array));
		cudaMallocManaged((char**)&this->m_Large_Node_Point_Index, this->m_Node_Array_Size * sizeof(*this->m_Large_Node_Point_Index));
		cudaMallocManaged((char**)&this->m_Large_Node_Point_Local_Index, this->m_Node_Array_Size * sizeof(*this->m_Large_Node_Point_Local_Index));

		int count = 0;
		for (int j = 0; j < this->m_Large_Node_Size; j++)
		{
			this->m_Large_Node_Index[j] = large_node_index[j];

			for (int k = 0; k < this->m_Octree_GPU[this->m_Large_Node_Index[j]].Point_Cloud_Num; k++)
			{
				this->m_Large_Node_Array[count] = this->m_Large_Node_Index[j];
				this->m_Large_Node_Point_Index[count] = this->m_Octree_GPU[this->m_Large_Node_Index[j]].Point_Cloud_Indices[k];
				this->m_Large_Node_Point_Local_Index[count] = k;
				count++;
			}
		}

	}

	std::cout << "Larger than " << threshold << ": " << total_num << std::endl;
}

void TurboLibiGWN::TurboLibiScreenedGWNLaunch(float3* query_points, float3* point_cloud, int query_size, float3* i_Normals, float* i_As, float* i_radii, float& beta, float& sigma, float* out_gwn, bool use_radii)
{
	int blockSize = 128;
	dim3 numBlocks = computeNbBlocks((unsigned int)query_size, blockSize);

	TurboLibiScreenedGWNKernal << <numBlocks, blockSize >> > (query_size, this->m_Octree_GPU, query_points, point_cloud, i_Normals, i_As, i_radii, beta, sigma, out_gwn, use_radii);

	cudaDeviceSynchronize();

	std::cout << "GWN Done" << std::endl;
}

void TurboLibiGWN::TurboLibiScreenedGWNCsLaunch(float3* query_points, float3* point_cloud, int query_size, float3* i_Normals, float* i_As, float* i_Cs, float* i_radii, float& beta, float& sigma, float* out_gwn, bool use_radii)
{
	int blockSize = 128;
	dim3 numBlocks = computeNbBlocks((unsigned int)query_size, blockSize);

	TurboLibiScreenedGWNCsKernal << <numBlocks, blockSize >> > (query_size, this->m_Octree_GPU, query_points, point_cloud, i_Normals, i_As, i_radii, i_Cs, beta, sigma, out_gwn, use_radii);

	cudaDeviceSynchronize();

	std::cout << "GWN Done" << std::endl;
}

__device__
void ScreenedGWNLibiAsCsBackward(OctreeNodeLibiGPU* __restrict__ root_node, float3& q, float3* __restrict__ i_points, float3* __restrict__ i_Normals, float* __restrict__ i_As, float* __restrict__ i_Cs, float* __restrict__ i_radii,
	float point_gwn, float beta, float sigma, bool use_radii, bool use_screened, float target_gwn, float grad_scale_Ai, float grad_scale_Ci, float* __restrict__ As_grad, float* __restrict__ Cs_grad, bool com_Ai, bool com_Ci, bool patch, int tid)
{
	float As_grad_s = 2.0f * grad_scale_Ai * (point_gwn - target_gwn);
	float Cs_grad_s = 2.0f * grad_scale_Ci * (point_gwn - target_gwn);

	bool continue_traversal = true;

	int iter_num = 0;
	int points_num = 0;

	int traversal_index = 0;

	int current_node_index = 0;

	float PI_4 = 4.0f * 3.14159265358979323;

	while (continue_traversal)
	{
		if (root_node[current_node_index].Children[0] == -1)
		{

			for (int i_point_in_cn = 0; i_point_in_cn < root_node[current_node_index].Point_Cloud_Num; i_point_in_cn++)
			{
				int p_idx = root_node[current_node_index].Point_Cloud_Indices[i_point_in_cn];
				float3& p = i_points[p_idx];
				float3& N = i_Normals[p_idx];
				float _A = i_As[p_idx];
				float _C = i_Cs[p_idx];

				float3 _R;
				_R.x = p.x - q.x; _R.y = p.y - q.y; _R.z = p.z - q.z;
				float _R_norm = sqrtf(_R.x * _R.x + _R.y * _R.y + _R.z * _R.z);

				if (_R_norm > 1e-10)
				{
					float dot_RN = _R.x * N.x + _R.y * N.y + _R.z * N.z;

					float screened = 0.0f;
					if (use_screened)
						screened = (use_radii) ? i_radii[p_idx] : 0.0015f;

					if (screened > _R_norm)
					{
						float up = (screened * sigma + 1.0f) * dot_RN;
						float down = expf(screened * sigma) * (PI_4 * screened * screened * screened) + 1e-7;
						float res = up / down;

						if (com_Ai)
						{
							float c_grad = As_grad_s * res * _C;
							atomicAdd(&(As_grad[p_idx]), c_grad);
						}

						if (com_Ci)
						{
							float c_grad = Cs_grad_s * res * _A;
							atomicAdd(&(Cs_grad[p_idx]), c_grad);
						}
					}
					else
					{
						float up = (_R_norm * sigma + 1.0f) * dot_RN;
						float down = expf(_R_norm * sigma) * (PI_4 * _R_norm * _R_norm * _R_norm) + 1e-7;
						float res = up / down;

						if (com_Ai)
						{
							float c_grad = As_grad_s * res * _C;
							atomicAdd(&(As_grad[p_idx]), c_grad);
						}

						if (com_Ci)
						{
							float c_grad = Cs_grad_s * res * _A;
							atomicAdd(&(Cs_grad[p_idx]), c_grad);
						}
					}
				}

				points_num++;
			}

			while (true)
			{
				bool is_last_child = true;

				if (root_node[current_node_index].IndexInFather == 7)
					is_last_child = true;
				else if (root_node[current_node_index].Father == -1)
				{
					continue_traversal = false;
					break;
				}
				else
				{
					traversal_index = root_node[current_node_index].IndexInFather + 1;
					current_node_index = root_node[current_node_index].Father;
					break;
				}
				if (root_node[current_node_index].Father != -1 && is_last_child)
				{
					current_node_index = root_node[current_node_index].Father;

					if (root_node[current_node_index].Father == -1)
					{
						continue_traversal = false;
						break;
					}
				}
				else
				{
					break;
				}
			}

		}
		else
		{

			bool descend = false;
			for (int c = traversal_index; c < 8; c++)
			{
				int cn_child = root_node[current_node_index].Children[c];
				if (root_node[cn_child].Point_Cloud_Num > 0)
				{
					float3& _massCenter = root_node[cn_child].Mass_Center;
					float _max_R = root_node[cn_child].Max_R;
					float3 MCToq;
					MCToq.x = _massCenter.x - q.x; MCToq.y = _massCenter.y - q.y; MCToq.z = _massCenter.z - q.z;
					float mctop_norm = sqrtf(MCToq.x * MCToq.x + MCToq.y * MCToq.y + MCToq.z * MCToq.z);

					if (mctop_norm > beta * _max_R)
					{
						if (root_node[cn_child].Children[0] == -1)
						{
							for (int i_point_in_cnc = 0; i_point_in_cnc < root_node[cn_child].Point_Cloud_Num; i_point_in_cnc++)
							{
								int p_idx = root_node[cn_child].Point_Cloud_Indices[i_point_in_cnc];
								float3& p = i_points[p_idx];
								float3& N = i_Normals[p_idx];
								float _A = i_As[p_idx];
								float _C = i_Cs[p_idx];

								float3 _R;
								_R.x = p.x - q.x; _R.y = p.y - q.y; _R.z = p.z - q.z;
								float _R_norm = sqrtf(_R.x * _R.x + _R.y * _R.y + _R.z * _R.z);

								if (_R_norm > 1e-10)
								{
									float dot_RN = _R.x * N.x + _R.y * N.y + _R.z * N.z;

									float screened = 0.0f;
									if (use_screened)
										screened = (use_radii) ? i_radii[p_idx] : 0.0015f;

									if (screened > _R_norm)
									{
										float up = (screened * sigma + 1.0f) * dot_RN;
										float down = expf(screened * sigma) * (PI_4 * screened * screened * screened) + 1e-7;
										float res = up / down;

										if (com_Ai)
										{
											float c_grad = As_grad_s * res * _C;
											atomicAdd(&(As_grad[p_idx]), c_grad);
										}

										if (com_Ci)
										{
											float c_grad = Cs_grad_s * res * _A;
											atomicAdd(&(Cs_grad[p_idx]), c_grad);
										}
									}
									else
									{
										float up = (_R_norm * sigma + 1.0f) * dot_RN;
										float down = expf(_R_norm * sigma) * (PI_4 * _R_norm * _R_norm * _R_norm) + 1e-7;
										float res = up / down;

										if (com_Ai)
										{
											float c_grad = As_grad_s * res * _C;
											atomicAdd(&(As_grad[p_idx]), c_grad);
										}

										if (com_Ci)
										{
											float c_grad = Cs_grad_s * res * _A;
											atomicAdd(&(Cs_grad[p_idx]), c_grad);
										}
									}
								}

								points_num++;
							}
						}
						else
						{
							float3& _massNormal = root_node[cn_child].Mass_Normal;

							float _R_norm = mctop_norm;
							if (_R_norm > 1e-10)
							{
								float dot_RN = MCToq.x * _massNormal.x + MCToq.y * _massNormal.y + MCToq.z * _massNormal.z;

								float screened = 0.0f;
								if (use_screened)
									screened = 0.0015f;

								float res = 0.0f;
								if (screened > _R_norm)
								{
									float up_wod = screened * sigma + 1.0f;
									float down = expf(screened * sigma) * (PI_4 * screened * screened * screened) + 1e-7;
									res = up_wod / down;

									if (com_Ai)
									{
										for (int i = 0; i < root_node[cn_child].Point_Cloud_Num; i++)
										{
											int curr_point_index = root_node[cn_child].Point_Cloud_Indices[i];

											float c_As_grad_mc_x = root_node[cn_child].As_Grad_x[i];
											float c_As_grad_x = c_As_grad_mc_x * root_node[cn_child].Mass_Normal.x + MCToq.x * i_Normals[curr_point_index].x * i_Cs[curr_point_index];

											float c_As_grad_mc_y = root_node[cn_child].As_Grad_y[i];
											float c_As_grad_y = c_As_grad_mc_y * root_node[cn_child].Mass_Normal.y + MCToq.y * i_Normals[curr_point_index].y * i_Cs[curr_point_index];

											float c_As_grad_mc_z = root_node[cn_child].As_Grad_z[i];
											float c_As_grad_z = c_As_grad_mc_z * root_node[cn_child].Mass_Normal.z + MCToq.z * i_Normals[curr_point_index].z * i_Cs[curr_point_index];

											float c_As_grad = As_grad_s * res * (c_As_grad_x + c_As_grad_y + c_As_grad_z);

											atomicAdd(&(As_grad[curr_point_index]), c_As_grad);
										}
									}

									if (com_Ci)
									{
										for (int i = 0; i < root_node[cn_child].Point_Cloud_Num; i++)
										{
											int curr_point_index = root_node[cn_child].Point_Cloud_Indices[i];
											float c_Cs_grad_mc_x = root_node[cn_child].Cs_Grad_x[i];
											float c_Cs_grad_x = c_Cs_grad_mc_x * root_node[cn_child].Mass_Normal.x + MCToq.x * i_Normals[curr_point_index].x * i_As[curr_point_index];

											float c_Cs_grad_mc_y = root_node[cn_child].Cs_Grad_y[i];
											float c_Cs_grad_y = c_Cs_grad_mc_y * root_node[cn_child].Mass_Normal.y + MCToq.y * i_Normals[curr_point_index].y * i_As[curr_point_index];

											float c_Cs_grad_mc_z = root_node[cn_child].Cs_Grad_z[i];
											float c_Cs_grad_z = c_Cs_grad_mc_z * root_node[cn_child].Mass_Normal.z + MCToq.z * i_Normals[curr_point_index].z * i_As[curr_point_index];

											float c_Cs_grad = Cs_grad_s * res * (c_Cs_grad_x + c_Cs_grad_y + c_Cs_grad_z);

											atomicAdd(&(Cs_grad[curr_point_index]), c_Cs_grad);
										}
									}
								}
								else
								{

									float f_b = 1.0f;
									float f_c =   1.0f;
									float f_d1 = PI_4 * _R_norm * _R_norm * _R_norm;
									float f_d1_2 = f_d1 * f_d1 ;
									float f_d = dot_RN / f_d1;

									if (com_Ai)
									{
										if (root_node[cn_child].Large_Node && patch)
										{
											int c_qi = atomicAdd(&(root_node[cn_child].Query_Index[0]), 1);
											root_node[cn_child].Query_Buffer[c_qi] = tid;
										}
										else
										{
											float R_norm_a = _R_norm  ;
											for (int i = 0; i < root_node[cn_child].Point_Cloud_Num; i++)
											{
												int curr_point_index = root_node[cn_child].Point_Cloud_Indices[i];

												float grad_a = (MCToq.x * root_node[cn_child].As_Grad_x[i] +
													MCToq.y * root_node[cn_child].As_Grad_y[i] +
													MCToq.z * root_node[cn_child].As_Grad_z[i]) / R_norm_a;

												float grad_d0 = i_Normals[curr_point_index].x * i_Cs[curr_point_index] * MCToq.x + root_node[cn_child].As_Grad_x[i] * _massNormal.x +
													i_Normals[curr_point_index].y * i_Cs[curr_point_index] * MCToq.y + root_node[cn_child].As_Grad_y[i] * _massNormal.y +
													i_Normals[curr_point_index].z * i_Cs[curr_point_index] * MCToq.z + root_node[cn_child].As_Grad_z[i] * _massNormal.z;
												float grad_d1 = 3.0f * PI_4 * _R_norm * _R_norm * grad_a;
												float grad_d = (grad_d0 * f_d1 - grad_d1 * dot_RN) / f_d1_2;

												float c_ai_grad =   f_b * f_c * grad_d;
												atomicAdd(&(As_grad[curr_point_index]), As_grad_s * c_ai_grad);
											}
										}
									}

									if (com_Ci)
									{
										if (root_node[cn_child].Large_Node && patch)
										{
											int c_qi = atomicAdd(&(root_node[cn_child].Query_Index[0]), 1);
											root_node[cn_child].Query_Buffer[c_qi] = tid;
										}
										else
										{
											for (int i = 0; i < root_node[cn_child].Point_Cloud_Num; i++)
											{
												int curr_point_index = root_node[cn_child].Point_Cloud_Indices[i];

												float grad_a = (MCToq.x * root_node[cn_child].Cs_Grad_x[i] +
													MCToq.y * root_node[cn_child].Cs_Grad_y[i] +
													MCToq.z * root_node[cn_child].Cs_Grad_z[i]) / (_R_norm  );

												float grad_b = -expf(-_R_norm * sigma) * sigma * grad_a;
												float grad_c = sigma * grad_a;
												float grad_d0 = i_Normals[curr_point_index].x * i_As[curr_point_index] * MCToq.x + root_node[cn_child].Cs_Grad_x[i] * _massNormal.x +
													i_Normals[curr_point_index].y * i_As[curr_point_index] * MCToq.y + root_node[cn_child].Cs_Grad_y[i] * _massNormal.y +
													i_Normals[curr_point_index].z * i_As[curr_point_index] * MCToq.z + root_node[cn_child].Cs_Grad_z[i] * _massNormal.z;
												float grad_d1 = 3.0f * PI_4 * _R_norm * _R_norm * grad_a;
												float grad_d = (grad_d0 * f_d1 - grad_d1 * dot_RN) / (f_d1 * f_d1);

												float c_ci_grad = grad_b * f_c * f_d + f_b * f_d * grad_c + f_b * f_c * grad_d;
												atomicAdd(&(Cs_grad[curr_point_index]), Cs_grad_s * c_ci_grad);
											}
										}
									}
								}

							}

							points_num++;
						}
					}
					else
					{
						current_node_index = cn_child;
						descend = true;
						break;
					}
				}
			}

			traversal_index = 0;

			if (!descend)
			{
				while (true)
				{
					bool is_last_child = true;

					if (root_node[current_node_index].IndexInFather == 7)
						is_last_child = true;
					else if (root_node[current_node_index].Father == -1)
					{
						continue_traversal = false;
						break;
					}
					else
					{
						traversal_index = root_node[current_node_index].IndexInFather + 1;
						current_node_index = root_node[current_node_index].Father;
						break;
					}
					if (root_node[current_node_index].Father != -1 && is_last_child)
					{
						current_node_index = root_node[current_node_index].Father;

						if (root_node[current_node_index].Father == -1)
						{
							continue_traversal = false;
							break;
						}
					}
					else
					{
						break;
					}
				}
			}
		}

		iter_num++;
	}
}

__device__
float3 GWNGradient(OctreeNodeLibiGPU* __restrict__ root_node, float3& q, float3* __restrict__ i_points, float3* __restrict__ i_Normals, float* __restrict__ i_As, float* __restrict__ i_Cs, float* __restrict__ i_radii, float beta, float sigma, bool use_radii, bool use_screened)
{
	float grad_gwn_x = 0.0f;
	float grad_gwn_y = 0.0f;
	float grad_gwn_z = 0.0f;

	bool continue_traversal = true;

	int iter_num = 0;
	int points_num = 0;

	int traversal_index = 0;

	int current_node_index = 0;

	float PI_4 = 4.0f * 3.14159265358979323;

	while (continue_traversal)
	{
		if (root_node[current_node_index].Children[0] == -1)
		{

			for (int i_point_in_cn = 0; i_point_in_cn < root_node[current_node_index].Point_Cloud_Num; i_point_in_cn++)
			{
				int p_idx = root_node[current_node_index].Point_Cloud_Indices[i_point_in_cn];
				float3& p = i_points[p_idx];
				float3& N = i_Normals[p_idx];
				float _A = i_As[p_idx];
				float _C = i_Cs[p_idx];

				float3 _R;
				_R.x = p.x - q.x; _R.y = p.y - q.y; _R.z = p.z - q.z;
				float _R_square = _R.x * _R.x + _R.y * _R.y + _R.z * _R.z;
				float _R_norm = sqrtf(_R_square);

				if (_R_norm > 1e-7)
				{
					float screened = 0.0f;
					if (use_screened)
					{
						screened = (use_radii) ? i_radii[p_idx] : 0.0015f;
					}

					if (screened > _R_norm)
					{
						float up =   _A * _C;
						float down =   (PI_4 * screened * screened * screened);
						float res = up / down;

						float c_grad_gwn_x = -res * N.x;
						float c_grad_gwn_y = -res * N.y;
						float c_grad_gwn_z = -res * N.z;

						grad_gwn_x += c_grad_gwn_x;
						grad_gwn_y += c_grad_gwn_y;
						grad_gwn_z += c_grad_gwn_z;
					}
					else
					{
						float gwn_dot = _R.x * N.x + _R.y * N.y + _R.z * N.z;
						float gwn_a = _A * _C;
						float gwn_b =   1.0f;
						float gwn_c = gwn_dot / (PI_4 * _R_norm * _R_norm * _R_norm  );

						float _R_norm_inv = __frsqrt_rn(_R_square);

						float grad_0_x = -_R_norm_inv * _R.x;
						float grad_0_y = -_R_norm_inv * _R.y;
						float grad_0_z = -_R_norm_inv * _R.z;

						float grad_c_f = PI_4 * _R_norm * _R_norm * _R_norm * _R_norm;
						float grad_c_x = (-N.x * _R_norm - 3.0f * gwn_dot * grad_0_x) / grad_c_f;
						float grad_c_y = (-N.y * _R_norm - 3.0f * gwn_dot * grad_0_y) / grad_c_f;
						float grad_c_z = (-N.z * _R_norm - 3.0f * gwn_dot * grad_0_z) / grad_c_f;

						float gwn_ab = gwn_a * gwn_b;

						float c_grad_gwn_x =   grad_c_x * gwn_ab;
						float c_grad_gwn_y =   grad_c_y * gwn_ab;
						float c_grad_gwn_z =   grad_c_z * gwn_ab;

						grad_gwn_x += c_grad_gwn_x;
						grad_gwn_y += c_grad_gwn_y;
						grad_gwn_z += c_grad_gwn_z;
					}
				}

				points_num++;
			}

			while (true)
			{
				bool is_last_child = true;

				if (root_node[current_node_index].IndexInFather == 7)
					is_last_child = true;
				else if (root_node[current_node_index].Father == -1)
				{
					continue_traversal = false;
					break;
				}
				else
				{
					traversal_index = root_node[current_node_index].IndexInFather + 1;
					current_node_index = root_node[current_node_index].Father;
					break;
				}
				if (root_node[current_node_index].Father != -1 && is_last_child)
				{
					current_node_index = root_node[current_node_index].Father;

					if (root_node[current_node_index].Father == -1)
					{
						continue_traversal = false;
						break;
					}
				}
				else
				{
					break;
				}
			}

		}
		else
		{

			bool descend = false;
			for (int c = traversal_index; c < 8; c++)
			{
				int cn_child = root_node[current_node_index].Children[c];
				if (root_node[cn_child].Point_Cloud_Num > 0)
				{
					float3& _massCenter = root_node[cn_child].Mass_Center;
					float _max_R = root_node[cn_child].Max_R;
					float3 MCToq;
					MCToq.x = _massCenter.x - q.x; MCToq.y = _massCenter.y - q.y; MCToq.z = _massCenter.z - q.z;

					float mctop_square = MCToq.x * MCToq.x + MCToq.y * MCToq.y + MCToq.z * MCToq.z;
					float mctop_norm = sqrtf(mctop_square);

					if (mctop_norm > beta * _max_R)
					{
						if (root_node[cn_child].Children[0] == -1)
						{
							for (int i_point_in_cnc = 0; i_point_in_cnc < root_node[cn_child].Point_Cloud_Num; i_point_in_cnc++)
							{
								int p_idx = root_node[cn_child].Point_Cloud_Indices[i_point_in_cnc];
								float3& p = i_points[p_idx];
								float3& N = i_Normals[p_idx];
								float _A = i_As[p_idx];
								float _C = i_Cs[p_idx];

								float3 _R;
								_R.x = p.x - q.x; _R.y = p.y - q.y; _R.z = p.z - q.z;
								float _R_square = _R.x * _R.x + _R.y * _R.y + _R.z * _R.z;
								float _R_norm = sqrtf(_R_square);

								if (_R_norm > 1e-7)
								{
									float screened = 0.0f;
									if (use_screened)
									{
										screened = (use_radii) ? i_radii[p_idx] : 0.0015f;
									}

									if (screened > _R_norm)
									{
										float up =   _A * _C;
										float down =   (PI_4 * screened * screened * screened);
										float res = up / down;

										float c_grad_gwn_x = -res * N.x;
										float c_grad_gwn_y = -res * N.y;
										float c_grad_gwn_z = -res * N.z;

										grad_gwn_x += c_grad_gwn_x;
										grad_gwn_y += c_grad_gwn_y;
										grad_gwn_z += c_grad_gwn_z;
									}
									else
									{
										float gwn_dot = _R.x * N.x + _R.y * N.y + _R.z * N.z;
										float gwn_a = _A * _C;
										float gwn_b =   1.0f;
										float gwn_c = gwn_dot / (PI_4 * _R_norm * _R_norm * _R_norm  );

										float _R_norm_inv = __frsqrt_rn(_R_square);

										float grad_0_x = -_R_norm_inv * _R.x;
										float grad_0_y = -_R_norm_inv * _R.y;
										float grad_0_z = -_R_norm_inv * _R.z;

										float grad_c_f = PI_4 * _R_norm * _R_norm * _R_norm * _R_norm;
										float grad_c_x = (-N.x * _R_norm - 3.0f * gwn_dot * grad_0_x) / grad_c_f;
										float grad_c_y = (-N.y * _R_norm - 3.0f * gwn_dot * grad_0_y) / grad_c_f;
										float grad_c_z = (-N.z * _R_norm - 3.0f * gwn_dot * grad_0_z) / grad_c_f;

										float gwn_ab = gwn_a * gwn_b;

										float c_grad_gwn_x =   grad_c_x * gwn_ab;
										float c_grad_gwn_y =   grad_c_y * gwn_ab;
										float c_grad_gwn_z =   grad_c_z * gwn_ab;

										grad_gwn_x += c_grad_gwn_x;
										grad_gwn_y += c_grad_gwn_y;
										grad_gwn_z += c_grad_gwn_z;
									}
								}

								points_num++;
							}
						}
						else
						{
							float3 _R = MCToq;
							float _R_square = mctop_square;
							float _R_norm = sqrtf(_R_square);
							float3 N = root_node[cn_child].Mass_Normal;

							if (_R_norm > 1e-7)
							{
								float screened = 0.0f;
								if (use_screened)
								{
									screened = 0.0015f;
								}

								if (screened > _R_norm)
								{
									float up = (  1.0f);
									float down =   (PI_4 * screened * screened * screened);
									float res = up / down;

									float c_grad_gwn_x = -res * N.x;
									float c_grad_gwn_y = -res * N.y;
									float c_grad_gwn_z = -res * N.z;

									grad_gwn_x += c_grad_gwn_x;
									grad_gwn_y += c_grad_gwn_y;
									grad_gwn_z += c_grad_gwn_z;
								}
								else
								{
									float gwn_dot = _R.x * N.x + _R.y * N.y + _R.z * N.z;
									float gwn_a = 1.0f;
									float gwn_b =  1.0f;
									float gwn_c = gwn_dot / (PI_4 * _R_norm * _R_norm * _R_norm  );

									float _R_norm_inv = __frsqrt_rn(_R_square);

									float grad_0_x = -_R_norm_inv * _R.x;
									float grad_0_y = -_R_norm_inv * _R.y;
									float grad_0_z = -_R_norm_inv * _R.z;

									float grad_c_f = PI_4 * _R_norm * _R_norm * _R_norm * _R_norm;
									float grad_c_x = (-N.x * _R_norm - 3.0f * gwn_dot * grad_0_x) / grad_c_f;
									float grad_c_y = (-N.y * _R_norm - 3.0f * gwn_dot * grad_0_y) / grad_c_f;
									float grad_c_z = (-N.z * _R_norm - 3.0f * gwn_dot * grad_0_z) / grad_c_f;

									float gwn_ab = gwn_a * gwn_b;

									float c_grad_gwn_x =   grad_c_x * gwn_ab;
									float c_grad_gwn_y =   grad_c_y * gwn_ab;
									float c_grad_gwn_z =   grad_c_z * gwn_ab;

									grad_gwn_x += c_grad_gwn_x;
									grad_gwn_y += c_grad_gwn_y;
									grad_gwn_z += c_grad_gwn_z;
								}
							}
						}
					}
					else
					{
						current_node_index = cn_child;
						descend = true;
						break;
					}
				}
			}

			traversal_index = 0;

			if (!descend)
			{
				while (true)
				{
					bool is_last_child = true;

					if (root_node[current_node_index].IndexInFather == 7)
						is_last_child = true;
					else if (root_node[current_node_index].Father == -1)
					{
						continue_traversal = false;
						break;
					}
					else
					{
						traversal_index = root_node[current_node_index].IndexInFather + 1;
						current_node_index = root_node[current_node_index].Father;
						break;
					}
					if (root_node[current_node_index].Father != -1 && is_last_child)
					{
						current_node_index = root_node[current_node_index].Father;

						if (root_node[current_node_index].Father == -1)
						{
							continue_traversal = false;
							break;
						}
					}
					else
					{
						break;
					}
				}
			}
		}

		iter_num++;
	}

	float3 grad_gwn;
	grad_gwn.x = grad_gwn_x;
	grad_gwn.y = grad_gwn_y;
	grad_gwn.z = grad_gwn_z;

	return grad_gwn;
}

__device__
void GWNGradientBackward(OctreeNodeLibiGPU* __restrict__ root_node, float3& q, float3* __restrict__ i_points, float3* __restrict__ i_Normals, float* __restrict__ i_As, float* __restrict__ i_Cs, float* __restrict__ i_radii,
	float3 gwn_grad,   float* __restrict__ Ai_grad, float* __restrict__ Ci_grad,
	float grad_scale_0, float grad_scale_1, float grad_scale_2, float beta, float sigma, bool use_radii, bool use_screened, bool com_Ai_grad, bool com_Ci_grad, bool com_normals_grad, int normal_mode, int tid)
{

	float grad_gwn_0 = 0.0f;
	float grad_gwn_1 = 0.0f;
	float grad_gwn_2 = 0.0f;
	grad_gwn_0 = 2.0f * grad_scale_0;
	grad_gwn_1 = 2.0f * grad_scale_1;
	grad_gwn_2 = 2.0f * grad_scale_2;

	bool continue_traversal = true;

	int iter_num = 0;
	int points_num = 0;

	int traversal_index = 0;

	int current_node_index = 0;

	float PI_4 = 4.0f * 3.14159265358979323;

	while (continue_traversal)
	{
		if (root_node[current_node_index].Children[0] == -1)
		{

			for (int i_point_in_cn = 0; i_point_in_cn < root_node[current_node_index].Point_Cloud_Num; i_point_in_cn++)
			{
				int p_idx = root_node[current_node_index].Point_Cloud_Indices[i_point_in_cn];
				float3& p = i_points[p_idx];
				float3& N = i_Normals[p_idx];
				float _A = i_As[p_idx];
				float _C = i_Cs[p_idx];

				float3 _R;
				_R.x = p.x - q.x; _R.y = p.y - q.y; _R.z = p.z - q.z;
				float _R_square = _R.x * _R.x + _R.y * _R.y + _R.z * _R.z;
				float _R_norm = sqrtf(_R_square);

				if (_R_norm > 1e-7)
				{
					float screened = 0.0f;
					if (use_screened)
					{
						screened = (use_radii) ? i_radii[p_idx] : 0.0015f;
					}

					if (screened > _R_norm)
					{
						float up =   1.0f;
						float down =   (PI_4 * screened * screened * screened);
						float res = up / down;

						if (com_Ai_grad)
						{
							float res_C = res * _C;

							float c_gwn_x_Ai_grad = (-res_C) * N.x;
							float c_gwn_y_Ai_grad = (-res_C) * N.y;
							float c_gwn_z_Ai_grad = (-res_C) * N.z;

							float c_gwn_xyz_Ai_grad = gwn_grad.x * c_gwn_x_Ai_grad + gwn_grad.y * c_gwn_y_Ai_grad + gwn_grad.z * c_gwn_z_Ai_grad;

							atomicAdd(&(Ai_grad[p_idx]), grad_gwn_1 * c_gwn_xyz_Ai_grad);
						}

						if (com_Ci_grad)
						{
							float res_A = res * _A;

							float c_gwn_x_Ci_grad = (-res_A) * N.x;
							float c_gwn_y_Ci_grad = (-res_A) * N.y;
							float c_gwn_z_Ci_grad = (-res_A) * N.z;

							float c_gwn_xyz_Ci_grad = gwn_grad.x * c_gwn_x_Ci_grad + gwn_grad.y * c_gwn_y_Ci_grad + gwn_grad.z * c_gwn_z_Ci_grad;

							atomicAdd(&(Ci_grad[p_idx]), grad_gwn_2 * c_gwn_xyz_Ci_grad);
						}
					}
					else
					{
						float gwn_b =   1.0f;

						float _R_norm_inv = __frsqrt_rn(_R_square);

						float grad_0_x = -_R_norm_inv * _R.x;
						float grad_0_y = -_R_norm_inv * _R.y;
						float grad_0_z = -_R_norm_inv * _R.z;

						float grad_c_f = PI_4 * _R_norm * _R_norm * _R_norm * _R_norm;

						if (com_Ai_grad)
						{
							float gwn_dot = _R.x * N.x + _R.y * N.y + _R.z * N.z;
							float gwn_c = gwn_dot / (PI_4 * _R_norm * _R_norm * _R_norm  );

							float gwn_a_grad = _C  ;

							float grad_c_x = (-N.x * _R_norm - 3.0f * gwn_dot * grad_0_x) / grad_c_f;
							float grad_c_y = (-N.y * _R_norm - 3.0f * gwn_dot * grad_0_y) / grad_c_f;
							float grad_c_z = (-N.z * _R_norm - 3.0f * gwn_dot * grad_0_z) / grad_c_f;

							float gwn_bc = gwn_b * gwn_c;
							float gwn_ac_Ai_grad = gwn_a_grad * gwn_c;
							float gwn_ab_Ai_grad = gwn_a_grad * gwn_b;

							float c_grad_gwn_x_Ai_grad =   grad_c_x * gwn_ab_Ai_grad;
							float c_grad_gwn_y_Ai_grad =   grad_c_y * gwn_ab_Ai_grad;
							float c_grad_gwn_z_Ai_grad =   grad_c_z * gwn_ab_Ai_grad;

							float c_grad_gwn_xyz_Ai_grad = gwn_grad.x * c_grad_gwn_x_Ai_grad + gwn_grad.y * c_grad_gwn_y_Ai_grad + gwn_grad.z * c_grad_gwn_z_Ai_grad;

							atomicAdd(&(Ai_grad[p_idx]), grad_gwn_1 * c_grad_gwn_xyz_Ai_grad);
						}

						if (com_Ci_grad)
						{
							float gwn_dot = _R.x * N.x + _R.y * N.y + _R.z * N.z;
							float gwn_c = gwn_dot / (PI_4 * _R_norm * _R_norm * _R_norm );

							float gwn_a_grad = _A  ;

							float grad_c_x = (-N.x * _R_norm - 3.0f * gwn_dot * grad_0_x) / grad_c_f;
							float grad_c_y = (-N.y * _R_norm - 3.0f * gwn_dot * grad_0_y) / grad_c_f;
							float grad_c_z = (-N.z * _R_norm - 3.0f * gwn_dot * grad_0_z) / grad_c_f;

							float gwn_bc = gwn_b * gwn_c;
							float gwn_ac_Ci_grad = gwn_a_grad * gwn_c;
							float gwn_ab_Ci_grad = gwn_a_grad * gwn_b;

							float c_grad_gwn_x_Ci_grad =   grad_c_x * gwn_ab_Ci_grad;
							float c_grad_gwn_y_Ci_grad =   grad_c_y * gwn_ab_Ci_grad;
							float c_grad_gwn_z_Ci_grad =   grad_c_z * gwn_ab_Ci_grad;

							float c_grad_gwn_xyz_Ci_grad = gwn_grad.x * c_grad_gwn_x_Ci_grad + gwn_grad.y * c_grad_gwn_y_Ci_grad + gwn_grad.z * c_grad_gwn_z_Ci_grad;

							atomicAdd(&(Ci_grad[p_idx]), grad_gwn_2 * c_grad_gwn_xyz_Ci_grad);
						}

					}
				}

				points_num++;
			}

			while (true)
			{
				bool is_last_child = true;

				if (root_node[current_node_index].IndexInFather == 7)
					is_last_child = true;
				else if (root_node[current_node_index].Father == -1)
				{
					continue_traversal = false;
					break;
				}
				else
				{
					traversal_index = root_node[current_node_index].IndexInFather + 1;
					current_node_index = root_node[current_node_index].Father;
					break;
				}
				if (root_node[current_node_index].Father != -1 && is_last_child)
				{
					current_node_index = root_node[current_node_index].Father;

					if (root_node[current_node_index].Father == -1)
					{
						continue_traversal = false;
						break;
					}
				}
				else
				{
					break;
				}
			}

		}
		else
		{

			bool descend = false;
			for (int c = traversal_index; c < 8; c++)
			{
				int cn_child = root_node[current_node_index].Children[c];
				if (root_node[cn_child].Point_Cloud_Num > 0)
				{
					float3& _massCenter = root_node[cn_child].Mass_Center;
					float _max_R = root_node[cn_child].Max_R;
					float3 MCToq;
					MCToq.x = _massCenter.x - q.x; MCToq.y = _massCenter.y - q.y; MCToq.z = _massCenter.z - q.z;

					float mctop_square = MCToq.x * MCToq.x + MCToq.y * MCToq.y + MCToq.z * MCToq.z;
					float mctop_norm = sqrtf(mctop_square);

					if (mctop_norm > beta * _max_R)
					{
						if (root_node[cn_child].Children[0] == -1)
						{
							for (int i_point_in_cnc = 0; i_point_in_cnc < root_node[cn_child].Point_Cloud_Num; i_point_in_cnc++)
							{
								int p_idx = root_node[cn_child].Point_Cloud_Indices[i_point_in_cnc];
								float3& p = i_points[p_idx];
								float3& N = i_Normals[p_idx];
								float _A = i_As[p_idx];
								float _C = i_Cs[p_idx];

								float3 _R;
								_R.x = p.x - q.x; _R.y = p.y - q.y; _R.z = p.z - q.z;
								float _R_square = _R.x * _R.x + _R.y * _R.y + _R.z * _R.z;
								float _R_norm = sqrtf(_R_square);

								float screened = 0.0f;
								if (use_screened)
								{
									screened = (use_radii) ? i_radii[p_idx] : 0.0015f;
								}

								if (screened > _R_norm)
								{
									float up =   1.0f;
									float down =  (PI_4 * screened * screened * screened);
									float res = up / down;

									if (com_Ai_grad)
									{
										float res_C = res * _C;

										float c_gwn_x_Ai_grad = (-res_C) * N.x;
										float c_gwn_y_Ai_grad = (-res_C) * N.y;
										float c_gwn_z_Ai_grad = (-res_C) * N.z;

										float c_gwn_xyz_Ai_grad = gwn_grad.x * c_gwn_x_Ai_grad + gwn_grad.y * c_gwn_y_Ai_grad + gwn_grad.z * c_gwn_z_Ai_grad;

										atomicAdd(&(Ai_grad[p_idx]), grad_gwn_1 * c_gwn_xyz_Ai_grad);
									}

									if (com_Ci_grad)
									{
										float res_A = res * _A;

										float c_gwn_x_Ci_grad = (-res_A) * N.x;
										float c_gwn_y_Ci_grad = (-res_A) * N.y;
										float c_gwn_z_Ci_grad = (-res_A) * N.z;

										float c_gwn_xyz_Ci_grad = gwn_grad.x * c_gwn_x_Ci_grad + gwn_grad.y * c_gwn_y_Ci_grad + gwn_grad.z * c_gwn_z_Ci_grad;

										atomicAdd(&(Ci_grad[p_idx]), grad_gwn_2 * c_gwn_xyz_Ci_grad);
									}
								}
								else
								{
									float gwn_b =   1.0f;

									float _R_norm_inv = __frsqrt_rn(_R_square);

									float grad_0_x = -_R_norm_inv * _R.x;
									float grad_0_y = -_R_norm_inv * _R.y;
									float grad_0_z = -_R_norm_inv * _R.z;

									float grad_c_f = PI_4 * _R_norm * _R_norm * _R_norm * _R_norm;

									if (com_Ai_grad)
									{
										float gwn_dot = _R.x * N.x + _R.y * N.y + _R.z * N.z;
										float gwn_c = gwn_dot / (PI_4 * _R_norm * _R_norm * _R_norm  );

										float gwn_a_grad = _C  ;

										float grad_c_x = (-N.x * _R_norm - 3.0f * gwn_dot * grad_0_x) / grad_c_f;
										float grad_c_y = (-N.y * _R_norm - 3.0f * gwn_dot * grad_0_y) / grad_c_f;
										float grad_c_z = (-N.z * _R_norm - 3.0f * gwn_dot * grad_0_z) / grad_c_f;

										float gwn_bc = gwn_b * gwn_c;
										float gwn_ac_Ai_grad = gwn_a_grad * gwn_c;
										float gwn_ab_Ai_grad = gwn_a_grad * gwn_b;

										float c_grad_gwn_x_Ai_grad =   grad_c_x * gwn_ab_Ai_grad;
										float c_grad_gwn_y_Ai_grad =   grad_c_y * gwn_ab_Ai_grad;
										float c_grad_gwn_z_Ai_grad =   grad_c_z * gwn_ab_Ai_grad;

										float c_grad_gwn_xyz_Ai_grad = gwn_grad.x * c_grad_gwn_x_Ai_grad + gwn_grad.y * c_grad_gwn_y_Ai_grad + gwn_grad.z * c_grad_gwn_z_Ai_grad;

										atomicAdd(&(Ai_grad[p_idx]), grad_gwn_1 * c_grad_gwn_xyz_Ai_grad);
									}

									if (com_Ci_grad)
									{
										float gwn_dot = _R.x * N.x + _R.y * N.y + _R.z * N.z;
										float gwn_c = gwn_dot / (PI_4 * _R_norm * _R_norm * _R_norm  );

										float gwn_a_grad = _A  ;

										float grad_c_x = (-N.x * _R_norm - 3.0f * gwn_dot * grad_0_x) / grad_c_f;
										float grad_c_y = (-N.y * _R_norm - 3.0f * gwn_dot * grad_0_y) / grad_c_f;
										float grad_c_z = (-N.z * _R_norm - 3.0f * gwn_dot * grad_0_z) / grad_c_f;

										float gwn_bc = gwn_b * gwn_c;
										float gwn_ac_Ci_grad = gwn_a_grad * gwn_c;
										float gwn_ab_Ci_grad = gwn_a_grad * gwn_b;

										float c_grad_gwn_x_Ci_grad =   grad_c_x * gwn_ab_Ci_grad;
										float c_grad_gwn_y_Ci_grad =   grad_c_y * gwn_ab_Ci_grad;
										float c_grad_gwn_z_Ci_grad =   grad_c_z * gwn_ab_Ci_grad;

										float c_grad_gwn_xyz_Ci_grad = gwn_grad.x * c_grad_gwn_x_Ci_grad + gwn_grad.y * c_grad_gwn_y_Ci_grad + gwn_grad.z * c_grad_gwn_z_Ci_grad;

										atomicAdd(&(Ci_grad[p_idx]), grad_gwn_2 * c_grad_gwn_xyz_Ci_grad);
									}
								}

								points_num++;
							}
						}
						else
						{
							float3 _R = MCToq;
							float _R_square = mctop_square;
							float _R_norm = sqrtf(_R_square);
							float3 N = root_node[cn_child].Mass_Normal;

							if (_R_norm > 1e-7)
							{
								float screened = 0.0f;
								if (use_screened)
								{
									screened = 0.0015f;
								}

								if (screened > _R_norm)
								{
									float up =   1.0f;
									float down =   (PI_4 * screened * screened * screened);
									float res = up / down;

									for (int i = 0; i < root_node[cn_child].Point_Cloud_Num; i++)
									{
										int curr_point_index = root_node[cn_child].Point_Cloud_Indices[i];

										if (com_Ai_grad)
										{
											float c_Ni_x = i_Normals[curr_point_index].x;
											float c_Ni_y = i_Normals[curr_point_index].y;
											float c_Ni_z = i_Normals[curr_point_index].z;
											float c_ci = i_Cs[curr_point_index];
											float c_grad_gwn_xyz_Ai_grad = gwn_grad.x * (-res) * c_ci * c_Ni_x + gwn_grad.y * (-res) * c_ci * c_Ni_y + gwn_grad.z * (-res) * c_ci * c_Ni_z;
											atomicAdd(&(Ai_grad[curr_point_index]), grad_gwn_1 * c_grad_gwn_xyz_Ai_grad);
										}

										if (com_Ci_grad)
										{
											float c_Ni_x = i_Normals[curr_point_index].x;
											float c_Ni_y = i_Normals[curr_point_index].y;
											float c_Ni_z = i_Normals[curr_point_index].z;
											float c_ai = i_As[curr_point_index];
											float c_grad_gwn_xyz_Ci_grad = gwn_grad.x * (-res) * c_ai * c_Ni_x + gwn_grad.y * (-res) * c_ai * c_Ni_y + gwn_grad.z * (-res) * c_ai * c_Ni_z;
											atomicAdd(&(Ci_grad[curr_point_index]), grad_gwn_2 * c_grad_gwn_xyz_Ci_grad);
										}
									}
								}
								else
								{
									float gwn_a = 1.0f;
									float gwn_b =   1.0f;

									float _R_norm_inv = __frsqrt_rn(_R_square);

									float grad_0_x = -_R_norm_inv * _R.x;
									float grad_0_y = -_R_norm_inv * _R.y;
									float grad_0_z = -_R_norm_inv * _R.z;

									float grad_c_f = PI_4 * _R_norm * _R_norm * _R_norm * _R_norm;

									float gwn_ab = gwn_a * gwn_b;

									if (com_Ai_grad)
									{
										if (root_node[cn_child].Large_Node)
										{
											int c_qi = atomicAdd(&(root_node[cn_child].Query_Index[0]), 1);
											root_node[cn_child].Query_Buffer[c_qi] = tid;
										}
										else
										{
											float R_norm_inv = __frsqrt_rn(_R_square);

											float gwn_dot = _R.x * N.x + _R.y * N.y + _R.z * N.z;

											float gwn_c0 = PI_4 * _R_norm * _R_norm * _R_norm;
											float gwn_c = gwn_dot / (gwn_c0 + 1e-7);

											float gwn_c_Ai_grad_a = gwn_c0 * _R_norm;

											float grad_c_Ai_grad_a = grad_c_f * grad_c_f;

											float grad_c_x_a = (-N.x * _R_norm - 3.0f * gwn_dot * grad_0_x);
											float grad_c_x = grad_c_x_a / grad_c_f;
											float grad_c_y_a = (-N.y * _R_norm - 3.0f * gwn_dot * grad_0_y);
											float grad_c_y = grad_c_y_a / grad_c_f;
											float grad_c_z_a = (-N.z * _R_norm - 3.0f * gwn_dot * grad_0_z);
											float grad_c_z = grad_c_z_a / grad_c_f;

											for (int i = 0; i < root_node[cn_child].Point_Cloud_Num; i++)
											{
												int curr_point_index = root_node[cn_child].Point_Cloud_Indices[i];

												float N_x_Ai_grad = i_Normals[curr_point_index].x * i_Cs[curr_point_index];
												float N_y_Ai_grad = i_Normals[curr_point_index].y * i_Cs[curr_point_index];
												float N_z_Ai_grad = i_Normals[curr_point_index].z * i_Cs[curr_point_index];

												float R_x_Ai_grad = root_node[cn_child].As_Grad_x[i];
												float R_y_Ai_grad = root_node[cn_child].As_Grad_y[i];
												float R_z_Ai_grad = root_node[cn_child].As_Grad_z[i];

												float R_norm_Ai_grad_a = _R.x * R_x_Ai_grad + _R.y * R_y_Ai_grad + _R.z * R_z_Ai_grad;
												float R_norm_Ai_grad = R_norm_inv * R_norm_Ai_grad_a;
												float R_norm_inv_Ai_grad = -R_norm_inv * R_norm_inv * R_norm_inv * R_norm_Ai_grad_a;

												float gwn_dot_Ai_grad = R_x_Ai_grad * N.x + N_x_Ai_grad * _R.x + R_y_Ai_grad * N.y + N_y_Ai_grad * _R.y + R_z_Ai_grad * N.z + N_z_Ai_grad * _R.z;

												float grad_0_x_Ai_grad = -(R_norm_inv_Ai_grad * _R.x + R_x_Ai_grad * R_norm_inv);
												float grad_0_y_Ai_grad = -(R_norm_inv_Ai_grad * _R.y + R_y_Ai_grad * R_norm_inv);
												float grad_0_z_Ai_grad = -(R_norm_inv_Ai_grad * _R.z + R_z_Ai_grad * R_norm_inv);

												float gwn_c_Ai_grad_b = gwn_dot_Ai_grad * _R_norm - 3.0f * R_norm_Ai_grad * gwn_dot;

												float grad_c_Ai_grad_b = 4.0f * gwn_c0 * R_norm_Ai_grad;

												float grad_c_x_Ai_grad_a = -(N_x_Ai_grad * _R_norm + R_norm_Ai_grad * N.x) - 3.0f * (gwn_dot_Ai_grad * grad_0_x + grad_0_x_Ai_grad * gwn_dot);
												float grad_c_x_Ai_grad_b = grad_c_x_Ai_grad_a * grad_c_f - grad_c_Ai_grad_b * grad_c_x_a;

												float grad_c_x_Ai_grad = grad_c_x_Ai_grad_b / grad_c_Ai_grad_a;

												float grad_c_y_Ai_grad_a = -(N_y_Ai_grad * _R_norm + R_norm_Ai_grad * N.y) - 3.0f * (gwn_dot_Ai_grad * grad_0_y + grad_0_y_Ai_grad * gwn_dot);
												float grad_c_y_Ai_grad_b = grad_c_y_Ai_grad_a * grad_c_f - grad_c_Ai_grad_b * grad_c_y_a;

												float grad_c_y_Ai_grad = grad_c_y_Ai_grad_b / grad_c_Ai_grad_a;

												float grad_c_z_Ai_grad_a = -(N_z_Ai_grad * _R_norm + R_norm_Ai_grad * N.z) - 3.0f * (gwn_dot_Ai_grad * grad_0_z + grad_0_z_Ai_grad * gwn_dot);
												float grad_c_z_Ai_grad_b = grad_c_z_Ai_grad_a * grad_c_f - grad_c_Ai_grad_b * grad_c_z_a;

												float grad_c_z_Ai_grad = grad_c_z_Ai_grad_b / grad_c_Ai_grad_a;

												float gwn_bc = gwn_b * gwn_c;
												float gwn_ac = gwn_a * gwn_c;

												float c_grad_gwn_x_Ai_grad_2 = grad_c_x_Ai_grad * gwn_ab  ;
												float c_grad_gwn_x_Ai_grad =   c_grad_gwn_x_Ai_grad_2;
												c_grad_gwn_x_Ai_grad = grad_c_x_Ai_grad;

												float c_grad_gwn_y_Ai_grad_2 = grad_c_y_Ai_grad * gwn_ab  ;
												float c_grad_gwn_y_Ai_grad =  c_grad_gwn_y_Ai_grad_2;
												c_grad_gwn_y_Ai_grad = grad_c_y_Ai_grad;

												float c_grad_gwn_z_Ai_grad_2 = grad_c_z_Ai_grad * gwn_ab  ;
												float c_grad_gwn_z_Ai_grad =   c_grad_gwn_z_Ai_grad_2;
												c_grad_gwn_z_Ai_grad = grad_c_z_Ai_grad;

												float c_grad_gwn_xyz_Ai_grad = gwn_grad.x * c_grad_gwn_x_Ai_grad + gwn_grad.y * c_grad_gwn_y_Ai_grad + gwn_grad.z * c_grad_gwn_z_Ai_grad;

												atomicAdd(&(Ai_grad[curr_point_index]), grad_gwn_1 * c_grad_gwn_xyz_Ai_grad);

											}
										}
									}

									if (com_Ci_grad)
									{
										if (root_node[cn_child].Large_Node)
										{
											int c_qi = atomicAdd(&(root_node[cn_child].Query_Index[0]), 1);
											root_node[cn_child].Query_Buffer[c_qi] = tid;
										}
										else
										{
											float R_norm_inv = __frsqrt_rn(_R_square);

											float gwn_dot = _R.x * N.x + _R.y * N.y + _R.z * N.z;

											float gwn_c0 = PI_4 * _R_norm * _R_norm * _R_norm;
											float gwn_c = gwn_dot / (gwn_c0  );

											float gwn_c_Ci_grad_a = gwn_c0 * _R_norm;

											float grad_c_Ci_grad_a = grad_c_f * grad_c_f;

											float grad_c_x_a = (-N.x * _R_norm - 3.0f * gwn_dot * grad_0_x);
											float grad_c_x = grad_c_x_a / grad_c_f;
											float grad_c_y_a = (-N.y * _R_norm - 3.0f * gwn_dot * grad_0_y);
											float grad_c_y = grad_c_y_a / grad_c_f;
											float grad_c_z_a = (-N.z * _R_norm - 3.0f * gwn_dot * grad_0_z);
											float grad_c_z = grad_c_z_a / grad_c_f;

											for (int i = 0; i < root_node[cn_child].Point_Cloud_Num; i++)
											{
												int curr_point_index = root_node[cn_child].Point_Cloud_Indices[i];

												float N_x_Ci_grad = i_Normals[curr_point_index].x * i_As[curr_point_index];
												float N_y_Ci_grad = i_Normals[curr_point_index].y * i_As[curr_point_index];
												float N_z_Ci_grad = i_Normals[curr_point_index].z * i_As[curr_point_index];

												float R_x_Ci_grad = root_node[cn_child].Cs_Grad_x[i];
												float R_y_Ci_grad = root_node[cn_child].Cs_Grad_y[i];
												float R_z_Ci_grad = root_node[cn_child].Cs_Grad_z[i];

												float R_norm_Ci_grad_a = _R.x * R_x_Ci_grad + _R.y * R_y_Ci_grad + _R.z * R_z_Ci_grad;
												float R_norm_Ci_grad = R_norm_inv * R_norm_Ci_grad_a;
												float R_norm_inv_Ci_grad = -R_norm_inv * R_norm_inv * R_norm_inv * R_norm_Ci_grad_a;

												float gwn_dot_Ci_grad = R_x_Ci_grad * N.x + N_x_Ci_grad * _R.x + R_y_Ci_grad * N.y + N_y_Ci_grad * _R.y + R_z_Ci_grad * N.z + N_z_Ci_grad * _R.z;

												float grad_0_x_Ci_grad = -(R_norm_inv_Ci_grad * _R.x + R_x_Ci_grad * R_norm_inv);
												float grad_0_y_Ci_grad = -(R_norm_inv_Ci_grad * _R.y + R_y_Ci_grad * R_norm_inv);
												float grad_0_z_Ci_grad = -(R_norm_inv_Ci_grad * _R.z + R_z_Ci_grad * R_norm_inv);

												float gwn_c_Ci_grad_b = gwn_dot_Ci_grad * _R_norm - 3.0f * R_norm_Ci_grad * gwn_dot;

												float grad_c_Ci_grad_b = 4.0f * gwn_c0 * R_norm_Ci_grad;

												float grad_c_x_Ci_grad_a = -(N_x_Ci_grad * _R_norm + R_norm_Ci_grad * N.x) - 3.0f * (gwn_dot_Ci_grad * grad_0_x + grad_0_x_Ci_grad * gwn_dot);
												float grad_c_x_Ci_grad_b = grad_c_x_Ci_grad_a * grad_c_f - grad_c_Ci_grad_b * grad_c_x_a;

												float grad_c_x_Ci_grad = grad_c_x_Ci_grad_b / grad_c_Ci_grad_a;

												float grad_c_y_Ci_grad_a = -(N_y_Ci_grad * _R_norm + R_norm_Ci_grad * N.y) - 3.0f * (gwn_dot_Ci_grad * grad_0_y + grad_0_y_Ci_grad * gwn_dot);
												float grad_c_y_Ci_grad_b = grad_c_y_Ci_grad_a * grad_c_f - grad_c_Ci_grad_b * grad_c_y_a;

												float grad_c_y_Ci_grad = grad_c_y_Ci_grad_b / grad_c_Ci_grad_a;

												float grad_c_z_Ci_grad_a = -(N_z_Ci_grad * _R_norm + R_norm_Ci_grad * N.z) - 3.0f * (gwn_dot_Ci_grad * grad_0_z + grad_0_z_Ci_grad * gwn_dot);
												float grad_c_z_Ci_grad_b = grad_c_z_Ci_grad_a * grad_c_f - grad_c_Ci_grad_b * grad_c_z_a;

												float grad_c_z_Ci_grad = grad_c_z_Ci_grad_b / grad_c_Ci_grad_a;

												float gwn_bc = gwn_b * gwn_c;
												float gwn_ac = gwn_a * gwn_c;

												float c_grad_gwn_x_Ci_grad_2 = grad_c_x_Ci_grad * gwn_ab  ;
												float c_grad_gwn_x_Ci_grad =   c_grad_gwn_x_Ci_grad_2;
												c_grad_gwn_x_Ci_grad = grad_c_x_Ci_grad;

												float c_grad_gwn_y_Ci_grad_2 = grad_c_y_Ci_grad * gwn_ab  ;
												float c_grad_gwn_y_Ci_grad =   c_grad_gwn_y_Ci_grad_2;
												c_grad_gwn_y_Ci_grad = grad_c_y_Ci_grad;

												float c_grad_gwn_z_Ci_grad_2 = grad_c_z_Ci_grad * gwn_ab  ;
												float c_grad_gwn_z_Ci_grad =   c_grad_gwn_z_Ci_grad_2;
												c_grad_gwn_z_Ci_grad = grad_c_z_Ci_grad;

												float c_grad_gwn_xyz_Ci_grad = gwn_grad.x * c_grad_gwn_x_Ci_grad + gwn_grad.y * c_grad_gwn_y_Ci_grad + gwn_grad.z * c_grad_gwn_z_Ci_grad;

												atomicAdd(&(Ci_grad[curr_point_index]), grad_gwn_2 * c_grad_gwn_xyz_Ci_grad);
											}
										}
									}
								}
							}
						}
					}
					else
					{
						current_node_index = cn_child;
						descend = true;
						break;
					}
				}
			}

			traversal_index = 0;

			if (!descend)
			{
				while (true)
				{
					bool is_last_child = true;

					if (root_node[current_node_index].IndexInFather == 7)
						is_last_child = true;
					else if (root_node[current_node_index].Father == -1)
					{
						continue_traversal = false;
						break;
					}
					else
					{
						traversal_index = root_node[current_node_index].IndexInFather + 1;
						current_node_index = root_node[current_node_index].Father;
						break;
					}
					if (root_node[current_node_index].Father != -1 && is_last_child)
					{
						current_node_index = root_node[current_node_index].Father;

						if (root_node[current_node_index].Father == -1)
						{
							continue_traversal = false;
							break;
						}
					}
					else
					{
						break;
					}
				}
			}
		}

		iter_num++;
	}
}

__device__
void GWNGradientAsPatchBackward(OctreeNodeLibiGPU* __restrict__ node, int point_index, int point_local_index, float3* __restrict__ query_points, float3* __restrict__ gwn_grad, float* __restrict__ grad_volume_scale,
	float* __restrict__ Ai_grad, float3 p_Ni, float p_Ci, float grad_scale)
{
	float3 mass_center = node->Mass_Center;
	float3 mass_normal = node->Mass_Normal;

	float PI_4 = 4.0f * 3.14159265358979323;

	float point_sum_grad = 0.0;
	for (int q_i = 0; q_i < node->Query_Index[0]; q_i++)
	{
		int c_qi = node->Query_Buffer[q_i];

		float3 q = query_points[c_qi];
		float3 c_q_gwn_grad = gwn_grad[c_qi];
		float c_q_grad_scale = 2.0f * grad_scale * grad_volume_scale[c_qi];

		float3 MCToq;
		MCToq.x = mass_center.x - q.x; MCToq.y = mass_center.y - q.y; MCToq.z = mass_center.z - q.z;

		float mctop_square = MCToq.x * MCToq.x + MCToq.y * MCToq.y + MCToq.z * MCToq.z;
		float mctop_norm = sqrtf(mctop_square);

		float3 _R = MCToq;
		float _R_square = mctop_square;
		float _R_norm = sqrtf(_R_square);

		float R_norm_inv = __frsqrt_rn(_R_square);

		float gwn_dot = _R.x * mass_normal.x + _R.y * mass_normal.y + _R.z * mass_normal.z;

		float gwn_c0 = PI_4 * _R_norm * _R_norm * _R_norm;
		float gwn_c = gwn_dot / (gwn_c0 + 1e-7);

		float gwn_c_Ai_grad_a = gwn_c0 * _R_norm;

		float gwn_a = 1.0f;
		float gwn_b = 1.0f;

		float _R_norm_inv = __frsqrt_rn(_R_square);

		float grad_0_x = -_R_norm_inv * _R.x;
		float grad_0_y = -_R_norm_inv * _R.y;
		float grad_0_z = -_R_norm_inv * _R.z;

		float grad_c_f = PI_4 * _R_norm * _R_norm * _R_norm * _R_norm;

		float gwn_ab = gwn_a * gwn_b;

		float grad_c_Ai_grad_a = grad_c_f * grad_c_f;

		float grad_c_x_a = (-mass_normal.x * _R_norm - 3.0f * gwn_dot * grad_0_x);

		float grad_c_y_a = (-mass_normal.y * _R_norm - 3.0f * gwn_dot * grad_0_y);

		float grad_c_z_a = (-mass_normal.z * _R_norm - 3.0f * gwn_dot * grad_0_z);

		float N_x_Ai_grad = p_Ni.x * p_Ci;
		float N_y_Ai_grad = p_Ni.y * p_Ci;
		float N_z_Ai_grad = p_Ni.z * p_Ci;

		float R_x_Ai_grad = node->As_Grad_x[point_local_index];
		float R_y_Ai_grad = node->As_Grad_y[point_local_index];
		float R_z_Ai_grad = node->As_Grad_z[point_local_index];

		float R_norm_Ai_grad_a = _R.x * R_x_Ai_grad + _R.y * R_y_Ai_grad + _R.z * R_z_Ai_grad;
		float R_norm_Ai_grad = R_norm_inv * R_norm_Ai_grad_a;
		float R_norm_inv_Ai_grad = -R_norm_inv * R_norm_inv * R_norm_inv * R_norm_Ai_grad_a;

		float gwn_dot_Ai_grad = R_x_Ai_grad * mass_normal.x + N_x_Ai_grad * _R.x + R_y_Ai_grad * mass_normal.y + N_y_Ai_grad * _R.y + R_z_Ai_grad * mass_normal.z + N_z_Ai_grad * _R.z;

		float grad_0_x_Ai_grad = -(R_norm_inv_Ai_grad * _R.x + R_x_Ai_grad * R_norm_inv);
		float grad_0_y_Ai_grad = -(R_norm_inv_Ai_grad * _R.y + R_y_Ai_grad * R_norm_inv);
		float grad_0_z_Ai_grad = -(R_norm_inv_Ai_grad * _R.z + R_z_Ai_grad * R_norm_inv);

		float gwn_c_Ai_grad_b = gwn_dot_Ai_grad * _R_norm - 3.0f * R_norm_Ai_grad * gwn_dot;

		float grad_c_Ai_grad_b = 4.0f * gwn_c0 * R_norm_Ai_grad;

		float grad_c_x_Ai_grad_a = -(N_x_Ai_grad * _R_norm + R_norm_Ai_grad * mass_normal.x) - 3.0f * (gwn_dot_Ai_grad * grad_0_x + grad_0_x_Ai_grad * gwn_dot);
		float grad_c_x_Ai_grad_b = grad_c_x_Ai_grad_a * grad_c_f - grad_c_Ai_grad_b * grad_c_x_a;
		float grad_c_x_Ai_grad = grad_c_x_Ai_grad_b / grad_c_Ai_grad_a;

		float grad_c_y_Ai_grad_a = -(N_y_Ai_grad * _R_norm + R_norm_Ai_grad * mass_normal.y) - 3.0f * (gwn_dot_Ai_grad * grad_0_y + grad_0_y_Ai_grad * gwn_dot);
		float grad_c_y_Ai_grad_b = grad_c_y_Ai_grad_a * grad_c_f - grad_c_Ai_grad_b * grad_c_y_a;
		float grad_c_y_Ai_grad = grad_c_y_Ai_grad_b / grad_c_Ai_grad_a;

		float grad_c_z_Ai_grad_a = -(N_z_Ai_grad * _R_norm + R_norm_Ai_grad * mass_normal.z) - 3.0f * (gwn_dot_Ai_grad * grad_0_z + grad_0_z_Ai_grad * gwn_dot);
		float grad_c_z_Ai_grad_b = grad_c_z_Ai_grad_a * grad_c_f - grad_c_Ai_grad_b * grad_c_z_a;
		float grad_c_z_Ai_grad = grad_c_z_Ai_grad_b / grad_c_Ai_grad_a;

		float c_grad_gwn_x_Ai_grad_2 = grad_c_x_Ai_grad * gwn_ab ;
		float c_grad_gwn_x_Ai_grad = c_grad_gwn_x_Ai_grad_2;
		c_grad_gwn_x_Ai_grad = grad_c_x_Ai_grad;

		float c_grad_gwn_y_Ai_grad_2 = grad_c_y_Ai_grad * gwn_ab;
		float c_grad_gwn_y_Ai_grad = c_grad_gwn_y_Ai_grad_2;
		c_grad_gwn_y_Ai_grad = grad_c_y_Ai_grad;

		float c_grad_gwn_z_Ai_grad_2 = grad_c_z_Ai_grad * gwn_ab;
		float c_grad_gwn_z_Ai_grad = c_grad_gwn_z_Ai_grad_2;
		c_grad_gwn_z_Ai_grad = grad_c_z_Ai_grad;

		float c_grad_gwn_xyz_Ai_grad = c_q_gwn_grad.x * c_grad_gwn_x_Ai_grad + c_q_gwn_grad.y * c_grad_gwn_y_Ai_grad + c_q_gwn_grad.z * c_grad_gwn_z_Ai_grad;

		float c_final_grad = c_q_grad_scale * c_grad_gwn_xyz_Ai_grad;

		point_sum_grad += c_final_grad;
	}
	atomicAdd(&(Ai_grad[point_index]), point_sum_grad);
}

__device__
void GWNGradientCsPatchBackward(OctreeNodeLibiGPU* __restrict__ node, int point_index, int point_local_index, float3* __restrict__ query_points, float3* __restrict__ gwn_grad, float* __restrict__ grad_volume_scale,
	float* __restrict__ Ci_grad, float3 p_Ni, float p_Ai, float grad_scale)
{
	float3 mass_center = node->Mass_Center;
	float3 mass_normal = node->Mass_Normal;

	float PI_4 = 4.0f * 3.14159265358979323;

	float point_sum_grad = 0.0f;
	for (int q_i = 0; q_i < node->Query_Index[0]; q_i++)
	{
		int c_qi = node->Query_Buffer[q_i];

		float3 q = query_points[c_qi];
		float3 c_q_gwn_grad = gwn_grad[c_qi];
		float c_q_grad_scale = 2.0f * grad_scale * grad_volume_scale[c_qi];

		float3 MCToq;
		MCToq.x = mass_center.x - q.x; MCToq.y = mass_center.y - q.y; MCToq.z = mass_center.z - q.z;

		float mctop_square = MCToq.x * MCToq.x + MCToq.y * MCToq.y + MCToq.z * MCToq.z;
		float mctop_norm = sqrtf(mctop_square);

		float3 _R = MCToq;
		float _R_square = mctop_square;
		float _R_norm = sqrtf(_R_square);

		float R_norm_inv = __frsqrt_rn(_R_square);

		float gwn_dot = _R.x * mass_normal.x + _R.y * mass_normal.y + _R.z * mass_normal.z;

		float gwn_c0 = PI_4 * _R_norm * _R_norm * _R_norm;
		float gwn_c = gwn_dot / (gwn_c0 + 1e-7);

		float gwn_c_Ci_grad_a = gwn_c0 * _R_norm;

		float gwn_a = 1.0f;
		float gwn_b = 1.0f;

		float _R_norm_inv = __frsqrt_rn(_R_square);

		float grad_0_x = -_R_norm_inv * _R.x;
		float grad_0_y = -_R_norm_inv * _R.y;
		float grad_0_z = -_R_norm_inv * _R.z;

		float grad_c_f = PI_4 * _R_norm * _R_norm * _R_norm * _R_norm;

		float gwn_ab = gwn_a * gwn_b;

		float grad_c_Ci_grad_a = grad_c_f * grad_c_f;

		float grad_c_x_a = (-mass_normal.x * _R_norm - 3.0f * gwn_dot * grad_0_x);

		float grad_c_y_a = (-mass_normal.y * _R_norm - 3.0f * gwn_dot * grad_0_y);

		float grad_c_z_a = (-mass_normal.z * _R_norm - 3.0f * gwn_dot * grad_0_z);

		float N_x_Ci_grad = p_Ni.x * p_Ai;
		float N_y_Ci_grad = p_Ni.y * p_Ai;
		float N_z_Ci_grad = p_Ni.z * p_Ai;

		float R_x_Ci_grad = node->Cs_Grad_x[point_local_index];
		float R_y_Ci_grad = node->Cs_Grad_y[point_local_index];
		float R_z_Ci_grad = node->Cs_Grad_z[point_local_index];

		float R_norm_Ci_grad_a = _R.x * R_x_Ci_grad + _R.y * R_y_Ci_grad + _R.z * R_z_Ci_grad;
		float R_norm_Ci_grad = R_norm_inv * R_norm_Ci_grad_a;
		float R_norm_inv_Ci_grad = -R_norm_inv * R_norm_inv * R_norm_inv * R_norm_Ci_grad_a;

		float gwn_dot_Ci_grad = R_x_Ci_grad * mass_normal.x + N_x_Ci_grad * _R.x + R_y_Ci_grad * mass_normal.y + N_y_Ci_grad * _R.y + R_z_Ci_grad * mass_normal.z + N_z_Ci_grad * _R.z;

		float grad_0_x_Ci_grad = -(R_norm_inv_Ci_grad * _R.x + R_x_Ci_grad * R_norm_inv);
		float grad_0_y_Ci_grad = -(R_norm_inv_Ci_grad * _R.y + R_y_Ci_grad * R_norm_inv);
		float grad_0_z_Ci_grad = -(R_norm_inv_Ci_grad * _R.z + R_z_Ci_grad * R_norm_inv);

		float gwn_c_Ci_grad_b = gwn_dot_Ci_grad * _R_norm - 3.0f * R_norm_Ci_grad * gwn_dot;

		float grad_c_Ci_grad_b = 4.0f * gwn_c0 * R_norm_Ci_grad;

		float grad_c_x_Ci_grad_a = -(N_x_Ci_grad * _R_norm + R_norm_Ci_grad * mass_normal.x) - 3.0f * (gwn_dot_Ci_grad * grad_0_x + grad_0_x_Ci_grad * gwn_dot);
		float grad_c_x_Ci_grad_b = grad_c_x_Ci_grad_a * grad_c_f - grad_c_Ci_grad_b * grad_c_x_a;
		float grad_c_x_Ci_grad = grad_c_x_Ci_grad_b / grad_c_Ci_grad_a;

		float grad_c_y_Ci_grad_a = -(N_y_Ci_grad * _R_norm + R_norm_Ci_grad * mass_normal.y) - 3.0f * (gwn_dot_Ci_grad * grad_0_y + grad_0_y_Ci_grad * gwn_dot);
		float grad_c_y_Ci_grad_b = grad_c_y_Ci_grad_a * grad_c_f - grad_c_Ci_grad_b * grad_c_y_a;
		float grad_c_y_Ci_grad = grad_c_y_Ci_grad_b / grad_c_Ci_grad_a;

		float grad_c_z_Ci_grad_a = -(N_z_Ci_grad * _R_norm + R_norm_Ci_grad * mass_normal.z) - 3.0f * (gwn_dot_Ci_grad * grad_0_z + grad_0_z_Ci_grad * gwn_dot);
		float grad_c_z_Ci_grad_b = grad_c_z_Ci_grad_a * grad_c_f - grad_c_Ci_grad_b * grad_c_z_a;
		float grad_c_z_Ci_grad = grad_c_z_Ci_grad_b / grad_c_Ci_grad_a;

		float c_grad_gwn_x_Ci_grad_2 = grad_c_x_Ci_grad * gwn_ab;
		float c_grad_gwn_x_Ci_grad = c_grad_gwn_x_Ci_grad_2;
		c_grad_gwn_x_Ci_grad = grad_c_x_Ci_grad;

		float c_grad_gwn_y_Ci_grad_2 = grad_c_y_Ci_grad * gwn_ab;
		float c_grad_gwn_y_Ci_grad = c_grad_gwn_y_Ci_grad_2;
		c_grad_gwn_y_Ci_grad = grad_c_y_Ci_grad;

		float c_grad_gwn_z_Ci_grad_2 = grad_c_z_Ci_grad * gwn_ab;
		float c_grad_gwn_z_Ci_grad = c_grad_gwn_z_Ci_grad_2;
		c_grad_gwn_z_Ci_grad = grad_c_z_Ci_grad;

		float c_grad_gwn_xyz_Ci_grad = c_q_gwn_grad.x * c_grad_gwn_x_Ci_grad + c_q_gwn_grad.y * c_grad_gwn_y_Ci_grad + c_q_gwn_grad.z * c_grad_gwn_z_Ci_grad;

		float c_final_grad = c_q_grad_scale * c_grad_gwn_xyz_Ci_grad;

		point_sum_grad += c_final_grad;
	}

	atomicAdd(&(Ci_grad[point_index]), point_sum_grad);
}

__device__
void GWNAsPatchBackward(OctreeNodeLibiGPU* __restrict__ node, int point_index, int point_local_index, float3* __restrict__ query_points,
	float* __restrict__ point_node_grad, int* __restrict__ point_grad_index, int large_node_size,
	float* __restrict__ gwn, float* __restrict__ Ai_grad, float3 p_Ni, float p_Ci, float target_gwn, float grad_scale)
{
	float3 mass_center = node->Mass_Center;
	float3 mass_normal = node->Mass_Normal;

	float PI_4 = 4.0f * 3.14159265358979323;

	float point_sum_grad = 0.0;
	for (int q_i = 0; q_i < node->Query_Index[0]; q_i++)
	{
		int c_qi = node->Query_Buffer[q_i];

		float3 q = query_points[c_qi];
		float c_q_gwn = gwn[c_qi];

		float3 MCToq;
		MCToq.x = mass_center.x - q.x; MCToq.y = mass_center.y - q.y; MCToq.z = mass_center.z - q.z;

		float mctop_square = MCToq.x * MCToq.x + MCToq.y * MCToq.y + MCToq.z * MCToq.z;
		float mctop_norm = sqrtf(mctop_square);

		float _R_square = mctop_square;
		float _R_norm = sqrtf(_R_square);

		float dot_RN = MCToq.x * mass_normal.x + MCToq.y * mass_normal.y + MCToq.z * mass_normal.z;

		float f_b = 1.0f;
		float f_c = 1.0f;
		float f_d1 = PI_4 * _R_norm * _R_norm * _R_norm;
		float f_d1_2 = f_d1 * f_d1  ;

		float R_norm_a = _R_norm  ;
		float grad_a = (MCToq.x * node->As_Grad_x[point_local_index] +
			MCToq.y * node->As_Grad_y[point_local_index] +
			MCToq.z * node->As_Grad_z[point_local_index]) / R_norm_a;

		float grad_d0 = p_Ni.x * p_Ci * MCToq.x + node->As_Grad_x[point_local_index] * mass_normal.x +
						p_Ni.y * p_Ci * MCToq.y + node->As_Grad_y[point_local_index] * mass_normal.y +
						p_Ni.z * p_Ci * MCToq.z + node->As_Grad_z[point_local_index] * mass_normal.z;
		float grad_d1 = 3.0f * PI_4 * _R_norm * _R_norm * grad_a;
		float grad_d = (grad_d0 * f_d1 - grad_d1 * dot_RN) / f_d1_2;

		float As_grad_s = 2.0f * grad_scale * (c_q_gwn - target_gwn);
		float c_ai_grad = As_grad_s * f_b * f_c * grad_d;

		point_sum_grad += c_ai_grad;
	}
	atomicAdd(&(Ai_grad[point_index]), point_sum_grad);

}

__device__
void GWNCsPatchBackward(OctreeNodeLibiGPU* __restrict__ node, int point_index, int point_local_index, float3* __restrict__ query_points, float* __restrict__ gwn,
	float* __restrict__ Ci_grad, float3 p_Ni, float p_Ai, float target_gwn, float grad_scale)
{
	float3 mass_center = node->Mass_Center;
	float3 mass_normal = node->Mass_Normal;

	float PI_4 = 4.0f * 3.14159265358979323;

	float point_sum_grad = 0.0f;
	for (int q_i = 0; q_i < node->Query_Index[0]; q_i++)
	{
		int c_qi = node->Query_Buffer[q_i];

		float3 q = query_points[c_qi];
		float c_q_gwn = gwn[c_qi];

		float3 MCToq;
		MCToq.x = mass_center.x - q.x; MCToq.y = mass_center.y - q.y; MCToq.z = mass_center.z - q.z;

		float mctop_square = MCToq.x * MCToq.x + MCToq.y * MCToq.y + MCToq.z * MCToq.z;
		float mctop_norm = sqrtf(mctop_square);

		float _R_square = mctop_square;
		float _R_norm = sqrtf(_R_square);

		float dot_RN = MCToq.x * mass_normal.x + MCToq.y * mass_normal.y + MCToq.z * mass_normal.z;

		float f_b = 1.0f;
		float f_c = 1.0f;
		float f_d1 = PI_4 * _R_norm * _R_norm * _R_norm;
		float f_d1_2 = f_d1 * f_d1  ;

		float R_norm_a = _R_norm  ;
		float grad_a = (MCToq.x * node->Cs_Grad_x[point_local_index] +
			MCToq.y * node->Cs_Grad_y[point_local_index] +
			MCToq.z * node->Cs_Grad_z[point_local_index]) / R_norm_a;

		float grad_d0 = p_Ni.x * p_Ai * MCToq.x + node->Cs_Grad_x[point_local_index] * mass_normal.x +
			p_Ni.y * p_Ai * MCToq.y + node->Cs_Grad_y[point_local_index] * mass_normal.y +
			p_Ni.z * p_Ai * MCToq.z + node->Cs_Grad_z[point_local_index] * mass_normal.z;
		float grad_d1 = 3.0f * PI_4 * _R_norm * _R_norm * grad_a;
		float grad_d = (grad_d0 * f_d1 - grad_d1 * dot_RN) / f_d1_2;

		float Cs_grad_s = 2.0f * grad_scale * (c_q_gwn - target_gwn);
		float c_ai_grad = Cs_grad_s * f_b * f_c * grad_d;

		point_sum_grad += c_ai_grad;
	}
	atomicAdd(&(Ci_grad[point_index]), point_sum_grad);
}

__global__
void TurboLibiScreenedGWNAsCsBackwardKernal(int query_size, OctreeNodeLibiGPU* root_node, float3* q, float3* i_points, float3* i_Normals, float* i_As, float* i_Cs, float* i_radii,
	float beta, float sigma, bool use_radii, bool use_screened, float* target_gwn, float* grad_scale_Ai, float grad_scale_Ci, float* gwn_results, float* Ai_grad, float* Ci_grad, bool com_Ai, bool com_Ci, bool patch)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	grad_scale_Ci *= 1.0f / ((float)query_size);
	for (int tid_i = tid; tid_i < query_size; tid_i += stride)
	{
		float c_grad_scale_Ai = grad_scale_Ai[tid_i] * (1.0f / ((float)query_size));
		ScreenedGWNLibiAsCsBackward(root_node, q[tid_i], i_points, i_Normals, i_As, i_Cs, i_radii,
			gwn_results[tid_i], beta, sigma, use_radii, use_screened, target_gwn[tid_i],
			c_grad_scale_Ai, grad_scale_Ci, Ai_grad, Ci_grad, com_Ai, com_Ci, patch, tid_i);
	}
}

__global__
void TurboGWNGradidentKernal(int query_size, OctreeNodeLibiGPU* root_node, float3* q, float3* i_points, float3* i_Normals, float* i_As, float* i_Cs, float* i_radii, float beta, float sigma, float3* gwn_grad, bool use_radii, bool use_screened)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int tid_i = tid; tid_i < query_size; tid_i += stride)
	{
		gwn_grad[tid_i] = GWNGradient(root_node, q[tid_i], i_points, i_Normals, i_As, i_Cs, i_radii, beta, sigma, use_radii, use_screened);
	}
}

__global__
void TurboGWNGradidentBackwardKernal(int query_size, OctreeNodeLibiGPU* root_node, float3* q, float3* i_points, float3* i_Normals, float* i_As, float* i_Cs, float* i_radii,
	float3* gwn_grads,   float* __restrict__ Ai_grad, float* __restrict__ Ci_grad,
	float* grad_scale_coff, float grad_volume_scale, float beta, float sigma, bool use_radii, bool use_screened, bool com_Ai_grad, bool com_Ci_grad, bool com_normals_grad, int normal_mode)
{
	int tid = threadIdx.x + blockIdx.x * blockDim.x;
	int stride = blockDim.x * gridDim.x;

	for (int tid_i = tid; tid_i < query_size; tid_i += stride)
	{
		float grad_scale_0 = grad_scale_coff[tid_i] * grad_volume_scale;
		float grad_scale_1 = grad_scale_coff[tid_i] * grad_volume_scale;
		float grad_scale_2 = grad_scale_coff[tid_i] * grad_volume_scale;
		GWNGradientBackward(root_node, q[tid_i], i_points, i_Normals, i_As, i_Cs, i_radii,
			gwn_grads[tid_i],   Ai_grad, Ci_grad,
			grad_scale_0, grad_scale_1, grad_scale_2, beta, sigma, use_radii, use_screened, com_Ai_grad, com_Ci_grad, com_normals_grad, normal_mode, tid_i);
	}
}

__global__
void TurboGWNGradidentAsPatchBackwardKernal(OctreeNodeLibiGPU* root_node, int* node_array, int* node_point_index, int* node_point_local_index, float3* query_points, float3* i_Normal, float* i_Cs, float3* gwn_grads, float* Ai_grad, float* grad_scale_coff, float grad_volume_scale, int query_size, int array_size)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < array_size)
	{
		OctreeNodeLibiGPU* tid_node = &root_node[node_array[tid]];
		int tid_p_index = node_point_index[tid];
		int tid_p_local_index = node_point_local_index[tid];

		GWNGradientAsPatchBackward(tid_node, tid_p_index, tid_p_local_index, query_points, gwn_grads, grad_scale_coff, Ai_grad, i_Normal[tid_p_index], i_Cs[tid_p_index], grad_volume_scale);
	}
}

__global__
void TurboGWNGradidentCsPatchBackwardKernal(OctreeNodeLibiGPU* root_node, int* node_array, int* node_point_index, int* node_point_local_index, float3* query_points, float3* i_Normal, float* i_As, float3* gwn_grads, float* Ci_grad, float* grad_scale_coff, float grad_volume_scale, int query_size, int array_size)
{

	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < array_size)
	{
		OctreeNodeLibiGPU* tid_node = &root_node[node_array[tid]];
		int tid_p_index = node_point_index[tid];
		int tid_p_local_index = node_point_local_index[tid];

		GWNGradientCsPatchBackward(tid_node, tid_p_index, tid_p_local_index, query_points, gwn_grads, grad_scale_coff, Ci_grad, i_Normal[tid_p_index], i_As[tid_p_index], grad_volume_scale);
	}
}

__global__
void TurboGWNAsPatchBackwardKernal(OctreeNodeLibiGPU* root_node, int* node_array, int* node_point_index, int* node_point_local_index, float3* query_points,
	float* point_node_grad, int* point_grad_index, int last_node_size,
	float3* i_Normal, float* i_Cs, float* gwn, float* Ai_grad, float grad_scale, float target_gwn, int array_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < array_size)
	{
		OctreeNodeLibiGPU* tid_node = &root_node[node_array[tid]];
		int tid_p_index = node_point_index[tid];
		int tid_p_local_index = node_point_local_index[tid];

		GWNAsPatchBackward(tid_node, tid_p_index, tid_p_local_index, query_points,
			point_node_grad, point_grad_index, last_node_size,
			gwn, Ai_grad, i_Normal[tid_p_index], i_Cs[tid_p_index], target_gwn, grad_scale);
	}
}

__global__
void TurboGWNCsPatchBackwardKernal(OctreeNodeLibiGPU* root_node, int* node_array, int* node_point_index, int* node_point_local_index, float3* query_points, float3* i_Normal, float* i_As, float* gwn, float* Ci_grad, float grad_scale, float target_gwn, int array_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < array_size)
	{
		OctreeNodeLibiGPU* tid_node = &root_node[node_array[tid]];
		int tid_p_index = node_point_index[tid];
		int tid_p_local_index = node_point_local_index[tid];

		GWNCsPatchBackward(tid_node, tid_p_index, tid_p_local_index, query_points, gwn, Ci_grad, i_Normal[tid_p_index], i_As[tid_p_index], target_gwn, grad_scale);
	}
}

__global__
void GradidentPatchResetKernal(OctreeNodeLibiGPU* __restrict__ node, int* __restrict__ large_node_index, int large_node_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < large_node_size)
	{
		OctreeNodeLibiGPU* c_node = &node[large_node_index[tid]];
		for (int i = 0; i <= c_node->Query_Index[0]; i++)
		{
			c_node->Query_Buffer[i] = -1;
		}

		c_node->Query_Index[0] = 0;
	}
}

void TurboLibiGWN::TurboLibiScreenedGWNAsCsBackwardLaunch(float3* query_points, float3* point_cloud, int query_size, float3* i_Normals, float* i_As, float* i_Cs, float* i_radii,
	float& beta, float& sigma, bool use_radii, bool use_screened, float* target_gwn, float* grad_scale_Ai, float grad_scale_Ci, float* gwn_results,
	float* Ai_grad, float* Ci_grad, bool com_Ai, bool com_Ci, bool patch)
{
	int blockSize = 128;
	dim3 numBlocks = computeNbBlocks((unsigned int)query_size, blockSize);

	TurboLibiScreenedGWNAsCsBackwardKernal << <numBlocks, blockSize >> > (query_size, this->m_Octree_GPU, query_points, point_cloud, i_Normals, i_As, i_Cs, i_radii,
		beta, sigma, use_radii, use_screened, target_gwn, grad_scale_Ai, grad_scale_Ci, gwn_results, Ai_grad, Ci_grad, com_Ai, com_Ci, patch);

	cudaDeviceSynchronize();

	if (com_Ai)
		std::cout << "GWN Ai Backward Done" << std::endl;
	else if (com_Ci)
		std::cout << "GWN Ci Backward Done" << std::endl;
}

void TurboLibiGWN::TurboGWNGradientLaunch(float3* query_points, float3* point_cloud, int query_size, float3* i_Normals, float* i_As, float* i_Cs, float* i_radii, float& beta, float& sigma, float3* out_gwn_grad, bool use_radii, bool use_screened)
{
	int blockSize = 128;
	dim3 numBlocks = computeNbBlocks((unsigned int)query_size, blockSize);

	TurboGWNGradidentKernal << <numBlocks, blockSize >> > (query_size, this->m_Octree_GPU, query_points, point_cloud, i_Normals, i_As, i_Cs, i_radii, beta, sigma, out_gwn_grad, use_radii, use_screened);

	cudaDeviceSynchronize();

	std::cout << "GWN Gradient Done" << std::endl;
}

void TurboLibiGWN::TurboGWNGradientBackwardLaunch(float3* query_points, float3* point_cloud, int query_size, float3* i_Normals, float* i_As, float* i_Cs, float* i_radii ,
	float3* gwn_grads,   float* Ai_grad, float* Ci_grad,
	float* grad_scale_coff, float grad_volume_scale, float& beta, float& sigma,
	bool use_radii, bool use_screened, bool com_Ai_grad, bool com_Ci_grad, bool com_normal_grad, int normal_mode)
{
	int blockSize = 128;
	dim3 numBlocks = computeNbBlocks((unsigned int)query_size, blockSize);

	TurboGWNGradidentBackwardKernal << <numBlocks, blockSize >> > (query_size, this->m_Octree_GPU, query_points, point_cloud, i_Normals, i_As, i_Cs, i_radii,
		gwn_grads,   Ai_grad, Ci_grad,
		grad_scale_coff, grad_volume_scale, beta, sigma, use_radii, use_screened, com_Ai_grad, com_Ci_grad, com_normal_grad, normal_mode);

	cudaDeviceSynchronize();

	std::cout << "GWN Gradient Backward Done" << std::endl;
}

void TurboLibiGWN::TurboGWNGradientAsPatchBackwardLaunch(float3* query_points, float3* point_cloud, float3* i_Normals, float* i_Cs, float3* gwn_grads, float* Ai_grad, float* grad_scale_coff, float grad_volume_scale, int query_size)
{

	const int batch_size = 5000000;
	int total = this->m_Node_Array_Size;
	std::cout << "Total work units: " << total << std::endl;

	int blockSize = 128;

	for (int offset = 0; offset < total; offset += batch_size)
	{
		int current_batch = std::min(batch_size, total - offset);
		dim3 numBlocks = computeNbBlocks((unsigned int)current_batch, blockSize);

		int* node_array_slice = this->m_Large_Node_Array + offset;
		int* point_index_slice = this->m_Large_Node_Point_Index + offset;
		int* point_local_index_slice = this->m_Large_Node_Point_Local_Index + offset;

		TurboGWNGradidentAsPatchBackwardKernal << <numBlocks, blockSize >> > (
			this->m_Octree_GPU,
			node_array_slice,
			point_index_slice,
			point_local_index_slice,
			query_points,
			i_Normals,
			i_Cs,
			gwn_grads,
			Ai_grad,
			grad_scale_coff,
			grad_volume_scale,
			query_size,
			current_batch
			);

		cudaDeviceSynchronize();

	}
	std::cout << "GWN Gradient Patch Backward Done" << std::endl;

	dim3 numBlocks = computeNbBlocks((unsigned int)this->m_Large_Node_Size, blockSize);
	GradidentPatchResetKernal << <numBlocks, blockSize >> > (this->m_Octree_GPU, this->m_Large_Node_Index, this->m_Large_Node_Size);
	cudaDeviceSynchronize();
	std::cout << "GWN Gradient Patch Reset Done" << std::endl;
}

void TurboLibiGWN::TurboGWNGradientCsPatchBackwardLaunch(float3* query_points, float3* point_cloud, float3* i_Normals, float* i_As, float3* gwn_grads, float* Ci_grad, float* grad_scale_coff, float grad_volume_scale, int query_size)
{
	const int batch_size = 5000000;
	int total = this->m_Node_Array_Size;
	std::cout << "Total work units: " << total << std::endl;

	int blockSize = 128;

	for (int offset = 0; offset < total; offset += batch_size)
	{
		int current_batch = std::min(batch_size, total - offset);
		dim3 numBlocks = computeNbBlocks((unsigned int)current_batch, blockSize);

		int* node_array_slice = this->m_Large_Node_Array + offset;
		int* point_index_slice = this->m_Large_Node_Point_Index + offset;
		int* point_local_index_slice = this->m_Large_Node_Point_Local_Index + offset;

		TurboGWNGradidentCsPatchBackwardKernal << <numBlocks, blockSize >> > (
			this->m_Octree_GPU,
			node_array_slice,
			point_index_slice,
			point_local_index_slice,
			query_points,
			i_Normals,
			i_As,
			gwn_grads,
			Ci_grad,
			grad_scale_coff,
			grad_volume_scale,
			query_size,
			current_batch
			);

		cudaDeviceSynchronize();
	}
	std::cout << "GWN Gradient Patch Backward Done" << std::endl;

	dim3 numBlocks = computeNbBlocks((unsigned int)this->m_Large_Node_Size, blockSize);
	GradidentPatchResetKernal << <numBlocks, blockSize >> > (this->m_Octree_GPU, this->m_Large_Node_Index, this->m_Large_Node_Size);
	cudaDeviceSynchronize();
	std::cout << "GWN Gradient Patch Reset Done" << std::endl;
}

void TurboLibiGWN::TurboGWNAsPatchBackwardLaunch(float3* query_points, float3* point_cloud, float3* i_Normals, float* i_Cs, float* gwn, float* Ai_grad, float target_gwn, float grad_scale, int batch_size)
{

	int total = this->m_Node_Array_Size;
	std::cout << "Total work units: " << total << std::endl;

	int blockSize = 128;

	for (int offset = 0; offset < total; offset += batch_size)
	{
		int current_batch = std::min(batch_size, total - offset);
		dim3 numBlocks = computeNbBlocks((unsigned int)current_batch, blockSize);

		int* node_array_slice = this->m_Large_Node_Array + offset;
		int* point_index_slice = this->m_Large_Node_Point_Index + offset;
		int* point_local_index_slice = this->m_Large_Node_Point_Local_Index + offset;

		TurboGWNAsPatchBackwardKernal << <numBlocks, blockSize >> > (
			this->m_Octree_GPU,
			node_array_slice,
			point_index_slice,
			point_local_index_slice,
			query_points,
			this->m_Point_Gradient,
			this->m_Point_Grad_Index,
			this->m_Large_Node_Size,
			i_Normals,
			i_Cs,
			gwn,
			Ai_grad,
			grad_scale,
			target_gwn,
			current_batch
			);

		cudaDeviceSynchronize();
	}

	dim3 numBlocks = computeNbBlocks((unsigned int)this->m_Large_Node_Size, blockSize);
	GradidentPatchResetKernal << <numBlocks, blockSize >> > (this->m_Octree_GPU, this->m_Large_Node_Index, this->m_Large_Node_Size);
	cudaDeviceSynchronize();

	std::cout << "GWN Patch Reset Done" << std::endl;
}

void TurboLibiGWN::TurboGWNCsPatchBackwardLaunch(float3* query_points, float3* point_cloud, float3* i_Normals, float* i_As, float* gwn, float* Ci_grad, float target_gwn, float grad_scale, int batch_size)
{
	int total = this->m_Node_Array_Size;
	std::cout << "Total work units: " << total << std::endl;

	int blockSize = 128;

	for (int offset = 0; offset < total; offset += batch_size)
	{
		int current_batch = std::min(batch_size, total - offset);
		dim3 numBlocks = computeNbBlocks((unsigned int)current_batch, blockSize);

		int* node_array_slice = this->m_Large_Node_Array + offset;
		int* point_index_slice = this->m_Large_Node_Point_Index + offset;
		int* point_local_index_slice = this->m_Large_Node_Point_Local_Index + offset;

		TurboGWNCsPatchBackwardKernal << <numBlocks, blockSize >> > (
			this->m_Octree_GPU,
			node_array_slice,
			point_index_slice,
			point_local_index_slice,
			query_points,
			i_Normals,
			i_As,
			gwn,
			Ci_grad,
			grad_scale,
			target_gwn,
			current_batch
			);

		cudaDeviceSynchronize();
	}
	std::cout << "GWN Patch Backward Done" << std::endl;

	dim3 numBlocks = computeNbBlocks((unsigned int)this->m_Large_Node_Size, blockSize);
	GradidentPatchResetKernal << <numBlocks, blockSize >> > (this->m_Octree_GPU, this->m_Large_Node_Index, this->m_Large_Node_Size);
	cudaDeviceSynchronize();

	std::cout << "GWN Patch Reset Done" << std::endl;
}
