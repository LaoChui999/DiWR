#include "RMSProp.h"

__global__
void RMSPropUpdateKernel(float* __restrict__ data, float* __restrict__ data_rms, float* __restrict__ data_grad,
						 float beta, float lr, float epsilon, float minval, int data_size)
{
	int tid = blockIdx.x * blockDim.x + threadIdx.x;
	if (tid < data_size)
	{
		float c_rms = data_rms[tid];
		float c_grad = data_grad[tid];
		float c_grad_sqr = c_grad * c_grad;
		float c_data = data[tid];

		c_rms = (c_rms == 0.0f) ? c_grad_sqr : fma(beta, c_rms, fma(-beta, c_grad_sqr, c_grad_sqr));
		data[tid] = c_data - lr * c_grad / (sqrtf(c_rms) + epsilon);

		data_rms[tid] = c_rms;
		data_grad[tid] = 0.0f;
	}
}

void RMSPropOpt::RMSPropUpdate(float* __restrict__ data, float* __restrict__ data_rms, float* __restrict__ data_grad, float beta, float lr, float epsilon, float minval, int data_size)
{
	int threadsPerBlock = 128;
	int blocksPerGrid = (data_size + threadsPerBlock - 1) / threadsPerBlock;
	RMSPropUpdateKernel << < blocksPerGrid, threadsPerBlock >> > (data, data_rms, data_grad, beta, lr, epsilon, minval, data_size);

	cudaDeviceSynchronize();
}
