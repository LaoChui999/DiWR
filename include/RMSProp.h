#pragma once

#include <cuda.h>
#include <cuda_runtime.h>

#include <device_launch_parameters.h>

class RMSPropOpt
{
public:
	RMSPropOpt() = default;
	void RMSPropUpdate(float* __restrict__ data,
					   float* __restrict__ data_rms,
					   float* __restrict__ data_grad,
		               float beta, float lr, float epsilon, float minval, int data_size);

private:

};
