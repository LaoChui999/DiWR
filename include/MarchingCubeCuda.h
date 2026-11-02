#pragma once

#include <cuda.h>
#include <cuda_runtime.h>
#include <device_launch_parameters.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <fstream>
#include <iostream>
#include <stdexcept>
#include <string>
#include <vector>

class TurboMC
{
public:
    int TurboMarchingCubes(float* voxel_values,
        float3* voxel_poses,
        int32_t res_x, int32_t res_y, int32_t res_z,
        float threshold, int iterations,
        std::string mesh_path,
        std::string mesh_full_path,
        float3* mid_Ps,
        float3* mid_Ns,
        int test_iter,
        cudaStream_t stream = 0);
};
