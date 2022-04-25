#pragma once

uint3 inline __device__ boundingBoxMap() {
    return (uint3) { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z };
}

uint3 inline __device__ hadoukenMap() {
    return (uint3) { 0, 0, 0 };
}

__device__ inline void work(MTYPE* data, size_t index, uint3 p) {
    data[index]++;
}

__global__ void kernelBoundingBox(MTYPE* data, const size_t n) {

    auto p = boundingBoxMap();
    if (p.x + p.y + p.z < n - 1) {
        size_t index = p.z * n * n + p.y * n + p.x;
        if (index < n * n * n) {
            work(data, index, p);
        }
    }
    return;
}

__global__ void kernelHadouken(const MTYPE* data, const size_t n) {

    return;
}

// Origin is the location of this orthotope origin inside the cube
// Depth is the current level being mapped
// level_n is the size of the orthotope at level depth
__global__ void kernelDynamicParallelism(MTYPE* data, const size_t n, const uint32_t depth, const uint32_t level_n, const uint3 origin) {
    // Map elements
    auto p = boundingBoxMap();
    if (p.x + p.y + p.z < level_n - 1) {
        // In the simplex part of the cube
        size_t index = (origin.z + p.z) * n * n + (origin.y + p.y) * n + (origin.x + p.x);
        if (index < n * n * n) {
            work(data, index, p);
        }
    } else {
        // Out of the simplex part, needs to be remmaped
        p = (uint3) { origin.x + p.y, origin.y + p.x, (2 * level_n) - 1 - p.z };
        size_t index = p.z * n * n + p.y * n + p.x;
        if (index < n * n * n) {
            // work(data, index, p);
        }
    }

    // Do work

    // Launch child kernels
    return;
}

// kernel non-linear Map
template <typename Lambda>
__global__ void kernel1(const float* data, char* mat, const unsigned long n, const unsigned long V, Lambda map) {
    // lambda map
    auto p = map();
    if (p.x < p.y && p.y < p.z) {
        unsigned long index = p.z * n * n + p.y * n + p.x;
        if (index < V) {
            work(data, mat, index, p);
        }
    }
    return;
}

__device__ float carmack_sqrtf(float nb) {
    float nb_half = nb * 0.5F;
    float y = nb;
    long i = *(long*)&y;
    // i = 0x5f3759df - (i >> 1);
    i = 0x5f375a86 - (i >> 1);
    y = *(float*)&i;
    // Repetitions increase accuracy(6)
    y = y * (1.5f - (nb_half * y * y));
    y = y * (1.5f - (nb_half * y * y));
    y = y * (1.5f - (nb_half * y * y));

    return nb * y;
}

__device__ inline float newton_sqrtf(const float number) {
    int i;
    float x, y;
    // const float f = 1.5F;
    x = number * 0.5f;
    i = *(int*)&number;
    i = 0x5f3759df - (i >> 1);
    y = *(float*)&i;
    y *= (1.5f - x * y * y);
    y *= (1.5f - x * y * y);
    y *= (1.5f - x * y * y);
    return number * y;
}

__device__ inline float newton1_sqrtf(const float number) {

    int i;
    float x, y;
    // const float f = 1.5F;
    x = number * 0.5f;
    i = *(int*)&number;
    i = 0x5f3759df - (i >> 1);
    y = *(float*)&i;
    y *= (1.5f - x * y * y);
    y *= number; // obteniendo resultado
    // arreglar
    if ((y + 1.0f) * (y + 1.0f) < number)
        return y;
    else
        return y - 0.5f;
}
