#pragma once

uint3 inline __device__ boundingBoxMap() {
    return (uint3) { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z };
}

uint3 inline __device__ hadoukenMap() {
    return (uint3) { 0, 0, 0 }
}

__device__ inline void work(const float* data, char* mat, unsigned long index, uint3 p) {
    mat[index] = 1;
}

__global__ void kernelBoundingBox(const MTYPE* data, const size_t n) {
    if (blockIdx.x > blockIdx.y || blockIdx.y > blockIdx.z) {
        return;
    }
    auto p = map();
    if (p.x < p.y && p.y < p.z) {
        unsigned long index = p.z * n * n + p.y * n + p.x;
        if (index < V) {
            work(data, mat, index, p);
        }
    }
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
#endif

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
