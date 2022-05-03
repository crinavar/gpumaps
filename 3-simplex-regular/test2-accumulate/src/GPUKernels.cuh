#pragma once

#define WORDLEN 31

__device__ inline bool isInSimplex(const uint3 coord, const uint32_t level_n) {
    return (coord.x + (coord.y) + coord.z < level_n - 1);
}

uint3 inline __device__ addThreadIdxOffset(uint3 blockCoord) {
    return (uint3) { blockCoord.x + threadIdx.x, blockCoord.y + threadIdx.y, blockCoord.z + threadIdx.z };
}
uint3 inline __device__ boundingBoxMap() {
    return (uint3) { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z };
}

uint3 inline __device__ hadoukenMap(uint3 coord, const size_t n) {

    if (coord.z < n / 2) {

        if (isInSimplex((uint3) { n / 2 - 1 - coord.x, coord.y, n / 2 - coord.z - 1 }, n / 2)) {
            printf("(%i,%i,%i) \n", coord.x, coord.y, coord.z);

            coord = (uint3) { coord.y, coord.x, (n)-blockIdx.z - 1 };
            coord.y = coord.y + n / 2;
            printf("(%i,%i,%i) \n", coord.x, coord.y, coord.z);

            return coord;
            //(levelNminusOne) - gridCoord.y, origin.y + (levelNminusOne)-gridCoord.x, (2 * levelN) - 1 - gridCoord.z return (uint3) { 11111, 0, 1111 };
        }

        // printf("(%i,%i,%i) \n", coord.x, coord.y, coord.z);
        coord.y = coord.y + n / 2;

        return coord;

    } else {
        if (coord.y == 0) {
            return (uint3) { 11111, 0, 1111 };
        }
        const uint32_t h = WORDLEN - __clz(coord.y);
        const uint32_t b = (1 << h);
        const uint32_t q = (coord.x >> h);
        coord = (uint3) { coord.x + q * b, coord.y + 2 * q * b, coord.z - n / 2 };
        if (isInSimplex(coord, n + 1)) {
            return coord;
        }
        // printf("(%i,%i,%i) q:%i b:%i -> %i, %i, %i\n", coord.x, coord.y, coord.z, q, b, coord.x + q * b, coord.y + 2 * q * b, coord.z - n / 2);
        return (uint3) { 11111, 0, 1111 };
    }
}

__device__ inline void work(MTYPE* data, size_t index, uint3 p) {
    data[index]++;
}

__global__ void kernelBoundingBox(MTYPE* data, const size_t n, const size_t blockedN) {

    if (isInSimplex(blockIdx, blockedN + 1)) {
        auto p = boundingBoxMap();
        if (isInSimplex(p, n)) {
            size_t index = p.z * n * n + p.y * n + p.x;
            if (index < n * n * n) {
                work(data, index, p);
            }
        }
    }
    return;
    // if (isInSimplex(p, blockedN)) {
    //     auto p = addThreadIdxOffset(p);
    //     size_t index = p.z * n * n + p.y * n + p.x;
    //     if (index < n * n * n) {
    //         work(data, index, p);
    //     }
    // }
    // return;
}

__global__ void kernelHadouken(MTYPE* data, const size_t n, const size_t blockedN) {
    auto p = hadoukenMap(blockIdx, blockedN);
    p = (uint3) { (p.x), ((blockedN - 1) - p.y), p.z };

    p = addThreadIdxOffset(p);
    size_t index = p.z * n * n + p.y * n + p.x;
    if (index < n * n * n) {
        work(data, index, p);
    }
    return;
}

// Origin is the location of this orthotope origin inside the cube
// Depth is the current level being mapped
// levelN is the size of the orthotope at level depth
// This kernel assumes that the grid axes direction coalign with data space
__global__ void kernelDynamicParallelism(MTYPE* data, const size_t n, const uint32_t depth, const uint32_t levelN, const uint3 origin) {
    const uint32_t halfLevelN = levelN >> 1;
    const uint32_t levelNminusOne = levelN - 1;

    // Map elements
    auto gridCoord = boundingBoxMap();

    // Check which elements from the grid are inside the simplex region of the orthotope with its origin in the oposite side of each axis
    if (isInSimplex((uint3) { levelNminusOne - gridCoord.x, levelNminusOne - gridCoord.y, levelNminusOne - gridCoord.z }, levelN)) {
        // In the simplex part of the cube
        // Performing the hinged map to data space
        gridCoord = (uint3) { origin.x + (levelNminusOne)-gridCoord.y, origin.y + (levelNminusOne)-gridCoord.x, (2 * levelN) - 1 - gridCoord.z };

        size_t index = gridCoord.z * n * n + gridCoord.y * n + gridCoord.x;
        if (index < n * n * n) {
            work(data, index, gridCoord);
        }
    } else {
        // Out of the simplex region of the grid
        // Directly map threads to data space
        size_t index = (origin.z + gridCoord.z) * n * n + (origin.y + gridCoord.y) * n + (origin.x + gridCoord.x);
        if (index < n * n * n) {
            work(data, index, gridCoord);
        }
    }

    // Launch child kernels

    if (levelN > 1) {
        if (threadIdx.x + threadIdx.x + threadIdx.x + blockIdx.x + blockIdx.y + blockIdx.z == 0) {
            dim3 blockSize(BSIZE3DX, BSIZE3DY, BSIZE3DZ);
            dim3 gridSize((halfLevelN + blockSize.x - 1) / blockSize.x, (halfLevelN + blockSize.y - 1) / blockSize.y, (halfLevelN + blockSize.z - 1) / blockSize.z);

            kernelDynamicParallelism<<<gridSize, blockSize>>>(data, n, depth + 1, halfLevelN, (uint3) { origin.x + levelN, origin.y, origin.z });
            kernelDynamicParallelism<<<gridSize, blockSize>>>(data, n, depth + 1, halfLevelN, (uint3) { origin.x, origin.y + levelN, origin.z });
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
