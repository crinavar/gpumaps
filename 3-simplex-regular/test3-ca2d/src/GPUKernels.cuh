#pragma once

#define WORDLEN 31

// Checks if the given coordinate is inside the regular 3-simplex of |side| = level_n.
// It compares with level_n-1 as we dont care about elements in the diagonal plane
__device__ inline bool isInSimplex(const uint3 coord, const uint32_t level_n) {
    return (coord.x + (coord.y) + coord.z < level_n - 1);
}

__device__ inline bool isOutsideSimplex(const uint3 coord, const uint32_t level_n) {
    return (coord.x + (coord.y) + coord.z >= level_n - 1);
}

// Transform a given block coordinate to individual thread coordinates, in order to access data.
uint3 inline __device__ addThreadIdxOffset(uint3 blockCoord) {
    return (uint3) { blockCoord.x * blockDim.x + threadIdx.x, blockCoord.y * blockDim.y + threadIdx.y, blockCoord.z * blockDim.z + threadIdx.z };
}

// Traditional bounding box map
uint3 inline __device__ boundingBoxMap() {
    return (uint3) { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z };
}

__device__ inline void work(MTYPE* data, MTYPE* dataPong, size_t index, uint3 p, uint32_t nWithHalo) {
    size_t indexNeighbourZp = (p.z + 2) * nWithHalo * nWithHalo + (p.y + 1) * nWithHalo + (p.x + 1);
    size_t indexNeighbourZm = (p.z) * nWithHalo * nWithHalo + (p.y + 1) * nWithHalo + (p.x + 1);

    size_t indexNeighbourYp = (p.z + 1) * nWithHalo * nWithHalo + (p.y + 2) * nWithHalo + (p.x + 1);
    size_t indexNeighbourYm = (p.z + 1) * nWithHalo * nWithHalo + (p.y) * nWithHalo + (p.x + 1);

    size_t indexNeighbourXp = (p.z + 1) * nWithHalo * nWithHalo + (p.y + 1) * nWithHalo + (p.x + 2);
    size_t indexNeighbourXm = (p.z + 1) * nWithHalo * nWithHalo + (p.y + 1) * nWithHalo + (p.x);

    uint32_t aliveNeighbors = data[indexNeighbourZp] + data[indexNeighbourZm] + data[indexNeighbourYp] + data[indexNeighbourYm] + data[indexNeighbourXp] + data[indexNeighbourXm];
    MTYPE val = data[index];
    if (val == 1) {
        if (aliveNeighbors < 2 || aliveNeighbors > 3) {
            dataPong[index] = 0;
        } else {
            dataPong[index] = 1;
        }
    } else {
        if (aliveNeighbors == 3) {
            dataPong[index] = 1;
        } else {
            dataPong[index] = 0;
        }
    }
    return;
}

// Hadouken 3D map: maps blocks from a reduced grid to the simplex region of data
// The standard 3d hadouken map is able to map blocks from a reduced grid (compared to BB) to the 3-simplex
// region of the matrix of data elements, without including the diagonal plane x+y+z=n.
// E.g. n=4
//        z=0         z=1         z=2         z=3
//      1 1 1 0     1 1 0 0     1 0 0 0     0 0 0 0
//      1 1 0 0     1 0 0 0     0 0 0 0     0 0 0 0
//      1 0 0 0     0 0 0 0     0 0 0 0     0 0 0 0
//      0 0 0 0     0 0 0 0     0 0 0 0     0 0 0 0
// However, in a blocked approach we would group up packets of BSIZE^3 elements into one block, thus working
// with a coarser version of the simplex.
// E.g. for BSIZE = 2 -> blocked_n = old_n/BSIZE
//      z=0     z=1
//      1  0*    0* 0
//      0* 0     0  0
// This introduces a problem, since we would only account for one block of data (denoted with 1) but there are
// still elements in the positions marked with '*'!
// Our solution consist in movin the blocked simplex down one position (of BSIZE elements) and then in parallel
// with a different kernel/stream, map the thick 2d version of the map at y=0.
// Although the corner of the regular 3-simplex is the origin, to simplify the calculation of this map, it assumes
// an origin at the corner y=n-1, with +y in the opposite direction.
uint3 inline __device__ hadoukenMap3d(uint3 coord, const uint32_t n) {
    const uint32_t nHalf = n >> 1;
    if (coord.z < nHalf) {
        // Block within the largest region of the grid

        // ***** VERSION 1 *****
        // Move the entire region to its position in the simplex
        // coord.y = coord.y + nHalf;
        // Then transform the coordinate to data space (with an origin at the corner of the 3-simplex)
        // coord = (uint3) { (coord.x), ((n - 1) - coord.y), coord.z };

        // ***** VERSION 2 [Faster] *****
        coord.y = ((n - 1) - (coord.y + nHalf));

        // To check which blocks are inside the simplex, and which ones have to be remapped
        // in a hinged manner to the top
        if (isOutsideSimplex(coord, n)) {
            // Here n/2 is the same as "b" in the paper, n/2 is faster tho.
            // coord = (uint3) { nHalf - coord.x - 1, nHalf - coord.y - 1, n - coord.z - 1 };
            coord.x = nHalf - coord.x - 1;
            coord.y = nHalf - coord.y - 1;
            coord.z = n - coord.z - 1;
        }
    } else {
        // Blocks that are on top of the largest region of the grid

        const uint32_t h = WORDLEN - __clz(coord.y);
        const uint32_t b = (1 << h);
        const uint32_t q = (coord.x >> h);

        // Check for blocks that do not belong to any region (wasted parallel space)
        if (coord.y == 0 || coord.z - nHalf >= b) {
            return (uint3) { 0, 1, 0xffffffff };
        }

        // The regions are mapped to their corresponding place within the simplex (note that for both x and y axis,
        // the map is the same as the 2d version) and then transform the coordinates to data space (with the ofrigin
        // at the corner of the simplex).
        // coord = (uint3) { coord.x + q * b, coord.y + 2 * q * b, coord.z - nHalf };
        // coord = (uint3) { (coord.x), ((n - 1) - coord.y), coord.z };
        // Or reusing the same memory location
        coord.x = coord.x + q * b;
        coord.y = ((n - 1) - (coord.y + 2 * q * b));
        coord.z = coord.z - nHalf;

        // With the regions placed and the coordinate in data space, check which blocks of said regions lie
        // outside of the simplex.
        if (isOutsideSimplex(coord, n)) {
            // Relative block coordinates within the region they belong to
            uint32_t bMinusOne = b - 1;
            uint32_t localX = coord.x & bMinusOne;
            uint32_t localY = coord.y & bMinusOne;

            // Relative hinged map of those blocks
            // uint2 coord2 = (uint2) { (b - 1) - localY, (b - 1) - localX };
            // Then add each offset to get the global coodinate
            // uint3 coord21 = (uint3) { coord.x - (localX - coord2.x), coord.y - (localY - coord2.y), (2 * b) - coord.z - 1 };
            // coord = coord21;

            // Or reusing memory
            coord.x = coord.x - (localX - (bMinusOne - localY));
            coord.y = coord.y - (localY - (bMinusOne - localX));
            coord.z = (2 * b) - coord.z - 1;
        }
    }
    // Finally shift everything one block down to allow the strip work at y=0
    // coord = (uint3) { coord.x, coord.y + 1, coord.z };
    coord.y = coord.y + 1;

    return coord;
}

uint3 inline __device__ hadoukenMap2d(uint3 coord, const uint32_t n) {
    if (coord.y == gridDim.y - 1) {
        return (uint3) { coord.x + gridDim.x, coord.y - 1, 0 };
    } else {
        const unsigned int h = WORDLEN - __clz(coord.y + 1);
        const unsigned int qb = (coord.x >> h) * (1 << h);
        return (uint3) { (coord.x + qb), (coord.y + (qb << 1)), 0 };
    }
}

__global__ void kernelBoundingBox(MTYPE* data, MTYPE* dataPong, const uint32_t n, const uint32_t blockedN, uint32_t nWithHalo) {

    if (isInSimplex(blockIdx, blockedN + 1)) {
        auto p = boundingBoxMap();
        if (isInSimplex(p, n)) {
            size_t index = (p.z + 1) * nWithHalo * nWithHalo + (p.y + 1) * nWithHalo + (p.x + 1);
            if (index < nWithHalo * nWithHalo * nWithHalo) {
                work(data, dataPong, index, p, nWithHalo);
            }
        }
    }
    return;
}

__global__ void kernelHadouken(MTYPE* data, MTYPE* dataPong, const uint32_t n, const uint32_t blockedN, uint32_t nWithHalo) {
    auto p = hadoukenMap3d(blockIdx, blockedN);
    if (p.z == 0xffffffff) {
        return;
    }
    p = addThreadIdxOffset(p);
    if (isInSimplex(p, n)) {
        size_t index = (p.z + 1) * nWithHalo * nWithHalo + (p.y + 1) * nWithHalo + (p.x + 1);
        if (index < nWithHalo * nWithHalo * nWithHalo) {
            work(data, dataPong, index, p, nWithHalo);
        }
    }
    return;
}

// Kernel to map the thick 2d version of hadouken at y=0
// This map works in an xy grid of blocks, to use it in out case, we need to use an xz version.
// It also asumes the origin at the top right corner, same as the 3d version
__global__ void kernelHadoukenStrip(MTYPE* data, MTYPE* dataPong, const uint32_t n, const uint32_t blockedN, uint32_t nWithHalo) {
    // Get the mapped coordinate of the region
    auto p = hadoukenMap2d(blockIdx, blockedN);

    // p = (uint3) { p.x, 0, ((blockedN - 1) - p.y) };
    // Then transform it toan xz version with the origin at the corner.
    p.z = ((blockedN - 1) - p.y);
    p.y = 0;
    p = addThreadIdxOffset(p);
    if (isInSimplex(p, n)) {
        size_t index = (p.z + 1) * nWithHalo * nWithHalo + (p.y + 1) * nWithHalo + (p.x + 1);
        if (index < nWithHalo * nWithHalo * nWithHalo) {
            work(data, dataPong, index, p, nWithHalo);
        }
    }
    return;
}

__global__ void kernelDynamicParallelismBruteForce(MTYPE* data, MTYPE* dataPong, const uint32_t n, const uint32_t originX, const uint32_t originY, uint32_t nWithHalo) {
#ifdef DP
    auto p = (uint3) { originX + blockIdx.x * blockDim.x + threadIdx.x, originY + blockIdx.y * blockDim.y + threadIdx.y, blockIdx.z * blockDim.z + threadIdx.z };

    if (isInSimplex(p, n)) {
        size_t index = (p.z + 1) * nWithHalo * nWithHalo + (p.y + 1) * nWithHalo + (p.x + 1);
        if (index < nWithHalo * nWithHalo * nWithHalo) {
            work(data, dataPong, index, p, nWithHalo);
        }
    }
#endif
    return;
}

// Origin is the location of this orthotope origin inside the cube
// Depth is the current level being mapped
// levelN is the size of the orthotope at level depth
// This kernel assumes that the grid axes direction coalign with data space
__global__ void kernelDynamicParallelism(MTYPE* data, MTYPE* dataPong, const uint32_t n, const uint32_t depth, const uint32_t levelN, uint32_t originX, uint32_t originY, uint32_t nWithHalo) {
#ifdef DP

    const uint32_t halfLevelN = levelN >> 1;
    const uint32_t levelNminusOne = levelN - 1;
    // Launch child kernels
    if (threadIdx.x + threadIdx.y + threadIdx.z + blockIdx.x + blockIdx.y + blockIdx.z == 0) {

        if (levelN > MIN_SIZE) {
            dim3 blockSize(BSIZE3DX, BSIZE3DY, BSIZE3DZ);
            dim3 gridSize(halfLevelN / BSIZE3DX, halfLevelN / BSIZE3DX, halfLevelN / BSIZE3DX);

            cudaStream_t s1, s2;
            cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
            kernelDynamicParallelism<<<gridSize, blockSize, 0, s1>>>(data, dataPong, n, depth + 1, halfLevelN, originX + levelN, originY, nWithHalo);
            kernelDynamicParallelism<<<gridSize, blockSize, 0, s2>>>(data, dataPong, n, depth + 1, halfLevelN, originX, originY + levelN, nWithHalo);

        } else {
            dim3 blockSize(BSIZE3DX, BSIZE3DY, BSIZE3DZ);
            dim3 gridSize(gridDim.x, gridDim.y, gridDim.z);

            cudaStream_t s1, s2;
            cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
            cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
            kernelDynamicParallelismBruteForce<<<gridSize, blockSize, 0, s1>>>(data, dataPong, n, originX + levelN, originY, nWithHalo);
            kernelDynamicParallelismBruteForce<<<gridSize, blockSize, 0, s2>>>(data, dataPong, n, originX, originY + levelN, nWithHalo);
        }
    }
    // Map elements directly to data space, both origins coalign.
    auto threadCoord = boundingBoxMap();
    auto dataCoord = (uint3) { originX + threadCoord.x, originY + threadCoord.y, threadCoord.z };
    if (threadCoord.x >= levelN || threadCoord.y >= levelN) {
        return;
    }
    if (isInSimplex(dataCoord, n)) {
        // In the simplex part of the cube
        // Performing the hinged map to data space
        size_t index = (dataCoord.z + 1) * nWithHalo * nWithHalo + (dataCoord.y + 1) * nWithHalo + (dataCoord.x + 1);
        if (index < nWithHalo * nWithHalo * nWithHalo) {
            work(data, dataPong, index, dataCoord, nWithHalo);
        }
    } else if (levelN >= BSIZE3DX) {
        // Out of the simplex region of the grid
        // Directly map threads to data space
        // threadCoord = (uint3) { originX + (levelNminusOne)-threadCoord.y, originY + (levelNminusOne)-threadCoord.x, (2 * levelN) - 1 - threadCoord.z };
        uint32_t bufferX = threadCoord.x;
        threadCoord.x = originX + (levelNminusOne)-threadCoord.y;
        threadCoord.y = originY + (levelNminusOne)-bufferX;
        threadCoord.z = (2 * levelN) - 1 - threadCoord.z;
        size_t index = (threadCoord.z + 1) * nWithHalo * nWithHalo + (threadCoord.y + 1) * nWithHalo + (threadCoord.x + 1);
        if (index < nWithHalo * nWithHalo * nWithHalo) {
            work(data, dataPong, index, threadCoord, nWithHalo);
        }
    }

#endif
    return;
}

__global__ void kernelDP_work(const uint32_t n, const uint32_t levelN, MTYPE* data, MTYPE* dataPong, unsigned int offX, unsigned int offY, uint32_t offZ, uint32_t nWithHalo) {
#ifdef DP
    // Process data
    auto p = (uint3) { offX + blockIdx.x * blockDim.x + threadIdx.x, offY + blockIdx.y * blockDim.y + threadIdx.y, offZ + blockIdx.z * blockDim.z + threadIdx.z };
    if (isInSimplex(p, n)) {
        size_t index = (p.z + 1) * nWithHalo * nWithHalo + (p.y + 1) * nWithHalo + (p.x + 1);
        if (index < nWithHalo * nWithHalo * nWithHalo) {
            work(data, dataPong, index, p, nWithHalo);
        }
    }
#endif
    return;
}

__global__ void kernelDP_exp(MTYPE* data, MTYPE* dataPong, const uint32_t n, const uint32_t depth, const uint32_t levelN, uint32_t x0, uint32_t y0, uint32_t z0, uint32_t nWithHalo) {
//__global__ void kernelDP_exp(unsigned int on, unsigned int n, MTYPE* data, unsigned int x0, unsigned int y0, unsigned int MIN_SIZE) {
#ifdef DP
    // 1) stopping case
    if (levelN <= MIN_SIZE) {
        dim3 bleaf(BSIZE3DX, BSIZE3DY, BSIZE3DZ), gleaf = dim3((levelN + bleaf.x - 1) / bleaf.x, (levelN + bleaf.y - 1) / bleaf.y, (levelN + bleaf.z - 1) / bleaf.z);
        // printf("leaf kernel at x=%i  y=%i   size %i x %i (grid (%i,%i,%i)  block(%i,%i,%i))\n", x0, y0, n, n, gleaf.x, gleaf.y, gleaf.z, bleaf.x, bleaf.y, bleaf.z);
        kernelDP_work<<<gleaf, bleaf>>>(n, levelN, data, dataPong, x0, y0, z0, nWithHalo);
        return;
    }
    // 2) explore up and right asynchronously
    cudaStream_t s1, s2, s3, s4;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s4, cudaStreamNonBlocking);
    // int subn = (levelN >> 1) + (levelN & 1);
    int n2 = levelN >> 1;
    // printf("subn %i\nn2 %i\n", subn, n2);
    //  up
    kernelDP_exp<<<1, 1, 0, s1>>>(data, dataPong, n, depth + 1, n2, x0, y0, z0 + n2, nWithHalo);
    // bottom right
    kernelDP_exp<<<1, 1, 0, s2>>>(data, dataPong, n, depth + 1, n2, x0 + n2, y0, z0, nWithHalo);
    // bottom left
    kernelDP_exp<<<1, 1, 0, s3>>>(data, dataPong, n, depth + 1, n2, x0, y0 + n2, z0, nWithHalo);

    // 3) work in the bot middle
    dim3 bnode(BSIZE3DX, BSIZE3DY, BSIZE3DZ);
    dim3 gnode = dim3((n2 + bnode.x - 1) / bnode.x, (n2 + bnode.y - 1) / bnode.y, (n2 + bnode.z - 1) / bnode.z);
    // printf("node kernel at x=%i  y=%i   size %i x %i\n", x0, y0+n2, n2, n2);
    kernelDP_work<<<gnode, bnode, 0, s4>>>(n, n2, data, dataPong, x0, y0, z0, nWithHalo);
#endif
}

//

//
__global__ void kernelDynamicParallelismHingedHYRBID(MTYPE* data, MTYPE* dataPong, const uint32_t n, const uint32_t levelN, const uint32_t originX, const uint32_t originY, uint32_t nWithHalo) {
#ifdef DP
    // Map elements directly to data space, both origins coalign.
    const uint32_t levelNminusOne = levelN - 1;
    auto threadCoord = boundingBoxMap();
    auto dataCoord = (uint3) { originX + threadCoord.x, originY + threadCoord.y, threadCoord.z };
    if (isInSimplex(dataCoord, n)) {
        // Out of the simplex region of the grid
        // Directly map threads to data space
        size_t index = (dataCoord.z + 1) * nWithHalo * nWithHalo + (dataCoord.y + 1) * nWithHalo + (dataCoord.x + 1);

        if (index < nWithHalo * nWithHalo * nWithHalo) {
            work(data, dataPong, index, dataCoord, nWithHalo);
        }
    } else if (levelN > BSIZE3DX) {
        // Out of the simplex region of the grid
        // Directly map threads to data space
        uint32_t bufferX = threadCoord.x;
        threadCoord.x = originX + (levelNminusOne)-threadCoord.y;
        threadCoord.y = originY + (levelNminusOne)-bufferX;
        threadCoord.z = (2 * levelN) - 1 - threadCoord.z;
        size_t index = (threadCoord.z + 1) * nWithHalo * nWithHalo + (threadCoord.y + 1) * nWithHalo + (threadCoord.x + 1);
        if (index < nWithHalo * nWithHalo * nWithHalo) {
            work(data, dataPong, index, threadCoord, nWithHalo);
        }
    }
#endif
    return;
}

// Origin is the location of this orthotope origin inside the cube
// Depth is the current level being mapped
// levelN is the size of the orthotope at level depth
// This kernel assumes that the grid axes direction coalign with data space
__global__ void kernelDynamicParallelismHYBRID(MTYPE* data, MTYPE* dataPong, const uint32_t n, const uint32_t depth, const uint32_t levelN, uint32_t x0, uint32_t y0, uint32_t nWithHalo) {
#ifdef DP

    // Launch child kernels
    // 1) stopping case
    int n2 = levelN >> 1;
    if (n2 <= MIN_SIZE) {
        dim3 bleaf(BSIZE3DX, BSIZE3DY, BSIZE3DZ), gleaf = dim3((levelN + bleaf.x - 1) / bleaf.x, (levelN + bleaf.y - 1) / bleaf.y, (levelN + bleaf.z - 1) / bleaf.z);
        // printf("[%i] leaf kernel at x=%i  y=%i   size %i x %i (grid (%i,%i,%i)  block(%i,%i,%i))\n", depth, x0, y0, levelN, levelN, gleaf.x, gleaf.y, gleaf.z, bleaf.x, bleaf.y, bleaf.z);
        kernelDynamicParallelismHingedHYRBID<<<gleaf, bleaf>>>(data, dataPong, n, levelN, x0, y0, nWithHalo);

        return;
    }
    // 2) explore up and right asynchronously
    cudaStream_t s1, s2, s3;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking);
    // int subn = (levelN >> 1) + (levelN & 1);
    //  up
    kernelDynamicParallelismHYBRID<<<1, 1, 0, s1>>>(data, dataPong, n, depth + 1, n2, x0, y0 + n2, nWithHalo);
    // bottom right
    kernelDynamicParallelismHYBRID<<<1, 1, 0, s2>>>(data, dataPong, n, depth + 1, n2, x0 + n2, y0, nWithHalo);

    // 3) work in the bot middl
    dim3 bnode(BSIZE3DX, BSIZE3DY, BSIZE3DZ);
    dim3 gnode = dim3((n2 + bnode.x - 1) / bnode.x, (n2 + bnode.y - 1) / bnode.y, (n2 + bnode.z - 1) / bnode.z);
    // printf("[%i] node kernel at x=%i  y=%i   size %i x %i\n", depth, x0, y0+n2, n2, n2);
    kernelDynamicParallelismHingedHYRBID<<<gnode, bnode, 0, s3>>>(data, dataPong, n, n2, x0, y0, nWithHalo);

#endif
    return;
}
