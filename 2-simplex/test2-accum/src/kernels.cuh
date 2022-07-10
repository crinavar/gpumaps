//////////////////////////////////////////////////////////////////////////////////
//  gpumaps                                                                     //
//  A GPU benchmark of mapping functions                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2018 Cristobal A. Navarro                                       //
//                                                                              //
//  This file is part of gpumaps.                                               //
//  gpumaps is free software: you can redistribute it and/or modify             //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  gpumaps is distributed in the hope that it will be useful,                  //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with gpumaps.  If not, see <http://www.gnu.org/licenses/>.            //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
#ifndef KERNELS_CUH
#define KERNELS_CUH

#define OFFSET -0.4999f
//#define OFFSET 0.5f
__device__ void work(DTYPE* data, MTYPE* mat, uint2 p, int n, const int a) {
    // (1) constant write
    unsigned long i = (unsigned long)p.y * n + (unsigned long)p.x;
    mat[i] += a;
    // mat[i] = a;

    // or (2) recursion level write
    // const int b = (int)log2f(blockIdx.y + 1) + 1;
    // mat[p.y * n + p.x] = b;
}

__device__ void workSubBlock(DTYPE* data, MTYPE* mat, uint2 p, int n, const int a) {
    // (1) constant write
    //unsigned long i = (unsigned long)p.y * n + (unsigned long)p.x;
    // mat[i] += a;
    // mat[i] = a;

    // or (2) recursion level write

    int subBlocksPerGPUblock_x = 32 >> SUBBLOCK_EXP;
    // 1024 threads available, 32 warps
    // int subBlockId = threadIdx.y / (SUBBLOCK_SIZE**2/32)
    int subBlockId = threadIdx.y >> ((SUBBLOCK_EXP << 1) - 5);

    int inner_coord_y = subBlockId / (subBlocksPerGPUblock_x);

    int subBlockIdx_y = inner_coord_y + blockIdx.y * subBlocksPerGPUblock_x;

    const int b = (int)log2f(subBlockIdx_y + 1) + 1;
    mat[p.y * n + p.x] = b;
}

// metodo kernel test
template <typename Lambda>
__global__ void kernel_test(const unsigned int n, const int a, const unsigned int msize, DTYPE* data, MTYPE* dmat, Lambda map, const unsigned int aux1, const unsigned int aux2, const unsigned int aux3) {
    auto p = map(n, msize, aux1, aux2, aux3, 0, 0);
    if (p.y >= p.x && p.y < n) {
        // if(p.y < n){
        work(data, dmat, p, n, 1);
    }
}

// metodo kernel test
template <typename Lambda>
__global__ void kernel_test_tensor_core(const unsigned int n, const int a, const unsigned int msize, DTYPE* data, MTYPE* dmat, Lambda map, const unsigned int aux1, const unsigned int aux2, const unsigned int aux3, const unsigned int subBlockGridSizex, const unsigned int subBlockGridSizey) {
    auto p = map(n, msize, aux1, aux2, aux3, subBlockGridSizex, subBlockGridSizey);
    if (p.y >= p.x && p.y < n) {
        // if(p.y < n){
        work(data, dmat, p, n, 1);
    }
}

// kernel test DP
// Root kernel process the square of ceil(n/2) in the middle of the triangle
// Then launches 2 child kernels to process the upper triangle and the lower right one.
// Well... actually, it launches the child kernel first, since there is no dependency
template <typename Lambda>
__global__ void kernel_test_DP(const unsigned int n, const unsigned int levelBlockedN, MTYPE* data, Lambda map, const uint32_t x0, const uint32_t y0) {
    int levelBlockedNHalf = ceil(levelBlockedN / 2.f);
    // Launch 2 child kernels

    if (levelBlockedN > 1) {
        if (threadIdx.x + blockIdx.x + threadIdx.y + blockIdx.y == 0) {
#ifdef DP

            cudaStream_t s1;
            cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
            cudaStream_t s2;
            cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);

            dim3 grid = dim3(levelBlockedNHalf, levelBlockedNHalf);
            kernel_test_DP<<<grid, blockDim, 0, s1>>>(n, levelBlockedNHalf, data, map, x0, y0 - levelBlockedNHalf);
            kernel_test_DP<<<grid, blockDim, 0, s2>>>(n, levelBlockedNHalf, data, map, x0 + levelBlockedN, y0 + levelBlockedN - levelBlockedNHalf);
#endif
        }
    }
    // Process data
    auto p = (uint2) { blockIdx.x + x0, blockIdx.y + y0 };
    if (p.x > p.y) {
        return;
    }
    p.x = p.x * blockDim.x + threadIdx.x;
    p.y = p.y * blockDim.y + threadIdx.y;

    if (p.y >= p.x && p.y < n) {
        // if(p.y < n){
        work(NULL, data, p, n, 1);
    }
}

// O(n^2) number of threads for work
__global__ void kernelDP_work(int n, int globalN, MTYPE *data, int offX, int offY){
    // Process data
    //if(threadIdx.x + threadIdx.y + blockIdx.x + blockIdx.y == 0)
    //printf("KernelWork n: %i, offX: %i, offY: %i  grid(%i, %i)\n", n, offX, offY, gridDim.x, gridDim.y);
    auto p = (uint2) {blockIdx.x, blockIdx.y};
    p.x = p.x * blockDim.x + threadIdx.x;
    p.y = p.y * blockDim.y + threadIdx.y;
    if (p.x >= n || p.y >= n){
        return;
    }
    p.x = p.x + offX;
    p.y = p.y + offY;
    //printf("KernelWork n: %i, offX: %i, offY: %i\ngrid(%i, %i)\n", n, offX, offY, gridDim.x, gridDim.y);
    //printf("p(%i,%i) - with off -> %i,%i\n", p.x-offX, p.y-offY, p.x, p.y);
    if(p.y >= p.x && p.y < globalN){
        work(NULL, data, p, globalN, 1);
    }
}

// 1 thread does exploration
__global__ void kernelDP_exp(int n, int globalN, MTYPE* data, int x0, int y0, int MIN_SIZE){
    #ifdef DP
        // 1) stopping case
        if(n <= MIN_SIZE){
            dim3 gridLeaf = dim3((n+BSIZE2D-1)/BSIZE2D, (n+BSIZE2D-1)/BSIZE2D, 1);
            dim3 blockLeaf = dim3(BSIZE2D, BSIZE2D, 1);
            //printf("minsizen: %i, offX: %i, offY: %i  grid(%i, %i)\n", n, x0, y0, gridDim.x, gridDim.y);
            kernelDP_work<<<gridLeaf, blockLeaf>>>(n, globalN, data, x0, y0);
            return;
        }
        // 2) explore up and right asynchronously
        cudaStream_t s1, s2, s3;
        cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
        cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking);
        int n2 = (n >> 1) + (n & 1);
        int nLeft = n-n2;
        // up
        kernelDP_exp<<<1,1,0,s1>>>(nLeft, globalN, data, x0     , y0     , MIN_SIZE);
        // bottom right
        kernelDP_exp<<<1,1,0,s2>>>(nLeft, globalN, data, x0 + n2, y0 + n2, MIN_SIZE);

        // 3) work in the bot middle
        //printf("n2: %i\n", n2);
        dim3 gridNode = dim3((n2+BSIZE2D-1)/BSIZE2D, (n2+BSIZE2D-1)/BSIZE2D, 1);
        //printf("KernelWork n: %i, offX: %i, offY: %i  grid(%i, %i)\n", n, x0, y0, gridDim.x, gridDim.y);
        dim3 blockLeaf = dim3(BSIZE2D, BSIZE2D, 1);
        kernelDP_work<<<gridNode, blockLeaf,0,s3>>>(n2, globalN, data, x0, y0 + nLeft);
    #endif
}

#endif
