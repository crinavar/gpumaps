//////////////////////////////////////////////////////////////////////////////////
//  gpumaps                                                                     //
//  A GPU benchmark of mapping functions                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
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


__device__ void work(DTYPE* data, MTYPE* mat, uint2 p, unsigned int n) {
    DTYPE a = data[p.x];
    DTYPE b = data[p.y];
    mat[p.y * (size_t)n + p.x] = sqrtf((float)(a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
    // printf("%f\n", mat[p.y*n + p.x]);
}

// metodo kernel test
template <typename Lambda>
__global__ void kernel_test(const unsigned int n, const int a, const unsigned int msize, DTYPE* data, MTYPE* dmat, Lambda map, const unsigned int aux1, const unsigned int aux2, const unsigned int aux3) {
    auto p = map(n, msize, aux1, aux2, aux3);
    if (p.y >= p.x && p.y < n) {
        // if(p.y < n){
        work(data, dmat, p, n);
    }
}

// kernel test DP
// Root kernel process the square of ceil(n/2) in the middle of the triangle
// Then launches 2 child kernels to process the upper triangle and the lower right one.
// Well... actually, it launches the child kernel first, since there is no dependency
template <typename Lambda>
__global__ void kernel_test_DP(const unsigned int n, const unsigned int levelBlockedN, MTYPE* data, DTYPE* wat, Lambda map, const unsigned int x0, const unsigned int y0) {
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
            kernel_test_DP<<<grid, blockDim, 0, s1>>>(n, levelBlockedNHalf, data, wat, map, x0, y0 - levelBlockedNHalf);
            kernel_test_DP<<<grid, blockDim, 0, s2>>>(n, levelBlockedNHalf, data, wat, map, x0 + levelBlockedN, y0 + levelBlockedN - levelBlockedNHalf);
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
        work(wat, data, p, n);
    }
}

// O(n^2) number of threads for work (on = original n)
__global__ void kernelDP_work(int on, int n, MTYPE* data, DTYPE* wat, int offX, int offY) {
    // Process data
    auto p = (uint2) { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
    // printf("thread at local x=%i  y=%i\n", p.x, p.y);
    if (p.x >= n || p.y >= n) {
        // printf("discarding thread at local x=%i  y=%i\n", p.x, p.y);
        return;
    }
    p.x = p.x + offX;
    p.y = p.y + offY;
    // printf("checking thread at global x=%i  y=%i\n", p.x, p.y);
    if (p.y >= p.x && p.y < on) {
        //printf("work at x=%i  y=%i\n", p.x, p.y);
        work(wat, data, p, on);
    }
}

// 1 thread does exploration (on = original n)
__global__ void kernelDP_exp(int on, int n, MTYPE* data, DTYPE* wat, int x0, int y0, int MIN_SIZE) {
#ifdef DP
    // 1) stopping case
    if (n <= MIN_SIZE) {
        dim3 bleaf(BSIZE2D, BSIZE2D), gleaf = dim3((n + bleaf.x - 1) / bleaf.x, (n + bleaf.y - 1) / bleaf.y, 1);
        // printf("leaf kernel at x=%i  y=%i   size %i x %i (grid (%i,%i,%i)  block(%i,%i,%i))\n", x0, y0, n, n, gleaf.x, gleaf.y, gleaf.z, bleaf.x, bleaf.y, bleaf.z);
        kernelDP_work<<<gleaf, bleaf>>>(on, n, data, wat, x0, y0);
        return;
    }
    // 2) explore up and right asynchronously
    cudaStream_t s1, s2, s3;
    cudaStreamCreateWithFlags(&s1, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s2, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&s3, cudaStreamNonBlocking);
    int subn = (n >> 1) + (n & 1);
    int n2 = n >> 1;
    // printf("subn %i\nn2 %i\n", subn, n2);
    //  up
    kernelDP_exp<<<1, 1, 0, s1>>>(on, n2, data, wat, x0, y0, MIN_SIZE);
    // bottom right
    kernelDP_exp<<<1, 1, 0, s2>>>(on, n2, data, wat, x0 + subn, y0 + subn, MIN_SIZE);
    // 3) work in the bot middle
    dim3 bnode(BSIZE2D, BSIZE2D);
    dim3 gnode = dim3((subn + bnode.x - 1) / bnode.x, (subn + bnode.y - 1) / bnode.y, 1);
    // printf("node kernel at x=%i  y=%i   size %i x %i\n", x0, y0+n2, subn, subn);
    kernelDP_work<<<gnode, bnode, 0, s3>>>(on, subn, data, wat, x0, y0 + n2);
#endif
}
#endif
