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
    // mat[i] += a;
    mat[i] = a;

    // or (2) recursion level write
    // const int b = (int)log2f(blockIdx.y + 1) + 1;
    // mat[p.y * n + p.x] = b;
}

__device__ void workSubBlock(DTYPE* data, MTYPE* mat, uint2 p, int n, const int a) {
    // (1) constant write
    unsigned long i = (unsigned long)p.y * n + (unsigned long)p.x;
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
#endif
