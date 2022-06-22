//////////////////////////////////////////////////////////////////////////////////
//  gpumaps                                                                     //
//  A GPU benchmark of mapping functions                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2018 Cristobal A. Navarro.                                      //
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
#ifndef GPUBENCHMARKS_CUH
#define GPUBENCHMARKS_CUH
#include <cuda.h>
#include <mma.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
using namespace nvcuda;

double bbox(const unsigned long n, const unsigned int REPEATS) {
#ifdef DEBUG
    printf("[Bounding Box]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
    unsigned long msize, trisize;
    dim3 block, grid;
    init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);
    gen_bbox_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3, const unsigned int subBlockGridSizex, const unsigned int subBlockGridSizey) {
        if (blockIdx.x > blockIdx.y) {
            return (uint2) { 1, 0 };
        }
        return (uint2) { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map, 0, 0, 0);
#ifdef DEBUG
    double check = (float)verify_result(n, REPEATS, msize, hdata, ddata, hmat, dmat, dim3(0, 0, 0), block);
#endif
    cudaFree(ddata);
    cudaFree(dmat);
#ifdef DEBUG
    return time * check;
#else
    return time;
#endif
}

double lambda(const unsigned long n, const unsigned int REPEATS) {
#ifdef DEBUG
    printf("[Lambda (inverse)]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
    unsigned long msize, trisize;
    dim3 block, grid;
    init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3, const unsigned int subBlockGridSizex, const unsigned int subBlockGridSizey) {
        uint2 p;
        unsigned int bc = blockIdx.x + blockIdx.y * gridDim.x;
        float arg = __fmaf_rn(2.0f, (float)bc, 0.25f);
        p.y = __fmaf_rn(arg, rsqrtf(arg), OFFSET); // + 0.001f;
        p.x = (bc - (p.y * (p.y + 1) >> 1));
        // p.x = (bc - (p.y * (p.y + 1) >> 1));
        // printf("Lambda Antes p.x=%lu, p.y=%lu\n", p.x, p.y);

        p.y = p.y * blockDim.y + threadIdx.y;
        p.x = p.x * blockDim.x + threadIdx.x;
        // printf("Lambda Despues p.x=%lu, p.y=%lu\n", p.x, p.y);
        return p;
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map, 0, 0, 0);
#ifdef DEBUG
    double check = (float)verify_result(n, REPEATS, msize, hdata, ddata, hmat, dmat, dim3(0, 0, 0), block);
#endif
    cudaFree(ddata);
    cudaFree(dmat);
#ifdef DEBUG
    return time * check;
#else
    return time;
#endif
}

double rectangle(const unsigned long n, const unsigned int REPEATS) {
#ifdef DEBUG
    printf("[Rectangle]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
    unsigned long msize, trisize;
    dim3 block, grid;
    init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);
    gen_rectangle_pspace(n, block, grid);
    // printf("n = %i\n", n);
    //  formulate map
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3, const unsigned int subBlockGridSizex, const unsigned int subBlockGridSizey) {
        uint2 p;
        p.y = blockIdx.y * blockDim.y + threadIdx.y;
        p.x = blockIdx.x * blockDim.x + threadIdx.x;
        // threads beyond n/2 (in grid coords, because of block padding) must be
        // filtered, otherwise we will have multiple threads on the same data elements at the n/2 region
        if (p.y > n || p.x >= n >> 1) {
            p = (uint2) { 1, 0 };
        } else {
            if (p.x >= p.y) {
                // (n & 1) applies the correct offset for odd or even 'n'
                p.x = (n - 1) - p.x - (n & 1);
                p.y = (n - 1) - p.y;
            } else {
                p.y = p.y - 1;
            }
        }
        return p;
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map, 0, 0, 0);
#ifdef DEBUG
    double check = (float)verify_result(n, REPEATS, msize, hdata, ddata, hmat, dmat, dim3(0, 0, 0), block);
#endif
    cudaFree(ddata);
    cudaFree(dmat);
#ifdef DEBUG
    return time * check;
#else
    return time;
#endif
}

#define MAX_UINT 4294967295
double hadouken(const unsigned long n, const unsigned int REPEATS) {
#ifdef DEBUG
    printf("[Hadouken]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
    unsigned long msize, trisize;
    dim3 block(BSIZE2D, BSIZE2D);
    init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);
#ifdef DEBUG
    printf("gen_hadouken_pspace(%i, ...)\n", n);
#endif
    // trapezoid map
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const int aux1, const int aux2, const int aux3, const unsigned int subBlockGridSizex, const unsigned int subBlockGridSizey) {
        // (1) optimzized version: just arithmetic and bit-level operations
        /*
        const unsigned int h = WORDLEN - __clz(blockIdx.y + 1);
        const unsigned int qb = blockIdx.x & (MAX_UINT << h);
        const unsigned int k = (aux1 - (int)blockIdx.y) >> 31;
        return (uint2) { (blockIdx.x + qb + (k & gridDim.x)) * blockDim.x + aux3 + threadIdx.x, (blockIdx.y - (k & aux2) + (qb << 1)) * blockDim.x + aux3 + threadIdx.y };
*/
        // (2) normal version: arithmetic, bit and logical operations
        /*
        const unsigned int h    = WORDLEN - __clz(blockIdx.y+1);
        const unsigned int qb   = (blockIdx.x >> h)*(1 << h);
        const unsigned int k = (int)blockIdx.y - aux1 > 0? 1 : 0;
        return (uint2){ aux3 + (blockIdx.x + qb + k*gridDim.x)*blockDim.x + threadIdx.x, aux3 + (blockIdx.y - k*aux2 + (qb << 1))*blockDim.y + threadIdx.y };
        */

        // (3) simple version: no programming tricks

        if (aux1 >= blockIdx.y) {
            const unsigned int h = WORDLEN - __clz(blockIdx.y + 1);
            const unsigned int qb = (blockIdx.x >> h) * (1 << h);
            return (uint2) { aux3 + (blockIdx.x + qb) * blockDim.x + threadIdx.x, aux3 + (blockIdx.y + (qb << 1)) * blockDim.y + threadIdx.y };
        } else {
            return (uint2) { aux3 + (blockIdx.x + gridDim.x) * blockDim.x + threadIdx.x, aux3 + (blockIdx.y - aux2) * blockDim.y + threadIdx.y };
        }
    };
    // benchmark
    double time = benchmark_map_hadouken(REPEATS, block, n, msize, trisize, ddata, dmat, map);
#ifdef DEBUG
    double check = (float)verify_result(n, REPEATS, msize, hdata, ddata, hmat, dmat, dim3(0, 0, 0), block);
#endif
    cudaFree(ddata);
    cudaFree(dmat);
#ifdef DEBUG
    return time * check;
#else
    return time;
#endif
}

#define MAX_UINT 4294967295
double hadouken_tensor_core(const unsigned long n, const unsigned int REPEATS) {
#ifdef DEBUG
    printf("[Hadouken Tensor Core]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
    unsigned long msize, trisize;
    dim3 block(BSIZE2D, BSIZE2D);
    init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);
#ifdef DEBUG
    printf("gen_hadouken_tensor_core_pspace(%i, ...)\n", n);
#endif
    // trapezoid map
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const int aux1, const int aux2, const int aux3, const unsigned int subBlockGridSizex, const unsigned int subBlockGridSizey) {
        __shared__ half mata[256];
        __shared__ half matb[256];
        __shared__ float matc[256];
        uint2 m;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_fragment;
        // wmma::fill_fragment(a_fragment, 0.f);
        // wmma::fill_fragment(b_fragment, 0.f);

        int tid = threadIdx.x + blockDim.x * threadIdx.y;
        int subBlocksPerGPUblock_x = 32 >> SUBBLOCK_EXP;
        // 1024 threads available, 32 warps
        // int subBlockId = threadIdx.y / (SUBBLOCK_SIZE * SUBBLOCK_SIZE / 32);
        int subBlockId = threadIdx.y >> ((SUBBLOCK_EXP << 1) - 5);

        int inner_coord_x = subBlockId & (subBlocksPerGPUblock_x - 1);
        int inner_coord_y = subBlockId / (subBlocksPerGPUblock_x);

        int subBlockIdx_x = inner_coord_x + blockIdx.x * subBlocksPerGPUblock_x;
        int subBlockIdx_y = inner_coord_y + blockIdx.y * subBlocksPerGPUblock_x;

        if (subBlockIdx_x >= subBlockGridSizex || subBlockIdx_y >= subBlockGridSizey) {
            return (uint2) { 1, 0 };
        }
        int elementsPerSubBlock = 1024 / (subBlocksPerGPUblock_x * subBlocksPerGPUblock_x);
        int subBlocktid = tid & (elementsPerSubBlock - 1);

        int subBlockThreadIdx_x = subBlocktid & (SUBBLOCK_SIZE - 1);
        int subBlockThreadIdx_y = subBlocktid >> SUBBLOCK_EXP;

        int warpId = tid >> 5;

        if (subBlocktid < 64) {
            // if (subBlocktid == 0) {
            //     printf("\nsubblock id %i\n", subBlockId);
            // }

            // THIS WARP WRITES MATRIX B
            if (subBlocktid < 32) {
                const unsigned int h = WORDLEN - __clz(subBlockIdx_y + 1);
                const unsigned int b = (1 << h);
                // const unsigned int k = (aux1 - (int)subBlockIdx_y) >> 31;
                // if (subBlocktid == 0) {
                //    printf("Subblockid: %i, warpid: %i\n", subBlockId, warpId);
                //}

                // FIRST HALF OF ROWS FOR X VALUES
                // matb[subBlockId] = subBlockIdx_x;
                // matb[subBlockId + 16] = b;

                // SECOND HALF OF ROWS FOR Y VALUES
                matb[subBlockId * 32 + 0] = subBlockIdx_x;
                matb[subBlockId * 32 + 1] = b;

                matb[subBlockId * 32 + 16 + 0] = subBlockIdx_y;
                matb[subBlockId * 32 + 16 + 1] = 2 * b;
                // mata[subBlockId * 32 + 2] = k & subBlockGridSizex;
                // const unsigned int q = (subBlockIdx_x >> h);
                // matb[subBlockId * 32 + 0] = 1;
                // matb[subBlockId * 32 + 1] = q;
                // matb[subBlockId * 32 + 2] = 1;

                // aux3 + (subBlockIdx_x + qb) * SUBBLOCK_SIZE + subBlockThreadIdx_x

                // THIS WARP WRITES MATRIX A
            } else {
                const unsigned int h = WORDLEN - __clz(subBlockIdx_y + 1);
                // const unsigned int k = (aux1 - (int)subBlockIdx_y) >> 31;
                // if (subBlocktid == 0) {
                //    printf("Subblockid: %i, warpid: %i\n", subBlockId, warpId);
                //}
                // mata[subBlockId * 32 + 16 + 0] = subBlockIdx_y;
                // mata[subBlockId * 32 + 16 + 1] = b2;
                // mata[subBlockId * 32 + 16 + 2] = -(k & aux2);
                const unsigned int q = (subBlockIdx_x >> h);
                // FIRST HALF OF ROWS FOR X VALUES

                mata[subBlockId * 16 + 0] = 1;
                mata[subBlockId * 16 + 1] = q;

                // SECOND HALF OF ROWS FOR Y VALUES
                // matb[subBlockId * 16 + 128 + 0] = 1;
                // matb[subBlockId * 16 + 128 + 1] = q;
                //  matb[subBlockId * 32 + 16 + 2] = 1;
                //   (blockIdx.y - (k & aux2) + (qb << 1)) * blockDim.x + aux3 + threadIdx.y
            }
        }

        __syncthreads();

        if (warpId == 0) {
            wmma::fill_fragment(c_fragment, 0.f);

            wmma::load_matrix_sync(a_fragment, &mata[0], 16);

            wmma::load_matrix_sync(b_fragment, &matb[0], 16);
            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);
            wmma::store_matrix_sync(matc, c_fragment, 16, wmma::mem_row_major);
        }
        __syncthreads();

        m = (uint2) { (int)(matc[subBlockId * 16 + subBlockId * 2]), (int)(matc[subBlockId * 16 + subBlockId * 2 + 1]) };

        /*if (tid == 0 && blockIdx.x + blockIdx.y == 0) {
            printf("\nMATRIX A\n");

            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", (float)mata[i * 16 + j]);
                }
                printf("\n");
            }

            printf("\nMATRIX B\n");

            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", (float)matb[i * 16 + j]);
                }
                printf("\n");
            }

            printf("\nMATRIX C\n");

            for (int i = 0; i < 16; i++) {
                for (int j = 0; j < 16; j++) {
                    printf("%.0f ", (float)matc[i * 16 + j]);
                }
                printf("\n");
            }
        }
*/
        //__syncthreads();
        if (aux1 >= subBlockIdx_y) {

            return (uint2) { m.x * SUBBLOCK_SIZE + aux3 + subBlockThreadIdx_x, m.y * SUBBLOCK_SIZE + aux3 + subBlockThreadIdx_y };
        } else {
            return (uint2) { aux3 + (subBlockIdx_x + subBlockGridSizex) * SUBBLOCK_SIZE + subBlockThreadIdx_x, aux3 + (subBlockIdx_y - aux2) * SUBBLOCK_SIZE + subBlockThreadIdx_y };
        }

        // (1) optimzized version: just arithmetic and bit-level operations
        /*
        const unsigned int h = WORDLEN - __clz(blockIdx.y + 1);
        const unsigned int qb = blockIdx.x & (MAX_UINT << h);
        const unsigned int k = (aux1 - (int)blockIdx.y) >> 31;
        return (uint2) { (blockIdx.x + qb + (k & gridDim.x)) * blockDim.x + aux3 + threadIdx.x, (blockIdx.y - (k & aux2) + (qb << 1)) * blockDim.x + aux3 + threadIdx.y };
        */

        // (2) normal version: arithmetic, bit and logical operations
        /*
        const unsigned int h    = WORDLEN - __clz(blockIdx.y+1);
        const unsigned int qb   = (blockIdx.x >> h)*(1 << h);
        const unsigned int k = (int)blockIdx.y - aux1 > 0? 1 : 0;
        return (uint2){ aux3 + (blockIdx.x + qb + k*gridDim.x)*blockDim.x + threadIdx.x, aux3 + (blockIdx.y - k*aux2 + (qb << 1))*blockDim.y + threadIdx.y };
        */

        // (3) simple version: no programming tricks
        if (aux1 >= subBlockIdx_y) {

            const unsigned int h = WORDLEN - __clz(subBlockIdx_y + 1);
            const unsigned int qb = (subBlockIdx_x >> h) * (1 << h);
            return (uint2) { aux3 + (subBlockIdx_x + qb) * SUBBLOCK_SIZE + subBlockThreadIdx_x, aux3 + (subBlockIdx_y + (qb << 1)) * SUBBLOCK_SIZE + subBlockThreadIdx_y };
        } else {
            return (uint2) { aux3 + (subBlockIdx_x + subBlockGridSizex) * SUBBLOCK_SIZE + subBlockThreadIdx_x, aux3 + (subBlockIdx_y - aux2) * SUBBLOCK_SIZE + subBlockThreadIdx_y };
        }
    };
    // benchmark
    double time = benchmark_map_hadouken_tensor_core(REPEATS, block, n, msize, trisize, ddata, dmat, map);
#ifdef DEBUG
    double check = (float)verify_result(n, REPEATS, msize, hdata, ddata, hmat, dmat, dim3(0, 0, 0), block);
#endif
    cudaFree(ddata);
    cudaFree(dmat);
#ifdef DEBUG
    return time * check;
#else
    return time;
#endif
}

double DynamicParallelism(const unsigned long n, const unsigned int REPEATS) {
#ifdef DEBUG
    printf("[Dynamic Parallelism]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
    unsigned long msize, trisize;
    dim3 block(BSIZE2D, BSIZE2D);
    init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);
#ifdef DEBUG
    printf("gen_hadouken_pspace(%i, ...)\n", n);
#endif
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3, const unsigned int subBlockGridSizex, const unsigned int subBlockGridSizey) {
        if (blockIdx.x > blockIdx.y) {
            return (uint2) { 1, 0 };
        }
        return (uint2) { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
    };
    // benchmark
    double time = benchmark_map_DP(REPEATS, block, n, msize, trisize, ddata, dmat, map);
#ifdef DEBUG
    double check = (float)verify_result(n, REPEATS, msize, hdata, ddata, hmat, dmat, dim3(0, 0, 0), block);
#endif
    cudaFree(ddata);
    cudaFree(dmat);
#ifdef DEBUG
    return time * check;
#else
    return time;
#endif
}

#endif
