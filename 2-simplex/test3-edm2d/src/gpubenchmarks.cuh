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
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3) {
        if (blockIdx.x > blockIdx.y) {
            return (uint2) { 1, 0 };
        }
        return (uint2) { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map, 0, 0, 0, "BB-");
    // check result
    double check = (float)verify_result(n, REPEATS, msize, hdata, ddata, hmat, dmat, grid, block);
    cudaFree(ddata);
    cudaFree(dmat);
    free(hdata);
    free(hmat);
#ifdef DEBUG
    return time;
#else
    return time * check;
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
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3) {
        uint2 p;
        unsigned int bc = blockIdx.x + blockIdx.y * gridDim.x;
#ifdef LAMBDA_FP32
        float arg = __fmaf_rn(2.0f, (float)bc, 0.25f);
        p.y = __fmaf_rn(arg, rsqrtf(arg), OFFSET); // + 0.001f;
#elif LAMBDA_FP64
        double arg = __fma_rn(2.0f, (double)bc, 0.25f);
        p.y = __fma_rn(arg, rsqrt(arg), OFFSET); // + 0.001f;
#endif
        p.x = (bc - (p.y * (p.y + 1) >> 1));

        p.y = p.y * blockDim.y + threadIdx.y;
        p.x = p.x * blockDim.x + threadIdx.x;
        return p;
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map, 0, 0, 0, "LAMB-");
    double check = (float)verify_result(n, REPEATS, msize, hdata, ddata, hmat, dmat, grid, block);
    cudaFree(ddata);
    cudaFree(dmat);
    free(hdata);
    free(hmat);
#ifdef DEBUG
    return time;
#else
    return time * check;
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
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3) {
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
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map, 0, 0, 0, "RECT-");
    // check result
    double check = (float)verify_result(n, REPEATS, msize, hdata, ddata, hmat, dmat, grid, block);
    cudaFree(ddata);
    cudaFree(dmat);
    free(hdata);
    free(hmat);
#ifdef DEBUG
    return time;
#else
    return time * check;
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
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const int aux1, const int aux2, const int aux3) {
        // (1) optimzized version: just arithmetic and bit-level operations
        const unsigned int h = WORDLEN - __clz(blockIdx.y + 1);
        const unsigned int qb = blockIdx.x & (MAX_UINT << h);
        const unsigned int k = (aux1 - (int)blockIdx.y) >> 31;
        return (uint2) { (blockIdx.x + qb + (k & gridDim.x)) * blockDim.x + aux3 + threadIdx.x, (blockIdx.y - (k & aux2) + (qb << 1)) * blockDim.x + aux3 + threadIdx.y };
    };
    // benchmark
    double time = benchmark_map_hadouken(REPEATS, block, n, msize, trisize, ddata, dmat, map, "HAD-");
    double check = (float)verify_result(n, REPEATS, msize, hdata, ddata, hmat, dmat, dim3(0, 0, 0), block);
    cudaFree(ddata);
    cudaFree(dmat);
    free(hdata);
    free(hmat);
#ifdef DEBUG
    return time;
#else
    return time * check;
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
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const int aux1, const int aux2, const int aux3) {
        if (blockIdx.x > blockIdx.y) {
            return (uint2) { 1, 0 };
        }
        return (uint2) { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
    };
    // benchmark
    double time = benchmark_map_DP(REPEATS, block, n, msize, trisize, ddata, dmat, map, "DP-");
    double check = (float)verify_result(n, REPEATS, msize, hdata, ddata, hmat, dmat, dim3(0, 0, 0), block);
    cudaFree(ddata);
    cudaFree(dmat);
    free(hdata);
    free(hmat);
#ifdef DEBUG
    return time;
#else
    return time * check;
#endif
}

#endif
