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
#ifndef GPUBENCHMARKS_CUH
#define GPUBENCHMARKS_CUH

double bbox(const unsigned int n, const unsigned int REPEATS, double DENSITY) {
#ifdef DEBUG
    printf("[Bounding Box]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
    unsigned long msize, trisize;
    dim3 block, grid;
    init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, DENSITY);
    gen_bbox_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__(const unsigned int n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3) {
        if (blockIdx.x > blockIdx.y) {
            return (int2) { -1, -2 };
        }
        return (int2) { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat1, dmat2, map, 0, 0, 0, "BB-");
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat2);
    cudaFree(ddata);
    cudaFree(dmat1);
    cudaFree(dmat2);

    return time * check;
}

double lambda(const unsigned int n, const unsigned int REPEATS, double DENSITY) {
#ifdef DEBUG
    printf("[Lambda (inverse)]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
    unsigned long msize, trisize;
    dim3 block, grid;
    init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, DENSITY);
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__(const unsigned int n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3) {
        int2 p;
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
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat1, dmat2, map, 0, 0, 0, "LAMB-");
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat2);
    cudaFree(ddata);
    cudaFree(dmat1);
    cudaFree(dmat2);

    return time * check;
}

double rectangle(const unsigned int n, const unsigned int REPEATS, double DENSITY) {
#ifdef DEBUG
    printf("[Rectangle]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
    unsigned long msize, trisize;
    dim3 block, grid;
    init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, DENSITY);
    gen_rectangle_pspace(n, block, grid);
#ifdef DEBUG
    printf("grid(%i, %i, %i)   block(%i, %i, %i)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);
#endif
    // formulate map
    auto map = [] __device__(const unsigned int n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3) {
        int2 p;
        p.y = blockIdx.y * blockDim.y + threadIdx.y;
        p.x = blockIdx.x * blockDim.x + threadIdx.x;
        if (p.x >= p.y) {
            p.x = n - p.x - 1;
            p.y = n - p.y - 1;
        } else {
            p.y = p.y - 1;
        }
        return p;
    };
    // benchmark
    double time = benchmark_map_rectangle(REPEATS, block, grid, n, msize, trisize, ddata, dmat1, dmat2, map, 0, 0, 0, "RECT-");
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat2);
    cudaFree(ddata);

    cudaFree(dmat1);
    cudaFree(dmat2);

    return time * check;
}

#define MAX_UINT 4294967295
double hadouken(const unsigned long n, const unsigned int REPEATS, double DENSITY) {
#ifdef DEBUG
    printf("[Hadouken]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
    unsigned long msize, trisize;
    dim3 block(BSIZE2D, BSIZE2D);
    init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, DENSITY);
#ifdef DEBUG
    printf("gen_hadouken_pspace(%i, ...)\n", n);
#endif
    // trapezoid map
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const int aux1, const int aux2, const int aux3) {
        // (1) optimzized version: just arithmetic and bit-level operations

        const int h = WORDLEN - __clz(blockIdx.y + 1);
        const int qb = blockIdx.x & (MAX_UINT << h);
        const int k = (aux1 - (int)blockIdx.y) >> 31;
        return (int2) { (blockIdx.x + qb + (k & gridDim.x)) * blockDim.x + aux3 + threadIdx.x, (blockIdx.y - (k & aux2) + (qb << 1)) * blockDim.x + aux3 + threadIdx.y };

        // (2) normal version: arithmetic, bit and logical operations
        /*
        const unsigned int h    = WORDLEN - __clz(blockIdx.y+1);
        const unsigned int qb   = (blockIdx.x >> h)*(1 << h);
        const unsigned int k = (int)blockIdx.y - aux1 > 0? 1 : 0;
        return (int2){ aux3 + (blockIdx.x + qb + k*gridDim.x)*blockDim.x + threadIdx.x, aux3 + (blockIdx.y - k*aux2 + (qb << 1))*blockDim.y + threadIdx.y };
        */

        // (3) simple version: no programming tricks
        /*
        if( aux1 >= blockIdx.y ){
            const unsigned int h    = WORDLEN - __clz(blockIdx.y+1);
            const unsigned int qb   = (blockIdx.x >> h)*(1 << h);
            return (int2){ aux3 + (blockIdx.x + qb)*blockDim.x + threadIdx.x, aux3 + (blockIdx.y + (qb << 1))*blockDim.y + threadIdx.y };
        }
        else{
            return (int2){ aux3 + (blockIdx.x + gridDim.x)*blockDim.x + threadIdx.x, aux3 + (blockIdx.y - aux2)*blockDim.y + threadIdx.y };
        }
        */
    };
    // benchmark
    double time = benchmark_map_hadouken(REPEATS, block, n, msize, trisize, ddata, dmat1, dmat2, map, "HAD-");
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat2);
    cudaFree(ddata);
    cudaFree(dmat1);
    cudaFree(dmat2);

#ifdef DEBUG
    return time;
#else
    return time * check;
#endif
}

double DynamicParallelism(const unsigned long n, const unsigned int REPEATS, double DENSITY) {
#ifdef DEBUG
    printf("[Dynamic Parallelism]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
    unsigned long msize, trisize;
    dim3 block(BSIZE2D, BSIZE2D);
    init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, DENSITY);
#ifdef DEBUG
    printf("gen_DP_pspace(%i, ...)\n", n);
#endif
    auto map = [] __device__(const unsigned long n, const unsigned long msize, const int aux1, const int aux2, const int aux3) {
        if (blockIdx.x > blockIdx.y) {
            return (uint2) { 1, 0 };
        }
        return (uint2) { blockIdx.x * blockDim.x + threadIdx.x, blockIdx.y * blockDim.y + threadIdx.y };
    };
    // benchmark
    double time = benchmark_map_DP(REPEATS, block, n, msize, trisize, ddata, dmat1, dmat2, map, "DP-");
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat2);
    cudaFree(ddata);
    cudaFree(dmat1);
    cudaFree(dmat2);

#ifdef DEBUG
    return time;
#else
    return time * check;
#endif
}

#endif
