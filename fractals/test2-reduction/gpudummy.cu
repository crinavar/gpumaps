#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <cuda.h>
#include <mma.h>
#include <cassert>

#include "gputools.cu"
#include "gpudummy.cuh"
#include "interface.h"
#include "kernels.cuh"

using namespace nvcuda;

#define PRINTLIMIT 7
#define INNER_REPEATS 10

unsigned int REPEATS; // Lazy programming
unsigned int WSIZE;  // Lazy programming

statistics gpudummy(unsigned int method, unsigned int repeats){

    RunningStat* r;
    statistics stat;

    WSIZE               = min(BSIZE1D, WARPSIZE);
    unsigned long n     = (int)ceil(pow(2, RLEVEL));        // Linear size
    unsigned long n2    = n*n;
    unsigned long rb    = max((int)(RLEVEL - BPOWER), 0);   // BlockTiled level
    unsigned long nb    = (int)(1 << rb);                   // N elements in BTiled level

    REPEATS = repeats;

#ifdef DEBUG
    printf("BPOWER=%i, BSIZE=%i, WSIZE=%i, scale r=%i, block-scale rb = %i   nb = %i\n", BPOWER, BSIZE2D, WSIZE, RLEVEL, rb, nb);
#endif 

    // int and float version of Euclidean space;
    MTYPE *Ec_h, *Ec_d, *res_d, *res_h;

#ifdef DEBUG
    printf("Allocating Memory Spaces of %lu x %lu = %lu -> %f MB in CPU & GPU.......", n, n, n2, n2*sizeof(MTYPE)/(1024.0*1024.0)); fflush(stdout); 
#endif

    Ec_h = (MTYPE*)malloc(sizeof(MTYPE)*n*n);
    res_h = (MTYPE*)calloc(1, sizeof(MTYPE));

    cudaMalloc(&Ec_d, sizeof(MTYPE)*n*n);
    cudaMalloc(&res_d, sizeof(MTYPE));

    last_cuda_error("cudaMalloc");
#ifdef DEBUG
    printf("ok\n"); fflush(stdout);

    // init data
    printf("Initializing Matrices......."); fflush(stdout);
#endif

    initmat<MTYPE>(Ec_h, n2, 0);

#ifdef DEBUG
    printf("ok\n"); fflush(stdout);
#endif

    cudaMemcpy(Ec_d, Ec_h, sizeof(MTYPE)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(res_d, res_h, sizeof(MTYPE), cudaMemcpyHostToDevice);

    switch (method){
        case 0:
            r = boundingBox(Ec_d, res_d, n, nb, rb);
            break;
        case 1:
            r = lambda(Ec_d, res_d, n, nb, rb);
            break;
        case 2:
            r = lambda_tc(Ec_d, res_d, n, nb, rb);
            break;
        case 3:
            r = lambda_tc_optimized(Ec_d, res_d, n, nb, rb);
            break;
        default:
            printf("Method can only take values 0, 1, 2 or 3\n");
            exit(2);
    }
    cudaMemcpy(Ec_h, Ec_d, sizeof(MTYPE)*n*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(res_h, res_d, sizeof(MTYPE), cudaMemcpyDeviceToHost);

#ifdef DEBUG
    if(RLEVEL <= PRINTLIMIT){
        printspace<MTYPE>(Ec_h, n);
    }
    printf("Result of reduction: %i\n", *res_h);
#endif

#ifdef DEBUG
    printf("gpudummy(): verifying results..."); fflush(stdout);
#endif

    //assert( verifyReduction<MTYPE>(res_h, n) ); // Verifying results

#ifdef DEBUG
    printf("ok\n\n"); fflush(stdout);
#endif

    free(Ec_h);
    free(res_h);
    cudaFree(Ec_d);

    // Return computing time
    stat.mean = r->Mean();
    stat.variance = r->Variance();
    stat.stdev = r->StandardDeviation();
    stat.sterr = r->StandardDeviation()/((double)sqrt(r->NumDataValues()));

    free(r);
    return stat;
}


RunningStat* boundingBox(MTYPE* M, MTYPE* res, unsigned long n, unsigned long nb, unsigned long rb){

    dim3 block, grid;

    auto psgen = [] (unsigned long rb, int BSIZE, dim3 &b, dim3 &g){ 
        b = dim3(BSIZE, BSIZE, 1);
        g = dim3(pow(2,rb), pow(2,rb), 1);
    };

    auto bbmap = [] __device__ (const int nb, const int rb, const int WSIZE){
        uint2 m;
        m.x = blockIdx.x*blockDim.x + threadIdx.x;
        m.y = blockIdx.y*blockDim.y + threadIdx.y;
        return m;
    };
    
    psgen(rb, 1<<BPOWER, block, grid);

    return performLoad(M, res, n, nb, rb, block, grid, bbmap);
}

RunningStat* lambda(MTYPE* M, MTYPE* res, unsigned long n, unsigned long nb, unsigned long rb){

    dim3 block, grid;

    // pspace: orthotope for lambda
    auto psgen =  [] (unsigned long rb, int BSIZE, dim3 &b, dim3 &g){ 
        b = dim3(BSIZE, BSIZE, 1);
        g = dim3((int)pow(3, ceil(rb/2.0)), (int)pow(3, floor(rb/2.0)), 1); 
    };

    // lambda map
    auto lambdamap = [] __device__ (const int nb, const int rb, const int WSIZE){
        __shared__ uint2 m;
        auto beta = [] __device__ (const int nb, const uint2 w, const int u){
            int b = (int)((blockIdx.x*(u & 1) + blockIdx.y*((u+1) & 1))/(pow3((u>>1) + (u&1) - 1)))%3;
            return b;
        };
        int lid = threadIdx.x + threadIdx.y*blockDim.x;
        int tid = lid;
        if(lid < WSIZE){
            uint2 lm = {0,0};
            while(lid < rb){
                int b = beta(nb, {blockIdx.x, blockIdx.y}, lid+1);
                lm.x += (b >> 1) * (1 << (lid));
                lm.y += (b - (b >> 1)) * (1 << (lid));
                lid += WSIZE;
            }
            lm = warp_reduce(lm,WSIZE); 
            if(tid == 0){ m = lm; }
        }
        __syncthreads();
        return (uint2){m.x * blockDim.x + threadIdx.x, m.y * blockDim.y + threadIdx.y};
    };

    psgen(rb, 1<<BPOWER, block, grid);

    return performLoad(M, res, n, nb, rb, block, grid, lambdamap);
}

RunningStat* lambda_tc(MTYPE* M, MTYPE* res, unsigned long n, unsigned long nb, unsigned long rb){

    dim3 block, grid;

    if (BSIZE1D < 32){
#ifdef DEBUG
        printf("Blocksize needs to be at least 32 to use warp sync mma operations.\n");
#endif
        return nullptr;
    }

    // pspace: orthotope for lambda
    auto psgen = [] (unsigned long rb, int BSIZE, dim3 &b, dim3 &g){ 
        b = dim3(BSIZE, BSIZE, 1);
        g = dim3((int)pow(3, ceil(rb/2.0)), (int)pow(3, floor(rb/2.0)), 1); 
    };

    // Tensor core lambda map
    // This map assumes that the block size is >=32, which is the minimum to perform tensor core mma,
    auto lambdamap_tc = [] __device__ (const int nb, const int rb, const int WSIZE){

        __shared__ half mata[256]; 
        __shared__ half matb[256];
        
        //Has to be declared after the matrices above to avoid 8-byte shifting
        __shared__ uint2 m;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_fragment;

        auto beta = [] __device__ (const int nb, const int u){
            int b = (int)((blockIdx.x*(u & 1) + blockIdx.y*((u+1) & 1))/(pow3((u>>1) + (u&1) - 1)))%3;
            return b;
        };
        
        int lid = threadIdx.x + threadIdx.y*blockDim.x;
        int index = lid;

        if (lid < 32) {
            //Has to be resetted to 0. Latter kernel calls were getting weird values
            #pragma unroll
            for (int i=0; i<2; i++){
                matb[i*32 + lid] = 0;
            }
            if (lid < rb){
                mata[lid] = 1 << lid;
            }
        } else {
            lid -= 32;
            if (lid < rb*2){
                char column = lid/rb;
                //lid = lid%rb;
                lid = lid-rb*column;
                char b = beta(nb, lid+1);
                int lm = ((b-(b >> 1))*(column & 1) + (b >> 1)*((column+1) & 1));
                matb[lid + column*16] = lm;
            }
        } 

        __syncthreads();
        if (index < 32) {
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
        
            wmma::load_matrix_sync(b_fragment, &matb[0], 16);
            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);
        }

        __syncthreads();
        if (index == 0){
            m = (uint2){(int)(c_fragment.x[0]), (int)(c_fragment.x[1])};
        }

        __syncthreads();

        return (uint2){m.x * blockDim.x + threadIdx.x, m.y * blockDim.y + threadIdx.y};
    };

    psgen(rb, 1<<BPOWER, block, grid);

    return performLoad(M, res, n, nb, rb, block, grid, lambdamap_tc );
}

RunningStat* lambda_tc_optimized(MTYPE* M, MTYPE* res, unsigned long n, unsigned long nb, unsigned long rb){

    dim3 block, grid;

    rb++;

    if (BSIZE1D < 32){
#ifdef DEBUG
        printf("Blocksize needs to be at least 32 to use warp sync mma operations.\n");
#endif
        return nullptr;
    }

    if (BPOWER != 5){
#ifdef DEBUG
        printf("BPOWER has to be 5 to use this method.\n");
#endif
        return nullptr;
    }

    // pspace: orthotope for lambda on the thing
    auto psgen = [] (unsigned long rb, int BSIZE, dim3 &b, dim3 &g){ 
        b = dim3(BSIZE, BSIZE, 1);
        g = dim3( (int)ceil(pow(3, ceil(rb/2.0)) / 2.0 ), (int)ceil(pow(3, floor(rb/2.0)) / 2.0), 1); 
    };

    // Tensor core lambda map
    // This map assumes that the block size is >=32, which is the minimum to perform tensor core mma,
    auto lambdamap_tc = [] __device__ (const int nb, const int rb, const int WSIZE){

        __shared__ half mata[256]; 
        __shared__ half matb[256];
        __shared__ float matc[256];
        
        //Has to be declared after the matrices above to avoid 8-byte shifting
        uint2 m;

        //Strange behaviour when fragments are shared, the entire iteration reads the same value
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_fragment;

        wmma::fill_fragment(c_fragment, 0.0f);

        auto beta = [] __device__ (const int nb, uint2 bids, const int u){
            int b = (int)((bids.x*(u & 1) + bids.y*((u+1) & 1))/(pow3((u>>1) + (u&1) - 1)))%3;
            return b;
        };
        
        int lid = threadIdx.x + threadIdx.y*blockDim.x;
        int index = lid;

        if (lid < 256) {
            if (lid < rb){
                mata[lid] = 1 << lid;
            }
        } else {
            lid -= 256;
            if (lid < 128){
                char row = lid >> 4;
                char aux = lid >> 5;
                char b = beta(nb, {(blockIdx.x<<1) + (aux & 1), (blockIdx.y<<1) + (aux>>1)}, (lid&15)+1);
                int lm = ((b-(b >> 1))*(row & 1) + (b >> 1)*((row+1) & 1));
                matb[lid] = lm;
            }
        } 

        __syncthreads();
        if (index < 32) {
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
        
            wmma::load_matrix_sync(b_fragment, &matb[0], 16);
            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

            wmma::store_matrix_sync(&matc[0], c_fragment, 16, wmma::mem_row_major);
        }
        __syncthreads();

        char x = threadIdx.x >> 4;
        char y = threadIdx.y >> 4;
        
        if ( (blockIdx.x == gridDim.x-1 && x==1) || (blockIdx.y == gridDim.y-1 && y==1)){ //fix
            return (uint2){0xFFFFFFFF, 0xFFFFFFFF};
        }
        char ss = (x<<1) + (y<<2);
        m = (uint2){(int)(matc[ss]), (int)(matc[ss+1])};

        return (uint2){(m.x << 4) + (threadIdx.x & 15), (m.y << 4) + (threadIdx.y & 15)};
    };

    psgen(rb, 1<<BPOWER, block, grid);

    return performLoad(M, res, n, nb, rb, block, grid, lambdamap_tc );
}


template<typename Lambda>
RunningStat* performLoad(MTYPE *M, MTYPE* res, unsigned long n, unsigned long nb, unsigned long rb, dim3 block, dim3 grid, Lambda map){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // runningstat statistics
    RunningStat *r = new RunningStat();
    float t = 0.0f;

#ifdef DEBUG
    printf("block: %i x %i x %i\n", block.x, block.y, block.z);
    printf("grid:  %i x %i x %i\n", grid.x, grid.y, grid.z);
    printf("\x1b[1mgpudummy(): REPEATS=%i...INNER_REPEATS=%i.......\x1b[0m\n", REPEATS, INNER_REPEATS); fflush(stdout);
#endif

    kernel_write_lambda<<< grid, block >>>(M, n, nb, rb, 1, WSIZE, map);

    n = max(n, (long)block.x);

    for(int i=0; i<REPEATS; ++i){
        cudaEventRecord(start);
        for(int i=0; i<INNER_REPEATS; ++i){
            kernel_block_reduction<<< grid, block >>>(M, res, n, nb, rb, WSIZE, map);
            cudaDeviceSynchronize();
        }
        last_cuda_error("lastError");
        cudaDeviceSynchronize();
        last_cuda_error("cudaDeviceSynchronize");
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&t, start, stop);
        r->Push(t/(1000.0f * INNER_REPEATS));
    }
#ifdef DEBUG
    printf("\x1b[1mok\n\x1b[0m"); fflush(stdout);
#endif

    return r;
}
