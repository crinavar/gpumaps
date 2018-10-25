#include <stdio.h>
#include <stdlib.h>
#include <vector>
#include <math.h>
#include <cuda.h>
#include <cassert>
#include "interface.h"
#include "gputools.cuh"
#include "kernels.cuh"
#include "running_stat.h"

#define PRINTLIMIT 6
#define INNER_REPEATS 10
#define REPEATS 100

//#define BPOWER 5
//#define BSIZE (1<<BPOWER)

statistics gpudummy(unsigned long r, int BPOWER){
    statistics stat;
    float t1=0.0f, t2=0.0f; 
    // linear block size
    int BSIZE = 1<<BPOWER;
    int WSIZE = min(BSIZE*BSIZE, WARPSIZE);
    // linear number of elements
    unsigned long n = (int)ceil(pow(2,r));
    // total number of elements
    unsigned long n2 = n*n;
    // block level rb
    unsigned long rb = max((int)(r-BPOWER),0);
    unsigned long nb = (int)(1 << rb);
    printf("scale r=%i, block-scale rb = %i   nb = %i\n", r, rb, nb);

    // int and float version of Euclidean space;
    char *Ec_h, *Ec_d;
    //int *Ei_h, *Ei_d;
    //float *Ef_h, *Ef_d;

    // runningstat statistics
    RunningStat r1, r2;

    // begin allocating the n x n memory
    printf("Allocating Memory Spaces of %lu x %lu = %lu -> %f MB in CPU & GPU.......", n, n, n2, n2*sizeof(char)/(1024.0*1024.0)); fflush(stdout); 
    //Ei_h = (int*)malloc(sizeof(int)*n*n);
    //Ef_h = (float*)malloc(sizeof(float)*n*n);
    Ec_h = (char*)malloc(sizeof(char)*n*n);
    //cudaMalloc(&Ei_d, sizeof(int)*n*n);
    //cudaMalloc(&Ef_d, sizeof(float)*n*n);
    cudaMalloc(&Ec_d, sizeof(char)*n*n);
    gpuErrchk( cudaPeekAtLastError() );
    //cudaMallocManaged(&Ei, n*n*sizeof(int)  );
    //cudaMallocManaged(&Ef, n*n*sizeof(float));
    //cudaDeviceSynchronize();
    printf("ok\n"); fflush(stdout);

    // init data
    printf("Initializing Matrices......."); fflush(stdout);
    //initmat<int>(Ei_h, n2, 0); 
    //initmat<float>(Ef_h, n2, 0.0); 
    initmat<char>(Ec_h, n2, 0); 
    printf("ok\n"); fflush(stdout);
    //cudaMemcpy(Ei_d, Ei_h, sizeof(int)*n*n, cudaMemcpyHostToDevice);
    //cudaMemcpy(Ef_d, Ef_h, sizeof(float)*n*n, cudaMemcpyHostToDevice);
    cudaMemcpy(Ec_d, Ec_h, sizeof(char)*n*n, cudaMemcpyHostToDevice);

    // common nb = linear number of blocks, BSIZE= blocksize
    dim3 block, grid;

    // pspace: orthotopes
    std::vector<void(*)(unsigned long n, int BSIZE, dim3 &b, dim3 &g)> psgen;  // parallel space generators
    psgen.push_back( [] (unsigned long rb, int BSIZE, dim3 &b, dim3 &g){ 
        b = dim3(BSIZE, BSIZE, 1);
        //g = dim3((n+b.x - 1)/b.x, (n+b.y - 1)/b.y, 1);
        g = dim3(pow(2,rb), pow(2,rb), 1);
    });

    // pspace: orthotope for lambda
    psgen.push_back( [] (unsigned long rb, int BSIZE, dim3 &b, dim3 &g){ 
        b = dim3(BSIZE, BSIZE, 1);
        g = dim3((int)pow(3, ceil(rb/2.0)), (int)pow(3, floor(rb/2.0)), 1); 
    });

    // bounding box map
    auto bbmap = [] __device__ (const int nb, const int rb, const int WSIZE){
        uint2 m;
        m.x = blockIdx.x*blockDim.x + threadIdx.x;
        m.y = blockIdx.y*blockDim.y + threadIdx.y;
        return m;
    };
    // lambda map
    auto lambdamap = [] __device__ (const int nb, const int rb, const int WSIZE){
        __shared__ uint2 m;
        auto beta = [] __device__ (const int nb, const uint2 w, const int u){
            //int b = (int)((w.x*( u % 2) + w.y*((u+1) % 2))/(pow(3, ceil(u/2.0) - 1)))%3;
            int b = (int)((blockIdx.x*(u & 1) + blockIdx.y*((u+1) & 1))/(pow3((u>>1) + (u&1) - 1)))%3;
            return b;
        };
        /*
        auto dx = [] __device__ (const unsigned int b){
            return (b>>1);
        };
        auto dy = [] __device__ (const unsigned int b){
            return b - (b>>1);
        };
        if(threadIdx.x + threadIdx.y == 0){
            m = {0,0};
            for(int u=1; u<=rb; ++u){
                int bindex = beta(nb, {blockIdx.x, blockIdx.y},u);
                //int bindex = (int)((blockIdx.x*(u & 1) + blockIdx.y*((u+1) & 1))/(pow3((u>>1) + (u&1) - 1)))%3;
                m.x += dx(bindex) * (1 << (u-1));
                m.y += dy(bindex) * (1 << (u-1));
                //printf("\nu=%i bindex=%i (dx,dy) = (%i,%i) block_xy (%i, %i) = d(%i, %i)", u, bindex, dxval, dyval, blockidx.x, blockidx.y, m.x, m.y); 
            }
        }

        __syncthreads();
        uint2 tid = {m.x * blockDim.x + threadIdx.x, m.y * blockDim.y + threadIdx.y};
        return tid;
        */
        int lid = threadIdx.x + threadIdx.y*blockDim.x;
        int tid = lid;
        //printf("WARPSIZE = %i   WSIZE=%i\n", WARPSIZE, WSIZE);
        if(lid < WSIZE){
            uint2 lm = {0,0};
            while(lid < rb){
                //printf("lid= %i\n", lid);
                int b = beta(nb, {blockIdx.x, blockIdx.y}, lid+1);
                lm.x += (b >> 1) * (1 << (lid));
                lm.y += (b - (b >> 1)) * (1 << (lid));
                lid += WSIZE;
            }
            lm = warp_reduce(lm,WSIZE); 
            /*
            for (int offset = WSIZE/2; offset > 0; offset /= 2){
                lm.x += __shfl_down(lm.x, offset, WSIZE);
                lm.y += __shfl_down(lm.y, offset, WSIZE);
            }
            */
            if(tid == 0){ m = lm; }
        }
        __syncthreads();
        return (uint2){m.x * blockDim.x + threadIdx.x, m.y * blockDim.y + threadIdx.y};

    };


    cudaEvent_t startbb, stopbb;
    cudaEvent_t startlam, stoplam;
    cudaEventCreate(&startbb);
    cudaEventCreate(&stopbb);
    cudaEventCreate(&startlam);
    cudaEventCreate(&stoplam);



    // BOUNDING-BOX MAP
    // call function by index
    psgen[0](rb, BSIZE, block, grid);
    printf("block: %i x %i x %i\n", block.x, block.y, block.z);
    printf("grid:  %i x %i x %i\n", grid.x, grid.y, grid.z);
    printf("\x1b[1mgpudummy(): bounding-box REPEATS=%i...INNER_REPEATS=%i.......\x1b[0m", REPEATS, INNER_REPEATS); fflush(stdout);
    for(int i=0; i<REPEATS; ++i){
        cudaEventRecord(startbb);
        for(int i=0; i<INNER_REPEATS; ++i){
            kernel_write_char<<< grid, block >>>(Ec_d, n, nb, rb, 1, WSIZE, bbmap);
            cudaDeviceSynchronize();
        }
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaEventRecord(stopbb);
        cudaEventSynchronize(stopbb);
        cudaEventElapsedTime(&t1, startbb, stopbb);
        r1.Push(t1/(1000.0f * INNER_REPEATS));
    }
    printf("\x1b[1mok\n\x1b[0m"); fflush(stdout);

    // print result of bounding-box
    //cudaMemcpy(Ei_h, Ei_d, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
    //cudaMemcpy(Ef_h, Ef_d, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ec_h, Ec_d, sizeof(char)*n*n, cudaMemcpyDeviceToHost);
    if(r <= PRINTLIMIT){
        printspace<char>(Ec_h, n);
    }

    // verify result for BB
    printf("gpudummy(): verifying result for BOUNDING-BOX..."); fflush(stdout);
    assert( verify<char>(Ec_h, n, 1) ); 
    printf("ok\n\n"); fflush(stdout);













    // LAMBDA MAP
    // call function by index
    psgen[1](rb, BSIZE, block, grid);
    printf("block: %i x %i x %i\n", block.x, block.y, block.z);
    printf("grid:  %i x %i x %i\n", grid.x, grid.y, grid.z);
    printf("\x1b[1mgpudummy(): lambda REPEATS=%i...INNER_REPEATS=%i.......\x1b[0m", REPEATS, INNER_REPEATS); fflush(stdout);
    for(int i=0; i<REPEATS; ++i){
        cudaEventRecord(startlam);
        for(int i=0; i<INNER_REPEATS; ++i){
            //kernel_single_write_int<<< grid, block >>>(Ei_d, n, nb, rb, 2, lambdamap);
            kernel_write_char_lambda<<< grid, block >>>(Ec_d, n, nb, rb, 3, WSIZE, lambdamap);
            cudaDeviceSynchronize();
        }
        gpuErrchk( cudaPeekAtLastError() );
        gpuErrchk( cudaDeviceSynchronize() );
        cudaEventRecord(stoplam);
        cudaEventSynchronize(stoplam);
        cudaEventElapsedTime(&t2, startlam, stoplam);
        r2.Push(t2/(1000.0f * INNER_REPEATS));
    }
    printf("\x1b[1mok\n\x1b[0m"); fflush(stdout);



    // verify result for Lambda
    //cudaMemcpy(Ei_h, Ei_d, sizeof(int)*n*n, cudaMemcpyDeviceToHost);
    //cudaMemcpy(Ef_h, Ef_d, sizeof(float)*n*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(Ec_h, Ec_d, sizeof(char)*n*n, cudaMemcpyDeviceToHost);
    if(r <= PRINTLIMIT){
        printspace<char>(Ec_h, n);
    }
    printf("gpudummy(): verifying result for LAMBDA..."); fflush(stdout);
    assert( verify<char>(Ec_h, n, 3) ); 
    printf("ok\n\n"); fflush(stdout);



    // clear 
    //free(Ei_h);
    //free(Ef_h);
    free(Ec_h);
    //cudaFree(Ei_d);
    //cudaFree(Ef_d);
    cudaFree(Ec_d);

    // return computing time for both methods
    stat.mean1 = r1.Mean();
    stat.variance1 = r1.Variance();
    stat.stdev1 = r1.StandardDeviation();
    stat.sterr1 = r1.StandardDeviation()/((double)sqrt(r1.NumDataValues()));
    stat.mean2 = r2.Mean();
    stat.variance2 = r2.Variance();
    stat.stdev2 = r2.StandardDeviation();
    stat.sterr2 = r2.StandardDeviation()/((double)sqrt(r2.NumDataValues()));
    return stat;
}
