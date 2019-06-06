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

unsigned int REPEATS; // Lazy programming
unsigned int WSIZE;  // Lazy programming

statistics gpudummy(unsigned int method, unsigned int repeats, double density){

    RunningStat* r;
    statistics stat;

    WSIZE               = min(BSIZE1D, WARPSIZE);
    unsigned long n     = (int)ceil(pow(2, RLEVEL));        // Linear size
    unsigned long rb    = max((int)(RLEVEL - BPOWER), 0);   // BlockTiled level
    unsigned long nb    = (int)(1 << rb);                   // N elements in BTiled level

    REPEATS = repeats;

#ifdef DEBUG
    printf("BPOWER=%i, BSIZE=%i, WSIZE=%i, scale r=%i, block-scale rb = %i   nb = %i\n", BPOWER, BSIZE2D, WSIZE, RLEVEL, rb, nb);

    printf("[Lambda (inverse)]\n");
#endif

    MTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
	unsigned long msize, trisize;

#ifdef DEBUG
    printf("ok\n"); fflush(stdout);

    // init data
    printf("Initializing Matrices......."); fflush(stdout);
#endif

	init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, density);	

#ifdef DEBUG
    printf("ok\n"); fflush(stdout);
#endif

    switch (method){
        case 0:
            r = boundingBox(ddata, dmat1, dmat2, n, nb, rb, msize, trisize);
            break;
        case 1:
            r = lambda(ddata, dmat1, dmat2, n, nb, rb, msize, trisize);
            break;
        case 2:
            r = lambda_tc(ddata, dmat1, dmat2, n, nb, rb, msize, trisize);
            break;
        case 3:
            r = lambda_tc_optimized(ddata, dmat1, dmat2, n, nb, rb, msize, trisize);
            break;
        default:
            printf("Method can only take values 0, 1, 2 or 3\n");
            exit(2);
    }
    //cudaMemcpy(Ec_h, Ec_d, sizeof(MTYPE)*n*n, cudaMemcpyDeviceToHost);
    //cudaMemcpy(res_h, res_d, sizeof(MTYPE), cudaMemcpyDeviceToHost);

#ifdef DEBUG
    printf("gpudummy(): verifying results..."); fflush(stdout);
#endif

    //assert(verifyCA(n, msize, hdata, ddata, hmat, dmat2));

#ifdef DEBUG
    printf("ok\n\n"); fflush(stdout);
#endif

    free(hdata);
    free(hmat);
    cudaFree(ddata);
    cudaFree(dmat1);
    cudaFree(dmat2);

    // Return computing time
    stat.mean = r->Mean();
    stat.variance = r->Variance();
    stat.stdev = r->StandardDeviation();
    stat.sterr = r->StandardDeviation()/((double)sqrt(r->NumDataValues()));

    free(r);
    return stat;
}


RunningStat* boundingBox(MTYPE *ddata, MTYPE *dmat1, MTYPE *dmat2, unsigned long n, unsigned long nb, unsigned long rb, unsigned long msize, unsigned long trisize){

    dim3 block, grid;

    auto psgen = [] (unsigned long rb, int BSIZE, dim3 &b, dim3 &g){ 
        b = dim3(BSIZE, BSIZE, 1);
        g = dim3(pow(2,rb), pow(2,rb), 1);
    };

    auto bbmap = [] __device__ (const int nb, const int rb, const int WSIZE, half *mata, half *matb, float *matc){
        uint2 m;
        m.x = blockIdx.x*blockDim.x + threadIdx.x;
        m.y = blockIdx.y*blockDim.y + threadIdx.y;
        return m;
    };
    
    psgen(rb, 1<<BPOWER, block, grid);

    return performLoad(ddata, dmat1, dmat2, n, nb, rb, msize, trisize, block, grid, bbmap, 0, 0, 0);
}

RunningStat* lambda(MTYPE *ddata, MTYPE *dmat1, MTYPE *dmat2, unsigned long n, unsigned long nb, unsigned long rb, unsigned long msize, unsigned long trisize){

    dim3 block, grid;

    // pspace: orthotope for lambda
    auto psgen =  [] (unsigned long rb, int BSIZE, dim3 &b, dim3 &g){ 
        b = dim3(BSIZE, BSIZE, 1);
        g = dim3((int)pow(3, ceil(rb/2.0)), (int)pow(3, floor(rb/2.0)), 1); 
    };

    // lambda map
    auto lambdamap = [] __device__ (const int nb, const int rb, const int WSIZE, half *mata, half *matb, float *matc){
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

    return performLoad(ddata, dmat1, dmat2, n, nb, rb, msize, trisize, block, grid, lambdamap, 0, 0, 0);
}

RunningStat* lambda_tc(MTYPE *ddata, MTYPE *dmat1, MTYPE *dmat2, unsigned long n, unsigned long nb, unsigned long rb, unsigned long msize, unsigned long trisize){

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
    auto lambdamap_tc = [] __device__ (const int nb, const int rb, const int WSIZE, half *mata, half *matb, float *matc){
        
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

    return performLoad(ddata, dmat1, dmat2, n, nb, rb, msize, trisize, block, grid, lambdamap_tc, 0, 0, 0 );
}

RunningStat* lambda_tc_optimized(MTYPE *ddata, MTYPE *dmat1, MTYPE *dmat2, unsigned long n, unsigned long nb, unsigned long rb, unsigned long msize, unsigned long trisize){

    dim3 block, grid;

    rb++;


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
    auto lambdamap_tc = [] __device__ (const int nb, const int rb, const int WSIZE, half *mata, half *matb, float *matc){
        
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
        
        if ( (blockIdx.x == gridDim.x-1 && x==1) || (blockIdx.y == gridDim.y-1 && y==1)){
            return (uint2){0xFFFFFFFF, 0xFFFFFFFF};
        }
        char ss = (x<<1) + (y<<2);
        m = (uint2){(int)(matc[ss]), (int)(matc[ss+1])};

        return (uint2){(m.x << 4) + (threadIdx.x & 15), (m.y << 4) + (threadIdx.y & 15)};
    };

    psgen(rb, 1<<BPOWER, block, grid);

    return performLoad(ddata, dmat1, dmat2, n, nb, rb, msize, trisize, block, grid, lambdamap_tc, 0, 0, 0);
}


template<typename Lambda>
RunningStat* performLoad( MTYPE *ddata, MTYPE *dmat1, MTYPE *dmat2, unsigned long n, unsigned long nb, unsigned long rb, unsigned long msize, unsigned int trisize, dim3 block, dim3 grid,
                            Lambda map, unsigned int aux1, unsigned int aux2, unsigned int aux3) {


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // runningstat statistics
    RunningStat *r = new RunningStat();
    float time = 0.0;

    // measure running time
    cudaEventRecord(start, 0);	
    for(int k=0; k<REPEATS; k++){
        kernel_update_ghosts<<< (n+BSIZE2D-1)/BSIZE2D, BSIZE2D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        kernel_test<<< grid, block >>>(n, msize, nb, rb, ddata, dmat1, dmat2, map, aux1, aux2, aux3);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("result ping\n");
            print_dmat(PRINTLIMIT, n, msize, dmat2, "PING dmat1 -> dmat2");
            getchar();
        #endif

        kernel_update_ghosts<<< (n+BSIZE2D-1)/BSIZE2D, BSIZE2D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        kernel_test<<< grid, block >>>(n, msize, nb, rb, ddata, dmat2, dmat1, map, aux1, aux2, aux3);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("result pong\n");
            print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG  dmat2 -> dmat1");
            getchar();
        #endif
    }
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
#endif
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // that's our time!
    last_cuda_error("benchmark-check");

    r->Push(time/(1000.0f * INNER_REPEATS));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
#ifdef DEBUG
    printf("\x1b[1mok\n\x1b[0m"); fflush(stdout);
#endif

    return r;
}
