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
#define INNER_REPEATS 1

unsigned int REPEATS;  
unsigned int WSIZE;  

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
    printf("BPOWER=%i, BSIZE=%i, WSIZE=%i, scale r=%i, block-scale rb = %i   nb = %i\n", BPOWER, BSIZE1D, WSIZE, RLEVEL, rb, nb);
#endif 

    // int and float version of Euclidean space;
    char *Ec_h, *Ec_d;

#ifdef DEBUG
    printf("Allocating Memory Spaces of %lu x %lu = %lu -> %f MB in CPU & GPU.......", n, n, n2, n2*sizeof(char)/(1024.0*1024.0)); fflush(stdout); 
#endif

    Ec_h = (char*)malloc(sizeof(char)*n*n);

    cudaMalloc(&Ec_d, sizeof(char)*n*n);

    last_cuda_error("cudaMalloc");
#ifdef DEBUG
    printf("ok\n"); fflush(stdout);

    // init data
    printf("Initializing Matrices......."); fflush(stdout);
#endif

    initmat<char>(Ec_h, n2, 0); 
#ifdef DEBUG
    printf("ok\n"); fflush(stdout);
#endif
    cudaMemcpy(Ec_d, Ec_h, sizeof(char)*n*n, cudaMemcpyHostToDevice);

    switch (method){
        case 1:
            r = boundingBox(Ec_d, n, nb, rb);
            break;
        case 2:
            r = lambda(Ec_d, n, nb, rb);
            break;
        case 3:
            r = lambda_tc(Ec_d, n, nb, rb);
            break;
        case 5:
            r = lambda_tc2(Ec_d, n, nb, rb);
            break;
        case 6:
            r = lambda_tc3(Ec_d, n, nb, rb);
            break;
        case 4:
            r = lambda_tc_optimized(Ec_d, n, nb, rb);
            break;
        case 7:
            r = lambda_tc_optimized2(Ec_d, n, nb, rb);
            break;
        default:
            printf("Method can only take values 1, 2, 3 or 4\n");
            exit(2);
    }
    cudaMemcpy(Ec_h, Ec_d, sizeof(char)*n*n, cudaMemcpyDeviceToHost);

#ifdef DEBUG
    if(RLEVEL <= PRINTLIMIT){
        printspace<char>(Ec_h, n);
    }
#endif

#ifdef DEBUG
    printf("gpudummy(): verifying results..."); fflush(stdout);
#endif

    assert( verify<char>(Ec_h, n, 1) ); // Verifying results

#ifdef DEBUG
    printf("ok\n\n"); fflush(stdout);
#endif

    free(Ec_h);
    cudaFree(Ec_d);

    // Return computing time
    stat.mean = r->Mean();
    stat.variance = r->Variance();
    stat.stdev = r->StandardDeviation();
    stat.sterr = r->StandardDeviation()/((double)sqrt(r->NumDataValues()));

    free(r);
    return stat;
}


RunningStat* boundingBox(char* M, unsigned long n, unsigned long nb, unsigned long rb){

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

    return performLoad(M, n, nb, rb, block, grid, bbmap, true);
}

RunningStat* lambda(char* M, unsigned long n, unsigned long nb, unsigned long rb){

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

    return performLoad(M, n, nb, rb, block, grid, lambdamap, false);
}

RunningStat* lambda_tc(char* M, unsigned long n, unsigned long nb, unsigned long rb){

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
            wmma::fill_fragment(c_fragment, 0.0f);
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

    return performLoad(M, n, nb, rb, block, grid, lambdamap_tc, false);
}

RunningStat* lambda_tc2(char* M, unsigned long n, unsigned long nb, unsigned long rb){

    dim3 block, grid;

    if (BSIZE2D != 16){
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
        __shared__ half matb[512];
        __shared__ float matc[512];
        
        //Has to be declared after the matrices above to avoid 8-byte shifting
        //__shared__ uint2 m;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_fragment;

        auto beta = [] __device__ (const int nb, const int u){
            int b = (int)((blockIdx.x*(u & 1) + blockIdx.y*((u+1) & 1))/(pow3((u>>1) + (u&1) - 1)))%3;
            return b;
        };
        
        int lid = threadIdx.x + threadIdx.y*blockDim.x;
        int lid2 = lid + 256;

        char col = lid & 15;
        char wid = (lid >> 5);


        if (col < rb){
            mata[lid] = 1 << (col+4);
            
            char b = beta(nb, col+1);
            matb[lid] = (b >> 1);
            matb[lid2] = (b-(b >> 1));
        } else {
            mata[lid] = 0;
        }

        matc[lid]  = threadIdx.x;
        matc[lid2] = threadIdx.y;
        __syncthreads();

        if (wid < 2) {
            int wid2 = wid << 8;
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
            wmma::load_matrix_sync(b_fragment, &matb[wid2], 16);
            wmma::load_matrix_sync(c_fragment, &matc[wid2], 16, wmma::mem_row_major);

            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);
            wmma::store_matrix_sync(&matc[wid2], c_fragment, 16, wmma::mem_row_major);
        } 
        __syncthreads();

        return (uint2){matc[lid], matc[lid2]};
    };

    psgen(rb, 1<<BPOWER, block, grid);

    return performLoad(M, n, nb, rb, block, grid, lambdamap_tc, false);
}

RunningStat* lambda_tc3(char* M, unsigned long n, unsigned long nb, unsigned long rb){

    dim3 block, grid;

    if (BSIZE2D != 16){
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
        __shared__ float matc[256];
        
        //Has to be declared after the matrices above to avoid 8-byte shifting
        uint2 m;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_fragment;

        auto beta = [] __device__ (const int nb, const int u){
            int b = (int)((blockIdx.x*(u & 1) + blockIdx.y*((u+1) & 1))/(pow3((u>>1) + (u&1) - 1)))%3;
            return b;
        };
        
        int index = threadIdx.x + threadIdx.y*blockDim.x;
        int col = index & 15;
        char b;

        if (col < rb){
            mata[index] = 1 << (col+4);
            
            b = beta(nb, col+1);
            matb[index] = (b >> 1);
        } else {
            mata[index] = 0;
        }
        
        matc[index] = threadIdx.x;

        __syncthreads();
        /*if (index == 0 && blockIdx.x == 1 && blockIdx.y == 1){
                printf("\n");
            for (int i=0; i<16; i++){
                for (int j=0; j<16; j++){
                    printf("%f ", matc[i*16 + j]);
                }
                printf("\n");
            }
                printf("\n");
        }
        __syncthreads();*/
        if (index < 32) {
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
            wmma::load_matrix_sync(b_fragment, &matb[0], 16);
            wmma::load_matrix_sync(c_fragment, &matc[0], 16, wmma::mem_row_major);

            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);
            wmma::store_matrix_sync(&matc[0], c_fragment, 16, wmma::mem_row_major);
        }
        __syncthreads();
        m.x = matc[index];

        if (col < rb){
            matb[index] = (b-(b >> 1));
        }


        matc[index] = threadIdx.y;
        __syncthreads();
        /*if (index == 0 && blockIdx.x == 1 && blockIdx.y == 1){
            for (int i=0; i<16; i++){
                for (int j=0; j<16; j++){
                    printf("%f ", matc[i*16 + j]);
                }
                printf("\n");
            }
        }
        __syncthreads();*/

        if (index < 32) {
            wmma::load_matrix_sync(b_fragment, &matb[0], 16);
            wmma::load_matrix_sync(c_fragment, &matc[0], 16, wmma::mem_row_major);

            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);
            wmma::store_matrix_sync(&matc[0], c_fragment, 16, wmma::mem_row_major);
        }
        __syncthreads();
        m.y = matc[index];

        return (uint2){m.x, m.y};
    };

    psgen(rb, 1<<BPOWER, block, grid);

    return performLoad(M, n, nb, rb, block, grid, lambdamap_tc, false);
}

RunningStat* lambda_tc_optimized(char* M, unsigned long n, unsigned long nb, unsigned long rb){

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
        } 
        else {
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
        char ss = (x<<1) + (y<<2);
        m = (uint2){(int)(matc[ss]), (int)(matc[ss+1])};

        return (uint2){(m.x << 4) + (threadIdx.x & 15), (m.y << 4) + (threadIdx.y & 15)};
    };

    psgen(rb, 1<<BPOWER, block, grid);

    return performLoad(M, n, nb, rb, block, grid, lambdamap_tc, false);
}

RunningStat* lambda_tc_optimized2(char* M, unsigned long n, unsigned long nb, unsigned long rb){

    dim3 block, grid;

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
        g = dim3( (int)ceil(pow(3, ceil(rb/2.0))), (int)ceil(pow(3, floor(rb/2.0))), 1); 
    };

    // Tensor core lambda map
    // This map assumes that the block size is >=32, which is the minimum to perform tensor core mma,
    auto lambdamap_tc = [] __device__ (const int nb, const int rb, const int WSIZE){

        __shared__ half mata[256]; 
        __shared__ half matb_x[256];
        __shared__ half matb_y[256];    //   256   256   256   256   
        __shared__ float matc_x[1024];    // |-----|-----|-----|-----|
        __shared__ float matc_y[1024];    // |-----|-----|-----|-----|
                                        //    0     1     2     3
        
        //Strange behaviour when fragments are shared, the entire iteration reads the same value
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, float> c_fragment;

        auto beta = [] __device__ (const int nb, const int u){
            int b = (int)((blockIdx.x*(u & 1) + blockIdx.y*((u+1) & 1))/(pow3((u>>1) + (u&1) - 1)))%3;
            return b;
        };
        
        int index = threadIdx.x + threadIdx.y*blockDim.x;
        int lid = index;
        
        char col = index & 15;

        __syncthreads();
        if (lid < 256) {
            if (col < rb){
                mata[lid] = 1 << (col+5);
            } else {
                mata[lid] = 0;
            }
        } else if (lid < 512){
            lid -= 256;
            char b = beta(nb, col+1);
            matb_x[lid] = (b >> 1);
            matb_y[lid] = (b-(b >> 1));
        } 

        __syncthreads();
        matc_x[index] = threadIdx.x;
        matc_y[index] = threadIdx.y;

        __syncthreads();

        if (index < 32) {
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
            wmma::load_matrix_sync(b_fragment, &matb_x[0], 16);
            wmma::load_matrix_sync(c_fragment, &matc_x[0], 16, wmma::mem_row_major);

            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

            wmma::store_matrix_sync(&matc_x[0], c_fragment, 16, wmma::mem_row_major);

        } else if (index < 64){
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
            wmma::load_matrix_sync(b_fragment, &matb_x[0], 16);
            wmma::load_matrix_sync(c_fragment, &matc_x[256], 16, wmma::mem_row_major);

            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

            wmma::store_matrix_sync(&matc_x[256], c_fragment, 16, wmma::mem_row_major);
        
        } else if (index < 96){
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
            wmma::load_matrix_sync(b_fragment, &matb_x[0], 16);
            wmma::load_matrix_sync(c_fragment, &matc_x[512], 16, wmma::mem_row_major);

            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

            wmma::store_matrix_sync(&matc_x[512], c_fragment, 16, wmma::mem_row_major);

        } else if (index < 128){
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
            wmma::load_matrix_sync(b_fragment, &matb_x[0], 16);
            wmma::load_matrix_sync(c_fragment, &matc_x[768], 16, wmma::mem_row_major);

            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

            wmma::store_matrix_sync(&matc_x[768], c_fragment, 16, wmma::mem_row_major);

        } else if (index < 160){
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
            wmma::load_matrix_sync(b_fragment, &matb_y[0], 16);
            wmma::load_matrix_sync(c_fragment, &matc_y[0], 16, wmma::mem_row_major);

            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

            wmma::store_matrix_sync(&matc_y[0], c_fragment, 16, wmma::mem_row_major);

        } else if (index < 192){
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
            wmma::load_matrix_sync(b_fragment, &matb_y[0], 16);
            wmma::load_matrix_sync(c_fragment, &matc_y[256], 16, wmma::mem_row_major);

            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);
            wmma::store_matrix_sync(&matc_y[256], c_fragment, 16, wmma::mem_row_major);

        } else if (index < 224){
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
            wmma::load_matrix_sync(b_fragment, &matb_y[0], 16);
            wmma::load_matrix_sync(c_fragment, &matc_y[512], 16, wmma::mem_row_major);

            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

            wmma::store_matrix_sync(&matc_y[512], c_fragment, 16, wmma::mem_row_major);

        } else if (index < 256){
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
            wmma::load_matrix_sync(b_fragment, &matb_y[0], 16);
            wmma::load_matrix_sync(c_fragment, &matc_y[768], 16, wmma::mem_row_major);

            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);

            wmma::store_matrix_sync(&matc_y[768], c_fragment, 16, wmma::mem_row_major);
        }
        __syncthreads();

        return (uint2){matc_x[index], matc_y[index]};
    };

    psgen(rb, 1<<BPOWER, block, grid);

    return performLoad(M, n, nb, rb, block, grid, lambdamap_tc, false);
}


template<typename Lambda>
RunningStat* performLoad(char *M, unsigned long n, unsigned long nb, unsigned long rb, dim3 block, dim3 grid, Lambda map, bool isBB){

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);


    // runningstat statistics
    RunningStat *r = new RunningStat();
    float t = 0.0f;

#ifdef DEBUG
    printf("block: %i x %i x %i\n", block.x, block.y, block.z);
    printf("grid:  %i x %i x %i\n", grid.x, grid.y, grid.z);
    printf("\x1b[1mgpudummy(): REPEATS=%i...INNER_REPEATS=%i.......\x1b[0m", REPEATS, INNER_REPEATS); fflush(stdout);
#endif

    if (!isBB)
        for(int i=0; i<REPEATS; ++i){
            cudaEventRecord(start);
            for(int i=0; i<INNER_REPEATS; ++i){
                kernel_write_char_lambda<<< grid, block >>>(M, n, nb, rb, 1, WSIZE, map); //TODO: fix
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
    else
        for(int i=0; i<REPEATS; ++i){
            cudaEventRecord(start);
            for(int i=0; i<INNER_REPEATS; ++i){
                kernel_write_char<<< grid, block >>>(M, n, nb, rb, 1, WSIZE, map); //TODO: fix
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
