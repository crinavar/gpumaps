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

#define INNER_REP 10
using namespace nvcuda;
#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true)
{
	   if (code != cudaSuccess) 
		      {
			            fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
				          if (abort) exit(code);
					     }
}
 int REPEATS; // Lazy programming
unsigned int WSIZE;  // Lazy programming
unsigned int nfract;
uint2 inv(const size_t xx, const size_t yy, const int nb, const int rb, const int WSIZE);
uint2 lamb(const size_t x, const size_t y, const int nb, const int rb, const int WSIZE);


RunningStat *r;

statistics gpudummy(unsigned int method, unsigned int repeats, double density){

    r = new RunningStat();
    statistics stat;

    srand(time(NULL));
    WSIZE        = min(BSIZE1D, 32);
    size_t n     = (int)ceil(pow(2, RLEVEL));        // Linear size
    size_t rb    = max((int)(RLEVEL - BPOWER), 0);   // BlockTiled level
    size_t nb    = (int)(1 << rb);                   // N elements in BTiled level

    REPEATS = repeats;
    nfract = n;
    //printf("N=%i\n", n);
#ifdef DEBUG
    printf("BPOWER=%i, BSIZE=%i, WSIZE=%i, scale r=%i, block-scale rb = %i   nb = %i\n", BPOWER, BSIZE2D, WSIZE, RLEVEL, rb, nb);

    printf("[Lambda (inverse)]\n");
#endif

    switch (method){
        case 1:
			for (int i=0; i<repeats; i++){
            	boundingBox(n, nb, rb, density);
			}
            break;
        case 2:
			for (int i=0; i<repeats; i++){
            	lambda(n, nb, rb, density);
			}
            break;
        case 3:
			for (int i=0; i<repeats; i++){
            	compressed(n, nb, rb, density);
			}
            break;
        case 4:
			for (int i=0; i<repeats; i++){
            	compressed_tc(n, nb, rb, density);
			}
            break;
        default:
            throw "Method not implemented.";
        
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

    if (r->NumDataValues() == 0){
        stat.mean = 0;
        stat.variance = 0;
        stat.stdev = 0;
        stat.sterr = 0;

    } else {
        // Return computing time
        stat.mean = r->Mean();
        stat.variance = r->Variance();
        stat.stdev = r->StandardDeviation();
        stat.sterr = r->StandardDeviation()/((double)sqrt(r->NumDataValues()));

        free(r);
    }
    return stat;
}


void boundingBox(size_t n, size_t nb, size_t rb, double density){
    MTYPE *mat_h, *mat1_d, *mat2_d;
    // AQUI DEBERIA SER N*N
    /*
	size_t cm, cn;

    size_t cm = (int)pow(3, ceil(rb/2.0))*BSIZE2D;
	size_t cn = (int)pow(3, floor(rb/2.0))*BSIZE2D; 
    size_t celements = cm*cn;
    */
    size_t cm = (int)pow(3, ceil(RLEVEL/2.0));
	size_t cn = (int)pow(3, floor(RLEVEL/2.0)); 

    n = (int)ceil(pow(2, RLEVEL));
    size_t nExtended = n+2;
    //printf("N is %i\n", n);
    size_t celements = nExtended*nExtended;

    gpuErrchk(cudaMalloc(&mat1_d, sizeof(MTYPE)*celements));
    gpuErrchk(cudaMalloc(&mat2_d, sizeof(MTYPE)*celements));

    mat_h = (int*)malloc(celements*sizeof(MTYPE));


    for(int i=0; i<nExtended; i++){
        for(int j=0; j<nExtended; j++){
            mat_h[i*(nExtended)+j] = 0x0;
        }
    }

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if ((double)rand() / (double)RAND_MAX < density && ((j) & (n-1-i)) == 0){
                mat_h[(i+1)*nExtended+(j+1)] = 0x1;
            }
        }
    }

    cudaMemcpy(mat1_d, mat_h, sizeof(MTYPE)*nExtended*nExtended, cudaMemcpyHostToDevice);

    #ifdef DEBUG
        printf("Initial state\n");
        print_dmat(PRINTLIMIT, RLEVEL, n+2, n+2, mat_h, "");
        getchar();
    #endif

    dim3 block, grid;
    block = dim3(BSIZE2D, BSIZE2D, 1);
    grid = dim3(pow(2,rb), pow(2,rb), 1);

    auto bbmap = [] __device__ (const int nb, const int rb, const int WSIZE, half* mata, half* matb){
        uint2 m;
        m.x = blockIdx.x*blockDim.x + threadIdx.x;
        m.y = blockIdx.y*blockDim.y + threadIdx.y;
        return m;
    };

    auto inv = [] __device__ (const int nb, const int rb, const int WSIZE, half* mata, half* matb){
        uint2 m;
        m.x = blockIdx.x*blockDim.x + threadIdx.x;
        m.y = blockIdx.y*blockDim.y + threadIdx.y;
        return m;
    };
    
    performLoad(mat_h, mat1_d, mat2_d, nb, rb, nExtended, nExtended, block, grid, bbmap, inv);
    gpuErrchk(cudaFree(mat1_d));
    gpuErrchk(cudaFree(mat2_d));
	free(mat_h);
}

void compressed(size_t n, size_t nb, size_t rb, double density){
    MTYPE *mat_h, *mat1_d, *mat2_d;

    dim3 block, grid;
    block = dim3(BSIZE2D, BSIZE2D, 1);
    grid = dim3((int)pow(3, ceil(rb/2.0)), (int)pow(3, floor(rb/2.0)), 1); 
    
    //printf("GRID (%i, %i, %i)\nBLOCK(%i, %i, %i)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

    size_t nx;
    size_t ny;
    if (rb == 0){
        nx = (size_t)ceil(pow(2, RLEVEL));
        ny = (size_t)ceil(pow(2, RLEVEL));
    } else {
        nx = pow(3, ceil(rb/2.0))*BSIZE2D;
        ny = pow(3, floor(rb/2.0))*BSIZE2D;
    }

    size_t nxExtended = nx+2;
    size_t nyExtended = ny+2;

    //printf("Nx is %i\nNy is %i\n", nx, ny);
    size_t celements = nxExtended*nyExtended;

    gpuErrchk(cudaMalloc(&mat1_d, sizeof(MTYPE)*celements));
    gpuErrchk(cudaMalloc(&mat2_d, sizeof(MTYPE)*celements));
    mat_h = (int*)malloc(celements*sizeof(MTYPE));

    for(int i=0; i<nyExtended; i++){
        for(int j=0; j<nxExtended; j++){
            mat_h[i*(nxExtended)+j] = 0x0;
        }
    }

    for(int i=0; i<ny; i++){
        for(int j=0; j<nx; j++){
            size_t coord = (i+1)*(nxExtended)+j+1;
            int x = j%BSIZE2D;
            int y = i%BSIZE2D;
            if ((double)rand() / (double)RAND_MAX < density && ((x) & (n-1-y)) == 0){
                mat_h[coord] = 0x1;
            } else {
                mat_h[coord ] = 0x0;
            }
        }
    }
    
    cudaMemcpy(mat1_d, mat_h, sizeof(MTYPE)*nxExtended*nyExtended, cudaMemcpyHostToDevice);
    #ifdef DEBUG
        printf("Initial state\n");
        print_dmat(PRINTLIMIT, RLEVEL, nx+2, ny+2, mat_h, "");
        getchar();
    #endif
    // compressed map
    auto lambdamap = [] __device__ (const int nb, const int rb, const int WSIZE, half* mata, half* matb){
        __shared__ int2 m;
        auto beta = [] __device__ (const int nb, const int u){
            int b = (int)((blockIdx.x*(u & 1) + blockIdx.y*((u+1) & 1))/(pow3((u>>1) + (u&1) - 1)))%3;
            return b;
        };
        int lid = threadIdx.x + threadIdx.y*blockDim.x;
        int tid = lid;
        //printf("%i- %i\n", lid, WSIZE);
        if(lid < WSIZE){
            int2 lm = {0,0};
            while(lid < rb){
                int b = beta(nb, lid+1);
                lm.x += (b >> 1) * (1 << (lid));
                lm.y += (b - (b >> 1)) * (1 << (lid));
                lid += WSIZE;
            }
            lm = warp_reduce(lm,WSIZE); 
            if(tid == 0){ m = lm; }
        }
        __syncthreads();
        //return (int2){m.x * blockDim.x + threadIdx.x, m.y * blockDim.y + threadIdx.y};
        return (int2){m.x, m.y};
    };

    auto inv = [] __device__ (const int x, const int y, const int nb, const int rb, const int WSIZE, int lid, half* mata, half* matb){
        int2 m = {0,0};
        auto H = [] __device__ (const int xx, const int yy, const int u){
            int res = (int) ((xx & ((1 << (u+1)) - 1))>>(u)) + ((yy & ((1 << (u+1)) - 1)) >> (u));
            return res;
        };
        //printf("lid: %i - wsize: %i - rb: %i\n", lid, WSIZE, rb);
        if(lid < WSIZE){
            while(lid < rb){
                int h = H(x, y, lid);
                int bx = ((lid+1) & 1);
                int by = ((lid) & 1);
                float p = pow3(lid >> 1);
                m.x += p * bx * h;
                m.y += p * by * h;
                lid += WSIZE;
            }
            m = warp_reduce(m, WSIZE); 
        }
        return m;
    };

    performLoadCompressed(mat_h, mat1_d, mat2_d, nb, rb, nxExtended, nyExtended, block, grid, lambdamap, inv);
    gpuErrchk(cudaFree(mat1_d));
    gpuErrchk(cudaFree(mat2_d));
	free(mat_h);
}

void compressed_tc(size_t n, size_t nb, size_t rb, double density){
    if (BSIZE2D < 16){

        //printf("Only implemented for 32x32 blockSize.\n");
        return;
    }
    MTYPE *mat_h, *mat1_d, *mat2_d;

    dim3 block, grid;
    block = dim3(BSIZE2D, BSIZE2D, 1);
    grid = dim3((int)pow(3, ceil(rb/2.0)), (int)pow(3, floor(rb/2.0)), 1); 
    
    //printf("GRID (%i, %i, %i)\nBLOCK(%i, %i, %i)\n", grid.x, grid.y, grid.z, block.x, block.y, block.z);

    size_t nx;
    size_t ny;
    if (rb == 0){
        nx = (size_t)ceil(pow(2, RLEVEL));
        ny = (size_t)ceil(pow(2, RLEVEL));
    } else {
        nx = pow(3, ceil(rb/2.0))*BSIZE2D;
        ny = pow(3, floor(rb/2.0))*BSIZE2D;
    }

    size_t nxExtended = nx+2;
    size_t nyExtended = ny+2;

    //printf("Nx is %i\nNy is %i\n", nx, ny);
    size_t celements = nxExtended*nyExtended;

    gpuErrchk(cudaMalloc(&mat1_d, sizeof(MTYPE)*celements));
    gpuErrchk(cudaMalloc(&mat2_d, sizeof(MTYPE)*celements));
    mat_h = (int*)malloc(celements*sizeof(MTYPE));

    for(int i=0; i<nyExtended; i++){
        for(int j=0; j<nxExtended; j++){
            mat_h[i*(nxExtended)+j] = 0x0;
        }
    }

    for(int i=0; i<ny; i++){
        for(int j=0; j<nx; j++){
            size_t coord = (i+1)*(nxExtended)+j+1;
            int x = j%BSIZE2D;
            int y = i%BSIZE2D;
            if ((double)rand() / (double)RAND_MAX < density && ((x) & (n-1-y)) == 0){
                mat_h[coord] = 0x1;
            } else {
                mat_h[coord ] = 0x0;
            }
        }
    }
    
    cudaMemcpy(mat1_d, mat_h, sizeof(MTYPE)*nxExtended*nyExtended, cudaMemcpyHostToDevice);
    #ifdef DEBUG
        printf("Initial state\n");
        print_dmat(PRINTLIMIT, RLEVEL, nx+2, ny+2, mat_h, "");
        getchar();
    #endif
    auto lambdamap_tc = [] __device__ (const int nb, const int rb, const int WSIZE, half* mata, half* matb){
        
        __shared__ uint2 m;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_fragment;

        auto beta = [] __device__ (const int nb, const int u){
            int b = (int)((blockIdx.x*(u & 1) + blockIdx.y*((u+1) & 1))/(pow3((u>>1) + (u&1) - 1)))%3;
            return b;
        };
        
        int lid = threadIdx.x + threadIdx.y*blockDim.x;
        int index = lid;

        if (lid < 32) {
            //Has to be resetted to 0. Latter kernel calls were getting weird values
            matb[lid] = 0;
            if (lid < rb){
                mata[lid] = 1 << lid;
            }
        } else {
            lid -= 32;
            if (lid < rb<<1){
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

        return (uint2){m.x, m.y};
    };

    auto inv_tc = [] __device__ (const int x, const int y, const int nb, const int rb, const int WSIZE, int lid, half* mata, half* matb){
        
        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_fragment;

        auto H = [] __device__ (const int xx, const int yy, const int u){
            int res = (int) ((xx & ((1 << (u+1)) - 1))>>(u)) + ((yy & ((1 << (u+1)) - 1)) >> (u));
            return res;
        };

        if (lid <256){
            char col = lid & 15;
            char row = lid >> 4;

            if (lid < rb){
                mata[lid] = pow3(col >> 1);
            } else {
                mata[lid] = 0;
            }

            if (col < rb){
                char column = (col + ((row+1) & 1)) & 1;
                char h = H(x, y, col);
                matb[lid] = h*column;
            }
        }
        __syncthreads();
        if(lid <32){
            wmma::fill_fragment(c_fragment, 0.f);
            wmma::load_matrix_sync(a_fragment, &mata[0], 16);
        
            wmma::load_matrix_sync(b_fragment, &matb[0], 16);
            wmma::mma_sync(c_fragment, a_fragment, b_fragment, c_fragment);
            wmma::store_matrix_sync(mata, c_fragment, 16, wmma::mem_row_major);
        }

    };
    // compressed map
    performLoadCompressed_tc(mat_h, mat1_d, mat2_d, nb, rb, nxExtended, nyExtended, block, grid, lambdamap_tc, inv_tc);
    gpuErrchk(cudaFree(mat1_d));
    gpuErrchk(cudaFree(mat2_d));
	free(mat_h);
}

void lambda(size_t n, size_t nb, size_t rb, double density){

    MTYPE *mat_h, *mat1_d, *mat2_d;
    size_t cm = (int)pow(3, ceil(RLEVEL/2.0));
	size_t cn = (int)pow(3, floor(RLEVEL/2.0)); 

    n = (int)ceil(pow(2, RLEVEL));
    size_t nExtended = n+2;
    //printf("N is %i\n", n);
    size_t celements = nExtended*nExtended;

    gpuErrchk(cudaMalloc(&mat1_d, sizeof(MTYPE)*celements));
    gpuErrchk(cudaMalloc(&mat2_d, sizeof(MTYPE)*celements));
    mat_h = (int*)malloc(celements*sizeof(MTYPE));

    for(int i=0; i<nExtended; i++){
        for(int j=0; j<nExtended; j++){
            mat_h[i*(nExtended)+j] = 0x0;
        }
    }

    for(int i=0; i<n; i++){
        for(int j=0; j<n; j++){
            if ((double)rand() / (double)RAND_MAX < density && ((j) & (n-1-i)) == 0){
                mat_h[(i+1)*nExtended+(j+1)] = 0x1;
            }
        }
    }
    cudaMemcpy(mat1_d, mat_h, sizeof(MTYPE)*nExtended*nExtended, cudaMemcpyHostToDevice);

    #ifdef DEBUG
        printf("Initial state\n");
        print_dmat(PRINTLIMIT, RLEVEL, n+2, n+2, mat_h, "");
        getchar();
    #endif

    dim3 block, grid;

    // pspace: orthotope for lambda
    auto psgen =  [] (unsigned long rb, int BSIZE, dim3 &b, dim3 &g){ 
        b = dim3(BSIZE, BSIZE, 1);
        g = dim3((int)pow(3, ceil(rb/2.0)), (int)pow(3, floor(rb/2.0)), 1); 
    };

    auto lambdamap_tc = [] __device__ (const int nb, const int rb, const int WSIZE, half* mata, half* matb){
        
        __shared__ uint2 m;

        wmma::fragment<wmma::matrix_a, 16, 16, 16, half, wmma::row_major> a_fragment;
        wmma::fragment<wmma::matrix_b, 16, 16, 16, half, wmma::col_major> b_fragment;
        wmma::fragment<wmma::accumulator, 16, 16, 16, half> c_fragment;

        auto beta = [] __device__ (const int nb, const int u){
            int b = (int)((blockIdx.x*(u & 1) + blockIdx.y*((u+1) & 1))/(pow3((u>>1) + (u&1) - 1)))%3;
            return b;
        };
        
        int lid = threadIdx.x + threadIdx.y*blockDim.x;
        int index = lid;

        if (lid < 32) {
            //Has to be resetted to 0. Latter kernel calls were getting weird values
            matb[lid] = 0;
            if (lid < rb){
                mata[lid] = 1 << lid;
            }
        } else {
            lid -= 32;
            if (lid < rb<<1){
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

    // lambda map
    auto lambdamap = [] __device__ (const int nb, const int rb, const int WSIZE, half* mata, half* matb){
        __shared__ int2 m;
        auto beta = [] __device__ (const int nb, const int u){
            int b = (int)((blockIdx.x*(u & 1) + blockIdx.y*((u+1) & 1))/(pow3((u>>1) + (u&1) - 1)))%3;
            return b;
        };
        int lid = threadIdx.x + threadIdx.y*blockDim.x;
        int tid = lid;
        if(lid < WSIZE){
            int2 lm = {0,0};
            while(lid < rb){
                int b = beta(nb, lid+1);
                lm.x += (b >> 1) * (1 << (lid));
                lm.y += (b - (b >> 1)) * (1 << (lid));
                lid += WSIZE;
            }
            lm = warp_reduce(lm,WSIZE); 
            if(tid == 0){ m = lm; }
        }
        __syncthreads();
        return (int2){m.x * blockDim.x + threadIdx.x, m.y * blockDim.y + threadIdx.y};
    };
    auto inv = [] __device__ (const int nb, const int rb, const int WSIZE){
        int2 m;
        m.x = blockIdx.x*blockDim.x + threadIdx.x;
        m.y = blockIdx.y*blockDim.y + threadIdx.y;
        return m;
    };
    psgen(rb, 1<<BPOWER, block, grid);

    if (BPOWER>=4){

        performLoadLambdaTC(mat_h, mat1_d, mat2_d, nb, rb, nExtended, nExtended, block, grid, lambdamap_tc, inv);
    } else {
        performLoad(mat_h, mat1_d, mat2_d, nb, rb, nExtended, nExtended, block, grid, lambdamap, inv);

    }
    gpuErrchk(cudaFree(mat1_d));
    gpuErrchk(cudaFree(mat2_d));
	free(mat_h);
}

template<typename Lambda, typename Inverse>
void performLoad(MTYPE *mat_h, MTYPE *mat1_d, MTYPE *mat2_d, size_t nb, size_t rb, size_t nx, size_t ny, dim3 block, dim3 grid,
                            Lambda map, Inverse inv) {


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // runningstat statistics
    float time = 0.0;

    // measure running time
    cudaEventRecord(start, 0);	
    for(int k=0; k<INNER_REP; k++){
        kernelBoundingBox<<< grid, block >>>(nx-2, ny-2, nb, rb, mat1_d, mat2_d, map, inv, WSIZE);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("result ping\n");
            print_dmat_gpu(PRINTLIMIT, RLEVEL, nx, ny, mat_h, mat2_d, "");
            getchar();
        #endif

        kernelBoundingBox<<< grid, block >>>(nx-2, ny-2, nb, rb, mat2_d, mat1_d, map, inv, WSIZE);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("result pong\n");
            print_dmat_gpu(PRINTLIMIT, RLEVEL, nx, ny, mat_h, mat1_d, "");
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

    r->Push(time/(2.f* INNER_REP));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
#ifdef DEBUG
    printf("\x1b[1mok\n\x1b[0m"); fflush(stdout);
#endif

}

template<typename Lambda, typename Inverse>
void performLoadLambdaTC(MTYPE *mat_h, MTYPE *mat1_d, MTYPE *mat2_d, size_t nb, size_t rb, size_t nx, size_t ny, dim3 block, dim3 grid,
                            Lambda map, Inverse inv) {


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // runningstat statistics
    float time = 0.0;

    // measure running time
    cudaEventRecord(start, 0);	
    for(int k=0; k<INNER_REP; k++){
        kernelLambdaTC<<< grid, block >>>(nx-2, ny-2, nb, rb, mat1_d, mat2_d, map, inv, WSIZE);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("result ping\n");
            print_dmat_gpu(PRINTLIMIT, RLEVEL, nx, ny, mat_h, mat2_d, "");
            getchar();
        #endif

        kernelLambdaTC<<< grid, block >>>(nx-2, ny-2, nb, rb, mat2_d, mat1_d, map, inv, WSIZE);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("result pong\n");
            print_dmat_gpu(PRINTLIMIT, RLEVEL, nx, ny, mat_h, mat1_d, "");
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

    r->Push(time/(2.f* INNER_REP));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
#ifdef DEBUG
    printf("\x1b[1mok\n\x1b[0m"); fflush(stdout);
#endif

}
template<typename Lambda, typename Inverse>
void performLoadCompressed(MTYPE *mat_h, MTYPE *mat1_d, MTYPE *mat2_d, size_t nb, size_t rb, size_t nx, size_t ny, dim3 block, dim3 grid,
                            Lambda map, Inverse inv) {


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // runningstat statistics
    float time = 0.0;
    #ifdef DEBUG

    printf("INITIAL DECOMP\n");
    print_dmat_gpu_comp(PRINTLIMIT, RLEVEL, nfract, rb, block.x, nx, ny, mat_h, mat1_d, "initial decomp");
    #endif

    // measure running time
    cudaEventRecord(start, 0);	
    for(int k=0; k<INNER_REP; k++){
        kernelCompressed<<< grid, block >>>(nfract, nx-2, ny-2, nb, rb, mat1_d, mat2_d, map, inv, WSIZE);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("result ping\n");
            print_dmat_gpu(PRINTLIMIT, RLEVEL, nx, ny, mat_h, mat2_d, "");
            print_dmat_gpu_comp(PRINTLIMIT, RLEVEL, nfract, rb, block.x, nx, ny, mat_h, mat2_d, "");
            getchar();
        #endif

        kernelCompressed<<< grid, block >>>(nfract, nx-2, ny-2, nb, rb, mat2_d, mat1_d, map, inv, WSIZE);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("result pong\n");
            print_dmat_gpu(PRINTLIMIT, RLEVEL, nx, ny, mat_h, mat1_d, "");
            print_dmat_gpu_comp(PRINTLIMIT, RLEVEL, nfract, rb, block.x, nx, ny, mat_h, mat1_d, "");
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

    r->Push(time/(2.f * INNER_REP));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
#ifdef DEBUG
    printf("\x1b[1mok\n\x1b[0m"); fflush(stdout);
#endif

}

template<typename Lambda, typename Inverse>
void performLoadCompressed_tc(MTYPE *mat_h, MTYPE *mat1_d, MTYPE *mat2_d, size_t nb, size_t rb, size_t nx, size_t ny, dim3 block, dim3 grid,
                            Lambda map, Inverse inv) {


	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);

    // runningstat statistics
    float time = 0.0;
    #ifdef DEBUG

    printf("INITIAL DECOMP\n");
    print_dmat_gpu_comp(PRINTLIMIT, RLEVEL, nfract, rb, block.x, nx, ny, mat_h, mat1_d, "initial decomp");
    #endif

    // measure running time
    cudaEventRecord(start, 0);	
    for(int k=0; k<INNER_REP; k++){
        kernelCompressed_tc<<< grid, block >>>(nfract, nx-2, ny-2, nb, rb, mat1_d, mat2_d, map, inv, WSIZE);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("result ping\n");
            print_dmat_gpu(PRINTLIMIT, RLEVEL, nx, ny, mat_h, mat2_d, "");
            print_dmat_gpu_comp(PRINTLIMIT, RLEVEL, nfract, rb, block.x, nx, ny, mat_h, mat2_d, "");
            getchar();
        #endif

        kernelCompressed_tc<<< grid, block >>>(nfract, nx-2, ny-2, nb, rb, mat2_d, mat1_d, map, inv, WSIZE);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("result pong\n");
            print_dmat_gpu(PRINTLIMIT, RLEVEL, nx, ny, mat_h, mat1_d, "");
            print_dmat_gpu_comp(PRINTLIMIT, RLEVEL, nfract, rb, block.x, nx, ny, mat_h, mat1_d, "");
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

    r->Push(time/(2.f * INNER_REP));

	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
#ifdef DEBUG
    printf("\x1b[1mok\n\x1b[0m"); fflush(stdout);
#endif
}

