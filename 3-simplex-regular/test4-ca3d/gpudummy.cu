#include <cuda.h>
#include <vector>
#include <stdio.h>
#include <time.h>
#include <sys/time.h>
#include <cassert>
#include "interface.h"
#include "gpudummy.cuh"
#include "tools.cuh"
#include "kernels.cuh"

// density
#define PRINTLIMIT 64

double gpudummy(int D, unsigned long N, int REPEATS, int maptype, double DENSITY){
	// set device id
	#ifdef DEBUG
    printf("gpudummy(): choosing device %i...", D); fflush(stdout);
	#endif
	gpuErrchk(cudaSetDevice(D));
	#ifdef DEBUG
    printf("ok\n");	fflush(stdout);
	#endif

    unsigned long Vcube = N*N*N;
    unsigned long Vsimplex = N*(N+1)*(N+2)/6;
    unsigned long sn = ceil(std::cbrt(Vsimplex));
	#ifdef DEBUG
    printf("gpudummy(): mapping to simplex of n=%lu   Vcube = %lu   Vsimplex = %lu\n", N, Vcube, Vsimplex);
    printf("gpudummy(): Cube size is %f MB\n", (float)Vcube*sizeof(MTYPE)/(1024.0*1024.0f));
	#endif

    // allocating unified memory CPU/GPU
    DTYPE *h_data, *d_data;
    MTYPE *h_outcube, *d_outcube1, *d_outcube2; 
    h_data = (DTYPE*)malloc(sizeof(DTYPE)*N);
    h_outcube = (MTYPE*)malloc(sizeof(MTYPE)*Vcube);
    
    cudaMalloc(&d_data, sizeof(DTYPE)*N);
    gpuErrchk(cudaPeekAtLastError());

    cudaMalloc(&d_outcube1, sizeof(MTYPE)*Vcube);
    cudaMalloc(&d_outcube2, sizeof(MTYPE)*Vcube);
    gpuErrchk(cudaPeekAtLastError());
    

    // parallel spaces 1D, 2D and 3D
    dim3 block1d( BSIZE1D, 1, 1);
    dim3 grid1d( (Vcube + block1d.x - 1)/block1d.x, 1, 1);
    dim3 block2d( BSIZE2DX, BSIZE2DY, 1);
    dim3 grid2d( (N+block2d.x - 1)/block2d.x, (N+block2d.y - 1)/block2d.y, 1);
    dim3 block3d, grid3d;



    // pspace: orthotope of bounding box 
    std::vector<void(*)(unsigned long n, dim3 &b, dim3 &g, int B)> psgen;  // parallel space generators
    psgen.push_back( [] (unsigned long n, dim3 &b, dim3 &g, int B){ 
        b = dim3(B, B, B);
        //b = dim3(32, 8, 4);
        g = dim3((n+b.x - 1)/b.x, (n+b.y - 1)/b.y, (n+b.z - 1)/b.z);
    });



    // pspace: orthotope of discrete orthogonal tetrahedron
    psgen.push_back( [] (unsigned long n, dim3 &b, dim3 &g, int bsize){ 
        b = dim3(bsize, bsize, bsize);
        unsigned long lx = (n + b.x - 1)/b.x;
        unsigned long ly = (n + b.y - 1)/b.y;
        unsigned long lz = (n + b.z - 1)/b.z;
        //printf("lx,y,z = %lu %lu %lu\n", lx, ly, lz);
        unsigned long sv = ceil((float)lx * ((float)ly + 1.0f) * ((float)lz + 2.0f) / 6.0f);
        //printf("sv = %lu\n", sv);
        unsigned long sn = ceil(std::cbrt(sv));
        //printf("sn = %lu\n", sn);
        g = dim3(sn,sn,sn);
    });







    // use the corresponding pspace generator
    psgen[maptype](N, block3d, grid3d, BSIZE3D);
	#ifdef DEBUG
    printf("gpudummy(): parallel space: b(%i, %i, %i) g(%i, %i, %i)\n", block3d.x, block3d.y, block3d.z, grid3d.x, grid3d.y, grid3d.z);
    printf("gpudummy(): set outdata init state..."); fflush(stdout);
	#endif



    // set cells initial states
    set_randconfig(h_outcube, N, DENSITY);

    /*
    set_alldead(h_outcube, N);
    set_cell(h_outcube, N, 5, 11, 14, 1);
    set_cell(h_outcube, N, 6, 11, 14, 1);
    set_cell(h_outcube, N, 5, 12, 14, 1);
    set_cell(h_outcube, N, 6, 12, 14, 1);

    set_cell(h_outcube, N, 5, 11, 15, 1);
    set_cell(h_outcube, N, 6, 11, 15, 1);
    set_cell(h_outcube, N, 5, 10, 15, 1);
    set_cell(h_outcube, N, 6, 10, 15, 1);
    //set_cell(h_outcube, N, 1, 2, 14, 1);
    //set_cell(h_outcube, N, 0, 2, 15, 1);
    //set_cell(h_outcube, N, 1, 2, 15, 1);
    */

    // set cube1 (in GPU) initial state to the one generated on CPU
    cudaMemcpy(d_outcube1,h_outcube, sizeof(MTYPE)*Vcube, cudaMemcpyHostToDevice);
    // set cube2 initial state to zero
    k_setoutdata<<<grid1d, block1d>>>(d_outcube2, Vcube, 0, [] __device__ (){ return (unsigned long) ((unsigned long)blockIdx.x*(unsigned long)blockDim.x + (unsigned long)threadIdx.x); } );
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
    //printf("ok\n");
	#ifdef DEBUG
    printf("ok\n");
	#endif






	#ifdef DEBUG
    // synchronize GPU/CPU mem
    printf("gpudummy(): synchronizing CPU/GPU mem..."); fflush(stdout);
	#endif
    cudaMemcpy(d_data,h_data, sizeof(DTYPE)*N, cudaMemcpyHostToDevice);
    gpuErrchk(cudaDeviceSynchronize());
    gpuErrchk(cudaPeekAtLastError());
	#ifdef DEBUG
    printf("ok\n");
	#endif






    // verify outdata initial state
    /*
    #ifdef VERIFY
        #ifdef DEBUG
        printf("gpudummy(): verifying matrix state..."); fflush(stdout);
        #endif
        assert( verify(h_outcube, N, [] (float d, float x, float y, float z){ if(d == 0.0f){ return true; } else{return false;}} ));
        #ifdef DEBUG
        printf("ok\n"); fflush(stdout);
        #endif
    #endif
    */







    // bounding box map
    auto f0 = [] __device__ (){
            return (uint3){blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y, blockIdx.z*blockDim.z + threadIdx.z};
    };


    // lambda map
    auto f1 = [] __device__ (){
        __shared__ uint3 m;
        if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
            unsigned int w = blockIdx.x + blockIdx.y*gridDim.x + blockIdx.z * gridDim.x * gridDim.z;
            //printf("w = %i\n", w);
            //if(w == 0){
            //    return (uint3){threadIdx.x, threadIdx.y, threadIdx.z};
            //}
            REAL wp = 27.0f*w;
            //REAL cv = powf(sqrtf(wp*wp - 3.0f) + wp, 1.0/3.0);
            REAL cv = powf(sqrtf(wp*wp - 3.0f) + wp, 0.33333333333f);
            //float cv = cbrt(sqrtf(wp*wp - 3.0f) + wp);
            //if(w < 100){
            //    printf("w=%i   cv = %f\n", w, cv);
            //}
            //printf("w=%i   cv = %f\n", w, cv);
            REAL fz = cv*CL + CR/cv - 1.0f;
            //printf("w=%i z = %f\n", w, fz);

            m.z = (unsigned int)fz;
            unsigned int w2d = w - m.z*(m.z+1)*(m.z+2)/6;

            /*
            m.y = (unsigned int) (sqrtf(0.25f + (w2d << 1) ) - 0.5f);
            m.x = w2d - m.y*(m.y+1)/2;
            */

            REAL arg = __fmaf_rn(2.0f, (REAL)w2d, 0.25f);
            m.y = __fmaf_rn(arg, rsqrtf(arg), OFFSET);// + 0.001f;
            m.x = w2d - (m.y*(m.y+1) >> 1);
            //printf("device: w = %i, b(%u, %u, %u)\n", w, m.x, m.y, m.z);
        }
        __syncthreads();
        return (uint3){m.x * blockDim.x + threadIdx.x, m.y * blockDim.y + threadIdx.y, m.z * blockDim.z + threadIdx.z};
    };

    // begin performance tests
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
	#ifdef DEBUG
    printf("\x1b[1mgpudummy(): kernel (map=%i, r=%i)...\x1b[0m", maptype, REPEATS); fflush(stdout);
	#endif
    if(maptype == 0){
        cudaEventRecord(start);
        for(int i=0; i<REPEATS; ++i){
            // mat1 --> mat2
			kernel0<<<grid3d, block3d>>>(d_data, d_outcube1, d_outcube2, N, Vcube, f0);
			cudaDeviceSynchronize();

            // mat2 --> mat1
			kernel0<<<grid3d, block3d>>>(d_data, d_outcube2, d_outcube1, N, Vcube, f0);
			cudaDeviceSynchronize();
		}
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
    }
    else{
        cudaEventRecord(start);
		gpuErrchk( cudaPeekAtLastError() );
        for(int i=0; i<REPEATS; ++i){
            // mat1 --> mat2
			kernel1<<<grid3d, block3d>>>(d_data, d_outcube1, d_outcube2, N, Vcube, f1);
			cudaDeviceSynchronize();

            // mat2 --> mat1
			kernel1<<<grid3d, block3d>>>(d_data, d_outcube2, d_outcube1, N, Vcube, f1);
			cudaDeviceSynchronize();
		}
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);
		gpuErrchk( cudaPeekAtLastError() );
		gpuErrchk( cudaDeviceSynchronize() );
    }
	#ifdef DEBUG
    printf("\x1b[1mok\n\x1b[0m"); fflush(stdout);
	#endif



	#ifdef DEBUG
    // synchronize GPU/CPU mem
    printf("gpudummy(): synchronizing CPU/GPU mem..."); fflush(stdout);
	#endif
    cudaMemcpy(h_data,d_data, sizeof(DTYPE)*N, cudaMemcpyDeviceToHost);
    gpuErrchk( cudaPeekAtLastError() );
    cudaMemcpy(h_outcube,d_outcube2, sizeof(MTYPE)*Vcube, cudaMemcpyDeviceToHost);
    gpuErrchk( cudaPeekAtLastError() );
    gpuErrchk(cudaDeviceSynchronize());
	#ifdef DEBUG
    printf("ok\n");
	#endif




    // verify result
    /*
	#ifdef VERIFY
        #ifdef DEBUG
            printf("gpudummy(): verifying result..."); fflush(stdout);
        #endif
        assert( verify(h_outcube, N, [REPEATS] (float d, float x, float y, float z){ return d != 0.0f;})    );
        #ifdef DEBUG
        printf("ok\n\n"); fflush(stdout);
        #endif
    #endif
    */




    // clear 
    free(h_data);
    free(h_outcube);
    cudaFree(d_data);
    cudaFree(d_outcube1);
    cudaFree(d_outcube2);




    // return computing time
    float msecs = 0;
    cudaEventElapsedTime(&msecs, start, stop);
    return msecs/((float)REPEATS);
}
