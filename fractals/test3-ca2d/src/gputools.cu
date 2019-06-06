#include <stdio.h>
#include "gpudummy.cuh"
#include "gputools.cuh"

template <typename T>
void initmat(T *E, unsigned long n2, T c){
    for(unsigned long i=0; i<n2; ++i){
        E[i] = c;
    }
}

template <typename T>
void printspace(T *E, unsigned long n){
    for(unsigned long i=0; i<n; ++i){
        for(unsigned long j=0; j<n; ++j){
            T val = E[i*n + j];
            if(val > 0){
                printf("%i ", val);
            }
            else{
                printf("  ");
            }
            //printf("%i ", val);
        }
        printf("\n");
    }
}

template <typename T>
int verifyWrite(T *E, int n, int c){
    for(unsigned long i=0; i<n; ++i){
        for(unsigned long j=0; j<n; ++j){
            if(E[i*n + j] == c){
                if(((n-1-i)&j) != 0){
                    return 0;
                }
            }
            if(E[i*n + j] != c){
                if(((n-1-i)&j) == 0){
                    return 0;
                }
            }
        }
    }
    return 1;
}

template <typename T>
int verifyReduction(T *E, int n){
    if (E[0] == pow(3, RLEVEL)){
        return 1;
    }
    return 0;
}

void last_cuda_error(const char *msg){
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
		printf("[%s]: CUDA error: %s\n", msg, cudaGetErrorString(error));
		exit(-1);
	}
}

void print_gpu_specs(int dev){
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device Number: %d\n", dev);
    printf("  Device name:                  %s\n", prop.name);
    printf("  Multiprocessor Count:         %d\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels:           %d\n", prop.concurrentKernels);
    printf("  Memory Clock Rate (KHz):      %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits):      %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0*prop.memoryClockRate*(prop.memoryBusWidth/8)/1.0e6);
}

void print_ghost_matrix(MTYPE *mat, const int n, const char *msg){
    printf("[%s]:\n", msg);
	for(int i=0; i<n+2; i++){
	    for(int j=0; j<n+2; j++){
            long w = i*(n+2) + j;
            if( i >= j && i<n+1 && j>0 ){
                //if(mat[w] == 1){
                //    printf("1 ");
                //}
                //else{
                //    printf("  ");
                //}
                if(mat[w] != 0){
                    printf("%i ", mat[w]);
                }
                else{
                    printf("  ", mat[w]);
                }
            }
            else{
                printf("%i ", mat[w]);
            }
        }
        printf("\n");
    }
}

int print_dmat(int PLIMIT, unsigned int n, unsigned long msize, MTYPE *dmat, const char *msg){
    if(n <= PLIMIT){
        MTYPE *hmat = (MTYPE*)malloc(sizeof(MTYPE)*msize);
        cudaMemcpy(hmat, dmat, sizeof(MTYPE)*msize, cudaMemcpyDeviceToHost);
        print_ghost_matrix(hmat, n, msg);
        free(hmat);
    }
    return 1;
}

void init(unsigned long n, MTYPE **hdata, MTYPE **hmat, MTYPE **ddata, MTYPE **dmat1, MTYPE **dmat2, unsigned long *msize, unsigned long *trisize, double DENSITY){
    // define ghost n, gn, for the (n+2)*(n+2) space for the ghost cells
	*msize = (n+2)*(n+2);

	*hmat = (MTYPE*)malloc(sizeof(MTYPE)*(*msize));
    set_everything(*hmat, n, 0);
    set_randconfig(*hmat, n, DENSITY);

	cudaMalloc((void **) dmat1, sizeof(MTYPE)*(*msize));
	cudaMalloc((void **) dmat2, sizeof(MTYPE)*(*msize));
    last_cuda_error("init: cudaMalloc dmat1 dmat2");
    cudaMemcpy(*dmat1, *hmat, sizeof(MTYPE)*(*msize), cudaMemcpyHostToDevice);
    last_cuda_error("init:memcpy hmat->dmat1");

#ifdef DEBUG
	printf("2-simplex: n=%i  msize=%lu (%f MBytes -> 2 lattices)\n", n, *msize, 2.0f * (float)sizeof(MTYPE)*(*msize)/(1024*1024));
    if(n <= PRINTLIMIT){
        print_ghost_matrix(*hmat, n, "\nhost ghost-matrix initialized");
    }
#endif
}

void set_everything(MTYPE *mat, const unsigned long n, MTYPE val){
    // set cells and ghost cells
    for(unsigned int y=0; y<n+2; ++y){
        for(unsigned int x=0; x<n+2; ++x){
            unsigned long i = y*(n+2) + x;
            mat[i] = val;
        }
    }
}
void set_randconfig(MTYPE *mat, const unsigned long n, double DENSITY){
    // note: this function does not act on the ghost cells (boundaries)
    for(unsigned int y=0; y<n; ++y){
        for(unsigned int x=0; x<n; ++x){
            unsigned long i = (y+1)*(n+2) + (x+1);
            if (((x) & (n-1-y)) == 0)
                mat[i] = (((double)rand()/RAND_MAX) <= DENSITY ? 1 : 0);
            else
                mat[i] = 0;
        }
    }
}

