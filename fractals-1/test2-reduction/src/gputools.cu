#include <stdio.h>
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
int verifyReduction(double res){
    if (abs(res - pow(3, RLEVEL)) < 0.0000001 ){
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


