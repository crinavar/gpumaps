#include <stdio.h>
#include <iostream>
#include "gpudummy.cuh"
#include "gputools.cuh"

template <typename T>
void initmat(T *E, size_t n2, T c){
    for(size_t i=0; i<n2; ++i){
        E[i] = c;
    }
}

template <typename T>
void printspace(T *E, size_t n){
    for(size_t i=0; i<n; ++i){
        for(size_t j=0; j<n; ++j){
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
    for(size_t i=0; i<n; ++i){
        for(size_t j=0; j<n; ++j){
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
/*
void print_ghost_matrix(MTYPE *mat, const int n, size_t cm, size_t cn, const char *msg){
    printf("[%s]:\n", msg);
    for(unsigned int i=0; i<cn*BSIZE2D; ++i){
		for(unsigned int j=0; j<cm*BSIZE2D; ++j){
			size_t w = (i)*(BSIZE2D*cm) + j;
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
        printf("\n");
	}
}*/
int print_dmat_gpu(int PLIMIT, int rlevel, size_t nx, size_t ny, MTYPE *mat, MTYPE *dmat, const char *msg){
    
    if(rlevel <= PLIMIT){
        cudaMemcpy(mat, dmat, sizeof(MTYPE)*nx*ny, cudaMemcpyDeviceToHost);
        for (int i=0; i<ny; i++){
            for (int j=0; j<nx; j++){
                if (mat[i*nx+j] == 0){
                    printf("  ");
                } else {
                    printf("%i ", mat[i*nx+j]);
                }
            }
            printf("\n");
        }
    }
    return 1;
}
uint2 inv(const int xx, const int yy, const int nb, const int rb, const int WSIZE){
    uint2 m = {0,0};
    
    int lid = 0;//threadIdx.x + threadIdx.y*blockDim.x;

    while(lid < rb){
        int h = (int) (xx%(1 << (lid+1)))/(1 << lid) + (yy%(1 << (lid+1))) /(1 << lid);
        //int b = beta(nb, {blockIdx.x, blockIdx.y}, lid+1);
        int bx = ((lid+1)%2);
        int by = ((lid)%2);
        m.x += pow(3, lid >> 1) * bx * h;
        m.y += pow(3, lid >> 1) * by * h;
        lid += 1;//WSIZE;
    }
        
    return m;
}



uint2 lamb(const size_t x, const size_t y, const int nb, const int rb, const int WSIZE){
    uint2 m = {0,0};
    
    int lid = 0;
    while(lid < rb){
        int b =  (int)((x*((lid+1) & 1) + y*(((lid+1)+1) & 1))/(   pow(3, ((lid+1)>>1) + ((lid+1)&1) - 1)   ))%3;
        m.x += (b >> 1) * (1 << (lid));
        m.y += (b - (b >> 1)) * (1 << (lid));
        lid += 1;
    }
    return m;
};
int print_dmat_gpu_comp(int PLIMIT, int rlevel, size_t n, size_t rb, size_t blockSize, size_t nx, size_t ny, MTYPE *mat, MTYPE *dmat, const char *msg){
    
    if(rlevel <= PLIMIT){
        cudaMemcpy(mat, dmat, sizeof(MTYPE)*nx*ny, cudaMemcpyDeviceToHost);
        for (int i=0; i<n; i++){
            for (int j=0; j<n; j++){
                if (((j) & (n-1-i)) == 0) {
                    int blockidx = j/blockSize;
                    int blockidy = i/blockSize;
                    uint2 coord = inv(blockidx, blockidy, 1, rb, 1);

                    int xx = coord.x*blockSize+j%blockSize;
                    int yy = coord.y*blockSize+i%blockSize;
                    //printf("1  ");
                    if (mat[(yy+1)*nx+xx+1] == 0){
                        printf("  ");
                    } else {
                        printf("%i ", mat[(yy+1)*nx+xx+1]);
                    }
                } else {
                    printf("  ");
                }
                
            }
            printf("\n");
        }
    }
    return 1;
}

int print_dmat(int PLIMIT, int rlevel, size_t nx, size_t ny, MTYPE *mat, const char *msg){
    if(rlevel <= PLIMIT){
        for (int i=0; i<ny; i++){
            for (int j=0; j<nx; j++){
                if (mat[i*nx+j] == 0){
                    printf("  ");
                } else {
                    printf("%i ", mat[i*nx+j]);
                }
            }
            printf("\n");
        }
    }
    return 1;
}

void init(size_t n, size_t rb, MTYPE **hdata, MTYPE **hmat, MTYPE **ddata, MTYPE **dmat1, MTYPE **dmat2, size_t *msize, size_t *trisize, size_t &cm, size_t &cn, double DENSITY){
    // define ghost n, gn, for the (n+2)*(n+2) space for the ghost cells
    cm = pow(3, ceil(rb/2.0));
	cn = pow(3, floor(rb/2.0)); 

	//printf("COMPAC SPACE: (cnxcm) %ix%i\n", cn, cm);
	*msize = cm*cn*BSIZE2D*BSIZE2D;
	*hmat = (MTYPE*)malloc(sizeof(MTYPE)*(*msize));
    set_everything(*hmat, *msize, 0);
    set_randconfig(*hmat, *msize, cm, cn, DENSITY);

	cudaMalloc((void **) dmat1, sizeof(MTYPE)*(*msize));
	cudaMalloc((void **) dmat2, sizeof(MTYPE)*(*msize));
    last_cuda_error("init: cudaMalloc dmat1 dmat2");
    cudaMemcpy(*dmat1, *hmat, sizeof(MTYPE)*(*msize), cudaMemcpyHostToDevice);
    last_cuda_error("init:memcpy hmat->dmat1");

#ifdef DEBUG
	printf("2-simplex: n=%i  msize=%lu (%f MBytes -> 2 lattices)\n", n, *msize, 2.0f * (float)sizeof(MTYPE)*(*msize)/(1024*1024));
    //print_ghost_matrix(*hmat, *msize, cm, cn, "host ghost-matrix initialized");
#endif
}

void set_everything(MTYPE *mat, const size_t n, MTYPE val){
    // set cells and ghost cells
    for(unsigned int i=0; i<n; ++i){
       	mat[i] = val;
    }
}
void set_randconfig(MTYPE *mat, const size_t n, size_t cm, size_t cn, double DENSITY){
    // note: this function does not act on the ghost cells (boundaries)

    for(unsigned int by=0; by<cn; ++by){
    for(unsigned int bx=0; bx<cm; ++bx){
    for(unsigned int y=0; y<BSIZE2D; ++y){
        for(unsigned int x=0; x<BSIZE2D; ++x){
            size_t i = (by)*(BSIZE2D*BSIZE2D*cm) + bx*BSIZE2D + y*cm*BSIZE2D + x;
            if (((x) & (n-1-y)) == 0)
                mat[i] = (((double)rand()/RAND_MAX) <= DENSITY ? 1 : 0);
            else
                mat[i] = 0;
        }
    }
	}}
}

