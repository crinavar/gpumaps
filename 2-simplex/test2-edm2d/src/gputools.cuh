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
#ifndef GPUTOOLS
#define GPUTOOLS

#define MAXSTREAMS 32

int print_dmat(unsigned int limit, unsigned int n, unsigned int msize, MTYPE *dmat){
    MTYPE *hmat = (MTYPE*)malloc(sizeof(MTYPE)*msize);
    cudaMemcpy(hmat, dmat, sizeof(MTYPE)*msize, cudaMemcpyDeviceToHost);
#ifdef DEBUG
    if(n <= limit){
        print_matrix(hmat, n, "host matrix");
    }
#endif
    free(hmat);
    return 1;
}

void last_cuda_error(const char *msg){
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
		printf("[%s]: CUDA error: %s\n", msg, cudaGetErrorString(error));
		exit(-1);
	}
}


void fill_random(DTYPE *array, int n){
	for(int i=0; i<n; i++){
		array[i] = (DTYPE){100.0f * (float)rand()/(float)RAND_MAX, 100.0f * (float)rand()/(float)RAND_MAX};
    }
} 

void init(unsigned int n, DTYPE **hdata, MTYPE **hmat, DTYPE **ddata, MTYPE **dmat, unsigned int *msize, unsigned int *trisize){
	*msize = n*n;
    *trisize = n*(n-1)/2;
	
	*hdata = (DTYPE*)malloc(sizeof(DTYPE)*n);
	*hmat = (MTYPE*)malloc(sizeof(MTYPE)*(*msize));
    for(int i=0; i<*msize; i++){
        (*hmat)[i] = 0;
    }
	fill_random(*hdata, n);
	cudaMalloc((void **) ddata, sizeof(DTYPE)*n);
    last_cuda_error("init: cudaMalloc ddata");
	cudaMalloc((void **) dmat, sizeof(MTYPE)*(*msize));
    last_cuda_error("init: cudaMalloc dmat");
    cudaMemcpy(*dmat, *hmat, sizeof(MTYPE)*(*msize), cudaMemcpyHostToDevice);
    last_cuda_error("init end:memcpy hmat->dmat");
	cudaMemcpy(*ddata, *hdata, sizeof(DTYPE)*n, cudaMemcpyHostToDevice);
    last_cuda_error("init end:memcpy hdata->ddata");
#ifdef DEBUG
	printf("2-simplex: n=%i  msize=%i (%f MBytes)\n", n, *msize, (float)sizeof(MTYPE)*(*msize)/(1024*1024));
    if(n <= 64){
        print_matrix(*hmat, n, "host matrix");
    }
#endif
}

void gen_lambda_pspace(const unsigned int n, dim3 &block, dim3 &grid){
    block = dim3(BSIZE2D, BSIZE2D, 1); 
	int sn = (n+block.x-1)/block.x;
	int sd = sn*(sn+1)/2;
	int s = ceil(sqrt((double)sd));	
   	grid = dim3(s, s, 1);
}

void gen_bbox_pspace(const unsigned int n, dim3 &block, dim3 &grid){
    block = dim3(BSIZE2D, BSIZE2D, 1); 
	grid = dim3( (n + block.x -1)/block.x, (n + block.y - 1)/block.y, 1);	
}

void gen_avril_pspace(const unsigned int n, dim3 &block, dim3 &grid){
    const int Na = n*(n-1)/2;
	block = dim3(BSIZE1D, 1, 1);
	unsigned long int sn2 = (Na+block.x-1)/block.x;
    grid = dim3(sn2, 1, 1);
}

void gen_rectangle_pspace(const unsigned int n, dim3 &block, dim3 &grid){
    int rect_evenx = n/2;
    int rect_oddx = (int)ceil((float)n/2.0f);
    block = dim3(BSIZE2D, BSIZE2D, 1);
    if(n%2==0){
        grid = dim3((rect_evenx+block.x-1)/block.x, ((n+1)+block.y-1)/block.y, 1);
    }
    else{
        grid = dim3((rect_oddx+block.x-1)/block.x, (n+block.y-1)/block.y, 1);
    }
}

void gen_recursive_pspace(const unsigned int n, dim3 &block, dim3 &grid){
    block = dim3(BSIZE2D, BSIZE2D, 1);
    // it is defined later, at each recursive level
    grid = dim3(1,1,1);
}

template<typename Lambda>
double benchmark_map(const int REPEATS, dim3 block, dim3 grid, unsigned int n, unsigned int msize, unsigned int trisize, DTYPE *ddata, MTYPE *dmat, Lambda map){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
#ifdef DEBUG
	printf("block= %i x %i x %i    grid = %i x %i x %i\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
#endif
    // Warmup
#ifdef DEBUG
    printf("warmup.........................."); fflush(stdout);
#endif
	for(int i=0; i<REPEATS; i++){
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat, map, 0, 0);	
        cudaThreadSynchronize();
    }
    last_cuda_error("warmup");
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
    printf("Benchmarking (%i REPEATS).......", REPEATS); fflush(stdout);
#endif
    float time = 0.0;
    // measure running time
    cudaEventRecord(start, 0);	
    for(int k=0; k<REPEATS; k++){
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat, map, 0, 0);	
        cudaThreadSynchronize();
    }
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
#endif
    last_cuda_error("benchmark-check");
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // that's our time!
    time = time/(float)REPEATS;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
    return time;
}


template<typename Lambda1, typename Lambda2>
double benchmark_map_recursive(const int REPEATS, dim3 block, dim3 grid, const unsigned int n, const unsigned int msize, const unsigned int trisize, DTYPE *ddata, MTYPE *dmat, Lambda1 maprec, Lambda2 mapdiag, const unsigned int m, const unsigned int k){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
    // Warmup
#ifdef DEBUG
    printf("warmup.........................."); fflush(stdout);
#endif
    last_cuda_error("begin warmup");

    unsigned int streamindex = 0;
    cudaStream_t streams[MAXSTREAMS];
    for(int i=0; i<MAXSTREAMS; ++i){
        cudaStreamCreate( &streams[i] );
    }
    int bx=0, by=0;
    int bm = m/block.x;
	for(int r=0; r<REPEATS; r++){
        by = 0;
        dim3 dg_diag(n/block.x, bm, 1);
        // recursive_diagonal <<< dg_diag, dimblock >>> (a_d, N, b_d, N2);
	    // printf("\t[diagonal, s=%i] block= %i x %i x %i    grid = %i x %i x %i\n", streamindex%MAXSTREAMS, block.x, block.y, block.z, dg_diag.x, dg_diag.y, dg_diag.z);
        kernel_test <<< dg_diag, block, 0, streams[(streamindex++) % MAXSTREAMS] >>> (n, msize, ddata, dmat, mapdiag, bx, by);
        last_cuda_error("diagonal");
        by += bm;

        // print_dmat(128, n, msize, dmat);
        // getchar();

        for(int i=0; i<k; i++){
            grid = dim3(n/(block.x*2), bm*pow(2, i), 1);
            //recursive_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2, bx, by);	
            //printf("\t[recursive k=%i, s=%i] block= %i x %i x %i    grid = %i x %i x %i\n", k, streamindex%MAXSTREAMS, block.x, block.y, block.z, grid.x, grid.y, grid.z);
            kernel_test <<< grid, block, 0, streams[(streamindex++) % MAXSTREAMS] >>> (n, msize, ddata, dmat, maprec, bx, by);	
            last_cuda_error("recursive");
            by += bm*pow(2,i);

            //print_dmat(128, n, msize, dmat);
            //getchar();

        }
        cudaThreadSynchronize();
    }
    last_cuda_error("warmup");
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
    printf("Benchmarking (%i REPEATS).......", REPEATS); fflush(stdout);
#endif
    float time = 0.0;
    // measure running time
    cudaEventRecord(start, 0);	
	for(int r=0; r<REPEATS; r++){
        by = 0;
        dim3 dg_diag(n/block.x, bm, 1);
        //recursive_diagonal <<< dg_diag, dimblock >>> (a_d, N, b_d, N2);
        kernel_test <<< dg_diag, block, 0, streams[(streamindex++) % MAXSTREAMS] >>> (n, msize, ddata, dmat, mapdiag, bx, by);
        by += bm;
        for(int i=0; i<k; i++){
            dim3 grid(n/(block.x*2), bm*pow(2, i), 1);
            //recursive_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2, bx, by);	
            kernel_test <<< grid, block, 0, streams[(streamindex++) % MAXSTREAMS] >>> (n, msize, ddata, dmat, maprec, bx, by);	
            by += bm*pow(2,i);
        }
        cudaThreadSynchronize();
    }
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
#endif
    last_cuda_error("benchmark-check");
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // that's our time!
    time = time/(float)REPEATS;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
    return time;
}
int verify_result(unsigned int n, unsigned int msize, DTYPE *hdata, DTYPE *ddata, MTYPE *hmat, MTYPE *dmat){
    return 1;
#ifdef DEBUG
    printf("verifying result................"); fflush(stdout);
#endif
    cudaMemcpy(hdata, ddata, sizeof(DTYPE)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hmat, dmat, sizeof(MTYPE)*msize, cudaMemcpyDeviceToHost);
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            unsigned int index = i*n + j;
            DTYPE a = hdata[i];
            DTYPE b = hdata[j];
            float res = sqrtf( (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) );
            if(i > j){
                if( fabs(hmat[index]-res) >= EPSILON ){
                    #ifdef DEBUG
                    fprintf(stderr, "[Verify] invalid element at hmat[%i,%i](%i) = %f  (res = %f)\n", i, j, index, hmat[index], res);
                    #endif
                    return 0;
                }
            }
            else if(i < j){
                if( hmat[index] != 0.0f ){
                    #ifdef DEBUG
                    fprintf(stderr, "[Verify] invalid element at hmat[%i,%i](%i) = %f\n", i, j, index, hmat[index], res);
                    #endif
                    return 0;
                }
            }
            else if(i == j){
                if(hmat[index] != 0.0f && fabs(hmat[index]-res) >= EPSILON){
                    #ifdef DEBUG
                    fprintf(stderr, "[Verify] invalid element at hmat[%i,%i](%i) = %f\n", i, j, index, hmat[index], res);
                    #endif
                    return 0;
                }
            }
        }
    }
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
    if(n <= 64){
        print_matrix(hmat, n, "host matrix");
    }
#endif
    return 1;
}

#endif
