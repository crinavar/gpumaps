#ifndef GPUTOOLS
#define GPUTOOLS

void last_cuda_error(const char *msg){
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
		printf("[%s]: CUDA error: %s\n", msg, cudaGetErrorString(error));
		exit(-1);
	}
}


void fill_random(float *array, int n){
	for(int i=0; i<n; i++){
		array[i] = 100.0f * (float)rand()/(float)RAND_MAX;
    }
} 

void init(unsigned int n, float **hdata, float **hmat, float **ddata, float **dmat, unsigned int *msize, unsigned int *trisize){
	*msize = n*n;
    *trisize = n*(n-1)/2;
	
	*hdata = (float*)malloc(sizeof(float)*n);
	*hmat = (float*)malloc(sizeof(float)*(*msize));
    for(int i=0; i<*msize; i++){
        (*hmat)[i] = 0.0f;
    }
	fill_random(*hdata, n);
	cudaMalloc((void **) ddata, sizeof(float)*n);
	cudaMalloc((void **) dmat, sizeof(float)*(*msize));
    last_cuda_error("mark1");
    cudaMemcpy(*dmat, *hmat, sizeof(float)*(*msize), cudaMemcpyHostToDevice);
	cudaMemcpy(*ddata, *hdata, sizeof(float)*n, cudaMemcpyHostToDevice);
	printf("2-simplex: n=%i  msize=%i (%f MBytes)\n", n, *msize, (float)sizeof(float)*(*msize)/(1024*1024));
    last_cuda_error("init end");
}

void gen_lambda_pspace(unsigned int n, dim3 &block, dim3 &grid){
    block = dim3(BSIZE2D, BSIZE2D, 1); 
	int sn = (n+block.x-1)/block.x;
	int sd = sn*(sn+1)/2;
	int s = ceil(sqrt((double)sd));	
   	grid = dim3(s, s, 1);
}

void gen_bbox_pspace(unsigned int n, dim3 &block, dim3 &grid){
    block = dim3(BSIZE2D, BSIZE2D, 1); 
	grid = dim3( (n + block.x -1)/block.x, (n + block.y - 1)/block.y, 1);	
}

template<typename Lambda>
double benchmark_map(const int REPEATS, dim3 block, dim3 grid, unsigned int n, unsigned int msize, unsigned int trisize, float *ddata, float *dmat, Lambda map){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("block= %i x %i x %i    grid = %i x %i x %i\n", n, block.x, block.y, block.z, grid.x, grid.y, grid.z);
    // Warmup
    printf("warmup........................."); fflush(stdout);
	for(int i=0; i<REPEATS; i++){
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat, map);	
        cudaThreadSynchronize();
    }
    last_cuda_error("warmup");
    printf("done\n"); fflush(stdout);
    printf("Benchmarking (%i REPEATS).......", REPEATS); fflush(stdout);
    float time = 0.0;
    // measure running time
    cudaEventRecord(start, 0);	
    for(int k=0; k<REPEATS; k++){
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat, map);	
        cudaThreadSynchronize();
    }
    printf("done\n"); fflush(stdout);
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // that's our time!
    time = time/(float)REPEATS;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
    return time;
}


int verify_result(unsigned int n, unsigned int msize, float *hdata, float *ddata, float *hmat, float *dmat){
    return 1;
}
#endif
