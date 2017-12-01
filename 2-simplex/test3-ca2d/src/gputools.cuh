#ifndef GPUTOOLS
#define GPUTOOLS

#define MAXSTREAMS 32
#define MAXPRINT 128

int print_dmat(unsigned int n, unsigned int msize, MTYPE *dmat){
    MTYPE *hmat = (MTYPE*)malloc(sizeof(MTYPE)*msize);
    cudaMemcpy(hmat, dmat, sizeof(MTYPE)*msize, cudaMemcpyDeviceToHost);
#ifdef DEBUG
    print_matrix(hmat, n, "host matrix");
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
        array[i] = ((double)rand()/RAND_MAX) <= 0.5 ? 1 : 0;
    }
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
double benchmark_map(const int REPEATS, dim3 block, dim3 grid, unsigned int n, unsigned int msize, unsigned int trisize, DTYPE *ddata, MTYPE *dmat1, MTYPE
        *dmat2, Lambda map){
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
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat1, dmat2, map, 0, 0);	
        cudaThreadSynchronize();
        #ifdef DEBUG
        printf("result ping\n");
        print_dmat(n, msize, dmat2);
        getchar();
        #endif
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat2, dmat1, map, 0, 0);	
        cudaThreadSynchronize();
        #ifdef DEBUG
        printf("result pong\n");
        print_dmat(n, msize, dmat1);
        getchar();
        #endif
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
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat1, dmat2, map, 0, 0);	
        cudaThreadSynchronize();
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat2, dmat1, map, 0, 0);	
        cudaThreadSynchronize();
    }
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
#endif
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // that's our time!
    last_cuda_error("benchmark-check");
    time = time/(float)REPEATS;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
    return time;
}



template<typename Lambda>
double benchmark_map_avril(const int REPEATS, dim3 block, dim3 grid, unsigned int n, unsigned int msize, unsigned int trisize, DTYPE *ddata, MTYPE *dmat1, MTYPE
        *dmat2, Lambda map){
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
        kernel_test_avril<<< grid, block >>>(n, msize, ddata, dmat1, dmat2, map, 0, 0);	
        cudaThreadSynchronize();
        #ifdef DEBUG
        printf("result ping\n");
        print_dmat(n, msize, dmat2);
        getchar();
        #endif
        kernel_test_avril<<< grid, block >>>(n, msize, ddata, dmat2, dmat1, map, 0, 0);	
        cudaThreadSynchronize();
        #ifdef DEBUG
        printf("result pong\n");
        print_dmat(n, msize, dmat1);
        getchar();
        #endif
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
        kernel_test_avril<<< grid, block >>>(n, msize, ddata, dmat1, dmat2, map, 0, 0);	
        cudaThreadSynchronize();
        kernel_test_avril<<< grid, block >>>(n, msize, ddata, dmat2, dmat1, map, 0, 0);	
        cudaThreadSynchronize();
    }
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
#endif
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // that's our time!
    last_cuda_error("benchmark-check");
    time = time/(float)REPEATS;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
    return time;
}



template<typename Lambda>
double benchmark_map_rectangle(const int REPEATS, dim3 block, dim3 grid, unsigned int n, unsigned int msize, unsigned int trisize, DTYPE *ddata, MTYPE *dmat1, MTYPE
        *dmat2, Lambda map){
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
        kernel_test_rectangle<<< grid, block >>>(n, msize, ddata, dmat1, dmat2, map, 0, 0);	
        cudaThreadSynchronize();
        #ifdef DEBUG
        printf("result ping\n");
        print_dmat(n, msize, dmat2);
        getchar();
        #endif
        kernel_test_rectangle<<< grid, block >>>(n, msize, ddata, dmat2, dmat1, map, 0, 0);	
        cudaThreadSynchronize();
        #ifdef DEBUG
        printf("result pong\n");
        print_dmat(n, msize, dmat1);
        getchar();
        #endif
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
        kernel_test_rectangle<<< grid, block >>>(n, msize, ddata, dmat1, dmat2, map, 0, 0);	
        cudaThreadSynchronize();
        kernel_test_rectangle<<< grid, block >>>(n, msize, ddata, dmat2, dmat1, map, 0, 0);	
        cudaThreadSynchronize();
    }
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
#endif
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // that's our time!
    last_cuda_error("benchmark-check");
    time = time/(float)REPEATS;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
    return time;
}


template<typename Lambda1, typename Lambda2>
double benchmark_map_recursive(const int REPEATS, dim3 block, dim3 grid, const unsigned int n, const unsigned int msize, const unsigned int trisize, DTYPE
        *ddata, MTYPE *dmat1, MTYPE *dmat2, Lambda1 maprec, Lambda2 mapdiag, const unsigned int m, const unsigned int k){
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
        kernel_test <<< dg_diag, block, 0, streams[(streamindex++) % MAXSTREAMS] >>> (n, msize, ddata, dmat1, dmat2, mapdiag, bx, by);
        last_cuda_error("diagonal");
        by += bm;

        // print_dmat(128, n, msize, dmat);
        // getchar();

        for(int i=0; i<k; i++){
            grid = dim3(n/(block.x*2), bm*pow(2, i), 1);
            //recursive_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2, bx, by);	
            //printf("\t[recursive k=%i, s=%i] block= %i x %i x %i    grid = %i x %i x %i\n", k, streamindex%MAXSTREAMS, block.x, block.y, block.z, grid.x, grid.y, grid.z);
            kernel_test <<< grid, block, 0, streams[(streamindex++) % MAXSTREAMS] >>> (n, msize, ddata, dmat1, dmat2, maprec, bx, by);	
            last_cuda_error("recursive");
            by += bm*pow(2,i);

            //print_dmat(128, n, msize, dmat);
            //getchar();

        }
        cudaThreadSynchronize();
        by = 0;
        // recursive_diagonal <<< dg_diag, dimblock >>> (a_d, N, b_d, N2);
	    // printf("\t[diagonal, s=%i] block= %i x %i x %i    grid = %i x %i x %i\n", streamindex%MAXSTREAMS, block.x, block.y, block.z, dg_diag.x, dg_diag.y, dg_diag.z);
        kernel_test <<< dg_diag, block, 0, streams[(streamindex++) % MAXSTREAMS] >>> (n, msize, ddata, dmat2, dmat1, mapdiag, bx, by);
        last_cuda_error("diagonal");
        by += bm;

        // print_dmat(128, n, msize, dmat);
        // getchar();

        for(int i=0; i<k; i++){
            grid = dim3(n/(block.x*2), bm*pow(2, i), 1);
            //recursive_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2, bx, by);	
            //printf("\t[recursive k=%i, s=%i] block= %i x %i x %i    grid = %i x %i x %i\n", k, streamindex%MAXSTREAMS, block.x, block.y, block.z, grid.x, grid.y, grid.z);
            kernel_test <<< grid, block, 0, streams[(streamindex++) % MAXSTREAMS] >>> (n, msize, ddata, dmat2, dmat1, maprec, bx, by);	
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
        kernel_test <<< dg_diag, block, 0, streams[(streamindex++) % MAXSTREAMS] >>> (n, msize, ddata, dmat1, dmat2, mapdiag, bx, by);
        by += bm;
        for(int i=0; i<k; i++){
            dim3 grid(n/(block.x*2), bm*pow(2, i), 1);
            //recursive_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2, bx, by);	
            kernel_test <<< grid, block, 0, streams[(streamindex++) % MAXSTREAMS] >>> (n, msize, ddata, dmat1, dmat2, maprec, bx, by);	
            by += bm*pow(2,i);
        }
        cudaThreadSynchronize();
        by = 0;
        //recursive_diagonal <<< dg_diag, dimblock >>> (a_d, N, b_d, N2);
        kernel_test <<< dg_diag, block, 0, streams[(streamindex++) % MAXSTREAMS] >>> (n, msize, ddata, dmat2, dmat1, mapdiag, bx, by);
        by += bm;
        for(int i=0; i<k; i++){
            dim3 grid(n/(block.x*2), bm*pow(2, i), 1);
            //recursive_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2, bx, by);	
            kernel_test <<< grid, block, 0, streams[(streamindex++) % MAXSTREAMS] >>> (n, msize, ddata, dmat2, dmat1, maprec, bx, by);	
            by += bm*pow(2,i);
        }
        cudaThreadSynchronize();
    }
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
#endif
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // that's our time!
    last_cuda_error("benchmark-check");
    time = time/(float)REPEATS;
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
    return time;
}
int verify_result(unsigned int n, unsigned int msize, DTYPE *hdata, DTYPE *ddata, MTYPE *hmat, MTYPE *dmat){
#ifdef DEBUG
    printf("verifying result................"); fflush(stdout);
#endif
    cudaMemcpy(hdata, ddata, sizeof(DTYPE)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hmat, dmat, sizeof(MTYPE)*msize, cudaMemcpyDeviceToHost);
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            //unsigned int index = i*n + j;
            //DTYPE a = hdata[i];
            //DTYPE b = hdata[j];
        }
    }
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
    if(n <= MAXPRINT){
        print_matrix(hmat, n, "host matrix");
    }
#endif
    return 1;
}

void set_randconfig(MTYPE *mat, const unsigned long n, double DENSITY){
    for(unsigned int y=0; y<n; ++y){
        for(unsigned int x=0; x<n; ++x){
            unsigned long i = y*n + x;
            mat[i] = x < y? (((double)rand()/RAND_MAX) <= DENSITY ? 1 : 0) : 0;
        }
    }
}

void set_alldead(MTYPE *mat, const unsigned long n){
    for(unsigned int y=0; y<n; ++y){
        for(unsigned int x=0; x<n; ++x){
            unsigned long i = y*n + x;
            mat[i] = 0;
        }
    }
}

void set_cell(MTYPE *mat, const unsigned long n, const int x, const int y, MTYPE val){
    unsigned long i = y*n + x;
    mat[i] = val;
}

unsigned long count_living_cells(MTYPE *mat, const unsigned long n){
    unsigned long c = 0;
    for(unsigned int y=0; y<n; ++y){
        for(unsigned int x=0; x<n; ++x){
            if(x<y){
                unsigned long i = y*n + x;
                c += mat[i];
            }
        }
    }
    return c;
}

void init(unsigned int n, DTYPE **hdata, MTYPE **hmat, DTYPE **ddata, MTYPE **dmat1, MTYPE **dmat2, unsigned int *msize, unsigned int *trisize, double DENSITY){
	*msize = n*n;
    *trisize = n*(n-1)/2;
	
	*hdata = (DTYPE*)malloc(sizeof(DTYPE)*n);
	*hmat = (MTYPE*)malloc(sizeof(MTYPE)*(*msize));

	fill_random(*hdata, n);
	cudaMalloc((void **) ddata, sizeof(DTYPE)*n);
    last_cuda_error("init: cudaMalloc ddata");
	cudaMemcpy(*ddata, *hdata, sizeof(DTYPE)*n, cudaMemcpyHostToDevice);
    last_cuda_error("init end:memcpy hdata->ddata");


    set_randconfig(*hmat, n, DENSITY);
    /*
    set_alldead(*hmat, n);
    set_cell(*hmat, n, 3, 6, 1);
    set_cell(*hmat, n, 4, 6, 1);
    set_cell(*hmat, n, 5, 6, 1);

    set_cell(*hmat, n, 3, 7, 1);
    set_cell(*hmat, n, 4, 7, 1);
    set_cell(*hmat, n, 5, 7, 1);

    set_cell(*hmat, n, 3, 8, 1);
    set_cell(*hmat, n, 4, 8, 1);
    set_cell(*hmat, n, 5, 8, 1);
    */

	cudaMalloc((void **) dmat1, sizeof(MTYPE)*(*msize));
	cudaMalloc((void **) dmat2, sizeof(MTYPE)*(*msize));
    last_cuda_error("init: cudaMalloc dmat");
    cudaMemcpy(*dmat1, *hmat, sizeof(MTYPE)*(*msize), cudaMemcpyHostToDevice);
    last_cuda_error("init end:memcpy hmat->dmat");

#ifdef DEBUG
	printf("2-simplex: n=%i  msize=%i (%f MBytes)\n", n, *msize, (float)sizeof(MTYPE)*(*msize)/(1024*1024));
    if(n <= MAXPRINT){
        print_matrix(*hmat, n, "host matrix");
    }
#endif
}

#endif
