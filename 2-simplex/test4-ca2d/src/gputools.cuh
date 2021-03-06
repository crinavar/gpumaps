//////////////////////////////////////////////////////////////////////////////////
//  gpumaps                                                                     //
//  A GPU benchmark of mapping functions                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2018 Cristobal A. Navarro.                                      //
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

// integer log2
__host__ __device__ int cf_log2i(const int val){
    int copy = val;
    int r = 0;
    while (copy >>= 1) 
        ++r;
    return r;
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

uint32_t cntsetbits(uint32_t i){
     // C or C++: use uint32_t
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

void print_array(DTYPE *a, const int n){
	for(int i=0; i<n; i++)
		printf("a[%d] = %f\n", i, a[i]);
}


void set_randconfig(MTYPE *mat, const unsigned long n, double DENSITY){
    // note: this function does not act on the ghost cells (boundaries)
    for(unsigned int y=0; y<n; ++y){
        for(unsigned int x=0; x<n; ++x){
            unsigned long i = (y+1)*(n+2) + (x+1);
            mat[i] = x <= y ? (((double)rand()/RAND_MAX) <= DENSITY ? 1 : 0) : 0;
        }
    }
}
void set_ids(MTYPE *mat, const unsigned long n){
    // note: this function does not act on the ghost cells (boundaries)
    int count = 0;
    for(unsigned int y=0; y<n; ++y){
        for(unsigned int x=0; x<n; ++x){
            unsigned long i = (y+1)*(n+2) + (x+1);
            if(y >= x){
                mat[i] = count++;
            }
        }
    }
}

void set_alldead(MTYPE *mat, const unsigned long n){
    // note: this function does not act on the ghost cells (boundaries)
    for(unsigned int y=0; y<n; ++y){
        for(unsigned int x=0; x<n; ++x){
            unsigned long i = (y+1)*(n+2) + (x+1);
            mat[i] = 0;
        }
    }
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

void set_cell(MTYPE *mat, const unsigned long n, const int x, const int y, MTYPE val){
    // the boundaries are ghost cells
    unsigned long i = (y+1)*(n+2) + (x+1);
    mat[i] = val;
}

unsigned long count_living_cells(MTYPE *mat, const unsigned long n){
    unsigned long c = 0;
    // note: this function does not act on the boundaries.
    for(unsigned int y=0; y<n; ++y){
        for(unsigned int x=0; x<n; ++x){
            if(x <= y){
                unsigned long i = (y+1)*(n+2) + (x+1);
                c += mat[i];
            }
        }
    }
    return c;
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

void last_cuda_error(const char *msg){
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
		printf("[%s]: CUDA error: %s\n", msg, cudaGetErrorString(error));
		exit(-1);
	}
}


void init(unsigned int n, DTYPE **hdata, MTYPE **hmat, DTYPE **ddata, MTYPE **dmat1, MTYPE **dmat2, unsigned long *msize, unsigned long *trisize, double DENSITY){
    // define ghost n, gn, for the (n+2)*(n+2) space for the ghost cells
	*msize = (n+2)*(n+2);

	*hmat = (MTYPE*)malloc(sizeof(MTYPE)*(*msize));
    set_everything(*hmat, n, 0);
    set_randconfig(*hmat, n, DENSITY);
    //set_ids(*hmat, n);

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

void gen_hadouken_pspace_lower(const unsigned int n, dim3 &block, dim3 &grid, unsigned int *aux1, unsigned int *aux2, unsigned int *aux3){
    // lower pow, tetrahedron covering (simplex + lower block)
    int nlow = 1 << ((int)floor(log2f(n)));
    block = dim3(BSIZE2D, BSIZE2D, 1);
    int nbo = (n + block.x - 1)/block.x;
    int nb = (nlow + block.x - 1)/block.x;
    int gx = nb <= 1? nb + 1 : nb;
    int gy = nbo;
    int extraby = nbo - nb;
    //printf("n %i   nlow %i   nbo %i  nb %i  s %i\n", n, nlow, nbo, nb, extraby);
    grid = dim3(ceil((gx-1.0)/2.0), gy+1+extraby, 1);
    /* big blocks for trapezoid tower */
    *aux1 = nb-1 + extraby;
    *aux2 = nbo-(nb-1);
    *aux3 = gy+extraby+1;
    //printf("extra segments = %i  aux1 = %i    aux2 = %i  aux3 = %i  extraby = %i\n", extraby, *aux1, *aux2, *aux3, extraby);
    /* a -- b  blocks (alternating to the side odd and even for the trapezoid)
    *aux1 = nb-1;
    *aux2 = nbo-(nb-1);
    *aux3 = gy+extraby+1;
    */
#ifdef DEBUG
	printf("[lower] block= %i x %i x %i    grid = %i x %i x %i\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
#endif
}

// covering from greater simplex
void gen_hadouken_pspace_upper(const unsigned int n, dim3 &block, dim3 &grid, unsigned int *aux1, unsigned int *aux2, unsigned int *aux3){
    block = dim3(BSIZE2D, BSIZE2D, 1);
    int nu = 1 << ((int)ceil(log2f(n)));
    int nb = (nu + block.x - 1)/block.x;
    int gx = nb <= 1? nb + 1 : nb;
    int gy = nb;
    grid = dim3(ceil((gx-1.0)/2.0), gy+1, 1);
    *aux1 = nb-1;
    *aux2 = 1;
#ifdef DEBUG
	printf("[upper] block= %i x %i x %i    grid = %i x %i x %i\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
#endif
}

void gen_recursive_pspace(const unsigned int n, dim3 &block, dim3 &grid){
    block = dim3(BSIZE2D, BSIZE2D, 1);
    // it is defined later, at each recursive level
    grid = dim3(1,1,1);
}

unsigned int count_recursions(unsigned int n, int bsize){
    unsigned int lpow = 1 << ((unsigned int)floor(log2f(n)));
    unsigned int hpow = 1 << ((unsigned int)ceil(log2f(n)));
    unsigned int nh = n;
    unsigned int numrec = 0;
    do{
        if( hpow - nh < HADO_TOL){
            nh = 0;
        }
        else{
            nh = nh - lpow;
            lpow = 1 << ((unsigned int)floor(log2f(nh)));
            hpow = 1 << ((unsigned int)ceil(log2f(nh)));
        }
        numrec++;
    }
    while(nh > 0);
    return numrec;
}

void create_grids_streams(unsigned int n, unsigned int numrec, dim3 *grids, dim3 block, unsigned int *aux1, unsigned int *aux2, unsigned int *aux3, cudaStream_t *streams, unsigned int *offsets){
    unsigned int lpow = 1 << ((unsigned int)floor(log2f(n)));
    unsigned int hpow = 1 << ((unsigned int)ceil(log2f(n)));
    unsigned int off = 0;
    unsigned int nh = n;
    int nr = 0;
    do{
        if( hpow - nh < HADO_TOL){
            // case n is close to the next power of two, we do all
            gen_hadouken_pspace_upper(nh, block, grids[nr], &aux1[nr], &aux2[nr], &aux3[nr]);
            offsets[nr] = off;
            nh = 0;
        }
        else{
            // case n is far from its next power of two, we continue in halves
            gen_hadouken_pspace_lower(nh, block, grids[nr], &aux1[nr], &aux2[nr], &aux3[nr]);
            offsets[nr] = off;
            nh = nh - lpow;
            off += lpow;
            lpow = 1 << ((unsigned int)floor(log2f(nh)));
            hpow = 1 << ((unsigned int)ceil(log2f(nh)));
        }
        cudaStreamCreate(&streams[nr]);
        nr++;
    }
    while(nh > 0);
}

void print_grids_offsets(unsigned int numrec, dim3 *grids, dim3 block, unsigned int *offsets){
    for(int i=0; i<numrec; ++i){
        printf("[%i] =  block(%i, %i, %i)  grid(%i, %i, %i)   offset = %i\n", i, block.x, block.y, block.z, grids[i].x, grids[i].y, grids[i].z, offsets[i]);
    }
}









template<typename Lambda>
double benchmark_map(const int REPEATS, dim3 block, dim3 grid, unsigned int n,
        unsigned long msize, unsigned int trisize, DTYPE *ddata, MTYPE *dmat1,
        MTYPE *dmat2, Lambda map, unsigned int aux1, unsigned int aux2, unsigned int aux3){
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
        kernel_update_ghosts<<< (n+BSIZE1D-1)/BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        #ifdef DEBUG
            //printf("result pong\n");
            //print_dmat(PRINTLIMIT, n, msize, dmat1, "ghost update dmat1");
            //getchar();
        #endif
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat1, dmat2, map, aux1, aux2, aux3);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("result ping\n");
            print_dmat(PRINTLIMIT, n, msize, dmat2, "PING dmat1 -> dmat2");
            getchar();
        #endif


        kernel_update_ghosts<<< (n+BSIZE1D-1)/BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        #ifdef DEBUG
            //printf("result pong\n");
            //print_dmat(PRINTLIMIT, n, msize, dmat2, "ghost update dmat2");
            //getchar();
        #endif
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat2, dmat1, map, aux1, aux2, aux3);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            printf("result pong\n");
            print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG  dmat2 -> dmat1");
            getchar();
        #endif
    }
    last_cuda_error("warmup");
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG  dmat2 -> dmat1");
    printf("Benchmarking (%i REPEATS).......", REPEATS); fflush(stdout);
#endif
    float time = 0.0;
    // measure running time
    cudaEventRecord(start, 0);	
    for(int k=0; k<REPEATS; k++){
        kernel_update_ghosts<<< (n+BSIZE1D-1)/BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat1, dmat2, map, aux1, aux2, aux3);	
        cudaDeviceSynchronize();

        kernel_update_ghosts<<< (n+BSIZE1D-1)/BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat2, dmat1, map, aux1, aux2, aux3);	
        cudaDeviceSynchronize();
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
double benchmark_map_rectangle(const int REPEATS, dim3 block, dim3 grid,
        unsigned int n, unsigned long msize, unsigned int trisize, DTYPE *ddata,
        MTYPE *dmat1, MTYPE *dmat2, Lambda map, 
        unsigned int aux1, unsigned int aux2, unsigned int aux3){
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
        kernel_update_ghosts<<< (n+BSIZE1D-1)/BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        kernel_test_rectangle<<< grid, block >>>(n, msize, ddata, dmat1, dmat2, map, aux1, aux2, aux3);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            //printf("result ping\n");
            //print_dmat(PRINTLIMIT, n, msize, dmat2, "PING");
        #endif
        kernel_update_ghosts<<< (n+BSIZE1D-1)/BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        kernel_test_rectangle<<< grid, block >>>(n, msize, ddata, dmat2, dmat1, map, aux1, aux2, aux3);	
        cudaDeviceSynchronize();
        #ifdef DEBUG
            //printf("result pong\n");
            //print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG");
        #endif
    }
    last_cuda_error("warmup");
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG");
    printf("Benchmarking (%i REPEATS).......", REPEATS); fflush(stdout);
#endif
    float time = 0.0;
    // measure running time
    cudaEventRecord(start, 0);	
    for(int k=0; k<REPEATS; k++){
        kernel_update_ghosts<<< (n+BSIZE1D-1)/BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        kernel_test_rectangle<<< grid, block >>>(n, msize, ddata, dmat1, dmat2, map, aux1, aux2, aux3);	
        cudaDeviceSynchronize();

        kernel_update_ghosts<<< (n+BSIZE1D-1)/BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        kernel_test_rectangle<<< grid, block >>>(n, msize, ddata, dmat2, dmat1, map, aux1, aux2, aux3);	
        cudaDeviceSynchronize();
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
double benchmark_map_hadouken(const int REPEATS, dim3 block, unsigned int n, unsigned long msize, unsigned int trisize, DTYPE *ddata, MTYPE *dmat1, MTYPE *dmat2, Lambda map){
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time = 0.0;
    unsigned int numrec = count_recursions(n, BSIZE2D);
    dim3 *grids = (dim3*)malloc(sizeof(dim3)*numrec);
    unsigned int *offsets  = (unsigned int*)malloc(sizeof(unsigned int)*numrec);
    unsigned int *auxs1  = (unsigned int*)malloc(sizeof(unsigned int)*numrec);
    unsigned int *auxs2  = (unsigned int*)malloc(sizeof(unsigned int)*numrec);
    unsigned int *auxs3  = (unsigned int*)malloc(sizeof(unsigned int)*numrec);
    cudaStream_t *streams = (cudaStream_t*)malloc(sizeof(cudaStream_t)*numrec);
    create_grids_streams(n, numrec, grids, block, auxs1, auxs2, auxs3, streams, offsets);
#ifdef DEBUG
    printf("HADO_TOL = %i\n", HADO_TOL);
    print_grids_offsets(numrec, grids, block, offsets);
#endif
    //printf("n = %i    lpow = %i, hpow = %i\n", n, lpow, hpow);
    //printf("numrecs = %i\n", numrec);
    // Warmup
#ifdef DEBUG
    printf("warmup.........................."); fflush(stdout);
#endif
    for(int k=0; k<REPEATS; k++){
        kernel_update_ghosts<<< (n+BSIZE1D-1)/BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
	    for(int i=0; i<numrec; ++i){
            kernel_test<<< grids[i], block, 0, streams[i] >>>(n, msize, ddata, dmat1, dmat2, map, auxs1[i], auxs2[i], offsets[i]);
        }
        cudaDeviceSynchronize();
        #ifdef DEBUG
            //printf("result ping\n");
            //print_dmat(PRINTLIMIT, n, msize, dmat2, "PING");
            //getchar();
        #endif
        kernel_update_ghosts<<< (n+BSIZE1D-1)/BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
	    for(int i=0; i<numrec; ++i){
            kernel_test<<< grids[i], block, 0, streams[i] >>>(n, msize, ddata, dmat2, dmat1, map, auxs1[i], auxs2[i], offsets[i]);
        }
        cudaDeviceSynchronize();
        #ifdef DEBUG
            //printf("result pong\n");
            //print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG");
            //getchar();
        #endif
        cudaDeviceSynchronize();
    }
    last_cuda_error("warmup");
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG");
    printf("Benchmarking (%i REPEATS).......", REPEATS); fflush(stdout);
#endif
    // measure running time
    cudaEventRecord(start, 0);	
    //numrec = count_recursions(n, BSIZE2D);
    //create_grids_streams(n, numrec, grids, block, auxs1, auxs2, auxs3, streams, offsets);
    #pragma loop unroll
    for(int k=0; k<REPEATS; ++k){
        kernel_update_ghosts<<< (n+BSIZE1D-1)/BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
	    for(int i=0; i<numrec; ++i){
	    //for(int i=numrec-1; i>=0; --i){
            kernel_test<<< grids[i], block, 0, streams[i] >>>(n, msize, ddata, dmat1, dmat2, map, auxs1[i], auxs2[i], offsets[i]);
        }
        kernel_update_ghosts<<< (n+BSIZE1D-1)/BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
	    for(int i=0; i<numrec; ++i){
	    //for(int i=numrec-1; i>=0; --i){
            kernel_test<<< grids[i], block, 0, streams[i] >>>(n, msize, ddata, dmat2, dmat1, map, auxs1[i], auxs2[i], offsets[i]);
	    }
        cudaDeviceSynchronize();
    }
    #ifdef DEBUG
        printf("done\n"); fflush(stdout);
    #endif
    cudaEventRecord(stop,0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    time = time/(float)REPEATS;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    last_cuda_error("benchmark_map: run");
    return time;
}

int verify_result(unsigned int n, const unsigned long msize, DTYPE *hdata, DTYPE *ddata, MTYPE *hmat, MTYPE *dmat){
    return 1;
}
#endif
