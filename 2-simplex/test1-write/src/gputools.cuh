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

#define DTYPE float
#define MTYPE char
#define WORDLEN 31
#define MAXSTREAMS 32
#define PRINTLIMIT 256
#define EXTRASPACE

template<typename Lambda>
__global__ void kernel_test(const unsigned int n, const unsigned int msize, DTYPE *data, MTYPE* dmat, Lambda map, unsigned int aux1, unsigned int aux2);
// integer log2
__host__ __device__ int cf_log2i(const int val){
    int copy = val;
    int r = 0;
    while (copy >>= 1) 
        ++r;
    return r;
}

void print_array(DTYPE *a, const int n){
	for(int i=0; i<n; i++)
		printf("a[%d] = %f\n", i, a[i]);
}

void print_matrix(MTYPE *mat, const int no, const char *msg){
    int n = no;
    // writing result to frame
	for(int i=0; i<n; i++){
	    for(int j=0; j<n; j++){
            if(mat[i*n+j] == 0){
                printf("* ");
            }
            else{
                printf("%i ", mat[i*n + j]);
            }
        }
        printf("\n");
    }
}

void print_map(MTYPE *mat, const int no, const char *msg, dim3 grid, dim3 block){
#ifdef EXTRASPACE
    // mostrar doble
    int n = 2*no;
#else
    // mostrar justo
    int n = no;
#endif
    int w = grid.x*block.x;
    int h = grid.y*block.y;
    const int gap = 3;
    const int framex = n + w + gap;
    const int framey = n > h? n + 5 : h + 5;

    int b=0;
    char frame[framey][framex];
	for(int i=0; i<framey; i++){
	    for(int j=0; j<framex; j++){
            frame[i][j] = ' ';
        }
    }

    // writing pspace to frame
	for(int i=0; i<h; i++){
	    for(int j=0; j<w; j++){
            b = 48 + (int)log2f(i/BSIZE2D + 1) + 1;
            frame[i+2][j] = b;
        }
    }

    // writing result to frame
	for(int i=0; i<n; i++){
	    for(int j=0; j<n; j++){
            if(mat[i*n+j] == 0){
                    frame[i+2][j+w+gap] = 42;
                    continue;
            }
            if(j>i){
                //if( mat[i*n + j] == 0 ){
                //    printf("  ");
                //}
                //else{
                    //printf("%i ", mat[i*n + j]);
                    frame[i+2][j+w+gap] = 48 + mat[i*n + j];
                //}
            }
            else{
                if( mat[i*n + j] == 0 ){
                    //printf("%i ", mat[i*n + j]);
                    frame[i+2][j+w+gap] = 48 + mat[i*n + j];
                }
                else{
                    //printf("%i ", mat[i*n + j]);
                    frame[i+2][j+w+gap] = 48 + mat[i*n + j];
                }
            }
        }
    }
	for(int i=0; i<framey; i++){
	    for(int j=0; j<framex; j++){
            printf("%c ", frame[i][j]);
        }
        printf("\n");
    }
    printf("[%s]:\n", msg); fflush(stdout);
}

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
		array[i] = 100.0f * (float)rand()/(float)RAND_MAX;
    }
} 

void init(unsigned long no, DTYPE **hdata, MTYPE **hmat, DTYPE **ddata, MTYPE **dmat, unsigned long *msize, unsigned long *trisize){
#ifdef EXTRASPACE
    // mostrar doble
    unsigned long n = 2*no;
#else
    // mostrar justo
    unsigned long n = no;
#endif
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
	printf("2-simplex: n=%i  msize=%lu (%f MBytes)\n", n, *msize, (float)sizeof(MTYPE)*(*msize)/(1024*1024));
    if(n <= PRINTLIMIT){
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

// powers of two assumed for now
void gen_hadouken_pspace(const unsigned int n, dim3 &block, dim3 &grid, unsigned int *bx, unsigned int *ex){
    /*
    // horizontal 
    int w = n-1;
    int h = n/2;
    block = dim3(BSIZE2D, BSIZE2D, 1);
    grid = dim3((w+block.x-1)/block.x + 2, (h+block.y-1)/block.y, 1);
    */

    // vertical
    //int n = no/2; 
    //int n = 1 << (WORDLEN - __builtin_clz(no)); 
    int w = (int)ceil((n+1)/2.0);
    //int h = n;
    int extra = 0;
    int nu = 1 << ((int)ceil(log2f(n)));
    printf("nu = %i\n", nu);
    //extra = ceil((float)(nu - n)/2.0);
    printf("extra = %i\n", extra);
    block = dim3(BSIZE2D, BSIZE2D, 1);
    int nb = (n + block.x - 1)/block.x;
    //grid = dim3((extra + w + block.x-1)/block.x, (h+block.y-1)/block.y, 1);
    grid = dim3(ceil((nb-1.0)/2.0), nb+1, 1);
    
    
    //// vertical any 'n'
    //const unsigned int nb = (n + BSIZE2D - 1)/BSIZE2D;

    //const unsigned int l = WORDLEN - __builtin_clz(n);
    //const unsigned int powl = 1 << l;
    //const unsigned int nbl = (powl + BSIZE2D - 1)/BSIZE2D; 
    //const unsigned int obx = nb - nbl;
    //unsigned int aux_exu=0;
    //unsigned int lobx = 0;
    //if(obx <= 1){
    //    aux_exu=0;
    //}
    //else{
    //    lobx = WORDLEN - __builtin_clz(obx-1);
    //    aux_exu = ((obx-1) - (1 << lobx) == 0) ?  (obx-1) : (1 << (lobx+1));
    //    printf("1 << lobx = %i\n", 1 << lobx);
    //}
    //const unsigned int exu = ceil((double)aux_exu/2.0);
    //const unsigned int nblhalf = nbl/2;
    //printf("aux_exu = %i, aux_exu/2.0f = %f  ceil(aux_exu/2.0f) = %f  exu = %i\n", aux_exu, aux_exu/2.0f, ceil(aux_exu/2.0f), exu);
    //block = dim3(BSIZE2D, BSIZE2D, 1);
    ////grid = dim3(nblhalf + obx + exu, nbl+1, 1);
    //grid = dim3(obx + nblhalf + exu, nbl+1, 1);
    //*bx = nblhalf;
    //*ex = nblhalf + obx;
    //*bx = obx;
    //*ex = obx + nblhalf;
    //printf("n=%u  l=%u  nb=%u  nbl=%u  nblhalf = %u obx=%u  lobx=%u  exu=%u bx=%u  ex=%u \n", n, l, nb, nbl, nblhalf, obx, lobx, exu, *bx, *ex);
#ifdef DEBUG
	printf("block= %i x %i x %i    grid = %i x %i x %i\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
#endif
}// powers of two assumed for now

void gen_hadouken_pspace_tall(const unsigned int n, dim3 &block, dim3 &grid, unsigned int *bmark, unsigned int *emark){
    int nl = 1 << (int)floor(log2f(n));
    printf("nl = %i\n", nl);
    int w = (int)ceil(nl/2.0);
    int extrah = n - nl;
    int h = nl + 3*extrah;
    int fakeh = nl - 1 + 1*extrah;
    block = dim3(BSIZE2D, BSIZE2D, 1);
    grid  = dim3((w + block.x-1)/block.x, (h + block.y-1)/block.y, 1);
    dim3 fakegrid  = dim3((w + block.x-1)/block.x, (fakeh + block.y-1)/block.y, 1);
    int hoffset = fakegrid.y - grid.x*2;
    *emark = grid.x*2;  
    *bmark = *emark + hoffset;
#ifdef DEBUG
	printf("block= %i x %i x %i    grid = %i x %i x %i\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
#endif
}

void gen_recursive_pspace(const unsigned int n, dim3 &block, dim3 &grid){
    block = dim3(BSIZE2D, BSIZE2D, 1);
    // it is defined later, at each recursive level
    grid = dim3(1,1,1);
}

template<typename Lambda>
double benchmark_map(const int REPEATS, dim3 block, dim3 grid, unsigned int n,
        unsigned int msize, unsigned int trisize, DTYPE *ddata, MTYPE *dmat,
        Lambda map, const unsigned int aux1, const unsigned int aux2){
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
    // Warmup
#ifdef DEBUG
    printf("warmup.........................."); fflush(stdout);
#endif
	for(int i=0; i<REPEATS; i++){
        //kernel_test<<< grid, block >>>(n, msize, ddata, dmat, map, aux1, aux2);	
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
        kernel_test<<< grid, block >>>(n, msize, ddata, dmat, map, aux1, aux2);	
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
int verify_result(unsigned int n, unsigned int msize, DTYPE *hdata, DTYPE *ddata, MTYPE *hmat, MTYPE *dmat, dim3 grid, dim3 block){
#ifdef DEBUG
    printf("verifying result................"); fflush(stdout);
#endif
    cudaMemcpy(hdata, ddata, sizeof(DTYPE)*n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hmat, dmat, sizeof(MTYPE)*msize, cudaMemcpyDeviceToHost);
#ifdef DEBUG
    printf("done\n"); fflush(stdout);
    if(n <= PRINTLIMIT){
        print_map(hmat, n, "host matrix", grid, block);
    }
#endif
    for(int i=0; i<n; ++i){
        for(int j=0; j<n; ++j){
            unsigned int index = i*n + j;
            if(i > j){
                if( hmat[index] != 1 ){
                    #ifdef DEBUG
                    fprintf(stderr, "[Verify] invalid element at hmat[%i,%i](%i) = %i\n", i, j, index, hmat[index]);
                    #endif
                    return 0;
                }
            }
            else if(i < j){
                if( hmat[index] != 0 ){
                    #ifdef DEBUG
                    fprintf(stderr, "[Verify] invalid element at hmat[%i,%i](%i) = %i\n", i, j, index, hmat[index]);
                    #endif
                    return 0;
                }
            }
            else if(i == j){
                if(hmat[index] != 0 && hmat[index] != 1){
                    #ifdef DEBUG
                    fprintf(stderr, "[Verify] invalid element at hmat[%i,%i](%i) = %i\n", i, j, index, hmat[index]);
                    #endif
                    return 0;
                }
            }
        }
    }
    return 1;
}

#endif
