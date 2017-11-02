#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include "custom_funcs.h"

#define SQRT_COEF 0.0001f
#define REPETITIONS 100
#define MEASURES 10
#define OFFSET -0.4999f
#define MAX_COLLISIONS 10240
#define R 5.0f
#define BLOCKSIZE 	128
#define BLOCKSIZE_AVRIl 256
#define BLOCKSIZE1D	128
#define BLOCKSIZE_REC 16
#define VOLUME_SIDE 100.0f
//#define OFFSET 0.5f



__device__ float carmack_sqrtf(float nb) {
    float nb_half = nb * 0.5F;
    float y = nb;
    long i = * (long *) &y;
    //i = 0x5f3759df - (i >> 1);
    i = 0x5f375a86 - (i >> 1);
    y = * (float *) &i;
    //Repetitions increase accuracy(6)
    y = y * (1.5f - (nb_half * y * y));
    y = y * (1.5f - (nb_half * y * y));
    y = y * (1.5f - (nb_half * y * y));

    return nb * y;
}

__device__ void cost_function(float4 a, float4 b, int i, int j, int *map, int N2){

    float distance = sqrtf( (a.x-b.x)*(a.x-b.x) );
	//+ (a.y-b.y)*(a.y-b.y) + (a.z-b.z)*(a.z-b.z) );
	//float distance = sqrtf( (a.x-b.x)*(a.x-b.x) );
    //printf("point (%i, %i)\n", i, j);
    //printf("comparing P%i(%f)     P%i(%f)        d=%f         R=%f\n", i, a.x, j, b.x, distance, R);
    if( distance < (a.w + b.w) ){
        //printf("comparing P%i(%f)     P%i(%f)        d=%f         R=%f\n", i, a.x, j, b.x, distance, R);
        
        int pos = ((i*(i+1)) >> 1) + j;
        if( pos < N2 )
            map[pos] = 1;
    }
	//else{
	//	int pos = ((i*(i+1)) >> 1) + j;
    //    if( pos < N2 )
    //        map[pos] = -1;
	//}
}



// avril map
__global__ void kernel_avril_map(float4 *a, int N, int* map, int N2){
	int k = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
    if(k<N2){
        int i = (-(2.0f*(float)N + 1.0f) + carmack_sqrtf(4.0f*(float)N*(float)N - 4.0f*(float)N - 8.0f*(float)k + 1.0f))/(-2.0f);
		//int i = (-(2.0*(double)N + 1.0) + sqrt(4.0*(double)N*(double)N - 4.0*(double)N - 8.0*(double)k + 1.0))/(-2.0);
        int j = (i+1) + k - ((i-1)*(2*N-i))/2;
        if(j > N){
            i++;
            j = i+(j-N);
        }
        if(i >= j){
            j = N - (i-j);
            i--;
        }
		// printf("threadIdx = %i   coord(%i, %i)\n", k, j-1, i-1);
		// transpose the coordinates for comparing outputs
		cost_function(a[i-1], a[j-1], j-1, i-1, map, N2+N);
	}
}


// metodo bounding box
__global__ void bb_method(float4 *a, int N, int *map, int N2){
	
	__shared__ float4 to_compare[BLOCKSIZE];
    if( blockIdx.x > blockIdx.y )
		return;
	int i = blockIdx.y*blockDim.x + threadIdx.x;
	int j = blockIdx.x*blockDim.x;
    
    // load data into shared memory
    float4 me = a[i];
    to_compare[threadIdx.x] = a[j+threadIdx.x];
    __syncthreads();

    for(int k=0; k<blockDim.x; k++){  
        int lpos = (threadIdx.x + k) % blockDim.x;
        if(i>j+lpos){
            cost_function(me, to_compare[lpos], i, j+lpos, map, N2);
        }
        __syncthreads();
    }
/*
    if( blockIdx.x > blockIdx.y )
		return;
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
    
	if( i != N || j != N )
		return;

    cost_function(a[i], a[j], i, j, map, N2);
*/
}

// metodo td con inversa square root
__global__ void td_method_2dr(float4 *a, int N, int *map, int N2){
    __shared__ float4 to_compare[BLOCKSIZE];

    unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
    float arg = __fmaf_rn(2.0f, (float)bc, 0.25f);
	//unsigned int bi = sqrtf(arg) - 0.5f;
    unsigned int bi = __fmaf_rn(arg, rsqrt(arg), OFFSET);// + 0.001f;
    unsigned int bj = bc - (bi*(bi+1) >> 1);
    
    unsigned int i = bi * blockDim.x + threadIdx.x;
	unsigned int j = bj * blockDim.x;
    
    // load data into shared memory

    float4 me = a[i];
    to_compare[threadIdx.x] = a[j+threadIdx.x];
    __syncthreads();
    for(int k=0; k<blockDim.x; k++){
        int lpos = (threadIdx.x + k) % blockDim.x;
        if(i>j+lpos){
            cost_function(me, to_compare[lpos], i, j+lpos, map, N2);
        }
        __syncthreads();
    }   
}


// metodo rectangle
__global__ void rect_method(float4 *a, int N, int *map, int N2){

    __shared__ float4 to_compare[BLOCKSIZE];
    int i = blockIdx.y * blockDim.x + threadIdx.x;
    int j = blockIdx.x * blockDim.x;
    //if(i >= N+1)
        //return;
    
    if(blockIdx.x == blockIdx.y){
        while(j < (blockIdx.x * blockDim.x) + blockDim.x ){
            if( j > i){
                cost_function(a[N-i-1], a[N-j-1], N-i-1, N-j-1, map, N2);
            }
            else if( i > j ){
                cost_function(a[i], a[j], i, j, map, N2);
            }
            j++;
        }
    }
    else{
        if(blockIdx.x > blockIdx.y){
            //j = N-j-1-blockDim.x-1;
			j = (gridDim.x - 1 - blockIdx.x)*blockDim.x + N/2; 
            i = N-i-1;
        }
		else{
			i = i-1;
		}
        //float4 me = a[i];
        to_compare[threadIdx.x] = a[j+threadIdx.x];
        __syncthreads();
		//printf("a=%i   j=%i      shared[%i] = (%.2f, %.2f, %.2f, %.2f)\n", i, j, threadIdx.x, to_compare[threadIdx.x].x, to_compare[threadIdx.x].y, to_compare[threadIdx.x].z, to_compare[threadIdx.x].w);
        for(int k=0; k<blockDim.x; k++){
            int lpos = (threadIdx.x + k) % blockDim.x;
    		int newj= j+lpos;
            if(i>newj){
				//printf("thread[%i]  (%i,%i) j=%i lpos = %i   to_compare[%i] = (%.2f, %.2f, %.2f, %.2f) \n", threadIdx.x, i, newj, j, lpos, lpos, to_compare[lpos].x, to_compare[lpos].y, to_compare[lpos].z, to_compare[lpos].w);
                cost_function(a[i], to_compare[lpos], i, newj, map, N2);
				//cost_function(me, a[newj], i, newj, map, N2);
				//if(i == 31 && newj == 7){
					//printf("xxxgpu %i (%.2f, %.2f, %.2f, %.2f) and %i (%.2f, %.2f, %.2f, %.2f) .....\n", 
						//	i, to_compare[i].x, to_compare[i].y, to_compare[i].z, to_compare[i].w,
							//newj, to_compare[lpos].x, to_compare[lpos].y, to_compare[lpos].z, to_compare[lpos].w);
				//} 
            }
            __syncthreads();
        }   
    }   
}


// metodo recursivo
__global__ void recursive_method(float4 *a, int N, int *map, int N2, int bx, int by){

    __shared__ float4 to_compare[BLOCKSIZE_REC];
    // calcula el indice del bloque recursivo, es division entera.
    int rec_index = blockIdx.x/gridDim.y;
    
    // calcula el offset, el valor x del bloque respecto al recblock que le corresponde.
    int dbx = blockIdx.x % gridDim.y;
    int j = (bx+(gridDim.y*rec_index*2) + dbx)*blockDim.x;
    int i = (by+(gridDim.y*rec_index*2) + blockIdx.y)*blockDim.x + threadIdx.x;

    float4 me = a[i];
    to_compare[threadIdx.x] = a[j+threadIdx.x];
    __syncthreads();
    for(int k=0; k<blockDim.x; k++){
        int lpos = (threadIdx.x + k) % blockDim.x;
        if(i>j+lpos){
            cost_function(me, to_compare[lpos], i, j+lpos, map, N2);
        }
        __syncthreads();
    }
}

__global__ void recursive_diagonal(float4 *a, int N, int *map, int N2){

    __shared__ float4 to_compare[BLOCKSIZE_REC];
    // calcula el indice del bloque recursivo, es division entera.
    int rec_index = blockIdx.x/gridDim.y;
    int j = blockIdx.x * blockDim.x;
    int i = (blockIdx.y + rec_index*gridDim.y)*blockDim.x + threadIdx.x;
    //if( j >= i )
    //    return;
    
    //float4 me = a[i];
    to_compare[threadIdx.x] = a[j+threadIdx.x];
    __syncthreads();
    for(int k=0; k<blockDim.x; k++){
        int lpos = (threadIdx.x + k) % blockDim.x;
        if(i>j+lpos){
            cost_function(a[i], to_compare[lpos], i, j+lpos, map, N2);
			//cost_function(a[i], a[j+lpos], i, j+lpos, map, N2);
        }
        __syncthreads();
    }
}

void export_result(int *b, int n, const char *filename){
    printf("exporting result vector.......");fflush(stdout);
    FILE *fw = fopen(filename, "w");
    if(!fw){
        fprintf(stderr, "error: cannot write to file %s\n", filename);
        exit(1);
    }
    for(int i=0; i<n; i++){
        fprintf(fw, "%i\n", b[i]);
    }
    printf("ok\n");
}


void fill_random(float4 *array, int n){
	for(int i=0; i<n; i++){
		array[i].x = VOLUME_SIDE * (float)rand()/(float)RAND_MAX;
        array[i].y = VOLUME_SIDE * (float)rand()/(float)RAND_MAX;
        array[i].z = VOLUME_SIDE * (float)rand()/(float)RAND_MAX;
        array[i].w = 1.0f + R * (float)rand()/(float)RAND_MAX;
    }
} 

void bb_computation(int *b_h, size_t sizeb, dim3 dimgrid, dim3 dimblock, float4 *a_d, int N, int *b_d, int N2){

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling kernel bb_method..b_h[0]=%i..block= %i x %i x %i    grid = %i x %i x %i\n", b_h[0], dimblock.x, dimblock.y, dimblock.z, dimgrid.x, dimgrid.y, dimgrid.z);
    //warmup
    printf("warmup...."); fflush(stdout);
	for(int i=0; i<MEASURES; i++){
    	bb_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
        cudaThreadSynchronize();
    }
    printf("ok\nmeasuring mean time of %i averages (each averages is computed from %i measures).......\n", REPETITIONS, MEASURES); fflush(stdout);
    float accum=0.0f, elapsedTime=0.0f, squared_accum=0.0f, mean=0.0f, time=0.0f, stdev=0.0f, div=(float)(REPETITIONS*(REPETITIONS-1));
    for(int j=0; j<REPETITIONS; j++){
        cudaEventRecord(start, 0);	
        for(int m=0; m<MEASURES; m++){
            bb_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
            cudaThreadSynchronize();
        }
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
        time = elapsedTime/(float)MEASURES;
        accum += time;
        squared_accum += time*time/((float)REPETITIONS-1.0f);
        if( j%100 == 0){ printf("."); fflush(stdout);}
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    mean = accum/(float)REPETITIONS;
    stdev = sqrtf(squared_accum - accum*accum/div);
	printf("\nAverage of %i averages(%i):\ncudaEventElapsedTime:\tmean=%f[ms]    stdev=%f    error=%f%%\n", 
    REPETITIONS, MEASURES, mean, stdev, 100.0f*stdev/mean);
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	else
		printf("OK!\n");
	cudaMemcpy(b_h, b_d, sizeb, cudaMemcpyDeviceToHost);
    FILE *fa;
    fa = fopen("bb_results.dat", "a");
    fprintf(fa, "%i        %f      %f\n", N, mean, stdev/mean);
    fclose(fa);
}

void avril_map(int *b_h, size_t sizeb, dim3 dimgrid, dim3 dimblock, float4 *a_d, int N, int *b_d, int N2){
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling kernel avril map....block= %i x %i x %i    grid = %i x %i x %i\n", dimblock.x, dimblock.y, dimblock.z, dimgrid.x, dimgrid.y, dimgrid.z);
    //warmup
    printf("warmup...."); fflush(stdout);
	for(int i=0; i<MEASURES; i++){
    	kernel_avril_map <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
        cudaThreadSynchronize();
    }
    printf("ok\nmeasuring mean time of %i averages (each averages is computed from %i measures).......\n", REPETITIONS, MEASURES); fflush(stdout);
    float accum=0.0f, elapsedTime=0.0f, squared_accum=0.0f, mean=0.0f, time=0.0f, stdev=0.0f, div=(float)(REPETITIONS*(REPETITIONS-1));
    for(int j=0; j<REPETITIONS; j++){
        cudaEventRecord(start, 0);	
        for(int k=0; k<MEASURES; k++){
            kernel_avril_map <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
            cudaThreadSynchronize();
        }

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
        time = elapsedTime/(float)MEASURES;
        accum += time;
        squared_accum += time*time/((float)REPETITIONS-1.0f);
        if( j%100 == 0){	printf("."); fflush(stdout);	}
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    mean = accum/(float)REPETITIONS;
    stdev = sqrtf(squared_accum - accum*accum/div);
	printf("\nAverage of %i averages(%i):\ncudaEventElapsedTime:\tmean=%f[ms]    stdev=%f    error=%f%%\n", 
    REPETITIONS, MEASURES, mean, stdev, 100.0f*stdev/mean);
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	else{
		printf("OK!\n");
	}
	cudaMemcpy(b_h, b_d, sizeb, cudaMemcpyDeviceToHost);
    FILE *fa;
    fa = fopen("avril_results.dat", "a");
    fprintf(fa, "%i        %f       %f\n", N, mean, stdev/mean);
    fclose(fa);

}

void td_computation_rsqrt(int *b_h, size_t sizeb, dim3 dimgrid, dim3 dimblock, float4 *a_d, int N, int *b_d, int N2){
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling kernel td_method_rsqrt....block= %i x %i x %i    grid = %i x %i x %i\n", dimblock.x, dimblock.y, dimblock.z, dimgrid.x, dimgrid.y, dimgrid.z);
    //warmup
    printf("warmup...."); fflush(stdout);
	for(int i=0; i<MEASURES; i++){
    	td_method_2dr <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
        cudaThreadSynchronize();
    }
    printf("ok\nmeasuring mean time of %i averages (each averages is computed from %i measures).......\n", REPETITIONS, MEASURES); fflush(stdout);
    float accum=0.0f, elapsedTime=0.0f, squared_accum=0.0f, mean=0.0f, time=0.0f, stdev=0.0f, div=(float)(REPETITIONS*(REPETITIONS-1));
    for(int j=0; j<REPETITIONS; j++){
        cudaEventRecord(start, 0);	
        for(int k=0; k<MEASURES; k++){
            td_method_2dr <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
            cudaThreadSynchronize();
        }

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
        time = elapsedTime/(float)MEASURES;
        accum += time;
        squared_accum += time*time/((float)REPETITIONS-1.0f);
        if( j%100 == 0){ printf("."); fflush(stdout);}
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    mean = accum/(float)REPETITIONS;
    stdev = sqrtf(squared_accum - accum*accum/div);
	printf("\nAverage of %i averages(%i):\ncudaEventElapsedTime:\tmean=%f[ms]    stdev=%f    error=%f%%\n", 
    REPETITIONS, MEASURES, mean, stdev, 100.0f*stdev/mean);
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	else
		printf("OK!\n");
	cudaMemcpy(b_h, b_d, sizeb, cudaMemcpyDeviceToHost);
    FILE *fa;
    fa = fopen("rsqrt_results.dat", "a");
    fprintf(fa, "%i        %f       %f\n", N, mean, stdev/mean);
    fclose(fa);

}

void td_computation_rectangle(int *b_h, size_t sizeb, dim3 dimgrid, dim3 dimblock, float4 *a_d, int N, int *b_d, int N2){
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling kernel rect_method....block= %i x %i x %i    grid = %i x %i x %i\n", dimblock.x, dimblock.y, dimblock.z, dimgrid.x, dimgrid.y, dimgrid.z);
    //warmup
    printf("warmup...."); fflush(stdout);
	for(int i=0; i<MEASURES; i++){
    	rect_method <<< dimgrid, dimblock>>> (a_d, N, b_d, N2);	
        cudaThreadSynchronize();
    }
    printf("ok\nmeasuring mean time of %i averages (each averages is computed from %i measures).......\n", REPETITIONS, MEASURES); fflush(stdout);
    float accum=0.0f, elapsedTime=0.0f, squared_accum=0.0f, mean=0.0f, time=0.0f, stdev=0.0f, div=(float)(REPETITIONS*(REPETITIONS-1));
    for(int j=0; j<REPETITIONS; j++){
        cudaEventRecord(start, 0);	
        for(int k=0; k<MEASURES; k++){
            rect_method <<< dimgrid, dimblock>>> (a_d, N, b_d, N2);	
            cudaThreadSynchronize();
        }

        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
        time = elapsedTime/(float)MEASURES;
        accum += time;
        squared_accum += time*time/((float)REPETITIONS-1.0f);
        if( j%100 == 0){ printf("."); fflush(stdout);}
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    mean = accum/(float)REPETITIONS;
    stdev = sqrtf(squared_accum - accum*accum/div);
	printf("\nAverage of %i averages(%i):\ncudaEventElapsedTime:\tmean=%f[ms]    stdev=%f    error=%f%%\n", REPETITIONS, MEASURES, mean, stdev, 100.0f*stdev/mean);
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	else
		printf("OK!\n");
	cudaMemcpy(b_h, b_d, sizeb, cudaMemcpyDeviceToHost);
    FILE *fa;
    fa = fopen("rectangle_results.dat", "a");
    fprintf(fa, "%i        %f       %f\n", N, mean, stdev/mean);
    fclose(fa);

}

void td_computation_recursive(int *b_h, size_t sizeb, dim3 dimblockrec, float4 *a_d, int N, int *b_d, int N2, int m, int kval){
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling kernel td_method_recursive....block= %i x %i x %i\n", dimblockrec.x, dimblockrec.y, dimblockrec.z);
    //warmup
    printf("warmup...."); fflush(stdout);
    int bx=0, by=0;
    int bm = m/dimblockrec.x;
	for(int r=0; r<MEASURES; r++){
        by = 0;
        dim3 dg_diag(N/dimblockrec.x, bm, 1);
        recursive_diagonal <<< dg_diag, dimblockrec >>> (a_d, N, b_d, N2);
        by += bm;
        for(int i=0; i<kval; i++){
            dim3 dimgrid(N/(dimblockrec.x*2), bm*pow(2, i), 1);
            recursive_method <<< dimgrid, dimblockrec >>> (a_d, N, b_d, N2, bx, by);	
            cudaThreadSynchronize();
            by += bm*pow(2,i);
        }
    }
    
    printf("ok\nmeasuring mean time of %i averages (each averages is computed from %i measures).......\n", 
            REPETITIONS, MEASURES); fflush(stdout);
    float accum=0.0f, elapsedTime=0.0f, squared_accum=0.0f, mean=0.0f, time=0.0f, stdev=0.0f, div=(float)(REPETITIONS*(REPETITIONS-1));
    for(int j=0; j<REPETITIONS; j++){
        cudaEventRecord(start, 0);	
        for(int k=0; k<MEASURES; k++){
            by = 0;
            dim3 dg_diag(N/dimblockrec.x, bm, 1);
            recursive_diagonal <<< dg_diag, dimblockrec >>> (a_d, N, b_d, N2);
            by += bm;
            for(int i=0; i<kval; i++){
                dim3 dimgrid(N/(dimblockrec.x*2), bm*pow(2, i), 1);
                recursive_method <<< dimgrid, dimblockrec >>> (a_d, N, b_d, N2, bx, by);	
                cudaThreadSynchronize();
                by += bm*pow(2,i);
            }
        }
        cudaEventRecord(stop,0);
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsedTime, start, stop); // that's our time!
        time = elapsedTime/(float)MEASURES;
        accum += time;
        squared_accum += time*time/((float)REPETITIONS-1.0f);
        if( j%100 == 0){ printf("."); fflush(stdout);}
	}
	cudaEventDestroy(start);
	cudaEventDestroy(stop);
    mean = accum/(float)REPETITIONS;
    stdev = sqrtf(squared_accum - accum*accum/div);
	printf("\nAverage of %i averages(%i):\ncudaEventElapsedTime:\tmean=%f[ms]    stdev=%f    error=%f%%\n", 
    REPETITIONS, MEASURES, mean, stdev, 100.0f*stdev/mean);
	cudaError_t error = cudaGetLastError();
	if(error != cudaSuccess){
		// print the CUDA error message and exit
		printf("CUDA error: %s\n", cudaGetErrorString(error));
		exit(-1);
	}
	else{
		printf("OK!\n");
    }
    
	cudaMemcpy(b_h, b_d, sizeb, cudaMemcpyDeviceToHost);
    
    FILE *fa;
    fa = fopen("recursive_results.dat", "a");
    fprintf(fa, "%i        %f       %f\n", N, mean, stdev/mean);
    fclose(fa);
    

}


void print_results(int *b, int n){
	for(int i=0; i<n; i++)
		printf("b[%d] = %i\n", i, b[i]);
}

void print_spheres(float4 *a, int n){
	for(int i=0; i<n; i++)
		printf("a[%i] = (%f, %f, %f, %f)\n", i, a[i].x, a[i].y, a[i].z, a[i].w);
}

// main routine that executes on the host
int main(int argc, char **argv){
	//srand ( time(NULL) );
	if( argc < 2 ){
		printf("arguments must be: <N> <method: 0 (bb), 1 (td)>\n");
		exit(1);
	}

    float4 *a_h, *a_d;
	int *b_h, *b_d;
	const int N = atoi(argv[1]);
    const int N2 = N*(N+1)/2;
	const int Na = N*(N-1)/2;
	const size_t size = N * sizeof(float4);
	const size_t sizeb = sizeof(int)*N2;
	//size_t sizeb = sizeof(float)*N;
	
	
	printf("doing a td-problem of:\nN=%i\nN2=%i\n", N, N2);
	a_h = (float4*)malloc(size);
	b_h = (int*)malloc(sizeb);
    
    for(int i=0; i<N2; i++)
        b_h[i] = 0;

	fill_random(a_h, N);
	//print_spheres(a_h, N);

	cudaMalloc((void **) &a_d, size);
	cudaMalloc((void **) &b_d, sizeb);
    cudaMemcpy(b_d, b_h, sizeb, cudaMemcpyHostToDevice);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);
	
	dim3 dimblock(BLOCKSIZE, 1, 1);
	dim3 dimblockrec(BLOCKSIZE_REC, 1, 1);
	dim3 dimblock2(BLOCKSIZE_AVRIl, 1, 1);
	dim3 block2d(BLOCKSIZE, BLOCKSIZE, 1);
	int sn2 = (Na+dimblock2.x-1)/dimblock2.x;
	int s2 = ceil(sqrt((double)sn2));

    dim3 dimgrid2(s2, s2, 1);

	dim3 dimgrid1((N+dimblock.x-1)/dimblock.x, (N+dimblock.x -1)/dimblock.x, 1);
	//dim3 dimgrid1((N+block2d.x-1)/block2d.x, (N+block2d.y -1)/block2d.y, 1);		

	int sn = (N+dimblock.x-1)/dimblock.x;
	int sd = sn*(sn+1)/2;
	int s = ceil(sqrt((double)sd));
   	dim3 dimgrid3(s, s, 1);

    int rect_evenx = N/2;
    int rect_oddx = (int)ceil((float)N/2.0f);
    dim3 dgrecteven((rect_evenx+dimblock.x-1)/dimblock.x, ((N+1)+dimblock.x-1)/dimblock.x, 1);
    dim3 dgrectodd((rect_oddx+dimblock.x-1)/dimblock.x, (N+dimblock.x-1)/dimblock.x, 1);
	printf("\n");
	// CPU computation
    if(atoi(argv[2])==0){
		struct timespec start, end;
        printf("cpu computation......."); fflush(stdout);
		clock_gettime(CLOCK_MONOTONIC, &start);
        for(int i=0; i<N; i++){
            for(int j=0; j<i; j++){
                float distance = sqrtf( (a_h[i].x - a_h[j].x)*(a_h[i].x - a_h[j].x) );
                if( distance < a_h[i].w + a_h[j].w ){
                    int pos = ((i*(i+1)) >> 1) + j;
                    b_h[pos] = 1;
                }
            }
        }
		clock_gettime(CLOCK_MONOTONIC, &end);
        printf("ok: %fs\n", ((float)end.tv_sec - (float)start.tv_sec) + ( (float)end.tv_nsec - (float)start.tv_nsec)/1000000000.0f); fflush(stdout);
        if(argc >= 4){
            export_result(b_h, N2, argv[3]);
		}
        exit(1);
    }
	// GPU computation BB method
    else if(atoi(argv[2])==1){
	    bb_computation(b_h, sizeb, dimgrid1, dimblock, a_d, N, b_d, N2);
		//bb_computation(b_h, sizeb, dimgrid1, block2d, a_d, N, b_d, N2);
        if(argc >= 4)
            export_result(b_h, N2, argv[3]);
    }
	else if(atoi(argv[2])==2){
	    avril_map(b_h, sizeb, dimgrid2, dimblock2, a_d, N, b_d, Na);
        if(argc >= 4)
            export_result(b_h, N2, argv[3]);
    }
    else if(atoi(argv[2])==3){
        if(N%2==0)
            td_computation_rectangle(b_h, sizeb, dgrecteven, dimblock, a_d, N, b_d, N2);
        else
            td_computation_rectangle(b_h, sizeb, dgrectodd, dimblock, a_d, N, b_d, N2);

        if(argc >= 4)
            export_result(b_h, N2, argv[3]);
    }
    else if(atoi(argv[2])==4){
        td_computation_rsqrt(b_h, sizeb, dimgrid3, dimblock, a_d, N, b_d, N2);
        if(argc >= 4)
            export_result(b_h, N2, argv[3]);

    }
    else if(atoi(argv[2])==5){
        int n=atoi(argv[3]);
        int m=N/n;
        if( (m % dimblockrec.x) != 0 ){
            fprintf(stderr, "error: m=%i, not a multiple of %i\n", m, dimblockrec.x);
            exit(1);
        }
        int k=cf_log2i(n);
        printf("N(dim)=%i  n=%i  m=%i   k=%i\n", N, n, m, k);
        td_computation_recursive(b_h, sizeb, dimblockrec, a_d, N, b_d, N2, m, k);
        if(argc >= 5){
            export_result(b_h, N2, argv[4]);
        }
    }
    printf("freeing memory.......");fflush(stdout);
	free(a_h); 
    free(b_h);
	cudaFree(a_d);
	cudaFree(b_d);
    printf("ok\n");
}

