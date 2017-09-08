#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include "custom_funcs.h"

#define SQRT_COEF 0.0001f
#define REPETITIONS 100
#define MEASURES 10
#define BB 2000
#define OFFSET -0.4999f
#define BSIZEX 16
#define BSIZEY 16
//#define OFFSET 0.5f

__device__ double cost_function(float4 *data, int i, int j){
    return 1.0f;
}


void export_result(float *b, int n, const char *filename){
    printf("exporting result vector.......");fflush(stdout);
    FILE *fw = fopen(filename, "w");
    if(!fw){
        fprintf(stderr, "error: cannot write to file %s\n", filename);
        exit(1);
    }
    for(int i=0; i<n; i++)
        fprintf(fw, "%f\n", b[i]);
    printf("ok\n");
}


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



__device__ inline float newton_sqrtf(const float number) {
    int i;
    float x,y;
    //const float f = 1.5F;
    x = number * 0.5f;
    i  = * ( int * ) &number;
    i  = 0x5f3759df - ( i >> 1 );
    y  = * ( float * ) &i;
    y  *= (1.5f -  x * y * y);
    y  *= (1.5f -  x * y * y); 
    y  *= (1.5f -  x * y * y); 
    return number * y;
}

__device__ inline float newton1_sqrtf(const float number){

    int i;
    float x,y;
    //const float f = 1.5F;
    x = number * 0.5f;
    i  = * ( int * ) &number;
    i  = 0x5f3759df - ( i >> 1 );
    y  = * ( float * ) &i;
    y *= (1.5f -  x * y * y);
    y *= number; //obteniendo resultado
    //arreglar
    if( (y+1.0f)*(y+1.0f) < number)
        return y;
    else
        return y-0.5f;
}

__global__ void kernel_work(int bi, int bj, float4 *a, int N, float *b, int N2){
    int i = bj * blockDim.x + threadIdx.x;
    int j = bi * blockDim.y + threadIdx.y;
    
    if(i >= j){
        b[0] = i+j;
    }
}

// block map with dynamic parallelism
__global__ void kernel_coarse_map(float4 *a, int blockN, int N, float *b, int N2){
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int idy = blockIdx.y * blockDim.y + threadIdx.y;
    if( idx >= blockN || idy >= blockN )
        return;

    //cudaStream_t s;
    //cudaStreamCreateWithFlags(&s, cudaStreamNonBlocking);

    unsigned int bc = idy*gridDim.x*blockDim.x + idx;
    unsigned int bi = sqrtf(2.0f*(float)bc + 0.25f) - 0.5f;
    unsigned int bj = bc - (bi*(bi+1) >> 1);
    
    dim3 workgrid(1,1,1);
    dim3 workblock(blockDim.x, blockDim.y, 1);
    //cudaStream_t s = bc;
    //kernel_work <<< workgrid, workblock, 0, s >>> (bi, bj, a, N, b, N2);
    
}



// metodo newton
__global__ void td_method_2dn(float4 *a, int N, float* b, int N2){
	unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
	unsigned int bi = __fadd_rn(newton_sqrtf(__fmaf_rn(2.0f, (float)bc, 0.25f)), OFFSET);
	unsigned int bj = (bc - (bi*(bi+1) >> 1));

	int i = bi * blockDim.y + threadIdx.y;
	int j = bj * blockDim.x + threadIdx.x;
    if(i>= j){
        b[0] = i + j;
    }
}

// metodo bounding box
__global__ void bb_method(float4 *a, int N, float* b, int N2){
	if( blockIdx.x > blockIdx.y )
		return;
	int i = blockIdx.y*blockDim.y + threadIdx.y;
	int j = blockIdx.x*blockDim.x + threadIdx.x;
	//if(i>=j){
	  //   b[0] = __fadd_rn(i, j);
    //}
    if(i>=j)
        b[0] = i+j;
}


// metodo td con grid lineal
// este m��todo no es muy bueno, la raiz requiere precision double sino falla muy luego
__global__ void td_method(float4 *data, int N, float* bdata, int N2){
	int k = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
    if(k<N2){
		//int a = __fadd_rn((float)N, __fadd_rn(0.5f, - (carmack_sqrtf(__fmaf_rn((float)N, (float)N, -(float)N) + __fmaf_rn(2.0f, (float)k, 0.25f)))));
	    //int b = (a+1) + k - (((a-1)*(2*N-a)) >> 1);  
        int a = (-(2.0f*(float)N + 1.0f) + carmack_sqrtf(4.0f*(float)N*(float)N - 4.0f*(float)N - 8.0f*(float)k + 1.0f))/(-2.0f);
        int b = (a+1) + k - ((a-1)*(2*N-a))/2;
        if(b > N){
            a++;
            b = a+(b-N);
        }
        if(a >= b){
            b = N - (a-b);
            a--;
        }
        //printf("writing into %i %i\n", a, b);
        //bdata[0] = __fadd_rn(a,b);
        bdata[0] = a+b;
	}
}






// metodo td con square root FP32
__global__ void td_method_2dx(float4 *a, int N, float* b, int N2){
    unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
    //unsigned int bi = __fadd_rn(sqrtf(__fmaf_rn(2.0f, (float)bc, 0.25f)), -0.5f);
    unsigned int bi = sqrtf(2.0f*(float)bc + 0.25f) - 0.5f;
	unsigned int bj = bc - (bi*(bi+1) >> 1);
    
    int i = bi * blockDim.y + threadIdx.y;
	int j = bj * blockDim.x + threadIdx.x;
    if(i>=j){
        //b[c] = cost_function(a, i, j);
        b[0] = i+j;
    }
}
// metodo td con inversa square root
__global__ void td_method_2dr(float4 *a, int N, float* b, int N2){
    
    __shared__ unsigned int bi[BSIZEX], bj[BSIZEX];
    unsigned int i = threadIdx.y;
	unsigned int j = threadIdx.x;
    if( threadIdx.y == 0 ){
        unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
        float arg = __fmaf_rn(2.0f, (float)bc, 0.25f);
        //float arg = 2.0f*(float)bc + 0.25f;
        bi[i] = __fmaf_rn(arg, rsqrtf(arg), OFFSET);// + 0.001f;
        //bi = arg*rsqrtf(arg) + OFFSET;
        bj[i] = bc - (bi[i]*(bi[i]+1) >> 1);
    }
    //__syncthreads();
    i += bi[0] * blockDim.y;
	j += bj[0] * blockDim.x;
    
   
    /*
    unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
    float arg = __fmaf_rn(2.0f, (float)bc, 0.25f);
    unsigned int bi = __fmaf_rn(arg, rsqrtf(arg), OFFSET);// + 0.001f;
    unsigned int bj = bc - (bi*(bi+1) >> 1);
    
    unsigned int i = bi * blockDim.y + threadIdx.y;
	unsigned int j = bj * blockDim.x + threadIdx.x;
    */
    //if(i>=j){
      //   b[0] = __fadd_rn(i, j);
    //}
    if(i>=j)
        b[0] = i+j;
}


// metodo rectangle
__global__ void rect_method(float4 *a, int N, float *b, int N2){

    int i = blockIdx.y * blockDim.y + threadIdx.y;
    int j = blockIdx.x * blockDim.x + threadIdx.x;

    if(i >= N+1 || j >= N/2)
        return;

    if( j >= i ){
        j = N - j -1;
        i = N - i -1;
    }
    else
        i = i-1;

    // b[0] = __fadd_rn(i, j);
    b[0] = i+j;
}


// metodo recursivo
__global__ void recursive_method(float4 *a, int N, float *b, int N2, int bx, int by){

    // calcula el indice del bloque recursivo, es division entera.
    int rec_index = blockIdx.x/gridDim.y;
    
    // calcula el offset, el valor x del bloque respecto al recblock que le corresponde.
    int dbx = blockIdx.x % gridDim.y;
    int tx = (bx+(gridDim.y*rec_index*2) + dbx)*blockDim.x + threadIdx.x;
    int ty = (by+(gridDim.y*rec_index*2) + blockIdx.y)*blockDim.y + threadIdx.y;

    //b[0] = __fadd_rn(ty, tx);
    b[0] = ty+tx;
}

__global__ void recursive_diagonal(float4 *a, int N, float *b, int N2){

    // calcula el indice del bloque recursivo, es division entera.
    int rec_index = blockIdx.x/gridDim.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = (blockIdx.y + rec_index*gridDim.y)*blockDim.y + threadIdx.y;
    if( tx > ty )
        return;
    
    //b[0] = __fadd_rn(ty, tx);
    b[0] = ty+tx;
    
}

int* load_table(const char *filename, int *n){
    int *table;
    FILE *fr = fopen(filename, "rb");
    if(!fr){
        printf("error loading table.... does file '%s' exist?\n", filename);
        exit(-1);
    }
    fread(n, sizeof(int), 1, fr);
    table = (int*)malloc((*n)*sizeof(int));
    fread(table, sizeof(int), *n, fr);
    return table;
}



void fill_random(float4 *array, int n){
	for(int i=0; i<n; i++){
		array[i].x = 100.0f * (float)rand()/(float)RAND_MAX;
        array[i].y = 100.0f * (float)rand()/(float)RAND_MAX;
        array[i].z = 100.0f * (float)rand()/(float)RAND_MAX;
        array[i].w = 100.0f * (float)rand()/(float)RAND_MAX;
    }
} 

void td_computationn(float *b_h, size_t sizeb, dim3 dimgrid, dim3 dimblock, float4 *a_d, int N, float *b_d, int N2){

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling kernel td_method_2dn....block= %i x %i x %i    grid = %i x %i x %i\n", dimblock.x, dimblock.y, dimblock.z, dimgrid.x, dimgrid.y, dimgrid.z);
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
        for(int k=0; k<MEASURES; k++){
            td_method_2dn <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
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
    stdev = sqrtf(squared_accum - (accum*accum)/div);
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
    fa = fopen("newton_results.dat", "a");
    fprintf(fa, "%i       %f      %f\n", N, mean, stdev/mean);
    fclose(fa);
}


void bb_computation(float *b_h, size_t sizeb, dim3 dimgrid, dim3 dimblock, float4 *a_d, int N, float *b_d, int N2){

    //printf("entering bb...\n");
	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling kernel bb_method....block= %i x %i x %i    grid = %i x %i x %i\n", dimblock.x, dimblock.y, dimblock.z, dimgrid.x, dimgrid.y, dimgrid.z);
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
void td_computationx(float *b_h, size_t sizeb, dim3 dimgrid, dim3 dimblock, float4 *a_d, int N, float *b_d, int N2){

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling kernel td_method_2dx....block= %i x %i x %i    grid = %i x %i x %i\n", dimblock.x, dimblock.y, dimblock.z, dimgrid.x, dimgrid.y, dimgrid.z);
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
        for(int k=0; k<MEASURES; k++){
            td_method_2dx <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
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
    fa = fopen("exactsqrt_results.dat", "a");
    fprintf(fa, "%i        %f         %f\n", N, mean, stdev/mean);
    fclose(fa);
}

void td_computation_rsqrt(float *b_h, size_t sizeb, dim3 dimgrid, dim3 dimblock, float4 *a_d, int N, float *b_d, int N2){
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
    float accum=0.0f, elapsedTime=0.0f, squared_accum=0.0f, mean=0.0f, time=0.0f, stdev=0.0f, div=(float)(REPETITIONS*(REPETITIONS-1.0f));
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

void td_computation_rectangle(float *b_h, size_t sizeb, dim3 dimgrid, dim3 dimblock, float4 *a_d, int N, float *b_d, int N2){
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling kernel rect_method....block= %i x %i x %i    grid = %i x %i x %i\n", dimblock.x, dimblock.y, dimblock.z, dimgrid.x, dimgrid.y, dimgrid.z);
    //warmup
    printf("warmup...."); fflush(stdout);
	for(int i=0; i<MEASURES; i++){
    	rect_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
        cudaThreadSynchronize();
    }
    printf("ok\nmeasuring mean time of %i averages (each averages is computed from %i measures).......\n", REPETITIONS, MEASURES); fflush(stdout);
    float accum=0.0f, elapsedTime=0.0f, squared_accum=0.0f, mean=0.0f, time=0.0f, stdev=0.0f, div=(float)(REPETITIONS*(REPETITIONS-1));
    for(int j=0; j<REPETITIONS; j++){
        cudaEventRecord(start, 0);	
        for(int k=0; k<MEASURES; k++){
            rect_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
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

void td_computation_flatrec(float *b_h, size_t sizeb, dim3 dimgrid, dim3 dimblock, float4 *a_d, int N, float *b_d, int N2){
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling kernel rect_flatrec....block= %i x %i x %i    grid = %i x %i x %i\n", dimblock.x, dimblock.y, dimblock.z, dimgrid.x, dimgrid.y, dimgrid.z);
    //warmup
    printf("warmup...."); fflush(stdout);
	for(int i=0; i<MEASURES; i++){
    	rect_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
        cudaThreadSynchronize();
    }
    printf("ok\nmeasuring mean time of %i averages (each averages is computed from %i measures).......\n", REPETITIONS, MEASURES); fflush(stdout);
    float accum=0.0f, elapsedTime=0.0f, squared_accum=0.0f, mean=0.0f, time=0.0f, stdev=0.0f, div=(float)(REPETITIONS*(REPETITIONS-1));
    for(int j=0; j<REPETITIONS; j++){
        cudaEventRecord(start, 0);	
        for(int k=0; k<MEASURES; k++){
            rect_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
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
void td_computation_simple(float *b_h, size_t sizeb, dim3 dimgrid, dim3 dimblock, float4 *a_d, int N, float *b_d, int N2){
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling kernel avril method....block= %i x %i x %i    grid = %i x %i x %i\n", dimblock.x, dimblock.y, dimblock.z, dimgrid.x, dimgrid.y, dimgrid.z);
    //warmup
    printf("warmup...."); fflush(stdout);
	for(int i=0; i<MEASURES; i++){
    	td_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
        cudaThreadSynchronize();
    }
    printf("ok\nmeasuring mean time of %i averages (each averages is computed from %i measures).......\n", REPETITIONS, MEASURES); fflush(stdout);
    float accum=0.0f, elapsedTime=0.0f, squared_accum=0.0f, mean=0.0f, time=0.0f, stdev=0.0f, div=(float)(REPETITIONS*(REPETITIONS-1));
    for(int j=0; j<REPETITIONS; j++){
        cudaEventRecord(start, 0);	
        for(int k=0; k<MEASURES; k++){
            td_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2);	
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
    fa = fopen("avril_results.dat", "a");
    fprintf(fa, "%i        %f       %f\n", N, mean, stdev/mean);
    fclose(fa);

}

void td_computation_recursive(float *b_h, size_t sizeb, dim3 dimblock, float4 *a_d, int N, float *b_d, int N2, int m, int kval){
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling kernel td_method_recursive....block= %i x %i x %i\n", dimblock.x, dimblock.y, dimblock.z);
    //warmup
    printf("warmup...."); fflush(stdout);
    int bx=0, by=0;
    int bm = m/dimblock.x;
	for(int r=0; r<MEASURES; r++){
        by = 0;
        dim3 dg_diag(N/dimblock.x, bm, 1);
        recursive_diagonal <<< dg_diag, dimblock >>> (a_d, N, b_d, N2);
        by += bm;
        for(int i=0; i<kval; i++){
            dim3 dimgrid(N/(dimblock.x*2), bm*pow(2, i), 1);
            recursive_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2, bx, by);	
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
            dim3 dg_diag(N/dimblock.x, bm, 1);
            recursive_diagonal <<< dg_diag, dimblock >>> (a_d, N, b_d, N2);
            by += bm;
            for(int i=0; i<kval; i++){
                dim3 dimgrid(N/(dimblock.x*2), bm*pow(2, i), 1);
                recursive_method <<< dimgrid, dimblock >>> (a_d, N, b_d, N2, bx, by);	
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

void coarse_strategy(float *b_h, size_t sizeb, dim3 dimgrid, dim3 dimblock, float4 *a_d, int N, float *b_d, int N2, int blockN){
    cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	// Start record
	printf("calling coarse map....block= %i x %i x %i    grid = %i x %i x %i\n", dimblock.x, dimblock.y, dimblock.z, dimgrid.x, dimgrid.y, dimgrid.z);
    //warmup
    printf("warmup...."); fflush(stdout);
	for(int i=0; i<MEASURES; i++){
    	kernel_coarse_map <<< dimgrid, dimblock >>> (a_d, blockN, N, b_d, N2);	
        cudaThreadSynchronize();
    }
    printf("ok\nmeasuring mean time of %i averages (each averages is computed from %i measures).......\n", REPETITIONS, MEASURES); fflush(stdout);
    float accum=0.0f, elapsedTime=0.0f, squared_accum=0.0f, mean=0.0f, time=0.0f, stdev=0.0f, div=(float)(REPETITIONS*(REPETITIONS-1.0f));
    for(int j=0; j<REPETITIONS; j++){
        cudaEventRecord(start, 0);	
        for(int k=0; k<MEASURES; k++){
            kernel_coarse_map <<< dimgrid, dimblock >>> (a_d, blockN, N, b_d, N2);	
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
    fa = fopen("coarsemap_results.dat", "a");
    fprintf(fa, "%i        %f       %f\n", N, mean, stdev/mean);
    fclose(fa);

}


void print_results(float *b, int n){
	for(int i=0; i<n; i++)
		printf("b[%d] = %f\n", i, b[i]);
}

// main routine that executes on the host
int main(int argc, char **argv){
	//srand ( time(NULL) );
	if( argc < 2 ){
		printf("arguments must be: <N> <method: 0 (bb), 1 (td)>\n");
		exit(1);
	}

    float4 *a_h, *a_d;
	float *b_h, *b_d;
	const int N = atoi(argv[1]);
	const int N2 = N*(N+1)/2;
    const int Na = N*(N-1)/2;
	size_t size = N * sizeof(float4);
	size_t sizeb = sizeof(float)*N2;
	
	
	printf("doing a td-problem of:\nN=%i\nN2=%i\n", N, N2);
	a_h = (float4*)malloc(size);
	b_h = (float*)malloc(sizeb);
    for(int i=0; i<N2; i++)
        b_h[i] = -1.0f;


	fill_random(a_h, N);
	cudaMalloc((void **) &a_d, size);
	cudaMalloc((void **) &b_d, sizeb);
    cudaMemcpy(b_d, b_h, sizeb, cudaMemcpyHostToDevice);
	cudaMemcpy(a_d, a_h, size, cudaMemcpyHostToDevice);

	
	dim3 dimblock1(BSIZEX, BSIZEY, 1);
	dim3 dimgrid1(N/dimblock1.x + (N%dimblock1.x == 0 ? 0:1), N/dimblock1.y + (N%dimblock1.y == 0 ? 0:1), 1);	

	dim3 dimblock2(256, 1, 1);
	int sn2 = (Na+dimblock2.x-1)/dimblock2.x;
	int s2 = ceil(sqrt((double)sn2));
    dim3 dimgrid2(s2, s2, 1);
	dim3 dimblock3(BSIZEX, BSIZEY, 1);
	dim3 dimblock5(BSIZEX, BSIZEY, 1);
	int sn = (N+dimblock3.x-1)/dimblock3.x;
	int sd = sn*(sn+1)/2;
	int s = ceil(sqrt((double)sd));	
	//dim3 dimgrid3((int)sd, 1, 1);	
   	dim3 dimgrid3(s, s, 1);

    // coarse grid
    dim3 coarsegrid((dimgrid3.x + dimblock3.x - 1) / dimblock3.x, (dimgrid3.y + dimblock3.y - 1)/dimblock3.y, 1);

    int rect_evenx = N/2;
    int rect_oddx = (int)ceil((float)N/2.0f);
    dim3 dbrect(BSIZEX, BSIZEY, 1);
    dim3 dgrecteven((rect_evenx+dbrect.x-1)/dbrect.x, ((N+1)+dbrect.y-1)/dbrect.y, 1);
    dim3 dgrectodd((rect_oddx+dbrect.x-1)/dbrect.x, (N+dbrect.y-1)/dbrect.y, 1);
	printf("\n");
	//compare_results(b_h1, b_h2, b_h3, N2);
    if(atoi(argv[2])==1){
	    bb_computation(b_h, sizeb, dimgrid1, dimblock1, a_d, N, b_d, N2);
        if(argc >= 4)
            export_result(b_h, N2, argv[3]);
    }
    else if(atoi(argv[2])==2){
	    td_computation_simple(b_h, sizeb, dimgrid2, dimblock2, a_d, N, b_d, Na);
        if(argc >= 4)
            export_result(b_h, Na, argv[3]);
    }
    else if(atoi(argv[2])==3){
	    td_computationn(b_h, sizeb, dimgrid3, dimblock3, a_d, N, b_d, N2);
        if(argc >= 4)
            export_result(b_h, N2, argv[3]);
    }
    else if(atoi(argv[2])==4){
        td_computationx(b_h, sizeb, dimgrid3, dimblock3, a_d, N, b_d, N2);
        if(argc >= 4)
            export_result(b_h, N2, argv[3]);
    }
    else if(atoi(argv[2])==5){
        int nb = (N+dimblock5.x-1)/dimblock5.x;
        dim3 dgflatrec(nb/2, 2*nb, 1);
        td_computation_flatrec(b_h, sizeb, dimgrid3, dimblock3, a_d, N, b_d, N2);
        if(argc >= 4)
            export_result(b_h, N2, argv[3]);
    }
    else if(atoi(argv[2])==6){
        td_computation_rsqrt(b_h, sizeb, dimgrid3, dimblock3, a_d, N, b_d, N2);
        if(argc >= 4)
            export_result(b_h, N2, argv[3]);

    }
    else if(atoi(argv[2])==7){
        if(N%2==0)
            td_computation_rectangle(b_h, sizeb, dgrecteven, dbrect, a_d, N, b_d, N2);
        else
            td_computation_rectangle(b_h, sizeb, dgrectodd, dbrect, a_d, N, b_d, N2);

        if(argc >= 4)
            export_result(b_h, N2, argv[3]);
    }
    else if(atoi(argv[2])==8){
        int n=atoi(argv[3]);
        int m=N/n;
        if( (m % dimblock3.x) != 0 ){
            fprintf(stderr, "error: m=%i, not a multiple of %i\n", m, dimblock3.x);
            exit(1);
        }
        int k=cf_log2i(n);
        printf("N(dim)=%i  n=%i  m=%i\n", N, n, m);
        td_computation_recursive(b_h, sizeb, dimblock3, a_d, N, b_d, N2, m, k);
        if(argc >= 5){
            export_result(b_h, N2, argv[4]);
        }
    }
    else if(atoi(argv[2])==9){
        coarse_strategy(b_h, sizeb, coarsegrid, dimblock3, a_d, N, b_d, N2, dimgrid3.x);
        if(argc >= 4)
            export_result(b_h, N2, argv[3]);
    }
    printf("cleaning memory.......");fflush(stdout);
	free(a_h); 
    free(b_h);
	cudaFree(a_d);
	cudaFree(b_d);
    printf("ok\n");
}

