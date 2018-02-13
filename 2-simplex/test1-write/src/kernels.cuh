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
#ifndef KERNELS_CUH
#define KERNELS_CUH

#define SQRT_COEF 0.0001f
#define BB 2000
#define OFFSET -0.4999f
//#define OFFSET 0.5f

__device__ void work(DTYPE *data, MTYPE *mat, uint2 p, int n){
    mat[p.y*n + p.x] = 1;
    
    /*
    const int b = (int)log2f(blockIdx.y-1);
    if(blockIdx.y < 2){
        mat[p.y*n + p.x] = 9;
    }
    else{
        mat[p.y*n + p.x] = b;
    }
    */
}

// metodo kernel test
template<typename Lambda>
__global__ void kernel_test(const unsigned int n, const unsigned int msize, DTYPE *data, MTYPE* dmat, Lambda map, unsigned int aux1, unsigned int aux2){
    auto p = map(n, msize, aux1, aux2); 
    /*
    if(p.y > n){
        printf("[ALERT] p(%i, %i) > %i \n", p.x, p.y, n);
    }
    */
    if(p.y >= p.x && p.y < n){
    //if(p.y < n){
        work(data, dmat, p, n);
    }
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


// metodo td con grid lineal (Avril et. al.)
// este metodo no es muy bueno, la raiz requiere mas precision sino falla muy luego
__global__ void td_method(float *data, int N, float* bdata, int N2){
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
        //work(data, bdata, a, b, N);
	}
}






// metodo td con square root FP32
__global__ void td_method_2dx(float *a, int N, float* b, int N2){
    unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
    //unsigned int bi = __fadd_rn(sqrtf(__fmaf_rn(2.0f, (float)bc, 0.25f)), -0.5f);
    unsigned int bi = sqrtf(2.0f*(float)bc + 0.25f) - 0.5f;
	unsigned int bj = bc - (bi*(bi+1) >> 1);
    
    int i = bi * blockDim.y + threadIdx.y;
	int j = bj * blockDim.x + threadIdx.x;
    if(i>=j){
        //work(a, b, i, j, N);
    }
}
// metodo td con inversa square root
__global__ void td_method_2dr(float *data, int N, float* mat, int N2){
    
    __shared__ uint2 b;
    if( threadIdx.y == 0 ){
        unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
        float arg = __fmaf_rn(2.0f, (float)bc, 0.25f);
        //float arg = 2.0f*(float)bc + 0.25f;
        b.y = __fmaf_rn(arg, rsqrtf(arg), OFFSET);// + 0.001f;
        //bi = arg*rsqrtf(arg) + OFFSET;
        b.x = bc - ((b.y*(b.y+1)) >> 1);
    }
    __syncthreads();
    uint2 t = {threadIdx.x + b.x*blockDim.x, threadIdx.y + b.y * blockDim.y};
    
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
    if(t.y >= t.x){
        //work(data, mat, t.y, t.x, N);
    }
}


// metodo rectangle
__global__ void rect_method(float *a, int N, float *b, int N2){

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

    //work(a, b, i, j, N);
}


// metodo recursivo
__global__ void recursive_method(float *a, int N, float *b, int N2, int bx, int by){

    // calcula el indice del bloque recursivo, es division entera.
    //int rec_index = blockIdx.x/gridDim.y;
    
    // calcula el offset, el valor x del bloque respecto al recblock que le corresponde.
    //int dbx = blockIdx.x % gridDim.y;
    //int tx = (bx+(gridDim.y*rec_index*2) + dbx)*blockDim.x + threadIdx.x;
    //int ty = (by+(gridDim.y*rec_index*2) + blockIdx.y)*blockDim.y + threadIdx.y;

    //b[0] = __fadd_rn(ty, tx);
    //work(a, b, ty, tx, N);
}

__global__ void recursive_diagonal(float *a, int N, float *b, int N2){

    // calcula el indice del bloque recursivo, es division entera.
    int rec_index = blockIdx.x/gridDim.y;
    int tx = blockIdx.x * blockDim.x + threadIdx.x;
    int ty = (blockIdx.y + rec_index*gridDim.y)*blockDim.y + threadIdx.y;
    if( tx > ty )
        return;
    
    //b[0] = __fadd_rn(ty, tx);
    //work(a, b, ty, tx, N);
    
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

#endif
