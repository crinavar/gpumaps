#ifndef KERNELS_CUH
#define KERNELS_CUH

#define SQRT_COEF 0.0001f
#define BB 2000
#define OFFSET -0.4999f
//#define OFFSET 0.5f

// rules
#define EL 2
#define EU 3
#define FL 3
#define FU 3

#define CLEN ((BSIZE2D)+2)
#define CSPACE (CLEN*CLEN)


#define CINDEX(x, y)     ((1+(y))*CLEN + (1+(x)))
#define GINDEX(x, y, n)  ((y)*(n) + (x))

__device__ __inline__ void set_halo(MTYPE *cache, MTYPE *mat, unsigned int N, uint3 lp, int2 p){
    // FREE BOUNDARY CONDITIONS
    // left side
    if(lp.x == 0){
        cache[CINDEX(lp.x - 1, lp.y)]      = p.x == 0 ? 0 : mat[ GINDEX(p.x-1, p.y, N) ];
        // bot-left corner
        if(lp.y == BSIZE2D-1){
            cache[CINDEX(lp.x - 1, lp.y + 1)]      = (p.x == 0 || p.y == N-1) ? 0 : mat[ GINDEX(p.x-1, p.y+1, N) ];
        }
        // top-left corner
        if(lp.y == 0){
            cache[CINDEX(lp.x - 1, lp.y - 1)]      = (p.x == 0 || p.y == 0) ? 0 : mat[ GINDEX(p.x-1, p.y-1, N) ];
        }
    }
    // right side
    if(lp.x == BSIZE2D-1){
        cache[CINDEX(lp.x + 1, lp.y)]      = p.x == N-1 ? 0 : mat[ GINDEX(p.x+1, p.y, N) ];
        // bot-right corner
        if(lp.y == BSIZE2D-1){
            cache[CINDEX(lp.x + 1, lp.y + 1)]      = (p.x == N-1 || p.y == N-1) ? 0 : mat[ GINDEX(p.x+1, p.y+1, N) ];
        }
        // top-right corner
        if(lp.y == 0){
            cache[CINDEX(lp.x + 1, lp.y - 1)]      = (p.x == N-1 || p.y == 0) ? 0 : mat[ GINDEX(p.x+1, p.y-1, N) ];
        }
    }
    // bottom side
    if(lp.y == BSIZE2D-1){
        cache[CINDEX(lp.x, lp.y + 1)]      = p.y == N-1 ? 0 : mat[ GINDEX(p.x, p.y+1, N) ];
    }
    // top side
    if(lp.y == 0){
        cache[CINDEX(lp.x, lp.y - 1)]      = p.y == 0 ? 0 : mat[ GINDEX(p.x, p.y-1, N) ];
    }
}

__device__ __inline__ void load_cache(MTYPE *cache, MTYPE *mat, unsigned int n, uint3 lp, int2 p){
    // loading the thread's element
	if(p.x > n-1 || p.y > n-1){return;}
    cache[CINDEX(lp.x, lp.y)] = mat[GINDEX(p.x, p.y, n)];
    // loading the cache's halo
    set_halo(cache, mat, n, lp, p);
    __syncthreads();
}

__device__ inline int h(int k, int a, int b){
    return (1 - (((k - a) >> 31) & 0x1)) * (1 - (((b - k) >> 31) & 0x1));
}

__device__ void work_cache(DTYPE *data, MTYPE *mat1, MTYPE *mat2, MTYPE *cache, uint3 lp, int2 p, int n){
    // neighborhood count 
    int nc =    
                cache[CINDEX(lp.x-1, lp.y-1)] + cache[CINDEX(lp.x, lp.y-1)] + cache[CINDEX(lp.x+1, lp.y-1)] + 
                cache[CINDEX(lp.x-1, lp.y  )] +                   0             + cache[CINDEX(lp.x+1, lp.y  )] + 
                cache[CINDEX(lp.x-1, lp.y+1)] + cache[CINDEX(lp.x, lp.y+1)] + cache[CINDEX(lp.x+1, lp.y+1)];
   unsigned int c = cache[CINDEX(lp.x, lp.y)];
    /* 
   if(c == 1){
       printf("cell (%i,%i):\n %i %i %i\n %i %i %i\n %i %i %i\n\n", p.x, p.y, 
                cache[CINDEX(lp.x-1, lp.y-1)], cache[CINDEX(lp.x, lp.y-1)], cache[CINDEX(lp.x+1, lp.y-1)],
                cache[CINDEX(lp.x-1, lp.y  )],                   c            , cache[CINDEX(lp.x+1, lp.y  )],
                cache[CINDEX(lp.x-1, lp.y+1)], cache[CINDEX(lp.x, lp.y+1)], cache[CINDEX(lp.x+1, lp.y+1)]);
   }
   */
   // transition function applied to state 'c' and written into mat2
   mat2[GINDEX(p.x, p.y, n)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
}

__device__ void work_nocache(DTYPE *data, MTYPE *mat1, MTYPE *mat2, int2 p, int n){
    // left
    int nc = 0;
    if(p.x > 0){
        nc += mat1[GINDEX(p.x-1, p.y, n)];
        // top left corner
        if(p.y > 0){
            nc += mat1[GINDEX(p.x-1, p.y-1, n)];
        }
        // bot left corner
        if(p.y < n-1){
            nc += mat1[GINDEX(p.x-1, p.y+1, n)];
        }
    }
    // right
    if(p.x < n-1){
        nc += mat1[GINDEX(p.x+1, p.y, n)];
        // top right corner
        if(p.y > 0){
            nc += mat1[GINDEX(p.x+1, p.y-1, n)];
        }
        // bot right corner
        if(p.y < n-1){
            nc += mat1[GINDEX(p.x+1, p.y+1, n)];
        }
    }
    // top
    if(p.y > 0){
        nc += mat1[GINDEX(p.x, p.y-1, n)];
    }
    // bottom
    if(p.y < n-1){
        nc += mat1[GINDEX(p.x, p.y+1, n)];
    }
    unsigned int c = mat1[GINDEX(p.x, p.y, n)];
    // transition function applied to state 'c' and written into mat2
    mat2[GINDEX(p.x, p.y, n)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
}

// metodo kernel cached test
template<typename Lambda>
__global__ void kernel_test(const unsigned int n, const unsigned int msize, DTYPE *data, MTYPE *dmat1, MTYPE *dmat2, Lambda map, unsigned int aux1, unsigned int aux2){
    __shared__ MTYPE cache[CSPACE];
	auto p = map(n, msize, aux1, aux2);
    if(p.x != -1){
        load_cache(cache, dmat1, n, threadIdx, p);
    }
    if(p.x < p.y && p.y < n){
        work_cache(data, dmat1, dmat2, cache, threadIdx, p, n);
    }
}

// metodo kernel cached test
template<typename Lambda>
__global__ void kernel_test_avril(const unsigned int n,const unsigned int msize,DTYPE *data,MTYPE *dmat1, MTYPE *dmat2,Lambda map,unsigned int aux1,unsigned int aux2){
	auto p = map(n, msize, aux1, aux2);
    if(p.x < p.y && p.y < n){
        work_nocache(data, dmat1, dmat2, p, n);
    }
}

// metodo kernel cached test
template<typename Lambda>
__global__ void kernel_test_rectangle(const unsigned int n,const unsigned int msize,DTYPE *data,MTYPE *dmat1, MTYPE *dmat2,Lambda map,unsigned int aux1,unsigned int aux2){
    // map
	auto p = map(n, msize, aux1, aux2);
    // cache
    __shared__ MTYPE cache[CSPACE];
    // mixed diagonal - no cache
    if(blockIdx.x == blockIdx.y){
        if(p.x < p.y && p.y < n){
            work_nocache(data, dmat1, dmat2, p, n);
        }
    }
    else if(blockIdx.x < blockIdx.y){
        // lower triangular - standard cache
        load_cache(cache, dmat1, n, threadIdx, p);
        if(p.x < p.y && p.y < n){
            work_cache(data, dmat1, dmat2, cache, threadIdx, p, n);
        }
    }
    else{
        // upper triangular - inverted cache
        uint3 invlp = (uint3){BSIZE2D-1 - threadIdx.x, BSIZE2D-1 - threadIdx.y, 0};
        load_cache(cache, dmat1, n, invlp, p);
        if(p.x < p.y && p.y < n){
            work_cache(data, dmat1, dmat2, cache, invlp, p, n);
        }
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
    
    __shared__ int2 b;
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
