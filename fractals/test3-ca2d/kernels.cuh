#ifndef KERNELS_H
#define KERNELS_H

#define WARPSIZE 32

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
#define GINDEX(x, y, n)  ((1+((unsigned long)y))*(((unsigned long)n)+2)  + (1+(x)))

__device__ inline int h(int k, int a, int b){
    return (1 - (((k - a) >> 31) & 0x1)) * (1 - (((b - k) >> 31) & 0x1));
}

__device__ __inline__ void set_halo(MTYPE *cache, MTYPE *mat, unsigned long N, uint3 lp, uint2 p){
    // EMPANADA FANTASMA BOUNDARY CONDITIONS
    // left side
    if(lp.x == 0){
        cache[CINDEX(lp.x - 1, lp.y)]      = mat[ GINDEX(p.x-1, p.y, N) ];
    }
    // right side
    if(lp.x == BSIZE2D-1){
        cache[CINDEX(lp.x + 1, lp.y)]      = mat[ GINDEX(p.x+1, p.y, N) ];
    }
    // bottom side
    if(lp.y == BSIZE2D-1){
        cache[CINDEX(lp.x, lp.y + 1)]      = mat[ GINDEX(p.x, p.y+1, N) ];
    }
    // top side
    if(lp.y == 0){
        cache[CINDEX(lp.x, lp.y - 1)]      = mat[ GINDEX(p.x, p.y-1, N) ];
    }
    // local thread 0 in charge of the four corners
    if(lp.x + lp.y == 0){
        // top-left
        cache[CINDEX(-1, -1)]           = mat[ GINDEX(p.x-1, p.y-1, N) ];
        // top-right
        cache[CINDEX(BSIZE2D, -1)]      = p.x + BSIZE2D <= N ? mat[ GINDEX(p.x + BSIZE2D, p.y - 1, N) ] : 0;
        // bot-lefj
        cache[CINDEX(-1, BSIZE2D)]      = p.y + BSIZE2D <= N ? mat[ GINDEX(p.x-1, p.y + (BSIZE2D), N) ] : 0;
        // bot-right
        cache[CINDEX(BSIZE2D, BSIZE2D)] = p.x + BSIZE2D <= N && p.y + BSIZE2D <= N ? mat[ GINDEX(p.x + BSIZE2D, p.y + BSIZE2D, N) ] : 0;
    }
}

__device__ void work_cache(MTYPE *data, MTYPE *mat1, MTYPE *mat2, MTYPE *cache, uint3 lp, uint2 p, int n){
    // neighborhood count 
    int nc =    cache[CINDEX(lp.x-1, lp.y-1)] + cache[CINDEX(lp.x, lp.y-1)] + cache[CINDEX(lp.x+1, lp.y-1)] + 
                cache[CINDEX(lp.x-1, lp.y  )] +                               cache[CINDEX(lp.x+1, lp.y  )] + 
                cache[CINDEX(lp.x-1, lp.y+1)] + cache[CINDEX(lp.x, lp.y+1)] + cache[CINDEX(lp.x+1, lp.y+1)];
   unsigned int c = cache[CINDEX(lp.x, lp.y)];
   // transition function applied to state 'c' and written into mat2
   //printf("p.x, p.y =  (%i, %i)\n", p.x, p.y);
   //if(p.x == 1 && p.y == 1){
   //    printf("cell (%i, %i) = %i   (%i neighbors alive)\n", p.x, p.y, c, nc);
   //}
   mat2[GINDEX(p.x, p.y, n)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
}


__device__ __inline__ void load_cache(MTYPE *cache, MTYPE *mat, unsigned long n, uint3 lp, uint2 p){
    // loading the thread's element
    if(p.x <= n && p.y <= n){
        cache[CINDEX(lp.x, lp.y)] = mat[GINDEX(p.x, p.y, n)];
        set_halo(cache, mat, n, lp, p);
    }
    // loading the cache's halo
    __syncthreads();
}
template<typename Lambda>
__global__ void kernel_test(const unsigned long n, const unsigned long msize, const unsigned long nb, const unsigned long rb, MTYPE *data, MTYPE *dmat1, MTYPE *dmat2, Lambda map, unsigned int aux1, unsigned int aux2, unsigned int aux3){
    __shared__ MTYPE auxs[6455];
    __shared__ MTYPE cache[CSPACE];
    auto p = map(nb, rb, WARPSIZE);
    load_cache(cache, dmat1, n, threadIdx, p);
    if(p.x <= n && p.y <= n && (p.x & (n-1-p.y)) == 0){
        work_cache(data, dmat1, dmat2, cache, threadIdx, p, n);
        //work_nocache(data, dmat1, dmat2, p, n);
    }
}

__global__ void kernel_update_ghosts(const unsigned int n, const unsigned long msize, MTYPE *dmat1, MTYPE *dmat2){
    // update ghosts cells
    // this kernel uses a linear grid of size n
    int tid = blockIdx.x * blockDim.x + threadIdx.x;
    if(tid >= n){return;}
    // left
    dmat2[(tid+1)*(n+2) + 0] = dmat1[(n)*(n+2) + n - tid];
    // bottom
    dmat2[(n+1)*(n+2) + (tid+1)] = dmat1[(n-tid)*(n+2) + 1];
    // diagonal (inner)
    dmat2[(tid)*(n+2) + (tid+1)] = dmat1[(n-tid)*(n+2) + (n-tid)];
    // diagonal (outer)
    dmat2[(tid)*(n+2) + (tid+2)] = dmat1[(n-tid)*(n+2) + (n-tid-1)];
    // special cells
    if(tid==0){
        // bot-left ghost
        dmat2[(n+1)*(n+2) + 0]   = dmat1[(n)*(n+2) + 0];
        // bot-right-most ghost
        dmat2[(n-1)*(n+2) + (n+1)] = dmat1[(n)*(n+2) + n];
    }
}

__device__ int pow3(const int u){
   int r=1; 
   for(int i=0; i<u; ++i){
       r *= 3;
   }
   return r;
}

__inline__ __device__
uint2 warp_reduce(uint2 lm, const int WSIZE) {
    //printf("WSIZE = %i\n", WSIZE);
    for (int offset = (WSIZE/2); offset > 0; offset /= 2){
        lm.x += __shfl_down_sync(0xFFFFFFFF, lm.x, offset, WSIZE);
        lm.y += __shfl_down_sync(0xFFFFFFFF, lm.y, offset, WSIZE);
    }
    return lm;
}

__inline__ __device__
int warp_reduce1D(int lm, const int WSIZE) {
    //printf("WSIZE = %i\n", WSIZE);
    for (int offset = (WSIZE/2); offset > 0; offset /= 2){
        lm += __shfl_down_sync(0xFFFFFFFF, lm, offset, WSIZE);
    }
    return lm;
}



#endif
