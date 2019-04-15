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
#define GINDEX(x, y, n)  ((1+((unsigned long)y))*(((unsigned long)n)+2)  + (1+(x)))


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

__device__ __inline__ void set_halo(MTYPE *cache, MTYPE *mat, unsigned int N, uint3 lp, int2 p){
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

__device__ __inline__ void set_halo_rectangle(MTYPE *cache, MTYPE *mat, unsigned int N, uint3 lp, int2 p){
    // EMPANADA FANTASMA BOUNDARY CONDITIONS
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

// metodo kernel cached test
/*
template<typename Lambda>
__global__ void kernel_test(const unsigned int n, const unsigned long msize, DTYPE *data, MTYPE *dmat1, MTYPE *dmat2, Lambda map, unsigned int aux1, unsigned int aux2, unsigned int aux3){
	auto p = map(n, msize, aux1, aux2, aux3);
    if(p.x <= p.y && p.y < n){
        work_nocache(data, dmat1, dmat2, p, n);
    }
}
*/

__device__ inline int h(int k, int a, int b){
    return (1 - (((k - a) >> 31) & 0x1)) * (1 - (((b - k) >> 31) & 0x1));
}

__device__ void work_cache(DTYPE *data, MTYPE *mat1, MTYPE *mat2, MTYPE *cache, uint3 lp, int2 p, int n){
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

__device__ void work_nocache(DTYPE *data, MTYPE *mat1, MTYPE *mat2, int2 p, int n){
    int nc = mat1[GINDEX(p.x-1, p.y-1, n)] + mat1[GINDEX(p.x, p.y-1, n)] + mat1[GINDEX(p.x+1, p.y-1, n)] + 
             mat1[GINDEX(p.x-1, p.y, n  )] +                               mat1[GINDEX(p.x+1, p.y, n  )] + 
             mat1[GINDEX(p.x-1, p.y+1, n)] + mat1[GINDEX(p.x, p.y+1, n)] + mat1[GINDEX(p.x+1, p.y+1, n)];
    unsigned int c = mat1[GINDEX(p.x, p.y, n)];
    // transition function applied to state 'c' and written into mat2
    mat2[GINDEX(p.x, p.y, n)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
}

__device__ __inline__ void load_cache(MTYPE *cache, MTYPE *mat, unsigned int n, uint3 lp, int2 p){
    // loading the thread's element
    if(p.x <= n && p.y <= n){
        cache[CINDEX(lp.x, lp.y)] = mat[GINDEX(p.x, p.y, n)];
        set_halo(cache, mat, n, lp, p);
    }
    // loading the cache's halo
    __syncthreads();
}
__device__ __inline__ void load_cache_rectangle(MTYPE *cache, MTYPE *mat, unsigned int n, uint3 lp, int2 p){
    // loading the thread's element
    if(p.x <= n && p.y <= n){
        cache[CINDEX(lp.x, lp.y)] = mat[GINDEX(p.x, p.y, n)];
        set_halo_rectangle(cache, mat, n, lp, p);
    }
    // loading the cache's halo
    __syncthreads();
}
// metodo kernel cached test
///*
template<typename Lambda>
__global__ void kernel_test(const unsigned int n, const unsigned long msize, DTYPE *data, MTYPE *dmat1, MTYPE *dmat2, Lambda map, unsigned int aux1, unsigned int aux2, unsigned int aux3){
    __shared__ MTYPE cache[CSPACE];
	auto p = map(n, msize, aux1, aux2, aux3);
    load_cache(cache, dmat1, n, threadIdx, p);
    if(p.x <= p.y && p.y < n){
        work_cache(data, dmat1, dmat2, cache, threadIdx, p, n);
        //work_nocache(data, dmat1, dmat2, p, n);
    }
}
//*/


// metodo kernel cached test
template<typename Lambda>
__global__ void kernel_test_rectangle(const unsigned int n,const unsigned long msize, DTYPE *data, MTYPE *dmat1, MTYPE *dmat2, Lambda map, unsigned int aux1, unsigned int aux2, unsigned int aux3){
    // map
	auto p = map(n, msize, aux1, aux2, aux3);
    // cache
    __shared__ MTYPE cache[CSPACE];
    if(blockIdx.x == blockIdx.y){
        //printf("THREAD IN DIAG\n");
        // block diagonal - no cache
        if(p.x <= p.y && p.y < n){
            work_nocache(data, dmat1, dmat2, p, n);
        }
    }
    else if(blockIdx.x < blockIdx.y){
        //printf("THREAD IN BOT\n");
        // lower triangular - standard cache
        load_cache_rectangle(cache, dmat1, n, threadIdx, p);
        if(p.x <= p.y && p.y < n){
            work_cache(data, dmat1, dmat2, cache, threadIdx, p, n);
            //work_nocache(data, dmat1, dmat2, p, n);
        }
    }
    else{
        //printf("THREAD IN UPPER\n");
        // upper triangular - inverted cache
        uint3 invlp = (uint3){BSIZE2D-1 - threadIdx.x, BSIZE2D-1 - threadIdx.y, 0};
        load_cache_rectangle(cache, dmat1, n, invlp, p);
        if(p.x <= p.y && p.y < n){
            work_cache(data, dmat1, dmat2, cache, invlp, p, n);
            //work_nocache(data, dmat1, dmat2, p, n);
        }
    }
}
#endif
