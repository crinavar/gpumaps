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
__global__ void kernel_test(const unsigned int n, const unsigned long msize, DTYPE *data, MTYPE *dmat1, MTYPE *dmat2, Lambda map, unsigned int aux1, unsigned int aux2, unsigned int aux3){
    __shared__ MTYPE cache[CSPACE];
	auto p = map(n, msize, aux1, aux2, aux3);
    if(p.x != -1){
        load_cache(cache, dmat1, n, threadIdx, p);
    }
    if(p.x < p.y && p.y < n){
        work_cache(data, dmat1, dmat2, cache, threadIdx, p, n);
    }
}

// metodo kernel cached test
template<typename Lambda>
__global__ void kernel_test_rectangle(const unsigned int n,const unsigned long msize, DTYPE *data, MTYPE *dmat1, MTYPE *dmat2, Lambda map, unsigned int aux1, unsigned int aux2, unsigned int aux3){
    // map
	auto p = map(n, msize, aux1, aux2, aux3);
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
#endif
