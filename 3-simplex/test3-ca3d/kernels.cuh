#ifndef KERNELS_CUH
#define KERNELS_CUH

#define ONETHIRD 0.3333333333f
#define COFF 1

// rules
#define EL 5
#define EU 7
#define FL 6
#define FU 6

#define CINDEX(x, y, z)     ((COFF+(z))*BSIZE3D*BSIZE3D + (COFF+(y))*BSIZE3D + (COFF+(x)))
#define GINDEX(x, y, z, n)  ((z)*(n)*(n) + (y)*(n) + (x))

// h() function checks if k belongs (inclusive) to [a,b] or not (1 or 0).
__device__ inline int h(int k, int a, int b){
    return (1 - (((k - a) >> 31) & 0x1)) * (1 - (((b - k) >> 31) & 0x1));
}

__device__ inline void work(const DTYPE *data, MTYPE *mat1, MTYPE *mat2, unsigned long index, unsigned int N, uint3 p){
    // ------------------------------------
    // phase 1: caching the block of cells from mat1
    // ------------------------------------
    __shared__ unsigned int cache[(BSIZE3D+1)*(BSIZE3D+1)*(BSIZE3D+1)];
    uint3 ltid = threadIdx;
    // cache left and right boundaries
    
    // left halo
    if(threadIdx.x == 0){
        cache[CINDEX(threadIdx.x - 1, threadIdx.y, threadIdx.z)]      = mat1[ GINDEX(p.x == 0     ? p.y-1 : p.x-1, p.y, p.z, N) ];
        //printf("p(%i, %i, %i) ---> (%i, %i, %i) --> gindex() = %i\n", p.x, p.y, p.z, p.x==0? p.y-1 : p.x-1, p.y, p.z, GINDEX(p.x==0? p.y-1 : p.x-1, p.y, p.z, N));
    }
    // right halo 
    if(threadIdx.x == BSIZE3D-1 || p.x == p.y-1){
        cache[CINDEX(threadIdx.x + 1, threadIdx.y, threadIdx.z)]      = mat1[ GINDEX(p.x == p.y-1 ? 0 : p.x+1,                           p.y,                             p.z, N) ];
    }
    // bottom halo
    if(threadIdx.y == BSIZE3D-1 || p.y == p.z-1){
        cache[CINDEX(threadIdx.x, threadIdx.y + 1, threadIdx.z)]      = mat1[ GINDEX(p.x                     ,   p.y == p.z-1? p.x+1 : p.y+1,                             p.z, N) ];
    }
    // top halo
    if(threadIdx.y == 0 || p.x == p.y-1){
        cache[CINDEX(threadIdx.x, threadIdx.y - 1, threadIdx.z)]      = mat1[ GINDEX(p.x                     ,     p.x == p.y-1? N-1 : p.y-1,                             p.z, N) ];
    }
    // front halo
    if(threadIdx.z == 0 || p.y == p.z-1){
        cache[CINDEX(threadIdx.x, threadIdx.y, threadIdx.z - 1)]      = mat1[ GINDEX(p.x                     ,                           p.y,       p.y == p.z-1? N-1 : p.z-1, N) ];
    }
    // back halo
    if(threadIdx.z == BSIZE3D-1 || p.z == N-1){
        cache[CINDEX(threadIdx.x, threadIdx.y, threadIdx.z + 1)]      = mat1[ GINDEX(                     p.x,                           p.y,       p.z == N-1? p.y+1 : p.z+1, N) ];
    }


    // cache the main cell
    cache[CINDEX(threadIdx.x, threadIdx.y, threadIdx.z)]              = mat1[ GINDEX(p.x, p.y, p.z, N) ];


    // cache sync point
    __syncthreads();
    // ----------------------------------
    // phase 2: simulate using the cache and write into mat2
    // ----------------------------------
    // neighborhood of cell
    int nc =    // z-1 layer
                cache[CINDEX(ltid.x-1, ltid.y-1, ltid.z-1)] + cache[CINDEX(ltid.x, ltid.y-1, ltid.z-1)] + cache[CINDEX(ltid.x+1, ltid.y-1, ltid.z-1)] + 
                cache[CINDEX(ltid.x-1, ltid.y,   ltid.z-1)] + cache[CINDEX(ltid.x, ltid.y,   ltid.z-1)] + cache[CINDEX(ltid.x+1, ltid.y,   ltid.z-1)] + 
                cache[CINDEX(ltid.x-1, ltid.y+1, ltid.z-1)] + cache[CINDEX(ltid.x, ltid.y+1, ltid.z-1)] + cache[CINDEX(ltid.x+1, ltid.y+1, ltid.z-1)] + 
                // z+0 layer 
                cache[CINDEX(ltid.x-1, ltid.y-1, ltid.z)] + cache[CINDEX(ltid.x, ltid.y-1, ltid.z)] + cache[CINDEX(ltid.x+1, ltid.y-1, ltid.z)] + 
                cache[CINDEX(ltid.x-1, ltid.y,   ltid.z)] +                   0                     + cache[CINDEX(ltid.x+1, ltid.y,   ltid.z)] + 
                cache[CINDEX(ltid.x-1, ltid.y+1, ltid.z)] + cache[CINDEX(ltid.x, ltid.y+1, ltid.z)] + cache[CINDEX(ltid.x+1, ltid.y+1, ltid.z)] + 
                // z+1 layer 
                cache[CINDEX(ltid.x-1, ltid.y-1, ltid.z+1)] + cache[CINDEX(ltid.x, ltid.y-1, ltid.z+1)] + cache[CINDEX(ltid.x+1, ltid.y-1, ltid.z+1)] + 
                cache[CINDEX(ltid.x-1, ltid.y,   ltid.z+1)] + cache[CINDEX(ltid.x, ltid.y,   ltid.z+1)] + cache[CINDEX(ltid.x+1, ltid.y,   ltid.z+1)] + 
                cache[CINDEX(ltid.x-1, ltid.y+1, ltid.z+1)] + cache[CINDEX(ltid.x, ltid.y+1, ltid.z+1)] + cache[CINDEX(ltid.x+1, ltid.y+1, ltid.z+1)];

   // cell state 
   char c = cache[CINDEX(ltid.x, ltid.y, ltid.z)];

   // transition function applied to state 'c' and written into mat2
   mat2[GINDEX(p.x, p.y, p.z, N)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);


}

template<typename Lambda>
__global__ void k_setoutdata(MTYPE *d, unsigned long n, char x, Lambda map){
    auto p = map();
    if(p < n){
        d[p] = x; 
    }
    return;
}

// kernel Bounding Box
template<typename Lambda>
__global__ void kernel0(const DTYPE *data, MTYPE *mat1, MTYPE *mat2, const unsigned long n, const unsigned long V, Lambda map){
    if(blockIdx.x > blockIdx.y || blockIdx.y > blockIdx.z){
        return;
    }
	auto p = map();
    if(p.x < p.y && p.y < p.z){
        unsigned long index = p.z*n*n + p.y*n + p.x;
        if(index < V){
            work(NULL, mat1, mat2, index, n, p);
        }
    }
    return;
}

// kernel non-linear Map
template<typename Lambda>
__global__ void kernel1(const DTYPE *data, MTYPE *mat1, MTYPE *mat2, const unsigned long n, const unsigned long V, Lambda map){
    // lambda map
	auto p = map();
    if(p.x < p.y && p.y < p.z){
        unsigned long index = p.z*n*n + p.y*n + p.x;
        if(index < V){
            work(NULL, mat1, mat2, index, n, p);
        }
    }
	return;
}
#endif
