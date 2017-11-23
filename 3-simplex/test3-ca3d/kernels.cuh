#ifndef KERNELS_CUH
#define KERNELS_CUH

#define ONETHIRD 0.3333333333f

// rules
#define EL 5
#define EU 7
#define FL 6
#define FU 6

#define CLEN ((BSIZE3D)+2)
#define CVOL (CLEN*CLEN*CLEN)


#define CINDEX(x, y, z)     ((1+(z))*CLEN*CLEN + (1+(y))*CLEN + (1+(x)))
#define GINDEX(x, y, z, n)  ((z)*(n)*(n) + (y)*(n) + (x))

// h() function checks if k belongs (inclusive) to [a,b] or not (1 or 0).
__device__ inline int h(int k, int a, int b){
    return (1 - (((k - a) >> 31) & 0x1)) * (1 - (((b - k) >> 31) & 0x1));
}

__device__ void reset_cache(unsigned int *cache, unsigned long cvol, unsigned int val){
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
        for(unsigned long i = 0; i<cvol; ++i){
            cache[i] = val;
        }
    }
}

__device__ unsigned int cache_living_cells(unsigned int *cache, unsigned long cvol){
    unsigned int cont = 0;
    if(threadIdx.x == 0 && threadIdx.y == 0 && threadIdx.z == 0){
        for(unsigned long i = 0; i<cvol; ++i){
            cont += cache[i];
        }
    }
    return cont;
}

__device__ void set_halo_faces(MTYPE *cache, MTYPE *mat, unsigned int N, uint3 ltid, uint3 p){
    // FREE BOUNDARY CONDITIONS
    // left face
    if(ltid.x == 0){
        cache[CINDEX(ltid.x - 1, ltid.y, ltid.z)]      = p.x == 0 ? 0 : mat[ GINDEX(p.x-1, p.y, p.z, N) ];
    }
    // right face
    if(ltid.x == BSIZE3D-1){
        cache[CINDEX(ltid.x + 1, ltid.y, ltid.z)]      = p.x > N-1 ? 0 : mat[ GINDEX(p.x+1, p.y, p.z, N) ];
    }
    // bottom face
    if(ltid.y == BSIZE3D-1){
        cache[CINDEX(ltid.x, ltid.y + 1, ltid.z)]      = p.y > N-1 ? 0 : mat[ GINDEX(p.x, p.y+1, p.z, N) ];
    }
    // top face
    if(ltid.y == 0){
        cache[CINDEX(ltid.x, ltid.y - 1, ltid.z)]      = p.y == 0 ? 0 : mat[ GINDEX(p.x, p.y-1, p.z, N) ];
    }
    // front face
    if(ltid.z == 0){
        cache[CINDEX(ltid.x, ltid.y, ltid.z - 1)]      = p.z == 0 ? 0 : mat[ GINDEX(p.x, p.y, p.z-1, N) ];
    }
    // back face
    if(ltid.z == BSIZE3D-1){
        cache[CINDEX(ltid.x, ltid.y, ltid.z + 1)]      = p.z == N-1 ? 0 : mat[ GINDEX(p.x,   p.y, p.z+1, N) ];
    }
}


__device__ void set_halo_edges(MTYPE *cache, MTYPE *mat, unsigned int N, uint3 ltid, uint3 p){
    // top left edge
    if((ltid.x == 0 && ltid.y == 0) ){
        // edge
        cache[CINDEX(ltid.x - 1, ltid.y - 1, ltid.z)]  = (p.x == 0 || p.y == 0)? 0 : mat[ GINDEX(p.x-1, p.y-1, p.z  , N) ];
        // top front left corner 
        if(ltid.z == 0){ cache[CINDEX(ltid.x - 1, ltid.y - 1, ltid.z - 1)]  = (p.x == 0 || p.y == 0 || p.z == 0) ? 0 : mat[ GINDEX(p.x-1, p.y-1, p.z-1  , N) ]; }
        // top back left corner
        if(ltid.z == BSIZE3D-1){ cache[CINDEX(ltid.x-1, ltid.y - 1, ltid.z + 1)]  = (p.x == 0 || p.y == 0 || p.z == N-1) ? 0 : mat[ GINDEX(p.x-1, p.y-1, p.z+1  , N) ]; }
    }

    // top right edge
    if(ltid.x == BSIZE3D-1 && ltid.y == 0){
        // edge
        cache[CINDEX(ltid.x + 1, ltid.y - 1, ltid.z)]  = (p.x == N-1 || p.y == 0) ? 0 : mat[ GINDEX(p.x+1, p.y-1, p.z  , N) ];
        // top front right corner
        if(ltid.z == 0){ cache[CINDEX(ltid.x + 1, ltid.y - 1, ltid.z - 1)]  = (p.x == N-1 || p.y == 0 || p.z == 0) ? 0 : mat[GINDEX(p.x + 1, p.y-1, p.z-1  , N) ]; }
        // top back right corner
        if(ltid.z == BSIZE3D-1){ cache[CINDEX(ltid.x+1, ltid.y - 1, ltid.z + 1)]  = (p.x == N-1 || p.y == 0 || p.z == N-1) ? 0 : mat[ GINDEX(p.x+1, p.y-1, p.z+1  , N) ]; }
    }

    // top front edge
    if((ltid.z == 0 && ltid.y == 0)){
        cache[CINDEX(ltid.x, ltid.y - 1, ltid.z - 1)]  = (p.y == 0 || p.z == 0) ? 0 : mat[ GINDEX(p.x, p.y-1, p.z-1  , N) ];
    }


    // top back edge
    if(ltid.z == BSIZE3D-1 && ltid.y == 0){
        cache[CINDEX(ltid.x, ltid.y - 1, ltid.z + 1)]  = (p.z == N-1 || p.y == 0) ? 0 : mat[ GINDEX(p.x, p.y-1, p.z+1  , N) ];
    }



    // bottom edges
    // bottom left edge
    if(ltid.x == 0 && ltid.y == BSIZE3D-1){
        cache[CINDEX(ltid.x - 1, ltid.y + 1, ltid.z)]  = (p.x == 0 || p.y == N-1) ? 0 : mat[ GINDEX(p.x-1, p.y+1, p.z  , N) ];
        // bottom front left corner
        if(ltid.z == 0){ cache[CINDEX(ltid.x-1, ltid.y + 1, ltid.z - 1)]  = (p.x == 0 || p.y == N-1 || p.z == 0) ? 0 : mat[ GINDEX(p.x-1, p.y+1, p.z-1  , N) ]; }
        // bottom back left corner
        if(ltid.z == BSIZE3D-1){ cache[CINDEX(ltid.x-1, ltid.y + 1, ltid.z + 1)]  = (p.x == 0 || p.y == N-1 || p.z == N-1) ? 0 : mat[GINDEX(p.x-1, p.y+1, p.z+1, N) ]; }
    }

    // bottom right edge
    if(ltid.x == BSIZE3D-1 && ltid.y == BSIZE3D-1){
        cache[CINDEX(ltid.x + 1, ltid.y + 1, ltid.z)]  = (p.x == N-1 || p.y == N-1) ? 0 : mat[ GINDEX(p.x+1, p.y+1, p.z  , N) ];
        // bottom front right corner
        if(ltid.z == 0){ cache[CINDEX(ltid.x+1, ltid.y + 1, ltid.z - 1)]  = (p.x == N-1 || p.y == N-1 || p.z == 0) ? 0 : mat[ GINDEX(p.x+1, p.y+1, p.z-1  , N) ]; }
        // bottom back right corner
        if(ltid.z == BSIZE3D-1){ cache[CINDEX(ltid.x+1, ltid.y + 1, ltid.z + 1)]  = (p.x == N-1 || p.y == N-1 || p.z == N-1) ? 0 : mat[ GINDEX(p.x+1, p.y+1, p.z+1  , N) ]; }
    }

    // bottom front edge
    if(ltid.z == 0 && ltid.y == BSIZE3D-1){
        cache[CINDEX(ltid.x, ltid.y + 1, ltid.z - 1)]  = (p.y == N-1 || p.z == 0) ? 0 : mat[ GINDEX(p.x, p.y+1, p.z-1  , N) ];
    }

    // bottom back edge
    if(ltid.z == BSIZE3D-1 && ltid.y == BSIZE3D-1){
        cache[CINDEX(ltid.x, ltid.y + 1, ltid.z + 1)]  = (p.y == N-1 || p.z == 0) ? 0 : mat[ GINDEX(p.x, p.y+1, p.z+1  , N) ];
    }




    // side edges
    // front left side
    if(ltid.x == 0 && ltid.z == 0){
        cache[CINDEX(ltid.x - 1, ltid.y, ltid.z - 1)]  = (p.x == 0 || p.z == 0) ? 0 : mat[ GINDEX(p.x-1, p.y, p.z-1  , N) ];
    }

    // front right side
    if(ltid.x == BSIZE3D-1 && ltid.z == 0){
        cache[CINDEX(ltid.x + 1, ltid.y, ltid.z - 1)]  = (p.x == N-1 || p.z == 0) ? 0 : mat[ GINDEX(p.x+1, p.y, p.z-1  , N) ];
    }

    // back left edge
    if( ltid.x == 0 && ltid.z == BSIZE3D-1 ){
        cache[CINDEX(ltid.x - 1, ltid.y, ltid.z + 1)]  = (p.x == 0 || p.z == N-1) ? 0 : mat[ GINDEX(p.x-1, p.y, p.z+1, N) ];
    }

    // back right edge
    if(ltid.x == BSIZE3D-1 && ltid.z == BSIZE3D-1){
        cache[CINDEX(ltid.x + 1, ltid.y, ltid.z + 1)]  = (p.x == N-1 || p.z == N-1) ? 0 : mat[ GINDEX(p.x+1, p.y, p.z+1, N) ];
    }
}

__device__ void load_cache(MTYPE *cache, MTYPE *mat, unsigned int n, uint3 p){
    // loading the thread's element
    cache[CINDEX(threadIdx.x, threadIdx.y, threadIdx.z)] = mat[ GINDEX(p.x, p.y, p.z, n) ];
    // loading the cache's halo
    set_halo_faces(cache, mat, n, threadIdx, p);
    set_halo_edges(cache, mat, n, threadIdx, p);
}

__device__ inline void work(const DTYPE *data, MTYPE *cache, MTYPE *mat2, unsigned long index, unsigned int N, uint3 p){
    uint3 ltid = threadIdx;
    // neighborhood count 
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

   unsigned int c = cache[CINDEX(ltid.x, ltid.y, ltid.z)];
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
    __shared__ MTYPE cache[CVOL];
    if(blockIdx.x > blockIdx.y || blockIdx.y > blockIdx.z){ return; }
	auto p = map();
    load_cache(cache, mat1, n, p);
    __syncthreads();
    if(p.x < p.y && p.y < p.z){
        unsigned long index = p.z*n*n + p.y*n + p.x;
        if(index < V){
            work(NULL, cache, mat2, index, n, p);
        }
    }
    return;
}

// kernel non-linear Map
template<typename Lambda>
__global__ void kernel1(const DTYPE *data, MTYPE *mat1, MTYPE *mat2, const unsigned long n, const unsigned long V, Lambda map){
    __shared__ MTYPE cache[CVOL];
    // lambda map
	auto p = map();
    load_cache(cache, mat1, n, p);
    __syncthreads();
    if(p.x < p.y && p.y < p.z){
        unsigned long index = p.z*n*n + p.y*n + p.x;
        if(index < V){
            work(NULL, cache, mat2, index, n, p);
        }
    }
	return;
}
#endif
