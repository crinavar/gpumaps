#ifndef KERNELS_H
#define KERNELS_H


#define SQRT_COEF 0.0001f
#define BB 2000
#define OFFSET -0.4999f
//#define OFFSET 0.5f

// rules
#define EL 2
#define EU 3
#define FL 2
#define FU 2

#define GINDEX(x, y, n)  ((y+1)*(n+2)  + x+1)
#define CINDEX(x, y) ((y+1)*(BSIZE2D+2) + x+1)

__device__ inline int h(int k, int a, int b){
    return (1 - (((k - a) >> 31) & 0x1)) * (1 - (((b - k) >> 31) & 0x1));
}

__device__ void work_nocache(MTYPE *data, MTYPE *mat1, MTYPE *mat2, uint2 p, int n){
    int nc = mat1[GINDEX(p.x-1, p.y-1, n)] + mat1[GINDEX(p.x, p.y-1, n)] + mat1[GINDEX(p.x+1, p.y-1, n)] + 
             mat1[GINDEX(p.x-1, p.y, n  )] +                               mat1[GINDEX(p.x+1, p.y, n  )] + 
             mat1[GINDEX(p.x-1, p.y+1, n)] + mat1[GINDEX(p.x, p.y+1, n)] + mat1[GINDEX(p.x+1, p.y+1, n)];
    unsigned int c = mat1[GINDEX(p.x, p.y, n)];
    // transition function applied to state 'c' and written into mat2
    mat2[GINDEX(p.x, p.y, n)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
}

template<typename Lambda, typename Inverse>
__global__ void kernelBoundingBox(const size_t nx, const size_t ny, const size_t nb, const size_t rb, MTYPE *dmat1, MTYPE *dmat2, Lambda map, Inverse inv, const int WARPSIZE){

    //__shared__ MTYPE cache[(BSIZE2D+2)*(BSIZE2D+2)];

    auto p = map(nb, rb, WARPSIZE);
    __syncthreads();


    if(p.x < nx && p.y < ny && (p.x & (nx-1-p.y)) == 0){

        int nc = dmat1[GINDEX(p.x-1, p.y-1, nx)] + dmat1[GINDEX(p.x, p.y-1, nx)] + dmat1[GINDEX(p.x+1, p.y-1, nx)] + 
                 dmat1[GINDEX(p.x-1, p.y,   nx)] +                                 dmat1[GINDEX(p.x+1, p.y,   nx)] + 
                 dmat1[GINDEX(p.x-1, p.y+1, nx)] + dmat1[GINDEX(p.x, p.y+1, nx)] + dmat1[GINDEX(p.x+1, p.y+1, nx)];
        unsigned int c = dmat1[GINDEX(p.x, p.y, nx)];

        // transition function applied to state 'c' and written into mat2
        dmat2[GINDEX(p.x, p.y, nx)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
        //work_cache(data, dmat1, dmat2, cache, threadIdx, p, nx);
        //work_nocache(data, dmat1, dmat2, p, n);
    }
}

template<typename Inverse>
__device__ int2 getUncompressedCoordinate(int x, int y, size_t n, size_t nx, size_t nb, size_t rb, Inverse inv, const int WARPSIZE, int elementInHaloSide){
    int2 m;
    //m = inv((x)>>BPOWER, (y)>>BPOWER, nb, rb, WARPSIZE);
    m = inv(x, y, nb, rb, WARPSIZE, elementInHaloSide);
    //int newx = (x) & (blockDim.x - 1);
    //int newy = (y) & (blockDim.y - 1);
    //m.x = (m.x<<BPOWER)+newx;
    //m.y = (m.y<<BPOWER)+newy;
    return m;
    
}

//TODO hay un error cuando bsize =1 
template<typename Lambda, typename Inverse>
__global__ void kernelCompressed(const size_t n, const size_t nx, const size_t ny, const size_t nb, const size_t rb, MTYPE *dmat1, MTYPE *dmat2, Lambda map, Inverse inv, const int WARPSIZE){
    
    // const char o int 
    int traducx[8] = {-1,  0,  1, -1, /*0,*/  1, -1,  0, 1};
    int traducy[8] = {-1, -1, -1,  0, /*0,*/  0,  1,  1, 1};

    //const int oobCheckX[9] = {-1,  127,  nb, -1, 127,  nb, -1,  127, nb};
    //const int oobCheckY[9] = {-1, -1, -1,  127, 127, 127, nb, nb, nb};

    __shared__ MTYPE cache[(BSIZE2D+2)*(BSIZE2D+2)];
    __shared__ int2 coords[8]; //vecindad de moore

    // In this method, this mapping function returns block level coordinates!
    auto p = map(nb, rb, WARPSIZE);
    uint2 local;
    local.x = blockIdx.x*blockDim.x + threadIdx.x;
    local.y = blockIdx.y*blockDim.y + threadIdx.y;

    int cacheSize = min(BSIZE2D, (int)n);

    int tid = blockDim.x*threadIdx.y + threadIdx.x;
    int wid = tid/WARPSIZE;

    int haloSideToCopy = BSIZE1D/32 < 4 ? threadIdx.y : tid/32;
    int elementInHaloSide = threadIdx.x;
    //detectar de que bloque es, traducir el blioque para ver su valor

    printf("%i, %i  --- > %i, %i\n", local.x, local.y, p.x, p.y);

    //NEW IDEA - Llenar halo con valores traducidos (SE CONSIDERA Q HAY UN HALO GLOBAL!!!)
    if (threadIdx.x < n && threadIdx.y < n)
        cache[CINDEX(threadIdx.x, threadIdx.y)] = dmat1[GINDEX(local.x, local.y, nx)];
    

    for (int neighbor=wid; neighbor<8; neighbor+=WARPSIZE){
        //traducx = (neighbor%3) - 1;     // -1,  0, +1, -1, 0, +1, -1,  0, +1
        //traducy = neighbor/3 - 1;       // -1, -1, -1,  0, 0,  0, +1, +1, +1
        /*if (traducx[neighbor] == 0 && traducy[neighbor] == 0){
            continue;
        }*/
        //int offtraducx = traducx[neighbor] + p.x;
        //int offtraducy = traducy[neighbor] + p.y;
        traducx[neighbor] += p.x;
        traducy[neighbor] += p.y;
        coords[neighbor] = getUncompressedCoordinate(traducx[neighbor], traducy[neighbor], n, nx, nb, rb, inv, WARPSIZE, elementInHaloSide);
    }


    // left side
    if(haloSideToCopy == 0%cacheSize){
        if (traducx[3] == -1 || (traducx[3] & (nb-1-traducy[3])) != 0) {
            cache[CINDEX(-1, elementInHaloSide)]           = 0;
        } else {
            int2 m = coord[3];
            //printf("left side tx: %i ty: %i | px: %i, py: %i - (%i, %i) -> %i,%i\n", threadIdx.x, threadIdx.y, p.x, p.y, p.x-1-elementInHaloSide, p.y-threadIdx.y+elementInHaloSide, m.x, m.y);
            m.x = m.x*blockDim.x + blockDim.x - 1;
            m.y = m.y*blockDim.y + elementInHaloSide;
            cache[CINDEX(-1, elementInHaloSide)]           = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
    // right side
    if(haloSideToCopy == 1%cacheSize){
        if (traducx[4] == nb || (traducx[4] & (nb-1-traducy[4])) != 0){
            cache[CINDEX(cacheSize, elementInHaloSide)] = 0;
        } else {
            int2 m = coord[4];
            //printf("right side tx: %i ty: %i | px: %i, py: %i - (%i, %i) -> %i,%i\n", threadIdx.x, threadIdx.y, p.x, p.y, traducx, traducy, m.x, m.y);
            m.x = m.x*blockDim.x;
            m.y = m.y*blockDim.y + elementInHaloSide;
            cache[CINDEX(cacheSize, elementInHaloSide)] = dmat1[ GINDEX(m.x, m.y, nx) ];
        }

    }
    // bottom side
    if(haloSideToCopy == 2%cacheSize){
        if (traducy[6] == nb || (traducx[6] & (nb-1-traducy[6])) != 0){
            cache[CINDEX(elementInHaloSide, cacheSize)] = 0;
        } else {
            int2 m = coord[6];
            m.x = m.x*blockDim.x + elementInHaloSide;
            m.y = m.y*blockDim.y;
            //printf("bottom side tx: %i ty: %i | px: %i, py: %i - (%i, %i) -> %i,%i\n", threadIdx.x, threadIdx.y, p.x, p.y, traducx, traducy, m.x, m.y);
            cache[CINDEX(elementInHaloSide, cacheSize)] = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
    // top side
    if(haloSideToCopy == 3%cacheSize){
        if (traducy[1] == -1 || (traducx[1] & (nb-1-traducy[1])) != 0) {
            cache[CINDEX(elementInHaloSide, -1)]         = 0;
        } else {
            int2 m = coord[1];
            //printf("top side tx: %i ty: %i | px: %i, py: %i - (%i, %i) -> %i,%i\n", threadIdx.x, threadIdx.y, p.x, p.y, traducx, traducy, m.x, m.y);
            m.x = m.x*blockDim.x + elementInHaloSide;
            m.y = m.y*blockDim.y + blockDim.y - 1;
            cache[CINDEX(elementInHaloSide, -1)]         = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
    // local thread 0 in charge of the four corners
    // compartir el valor del bloque traducido al fractal para el warp asi se aprovecha la reduccion y el loglog(n)
    if(tid < WARPSIZE){
        // top-left
        
        traducx = p.x-1;
        traducy = p.y-1;
        if (traducx == -1 || traducy == -1 || (traducx & (nb-1-traducy)) != 0){
            if (tid == 0)
                cache[CINDEX(-1, -1)]           = 0;
        } else {
            int2 m = getUncompressedCoordinate(traducx, traducy, n, nx, nb, rb, inv, WARPSIZE, elementInHaloSide);
            m.x = m.x*blockDim.x + blockDim.x - 1;
            m.y = m.y*blockDim.y + blockDim.y - 1;
            if (tid == 0)
                cache[CINDEX(-1, -1)]           = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
        
        // top-right
        traducx = p.x + 1;
        traducy = p.y - 1;
        if (traducx > nb || traducy-1 == -1 || (traducx & (nb-1-traducy)) != 0){
            if (tid == 0)
                cache[CINDEX(cacheSize, -1)]      = 0;
        } else {
            int2 m = getUncompressedCoordinate(traducx, traducy, n, nx, nb, rb, inv, WARPSIZE, elementInHaloSide);
            m.x = m.x*blockDim.x;
            m.y = m.y*blockDim.y + blockDim.y - 1;
            if (tid == 0)
                cache[CINDEX(cacheSize, -1)]      = dmat1[ GINDEX(m.x, m.y, nx) ];
        }

        // bot-left
        traducx = p.x-1;
        traducy = p.y+1;
        if (traducx == -1 || traducy == nb || (traducx & (nb-1-traducy)) != 0){
            if (tid == 0)
            cache[CINDEX(-1, cacheSize)]      = 0;
        } else {
            int2 m = getUncompressedCoordinate(traducx, traducy, n, nx, nb, rb, inv, WARPSIZE, elementInHaloSide);
            m.x = m.x*blockDim.x + blockDim.x - 1;
            m.y = m.y*blockDim.y;
            if (tid == 0)
            cache[CINDEX(-1, cacheSize)]      = dmat1[ GINDEX(m.x, m.y, nx) ];
        }

        // bot-right
        traducx = p.x + 1;
        traducy = p.y + 1;
        if (traducx == nb || traducy == nb || (traducx & (nb-1-traducy)) != 0){
            if (tid == 0)
            cache[CINDEX(cacheSize, cacheSize)] = 0;
        } else {
            int2 m = getUncompressedCoordinate(traducx, traducy, n, nx, nb, rb, inv, WARPSIZE, elementInHaloSide);
            m.x = m.x*blockDim.x;
            m.y = m.y*blockDim.y;
            if (tid == 0)
            cache[CINDEX(cacheSize, cacheSize)] = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }

    __syncthreads();
    //if (threadIdx.x%BSIZE2D & (BSIZE2D-1-threadIdx.y % BSIZE2D) != 0){
    //    return;
    //}
    if (local.x > n || local.y > n){
        return;
    }

    if (p.x & (n - 1 - p.y)){
        return;
    }
    int nc =    cache[CINDEX(threadIdx.x-1, threadIdx.y-1)] + cache[CINDEX(threadIdx.x, threadIdx.y-1)] + cache[CINDEX(threadIdx.x+1, threadIdx.y-1)] + 
                cache[CINDEX(threadIdx.x-1, threadIdx.y  )] +                               cache[CINDEX(threadIdx.x+1, threadIdx.y  )] + 
                cache[CINDEX(threadIdx.x-1, threadIdx.y+1)] + cache[CINDEX(threadIdx.x, threadIdx.y+1)] + cache[CINDEX(threadIdx.x+1, threadIdx.y+1)];
    unsigned int c = cache[CINDEX(threadIdx.x, threadIdx.y)];
/*
    for (int i=0; i<10000; i++);
    if (tid ==0 && blockIdx.x == 0 && blockIdx.y == 0){
        for (int i=0; i<BSIZE2D+2; i++){
            for (int j=0; j<BSIZE2D+2; j++){
                printf("%i ", cache[i*(BSIZE2D+2)+j]);
            }
            printf("\n");
        }
    }*/
    // transition function applied to state 'c' and written into mat2
    //dmat2[GINDEX(local.x, local.y, nx)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
    
    dmat2[GINDEX(local.x, local.y, nx)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
    //work_cache(data, dmat1, dmat2, cache, threadIdx, p, nx);
    //work_nocache(data, dmat1, dmat2, p, n);
}

__device__ int pow3(const int u){
   int r=1; 
   for(int i=0; i<u; ++i){
       r *= 3;
   }
   return r;
}

__inline__ __device__
int2 warp_reduce(int2 lm, const int WSIZE) {
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
