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
__global__ void kernelLambdaTC(const size_t nx, const size_t ny, const size_t nb, const size_t rb, MTYPE *dmat1, MTYPE *dmat2, Lambda map, Inverse inv, const int WARPSIZE){

    //__shared__ MTYPE cache[(BSIZE2D+2)*(BSIZE2D+2)];

    __shared__ half mata[256];
    __shared__ half matb[256];
    auto p = map(nb, rb, WARPSIZE, mata, matb);
    /*const uint3 lp = threadIdx;
    if(p.x < nx && p.y < ny){
        cache[CINDEX(threadIdx.x, threadIdx.y)] = dmat1[GINDEX(p.x, p.y, nx)];

        if(lp.x == 0){
            cache[CINDEX(lp.x - 1, lp.y)]      = dmat1[ GINDEX(p.x-1, p.y, nx) ];
        }
        // right side
        if(lp.x == BSIZE2D-1){
            cache[CINDEX(lp.x + 1, lp.y)]      = dmat1[ GINDEX(p.x+1, p.y, nx) ];
        }
        // bottom side
        if(lp.y == BSIZE2D-1){
            cache[CINDEX(lp.x, lp.y + 1)]      = dmat1[ GINDEX(p.x, p.y+1, nx) ];
        }
        // top side
        if(lp.y == 0){
            cache[CINDEX(lp.x, lp.y - 1)]      = dmat1[ GINDEX(p.x, p.y-1, nx) ];
        }
        // local thread 0 in charge of the four corners
        if(lp.x + lp.y == 0){
            // top-left
            cache[CINDEX(-1, -1)]           = dmat1[ GINDEX(p.x-1, p.y-1, nx) ];
            // top-right
            cache[CINDEX(BSIZE2D, -1)]      = dmat1[ GINDEX(p.x + BSIZE2D, p.y - 1, nx) ];
            // bot-lefj
            cache[CINDEX(-1, BSIZE2D)]      = dmat1[ GINDEX(p.x-1, p.y + (BSIZE2D), nx) ];
            // bot-right
            cache[CINDEX(BSIZE2D, BSIZE2D)] = dmat1[ GINDEX(p.x + BSIZE2D, p.y + BSIZE2D, nx) ];
        }
    }*/
    //__syncthreads();


    if(p.x < nx && p.y < ny && (p.x & (nx-1-p.y)) == 0){

        /*int nc =    cache[CINDEX(lp.x-1, lp.y-1)] + cache[CINDEX(lp.x, lp.y-1)] + cache[CINDEX(lp.x+1, lp.y-1)] + 
                    cache[CINDEX(lp.x-1, lp.y  )] +                               cache[CINDEX(lp.x+1, lp.y  )] + 
                    cache[CINDEX(lp.x-1, lp.y+1)] + cache[CINDEX(lp.x, lp.y+1)] + cache[CINDEX(lp.x+1, lp.y+1)];
        unsigned int c = cache[CINDEX(lp.x, lp.y)];
        dmat2[GINDEX(p.x, p.y, nx)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);*/
        int nc = dmat1[GINDEX(p.x-1, p.y-1, nx)] + dmat1[GINDEX(p.x, p.y-1, nx)] + dmat1[GINDEX(p.x+1, p.y-1, nx)] + 
                 dmat1[GINDEX(p.x-1, p.y,   nx)] +                                 dmat1[GINDEX(p.x+1, p.y,   nx)] + 
                 dmat1[GINDEX(p.x-1, p.y+1, nx)] + dmat1[GINDEX(p.x, p.y+1, nx)] + dmat1[GINDEX(p.x+1, p.y+1, nx)];
        unsigned int c = dmat1[GINDEX(p.x, p.y, nx)];
        dmat2[GINDEX(p.x, p.y, nx)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
    }
}

template<typename Lambda, typename Inverse>
__global__ void kernelBoundingBox(const size_t nx, const size_t ny, const size_t nb, const size_t rb, MTYPE *dmat1, MTYPE *dmat2, Lambda map, Inverse inv, const int WARPSIZE){

    //__shared__ MTYPE cache[(BSIZE2D+2)*(BSIZE2D+2)];

    auto p = map(nb, rb, WARPSIZE, nullptr, nullptr);
    /*const uint3 lp = threadIdx;
    if(p.x < nx && p.y < ny){
        cache[CINDEX(threadIdx.x, threadIdx.y)] = dmat1[GINDEX(p.x, p.y, nx)];

        if(lp.x == 0){
            cache[CINDEX(lp.x - 1, lp.y)]      = dmat1[ GINDEX(p.x-1, p.y, nx) ];
        }
        // right side
        if(lp.x == BSIZE2D-1){
            cache[CINDEX(lp.x + 1, lp.y)]      = dmat1[ GINDEX(p.x+1, p.y, nx) ];
        }
        // bottom side
        if(lp.y == BSIZE2D-1){
            cache[CINDEX(lp.x, lp.y + 1)]      = dmat1[ GINDEX(p.x, p.y+1, nx) ];
        }
        // top side
        if(lp.y == 0){
            cache[CINDEX(lp.x, lp.y - 1)]      = dmat1[ GINDEX(p.x, p.y-1, nx) ];
        }
        // local thread 0 in charge of the four corners
        if(lp.x + lp.y == 0){
            // top-left
            cache[CINDEX(-1, -1)]           = dmat1[ GINDEX(p.x-1, p.y-1, nx) ];
            // top-right
            cache[CINDEX(BSIZE2D, -1)]      = dmat1[ GINDEX(p.x + BSIZE2D, p.y - 1, nx) ];
            // bot-lefj
            cache[CINDEX(-1, BSIZE2D)]      = dmat1[ GINDEX(p.x-1, p.y + (BSIZE2D), nx) ];
            // bot-right
            cache[CINDEX(BSIZE2D, BSIZE2D)] = dmat1[ GINDEX(p.x + BSIZE2D, p.y + BSIZE2D, nx) ];
        }
    }*/
    //__syncthreads();


    if(p.x < nx && p.y < ny && (p.x & (nx-1-p.y)) == 0){

        /*int nc =    cache[CINDEX(lp.x-1, lp.y-1)] + cache[CINDEX(lp.x, lp.y-1)] + cache[CINDEX(lp.x+1, lp.y-1)] + 
                    cache[CINDEX(lp.x-1, lp.y  )] +                               cache[CINDEX(lp.x+1, lp.y  )] + 
                    cache[CINDEX(lp.x-1, lp.y+1)] + cache[CINDEX(lp.x, lp.y+1)] + cache[CINDEX(lp.x+1, lp.y+1)];
        unsigned int c = cache[CINDEX(lp.x, lp.y)];
        dmat2[GINDEX(p.x, p.y, nx)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);*/
        int nc = dmat1[GINDEX(p.x-1, p.y-1, nx)] + dmat1[GINDEX(p.x, p.y-1, nx)] + dmat1[GINDEX(p.x+1, p.y-1, nx)] + 
                 dmat1[GINDEX(p.x-1, p.y,   nx)] +                                 dmat1[GINDEX(p.x+1, p.y,   nx)] + 
                 dmat1[GINDEX(p.x-1, p.y+1, nx)] + dmat1[GINDEX(p.x, p.y+1, nx)] + dmat1[GINDEX(p.x+1, p.y+1, nx)];
        unsigned int c = dmat1[GINDEX(p.x, p.y, nx)];
        dmat2[GINDEX(p.x, p.y, nx)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
    }
}

template<typename Inverse>
__device__ int2 getUncompressedCoordinate(int x, int y, size_t n, size_t nx, size_t nb, size_t rb, Inverse inv, const int WARPSIZE, int lid, half* mata, half* matb){
    int2 m;
    //m = inv((x)>>BPOWER, (y)>>BPOWER, nb, rb, WARPSIZE);
    m = inv(x, y, nb, rb, WARPSIZE, lid, mata, matb);
    //m={1,1};
    //int newx = (x) & (blockDim.x - 1);
    //int newy = (y) & (blockDim.y - 1);
    //m.x = (m.x<<BPOWER)+newx;
    //m.y = (m.y<<BPOWER)+newy;
    return m;
    
}

template<typename Lambda, typename Inverse>
__global__ void kernelCompressed(const size_t n, const size_t nx, const size_t ny, const size_t nb, const size_t rb, MTYPE *dmat1, MTYPE *dmat2, Lambda map, Inverse inv, const int WARPSIZE){
    
    // const char o int, tambien podria ser 1 solo arreglo y usar indexacion para extenderlo
    int traducx[8] = {-1,  0,  1, -1, /*0,*/  1, -1,  0, 1};
    int traducy[8] = {-1, -1, -1,  0, /*0,*/  0,  1,  1, 1};


    __shared__ MTYPE cache[(BSIZE2D+2)*(BSIZE2D+2)];
    __shared__ int2 coords[8]; //vecindad de moore

    // In this method, this mapping function returns block level coordinates!
    int2 p = map(nb, rb, WARPSIZE, nullptr, nullptr);
    int lpx = p.x*blockDim.x + threadIdx.x;
    int lpy = p.y*blockDim.y + threadIdx.y;

    uint2 local;
    local.x = blockIdx.x*blockDim.x + threadIdx.x;
    local.y = blockIdx.y*blockDim.y + threadIdx.y;

    int cacheSize = min(BSIZE2D, (int)n);

    int tid = blockDim.x*threadIdx.y + threadIdx.x;
    int wid = tid/WARPSIZE;

    int haloSideToCopy = BSIZE1D/32 < 4 ? threadIdx.y : tid/32;

    //printf("%i, %i  --- > %i, %i\n", local.x, local.y, p.x, p.y);

    if (local.x < nx && local.y < ny)
        cache[CINDEX(threadIdx.x, threadIdx.y)] = dmat1[GINDEX(local.x, local.y, nx)];
    

    int lid = (threadIdx.x + threadIdx.y*blockDim.x) - wid*WARPSIZE;
    for (int neighbor=wid; neighbor<8; neighbor+=BSIZE1D/WARPSIZE){
        int neighborCoordx = traducx[neighbor] + p.x;
        int neighborCoordy = traducy[neighbor] + p.y;
        
        int2 m = inv(neighborCoordx, neighborCoordy, nb, rb, WARPSIZE, lid, nullptr, nullptr);
        if (lid == 0){
            coords[neighbor] = m;
        }
        //if (lid ==0 && blockIdx.x == 1 && blockIdx.y == 0){
        //    printf("%i: b(%i, %i) con p(%i,%i) -> %i, %i -----> %i, %i\n", neighbor, blockIdx.x, blockIdx.y, p.x, p.y, neighborCoordx, neighborCoordy, coords[neighbor].x, coords[neighbor].y);
            
        //}
    }

    __syncthreads();
    // top side
    if(haloSideToCopy == 0 & (cacheSize-1)){
        int neighborCoordx = traducx[1] + p.x;
        int neighborCoordy = traducy[1] + p.y;
        if (neighborCoordy == -1 || (neighborCoordx & (nb-1-neighborCoordy)) != 0) {
            cache[CINDEX(threadIdx.x, -1)]         = 0;
        } else {
            int2 m = coords[1];
            //printf("top side tx: %i ty: %i | px: %i, py: %i - (%i, %i) -> %i,%i\n", threadIdx.x, threadIdx.y, p.x, p.y, traducx, traducy, m.x, m.y);
            m.x = m.x*blockDim.x + threadIdx.x;
            m.y = m.y*blockDim.y + blockDim.y - 1;
            cache[CINDEX(threadIdx.x, -1)]         = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
    // right side
    if(haloSideToCopy == 1 & (cacheSize-1)){
        int neighborCoordx = traducx[4] + p.x;
        int neighborCoordy = traducy[4] + p.y;
        if (neighborCoordx == nb || (neighborCoordx & (nb-1-neighborCoordy)) != 0){
            cache[CINDEX(cacheSize, threadIdx.x)] = 0;
        } else {
            int2 m = coords[4];
            //printf("right side tx: %i ty: %i | px: %i, py: %i - (%i, %i) -> %i,%i\n", threadIdx.x, threadIdx.y, p.x, p.y, traducx, traducy, m.x, m.y);
            m.x = m.x*blockDim.x;
            m.y = m.y*blockDim.y + threadIdx.x;
            cache[CINDEX(cacheSize, threadIdx.x)] = dmat1[ GINDEX(m.x, m.y, nx) ];
        }

    }
    // left side
    if(haloSideToCopy == 2 & (cacheSize-1)){
        int neighborCoordx = traducx[3] + p.x;
        int neighborCoordy = traducy[3] + p.y;
        if (neighborCoordx == -1 || (neighborCoordx & (nb-1-neighborCoordy)) != 0) {
            //cache[CINDEX(-1, threadIdx.x)]           = 0;
            cache[CINDEX(0, threadIdx.x) - 1]           = 0;
        } else {
            int2 m = coords[3];
            //printf("left side tx: %i ty: %i | px: %i, py: %i - (%i, %i) -> %i,%i\n", threadIdx.x, threadIdx.y, p.x, p.y, p.x-1-threadIdx.x, p.y-threadIdx.y+elementInHaloSide, m.x, m.y);
            m.x = m.x*blockDim.x + blockDim.x - 1;
            m.y = m.y*blockDim.y + threadIdx.x;
            cache[CINDEX(0, threadIdx.x)-1]           = dmat1[ GINDEX(m.x, m.y, nx) ];
            //cache[CINDEX(-1, threadIdx.x)]           = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
    // bottom side
    if(haloSideToCopy == 3 & (cacheSize-1)){
        int neighborCoordx = traducx[6] + p.x;
        int neighborCoordy = traducy[6] + p.y;
        if (neighborCoordy == nb || (neighborCoordx & (nb-1-neighborCoordy)) != 0){
            cache[CINDEX(threadIdx.x, cacheSize)] = 0;
        } else {
            int2 m = coords[6];
            m.x = m.x*blockDim.x + threadIdx.x;
            m.y = m.y*blockDim.y;
            //printf("bottom side tx: %i ty: %i | px: %i, py: %i - (%i, %i) -> %i,%i\n", threadIdx.x, threadIdx.y, p.x, p.y, traducx, traducy, m.x, m.y);
            cache[CINDEX(threadIdx.x, cacheSize)] = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
    // local thread 0 in charge of the four corners
    // compartir el valor del bloque traducido al fractal para el warp asi se aprovecha la reduccion y el loglog(n)
    if(haloSideToCopy == 4 & (cacheSize-1)){
        // top-left
        int neighborCoordx = traducx[0] + p.x;
        int neighborCoordy = traducy[0] + p.y;
        if (neighborCoordx == -1 || neighborCoordy == -1 || (neighborCoordx & (nb-1-neighborCoordy)) != 0){
            cache[CINDEX(-1, -1)]           = 0;
        } else {
            int2 m = coords[0];
            m.x = m.x*blockDim.x + blockDim.x - 1;
            m.y = m.y*blockDim.y + blockDim.y - 1;
            cache[CINDEX(-1, -1)]           = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }   
        // top-right
    if(haloSideToCopy == 5 & (cacheSize-1)){
        int neighborCoordx = traducx[2] + p.x;
        int neighborCoordy = traducy[2] + p.y;
        if ( neighborCoordx == nb || neighborCoordy == -1 || ( neighborCoordx & (nb-1-neighborCoordy)) != 0){
            cache[CINDEX(cacheSize, -1)]      = 0;
        } else {
            int2 m = coords[2];
            m.x = m.x*blockDim.x;
            m.y = m.y*blockDim.y + blockDim.y - 1;
            cache[CINDEX(cacheSize, -1)]      = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
        // bot-left
    if(haloSideToCopy == 6 & (cacheSize-1)){
        int neighborCoordx = traducx[5] + p.x;
        int neighborCoordy = traducy[5] + p.y;
        if ( neighborCoordx == -1 || neighborCoordy == nb || ( neighborCoordx & (nb-1-neighborCoordy)) != 0){
            cache[CINDEX(-1, cacheSize)]      = 0;
        } else {
            int2 m = coords[5];
            m.x = m.x*blockDim.x + blockDim.x - 1;
            m.y = m.y*blockDim.y;
            cache[CINDEX(-1, cacheSize)]      = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }

        // bot-right
    if(haloSideToCopy == 7 & (cacheSize-1)){
        int neighborCoordx = traducx[7] + p.x;
        int neighborCoordy = traducy[7] + p.y;
        if ( neighborCoordx == nb || neighborCoordy == nb || ( neighborCoordx & (nb-1-neighborCoordy)) != 0){
            cache[CINDEX(cacheSize, cacheSize)] = 0;
        } else {
            int2 m = coords[7];
            m.x = m.x*blockDim.x;
            m.y = m.y*blockDim.y;
            cache[CINDEX(cacheSize, cacheSize)] = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
    /*for (int i=0; i<10000; i++);
    if (tid ==0 && blockIdx.x == 1 && blockIdx.y == 0){
        for (int i=0; i<BSIZE2D+2; i++){
            for (int j=0; j<BSIZE2D+2; j++){
                printf("%i ", cache[i*(BSIZE2D+2)+j]);
            }
            printf("\n");
        }
    }*/


    // transition function applied to state 'c' and written into mat2
    //dmat2[GINDEX(local.x, local.y, nx)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
    
    __syncthreads();
    if(lpx < n && lpy < n && (lpx & (n-1-lpy)) == 0){
        //printf("%i, %i, %i\n", threadIdx.x, threadIdx.y, (int)BSIZE2D);
        unsigned int c = cache[CINDEX(threadIdx.x, threadIdx.y)];
        int nc =    cache[CINDEX(threadIdx.x-1, threadIdx.y-1)] + cache[CINDEX(threadIdx.x, threadIdx.y-1)] + cache[CINDEX(threadIdx.x+1, threadIdx.y-1)] + 
                    cache[CINDEX(threadIdx.x-1, threadIdx.y  )] +                                             cache[CINDEX(threadIdx.x+1, threadIdx.y  )] + 
                    cache[CINDEX(threadIdx.x-1, threadIdx.y+1)] + cache[CINDEX(threadIdx.x, threadIdx.y+1)] + cache[CINDEX(threadIdx.x+1, threadIdx.y+1)];
        
        dmat2[GINDEX(local.x, local.y, nx)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
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

template<typename Lambda, typename Inverse>
__global__ void kernelCompressed_tc(const size_t n, const size_t nx, const size_t ny, const size_t nb, const size_t rb, MTYPE *dmat1, MTYPE *dmat2, Lambda map, Inverse inv, const int WARPSIZE){
    
    // const char o int, tambien podria ser 1 solo arreglo y usar indexacion para extenderlo
    int traducx[8] = {-1,  0,  1, -1, /*0,*/  1, -1,  0, 1};
    int traducy[8] = {-1, -1, -1,  0, /*0,*/  0,  1,  1, 1};

    __shared__ half mata[256];
    __shared__ half matb[256];

    __shared__ MTYPE cache[(BSIZE2D+2)*(BSIZE2D+2)];
    __shared__ int2 coords[8]; //vecindad de moore

    // In this method, this mapping function returns block level coordinates!
    auto p = map(nb, rb, WARPSIZE, mata, matb);
    uint2 local;
    local.x = blockIdx.x*blockDim.x + threadIdx.x;
    local.y = blockIdx.y*blockDim.y + threadIdx.y;

    int tid = blockDim.x*threadIdx.y + threadIdx.x;
    int haloSideToCopy = tid>>5;

    if (local.x < nx && local.y < ny)
        cache[CINDEX(threadIdx.x, threadIdx.y)] = dmat1[GINDEX(local.x, local.y, nx)];
    int neighborCoordx;
    int neighborCoordy;

    neighborCoordx = traducx[haloSideToCopy&7] + p.x;
    neighborCoordy = traducy[haloSideToCopy&7] + p.y;

    inv(neighborCoordx, neighborCoordy, nb, rb, WARPSIZE, tid, mata, matb);

    __syncthreads();
    if (tid <256){
        coords[haloSideToCopy] = (int2){mata[haloSideToCopy<<1], mata[(haloSideToCopy<<1)+1]};
    }
    __syncthreads();
       /*if (tid ==0 && blockIdx.x == 2 && blockIdx.y == 0){
            for (int neighbor=0; neighbor<8; neighbor++)
            printf("%i: b(%i, %i) con p(%i,%i) -> %i, %i -----> %i, %i\n", neighbor, blockIdx.x, blockIdx.y, p.x, p.y, traducx[neighbor] + p.x, traducy[neighbor] + p.y, coords[neighbor].x, coords[neighbor].y);
        }*/
    // top side
    if(haloSideToCopy==0){
        int neighborCoordx = traducx[1] + p.x;
        int neighborCoordy = traducy[1] + p.y;
        if (neighborCoordy == -1 || (neighborCoordx & (nb-1-neighborCoordy)) != 0) {
            cache[CINDEX(threadIdx.x, -1)]         = 0;
        } else {
            int2 m = coords[1];
            //printf("top side tx: %i ty: %i | px: %i, py: %i - (%i, %i) -> %i,%i\n", threadIdx.x, threadIdx.y, p.x, p.y, traducx, traducy, m.x, m.y);
            m.x = m.x*blockDim.x + threadIdx.x;
            m.y = m.y*blockDim.y + blockDim.y - 1;
            cache[CINDEX(threadIdx.x, -1)]         = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
    // right side
    else if(haloSideToCopy == 1){
        int neighborCoordx = traducx[4] + p.x;
        int neighborCoordy = traducy[4] + p.y;
        if (neighborCoordx == nb || (neighborCoordx & (nb-1-neighborCoordy)) != 0){
            cache[CINDEX(BSIZE2D, threadIdx.x)] = 0;
        } else {
            int2 m = coords[4];
            //printf("right side tx: %i ty: %i | px: %i, py: %i - (%i, %i) -> %i,%i\n", threadIdx.x, threadIdx.y, p.x, p.y, traducx, traducy, m.x, m.y);
            m.x = m.x*blockDim.x;
            m.y = m.y*blockDim.y + threadIdx.x;
            cache[CINDEX(BSIZE2D, threadIdx.x)] = dmat1[ GINDEX(m.x, m.y, nx) ];
        }

    }
    // left side
    else if(haloSideToCopy == 2){
        int neighborCoordx = traducx[3] + p.x;
        int neighborCoordy = traducy[3] + p.y;
        if (neighborCoordx == -1 || (neighborCoordx & (nb-1-neighborCoordy)) != 0) {
            //cache[CINDEX(-1, threadIdx.x)]           = 0;
            cache[CINDEX(0, threadIdx.x)-1]           = 0;
        } else {
            int2 m = coords[3];
            //printf("left side tx: %i ty: %i | px: %i, py: %i - (%i, %i) -> %i,%i\n", threadIdx.x, threadIdx.y, p.x, p.y, p.x-1-threadIdx.x, p.y-threadIdx.y+elementInHaloSide, m.x, m.y);
            m.x = m.x*blockDim.x + blockDim.x - 1;
            m.y = m.y*blockDim.y + threadIdx.x;
            //cache[CINDEX(-1, threadIdx.x)]           = dmat1[ GINDEX(m.x, m.y, nx) ];
            cache[CINDEX(0, threadIdx.x)-1]           = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
    // bottom side
    else if(haloSideToCopy == 3){
        int neighborCoordx = traducx[6] + p.x;
        int neighborCoordy = traducy[6] + p.y;
        if (neighborCoordy == nb || (neighborCoordx & (nb-1-neighborCoordy)) != 0){
            cache[CINDEX(threadIdx.x, BSIZE2D)] = 0;
        } else {
            int2 m = coords[6];
            m.x = m.x*blockDim.x + threadIdx.x;
            m.y = m.y*blockDim.y;
            //printf("bottom side tx: %i ty: %i | px: %i, py: %i - (%i, %i) -> %i,%i\n", threadIdx.x, threadIdx.y, p.x, p.y, traducx, traducy, m.x, m.y);
            cache[CINDEX(threadIdx.x, BSIZE2D)] = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
    // local thread 0 in charge of the four corners
    // compartir el valor del bloque traducido al fractal para el warp asi se aprovecha la reduccion y el loglog(n)
    else if(haloSideToCopy == 4){
        // top-left
        int neighborCoordx = traducx[0] + p.x;
        int neighborCoordy = traducy[0] + p.y;
        if (neighborCoordx == -1 || neighborCoordy == -1 || (neighborCoordx & (nb-1-neighborCoordy)) != 0){
            cache[CINDEX(-1, -1)]           = 0;
        } else {
            int2 m = coords[0];
            m.x = m.x*blockDim.x + blockDim.x - 1;
            m.y = m.y*blockDim.y + blockDim.y - 1;
            cache[CINDEX(-1, -1)]           = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }   
        // top-right
    else if(haloSideToCopy == 5){
        int neighborCoordx = traducx[2] + p.x;
        int neighborCoordy = traducy[2] + p.y;
        if ( neighborCoordx == nb || neighborCoordy == -1 || ( neighborCoordx & (nb-1-neighborCoordy)) != 0){
            cache[CINDEX(BSIZE2D, -1)]      = 0;
        } else {
            int2 m = coords[2];
            m.x = m.x*blockDim.x;
            m.y = m.y*blockDim.y + blockDim.y - 1;
            cache[CINDEX(BSIZE2D, -1)]      = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
        // bot-left
    else if(haloSideToCopy == 6){
        int neighborCoordx = traducx[5] + p.x;
        int neighborCoordy = traducy[5] + p.y;
        if ( neighborCoordx == -1 || neighborCoordy == nb || ( neighborCoordx & (nb-1-neighborCoordy)) != 0){
            cache[CINDEX(-1, BSIZE2D)]      = 0;
        } else {
            int2 m = coords[5];
            m.x = m.x*blockDim.x + blockDim.x - 1;
            m.y = m.y*blockDim.y;
            cache[CINDEX(-1, BSIZE2D)]      = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }

        // bot-right
   else if(haloSideToCopy == 7){
        int neighborCoordx = traducx[7] + p.x;
        int neighborCoordy = traducy[7] + p.y;
        if ( neighborCoordx == nb || neighborCoordy == nb || ( neighborCoordx & (nb-1-neighborCoordy)) != 0){
            cache[CINDEX(BSIZE2D, BSIZE2D)] = 0;
        } else {
            int2 m = coords[7];
            m.x = m.x*blockDim.x;
            m.y = m.y*blockDim.y;
            cache[CINDEX(BSIZE2D, BSIZE2D)] = dmat1[ GINDEX(m.x, m.y, nx) ];
        }
    }
    /*__syncthreads();
    __syncthreads();*/
    /*for (int i=0; i<10000; i++);
    if (tid ==0 && blockIdx.x == 1 && blockIdx.y == 0){
        for (int i=0; i<BSIZE2D+2; i++){
            for (int j=0; j<BSIZE2D+2; j++){
                printf("%i ", cache[i*(BSIZE2D+2)+j]);
            }
            printf("\n");
        }
    }*/


    // transition function applied to state 'c' and written into mat2
    //dmat2[GINDEX(local.x, local.y, nx)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
    
    //__syncthreads();
    __syncthreads();
    if((threadIdx.x & (BSIZE2D-1-threadIdx.y)) == 0){
        //printf("%i, %i, %i\n", threadIdx.x, threadIdx.y, (int)BSIZE2D);
        unsigned int c = cache[CINDEX(threadIdx.x, threadIdx.y)];
        int nc =    cache[CINDEX(threadIdx.x-1, threadIdx.y-1)] + cache[CINDEX(threadIdx.x, threadIdx.y-1)] + cache[CINDEX(threadIdx.x+1, threadIdx.y-1)] + 
                    cache[CINDEX(threadIdx.x-1, threadIdx.y  )] +                                             cache[CINDEX(threadIdx.x+1, threadIdx.y  )] + 
                    cache[CINDEX(threadIdx.x-1, threadIdx.y+1)] + cache[CINDEX(threadIdx.x, threadIdx.y+1)] + cache[CINDEX(threadIdx.x+1, threadIdx.y+1)];
        
        dmat2[GINDEX(local.x, local.y, nx)] = c*h(nc, EL, EU) + (1-c)*h(nc, FL, FU);
    }
}


#endif
