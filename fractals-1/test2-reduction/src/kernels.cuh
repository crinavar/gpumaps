#ifndef KERNELS_H
#define KERNELS_H

#define WARPSIZE 32
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

template<typename Lambda>
__global__ void kernel_block_reduction(MTYPE *E, double* res, const unsigned int n, const unsigned int nb, const unsigned int rb, const int WSIZE, Lambda map){
    __shared__ MTYPE shmem[BSIZE1D];


    int val;
    int tid = threadIdx.y*blockDim.x + threadIdx.x;
    int lane = tid & (WSIZE - 1);
    int wid = tid / WSIZE;

    auto p = map(nb, rb, WSIZE);

    if (p.y == 0xFFFFFFFF || p.x == 0xFFFFFFFF) //fix
        shmem[tid] = 0;
    else{
        shmem[tid] = *(E + p.y*n + p.x);
    }
    __syncthreads();
    val = warp_reduce1D(shmem[tid], WSIZE);

    __syncthreads();
    
    shmem[tid] = 0;
    __syncthreads();

    if (lane == 0 ) {
        shmem[wid] = val;
    }

    //if (lane==0 && wid==0 )
    //    printf("block: %i,%i  val es: %i\n", blockIdx.x, blockIdx.y, val);

    __syncthreads();

    if (wid == 0){
        val = warp_reduce1D(shmem[tid], WSIZE);
    }

    __syncthreads();

    if (tid == 0)
        atomicAdd(res, val);
    
}

template<typename Lambda>
__global__ void kernel_write_int(int *E, const unsigned int n, const unsigned int nb, const unsigned int rb, const int c, const int WSIZE, Lambda map){
    if(blockIdx.x > blockIdx.y)
        return;
    auto p = map(nb, rb, WSIZE);
    //printf("\n(thread %i %i %i)= d(%i, %i, %i) = (%f, %f, %f, %f)\n", threadIdx.x, threadIdx.y, threadIdx.z, p.x, p.y, p.z, d[p.x].x, d[p.x].y, d[p.x].z, d[p.x].w);
    if( (p.x & (n-1-p.y)) == 0 ){
        E[p.y*n + p.x] = c;
    }
    return;
}

template<typename Lambda>
__global__ void kernel_write_float(float *E, const unsigned int n, const unsigned int nb, const unsigned int rb, const float c, const int WSIZE, Lambda map){
    auto p = map(nb, rb, WSIZE);
    //printf("\n(thread %i %i %i)= d(%i, %i, %i) = (%f, %f, %f, %f)\n", threadIdx.x, threadIdx.y, threadIdx.z, p.x, p.y, p.z, d[p.x].x, d[p.x].y, d[p.x].z, d[p.x].w);
    if( ((int)p.x & (int)(n-1-p.y)) == 0 ){
        E[p.y*n + p.x] = c;
    }
    return;
}

template<typename Lambda>
__global__ void kernel_write_char(char *E, const unsigned int n, const unsigned int nb, const unsigned int rb, const char c, const int WSIZE, Lambda map){
    if(blockIdx.x > blockIdx.y)
        return;
    auto p = map(nb, rb, WSIZE);
    //printf("\n(thread %i %i %i)= d(%i, %i, %i) = (%f, %f, %f, %f)\n", threadIdx.x, threadIdx.y, threadIdx.z, p.x, p.y, p.z, d[p.x].x, d[p.x].y, d[p.x].z, d[p.x].w);
    if( p.x < n && p.y < n && (p.x & (n-1-p.y)) == 0 ){
        E[p.y*n + p.x] = c;
    }
    return;
}

template<typename Lambda>
__global__ void kernel_write_lambda(MTYPE *E, const unsigned int n, const unsigned int nb, const unsigned int rb, const MTYPE c, const int WSIZE, Lambda map){
    auto p = map(nb, rb, WSIZE);
    if( p.x < n && p.y < n && (p.x & (n-1-p.y)) == 0 ){
        E[p.y*n + p.x] = c;
    }
    return;
}
#endif
