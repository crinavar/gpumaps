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
        lm.x += __shfl_down(lm.x, offset, WSIZE);
        lm.y += __shfl_down(lm.y, offset, WSIZE);
    }
    return lm;
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
__global__ void kernel_write_char_lambda(char *E, const unsigned int n, const unsigned int nb, const unsigned int rb, const char c, const int WSIZE, Lambda map){
    auto p = map(nb, rb, WSIZE);
    //printf("\n(thread %i %i %i)= d(%i, %i, %i) = (%f, %f, %f, %f)\n", threadIdx.x, threadIdx.y, threadIdx.z, p.x, p.y, p.z, d[p.x].x, d[p.x].y, d[p.x].z, d[p.x].w);
    if( p.x < n && p.y < n && (p.x & (n-1-p.y)) == 0 ){
        E[p.y*n + p.x] = c;
    }
    return;
}
#endif
