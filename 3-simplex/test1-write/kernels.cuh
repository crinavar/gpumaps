#ifndef KERNELS_CUH
#define KERNELS_CUH
__device__ inline void work(const float *data, char *mat, unsigned long index, uint3 p){
    mat[index] = 1;
}

template<typename Lambda>
__global__ void k_setoutdata(char *d, unsigned long n, char x, Lambda map){
    auto p = map();
    if(p < n){
        d[p] = x; 
    }
    return;
}

// kernel Bounding Box
template<typename Lambda>
__global__ void kernel0(const float *data, char *mat, const unsigned long n, const unsigned long V, Lambda map){
    if(blockIdx.x > blockIdx.y || blockIdx.y > blockIdx.z){
        return;
    }
	auto p = map();
    if(p.x < p.y && p.y < p.z){
        unsigned long index = p.z*n*n + p.y*n + p.x;
        if(index < V){
            work(data, mat, index, p);
        }
    }
    return;
}

// kernel non-linear Map
template<typename Lambda>
__global__ void kernel1(const float *data, char *mat, const unsigned long n, const unsigned long V, Lambda map){
    // lambda map
	auto p = map();
    if(p.x < p.y && p.y < p.z){
        unsigned long index = p.z*n*n + p.y*n + p.x;
        if(index < V){
            work(data, mat, index, p);
        }
    }
	return;
}
#endif
