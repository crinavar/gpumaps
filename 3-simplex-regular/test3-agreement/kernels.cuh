#ifndef KERNELS_CUH
#define KERNELS_CUH

#define ONETHIRD 0.3333333333f

__device__ inline void work(const float *data, float *mat, unsigned long index, uint3 p){
    float3 dp = (float3){mat[p.x], mat[p.y], mat[p.z]};
    float x = (dp.x+dp.y+dp.z)*ONETHIRD;
    mat[index] = 1.0f - sqrtf(ONETHIRD * ((x-dp.x)*(x-dp.x) + (x-dp.y)*(x-dp.y) + (x-dp.z)*(x-dp.z)));
}

template<typename Lambda>
__global__ void k_setoutdata(float *d, unsigned long n, char x, Lambda map){
    auto p = map();
    if(p < n){
        d[p] = x; 
    }
    return;
}

// kernel Bounding Box
template<typename Lambda>
__global__ void kernel0(const float *data, float *mat, const unsigned long n, const unsigned long V, Lambda map){
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
__global__ void kernel1(const float *data, float *mat, const unsigned long n, const unsigned long V, Lambda map){
    // lambda map
	auto p = map();
    //if(p.x == 516 && p.y == 518){
    //    printf("thread %i  %i  %i\n", p.x, p.y, p.z);
    //}
    if(p.x < p.y && p.y < p.z){
        unsigned long index = p.z*n*n + p.y*n + p.x;
        if(index < V){
            work(data, mat, index, p);
        }
    }
	return;
}
#endif
