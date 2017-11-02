#ifndef GPUBENCHMARKS_CUH
#define GPUBENCHMARKS_CUH

double bbox(const unsigned int n, const unsigned int REPEATS){
    printf("Bounding Box\n");
    float *hdata, *ddata, *hmat, *dmat;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_bbox_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (){
            return (uint2){blockIdx.y*blockDim.y + threadIdx.y, blockIdx.x*blockDim.x + threadIdx.x};
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map);
    // check result
    verify_result(n, msize, hdata, ddata, hmat, dmat);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    return time;
}

double avril(const unsigned int n, const unsigned int REPEATS){
    printf("Avril\n");
    float *hdata, *ddata, *hmat, *dmat;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &ddata, &hmat, &dmat, &msize, &trisize);	
    gen_bbox_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (){
            return (uint2){blockIdx.y*blockDim.y + threadIdx.y, blockIdx.x*blockDim.x + threadIdx.x};
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map);
    // check result
    verify_result(n, msize, hdata, ddata, hmat, dmat);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    return time;
}

double lambda_newton(const unsigned int n, const unsigned int REPEATS){
    printf("Lambda (Newton)\n");
    float *hdata, *ddata, *hmat, *dmat;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &ddata, &hmat, &dmat, &msize, &trisize);	
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (){
            uint2 p;
            unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
            p.y = __fadd_rn(newton_sqrtf(__fmaf_rn(2.0f, (float)bc, 0.25f)), OFFSET);
            p.x = (bc - (p.y*(p.y+1) >> 1));

            p.y = p.y * blockDim.y + threadIdx.y;
            p.x = p.x * blockDim.x + threadIdx.x;
            return p;
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map);
    // check result
    verify_result(n, msize, hdata, ddata, hmat, dmat);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    return time;
}

double lambda_standard(const unsigned int n, const unsigned int REPEATS){
    printf("Lambda (standard)\n");
    float *hdata, *ddata, *hmat, *dmat;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &ddata, &hmat, &dmat, &msize, &trisize);	
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (){
        
            uint2 p;
            unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
            p.y = __fadd_rn(newton_sqrtf(__fmaf_rn(2.0f, (float)bc, 0.25f)), OFFSET);
            p.x = (bc - (p.y*(p.y+1) >> 1));

            p.y = p.y * blockDim.y + threadIdx.y;
            p.x = p.x * blockDim.x + threadIdx.x;
            return p;
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map);
    // check result
    verify_result(n, msize, hdata, ddata, hmat, dmat);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    return time;
}

double lambda_inverse(const unsigned int n, const unsigned int REPEATS){
    printf("Lambda (inverse)\n");
    float *hdata, *ddata, *hmat, *dmat;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &ddata, &hmat, &dmat, &msize, &trisize);	
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (){
        
            uint2 p;
            unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
            p.y = __fadd_rn(newton_sqrtf(__fmaf_rn(2.0f, (float)bc, 0.25f)), OFFSET);
            p.x = (bc - (p.y*(p.y+1) >> 1));

            p.y = p.y * blockDim.y + threadIdx.y;
            p.x = p.x * blockDim.x + threadIdx.x;
            return p;
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map);
    // check result
    verify_result(n, msize, hdata, ddata, hmat, dmat);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    return time;
}


double lambda_flatrec(const unsigned int n, const unsigned int REPEATS){
    return 0.0;
}

double rectangle_map(const unsigned int n, const unsigned int REPEATS){
    return 0.0;
}


double recursive_map(const unsigned int n, int k, const unsigned int REPEATS){
    return 0.0;
}

#endif
