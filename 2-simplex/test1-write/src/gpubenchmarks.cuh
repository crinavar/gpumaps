#ifndef GPUBENCHMARKS_CUH
#define GPUBENCHMARKS_CUH

double bbox(const unsigned int n, const unsigned int REPEATS){
#ifdef DEBUG
    printf("[Bounding Box]\n");
#endif
    float *hdata, *ddata;
    char *hmat, *dmat;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_bbox_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int a1, const unsigned int a2){
        if(blockIdx.x > blockIdx.y){
            return (uint2){1,0};
        }
        return (uint2){blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y};
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
#ifdef DEBUG
    printf("[Avril]\n");
#endif
    float *hdata, *ddata;
    char *hmat, *dmat;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_avril_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int a1, const unsigned int a2){
        int k = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
        //int a = __fadd_rn((float)N, __fadd_rn(0.5f, - (carmack_sqrtf(__fmaf_rn((float)N, (float)N, -(float)N) + __fmaf_rn(2.0f, (float)k, 0.25f)))));
        uint2 p;
        //float res = (-(2.0f*(float)n + 1.0f) + sqrtf(4.0f*(float)n*(float)n - 4.0f*(float)n - 8.0f*(float)k + 1.0f))/(-2.0f);
        p.x = (-(2.0f*(float)n + 1.0f) + sqrtf(4.0f*(float)n*(float)n - 4.0f*(float)n - 8.0f*(float)k + 1.0f))/(-2.0f);
        p.y = (p.x+1) + k - ((p.x-1)*(2*n-p.x))/2;
        if(p.y > n){
            p.x++;
            p.y = p.x+(p.y-n);
        }
        if(p.x >= p.y){
            p.y = n - (p.x-p.y);
            p.x--;
        }
        return (uint2){p.x-1, p.y-1};
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
#ifdef DEBUG
    printf("[Lambda (Newton)]\n");
#endif
    float *hdata, *ddata;
    char *hmat, *dmat;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int a1, const unsigned int a2){
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
#ifdef DEBUG
    printf("[Lambda (standard)]\n");
#endif
    float *hdata, *ddata;
    char *hmat, *dmat;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int a1, const unsigned int a2){
        uint2 p;
        unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
        //p.y = sqrtf(0.25 + 2.0f*(float)bc) - 0.5f;
        p.y = __fadd_rn(sqrtf(__fmaf_rn(2.0f, (float)bc, 0.25f)), OFFSET);
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
#ifdef DEBUG
    printf("[Lambda (inverse)]\n");
#endif
    float *hdata, *ddata;
    char *hmat, *dmat;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int a1, const unsigned int a2){
        uint2 p;
        unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
        float arg = __fmaf_rn(2.0f, (float)bc, 0.25f);
        p.y = __fmaf_rn(arg, rsqrtf(arg), OFFSET);// + 0.001f;
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
#ifdef DEBUG
    printf("[Flat Recursive] (TODO)\n");
#endif
    return 0.0;
}

double rectangle_map(const unsigned int n, const unsigned int REPEATS){
#ifdef DEBUG
    printf("[Rectangle]\n");
#endif
    float *hdata, *ddata;
    char *hmat, *dmat;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_rectangle_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int a1, const unsigned int a2){
        uint2 p;
        p.y = blockIdx.y * blockDim.y + threadIdx.y;
        p.x = blockIdx.x * blockDim.x + threadIdx.x;
        if(p.y >= n+1 || p.x >= n/2){
            p = (uint2){1,0};
        }
        else{
            if( p.x >= p.y ){
                p.x = n - p.x -1;
                p.y = n - p.y -1;
            }
            else{
                p.y = p.y-1;
            }
        }
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


double recursive_map(const unsigned int n, int recn, const unsigned int REPEATS){
#ifdef DEBUG
    printf("[Recursive]\n");
#endif
    float *hdata, *ddata;
    char *hmat, *dmat;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_recursive_pspace(n, block, grid);

    unsigned int m=n/recn;
    if( (m % block.x) != 0 ){
        fprintf(stderr, "error: m=%i, not a multiple of %i\n", m, block.x);
        exit(1);
    }
    unsigned int k=cf_log2i(recn);
#ifdef DEBUG
    printf("[recursive params] --> n=%i recn = %i = 2^k  m=%i  k=%i\n", n, recn, m, k);
#endif
    // formulate map
    auto maprec = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int bx, const unsigned int by){
        uint2 p;
        // calcula el indice del bloque recursivo, es division entera.
        int rec_index = blockIdx.x/gridDim.y;
        // calcula el offset, el valor x del bloque respecto al recblock que le corresponde.
        int dbx = blockIdx.x % gridDim.y;
        p.x = (bx+(gridDim.y*rec_index*2) + dbx)*blockDim.x + threadIdx.x;
        p.y = (by+(gridDim.y*rec_index*2) + blockIdx.y)*blockDim.y + threadIdx.y;
        return p;
    };
    auto mapdiag = [] __device__ (const unsigned int n, const unsigned int msize, unsigned int aux1, unsigned int aux2){
        // calcula el indice del bloque recursivo, es division entera.
        uint2 p;
        int rec_index = blockIdx.x/gridDim.y;
        p.x = blockIdx.x * blockDim.x + threadIdx.x;
        p.y = (blockIdx.y + rec_index*gridDim.y)*blockDim.y + threadIdx.y;
        if( p.x > p.y )
            return (uint2){1,0};
        else
            return p;
    };
    // benchmark
    double time = benchmark_map_recursive(REPEATS, block, grid, n, msize, trisize, ddata, dmat, maprec, mapdiag, m, k);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    return time*check;
}

#endif
