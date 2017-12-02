#ifndef GPUBENCHMARKS_CUH
#define GPUBENCHMARKS_CUH

#define AVRIL_LIMIT 8192

double bbox(const unsigned int n, const unsigned int REPEATS, double DENSITY){
    #ifdef DEBUG
        printf("[Bounding Box]\n");
    #endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, DENSITY);	
    gen_bbox_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int a1, const unsigned int a2){
        if(blockIdx.x > blockIdx.y){
            return (int2){-1,-2};
        }
        return (int2){blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y};
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat1, dmat2, map);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat2);
	cudaFree(ddata);
	cudaFree(dmat1);
	cudaFree(dmat2);
	free(hdata); 
    free(hmat);
    return time*check;
}

double avril(const unsigned int n, const unsigned int REPEATS, double DENSITY){
#ifdef DEBUG
    printf("[Avril]\n");
#endif
    if(n > AVRIL_LIMIT){ 
	return 0.0f;
    }
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, DENSITY);	
    gen_avril_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int a1, const unsigned int a2){
        int k = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
        //int a = __fadd_rn((float)N, __fadd_rn(0.5f, - (carmack_sqrtf(__fmaf_rn((float)N, (float)N, -(float)N) + __fmaf_rn(2.0f, (float)k, 0.25f)))));
        int2 p;
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
        return (int2){p.x-1, p.y-1};
	};
    // benchmark
    double time = benchmark_map_avril(REPEATS, block, grid, n, msize, trisize, ddata, dmat1, dmat2, map);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat2);
	cudaFree(ddata);
	cudaFree(dmat1);
	cudaFree(dmat2);
	free(hdata); 
    free(hmat);
    return time*check;
}

double lambda_newton(const unsigned int n, const unsigned int REPEATS, double DENSITY){
#ifdef DEBUG
    printf("[Lambda (Newton)]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, DENSITY);	
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int a1, const unsigned int a2){
            int2 p;
            unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
            p.y = __fadd_rn(newton_sqrtf(__fmaf_rn(2.0f, (float)bc, 0.25f)), OFFSET);
            p.x = (bc - (p.y*(p.y+1) >> 1));

            p.y = p.y * blockDim.y + threadIdx.y;
            p.x = p.x * blockDim.x + threadIdx.x;
            return p;
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat1, dmat2, map);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat2);
	cudaFree(ddata);
	cudaFree(dmat1);
	cudaFree(dmat2);
	free(hdata); 
    free(hmat);
    return time*check;
}

double lambda_standard(const unsigned int n, const unsigned int REPEATS, double DENSITY){
#ifdef DEBUG
    printf("[Lambda (standard)]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, DENSITY);	
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int a1, const unsigned int a2){
        int2 p;
        unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
        //p.y = sqrtf(0.25 + 2.0f*(float)bc) - 0.5f;
        p.y = __fadd_rn(sqrtf(__fmaf_rn(2.0f, (float)bc, 0.25f)), OFFSET);
        p.x = (bc - (p.y*(p.y+1) >> 1));
        p.y = p.y * blockDim.y + threadIdx.y;
        p.x = p.x * blockDim.x + threadIdx.x;
        return p;
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat1, dmat2, map);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat2);
	cudaFree(ddata);
	cudaFree(dmat1);
	cudaFree(dmat2);
	free(hdata); 
    free(hmat);
    return time*check;
}

double lambda_inverse(const unsigned int n, const unsigned int REPEATS, double DENSITY){
#ifdef DEBUG
    printf("[Lambda (inverse)]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, DENSITY);	
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int a1, const unsigned int a2){
        int2 p;
        unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
        float arg = __fmaf_rn(2.0f, (float)bc, 0.25f);
        p.y = __fmaf_rn(arg, rsqrtf(arg), OFFSET);// + 0.001f;
        p.x = (bc - (p.y*(p.y+1) >> 1));

        p.y = p.y * blockDim.y + threadIdx.y;
        p.x = p.x * blockDim.x + threadIdx.x;
        return p;
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat1, dmat2, map);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat2);
	cudaFree(ddata);
	cudaFree(dmat1);
	cudaFree(dmat2);
	free(hdata); 
    free(hmat);
    return time*check;
}


double lambda_flatrec(const unsigned int n, const unsigned int REPEATS, double DENSITY){
#ifdef DEBUG
    printf("[Flat Recursive] (TODO)\n");
#endif
    return 0.0;
}

double rectangle_map(const unsigned int n, const unsigned int REPEATS, double DENSITY){
#ifdef DEBUG
    printf("[Rectangle]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, DENSITY);	
    gen_rectangle_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned int n, const unsigned int msize, const unsigned int a1, const unsigned int a2){
        int2 p;
        p.y = blockIdx.y * blockDim.y + threadIdx.y;
        p.x = blockIdx.x * blockDim.x + threadIdx.x;
        if(p.y >= n+1 || p.x >= n/2){
            p = (int2){1,0};
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
    double time = benchmark_map_rectangle(REPEATS, block, grid, n, msize, trisize, ddata, dmat1, dmat2, map);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat2);
	cudaFree(ddata);
	cudaFree(dmat1);
	cudaFree(dmat2);
	free(hdata); 
    free(hmat);
    return time*check;
}


double recursive_map(const unsigned int n, int recn, const unsigned int REPEATS, double DENSITY){
#ifdef DEBUG
    printf("[Recursive] cnb(%i) = %i\n", n, cntsetbits(n));
#endif
    if(cntsetbits(n) != 1){
	return 0.0;
    }
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat1, *dmat2;
	unsigned int msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat1, &dmat2, &msize, &trisize, DENSITY);	
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
        int2 p;
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
        int2 p;
        int rec_index = blockIdx.x/gridDim.y;
        p.x = blockIdx.x * blockDim.x + threadIdx.x;
        p.y = (blockIdx.y + rec_index*gridDim.y)*blockDim.y + threadIdx.y;
        if( p.x > p.y )
            return (int2){1,0};
        else
            return p;
    };
    // benchmark
    double time = benchmark_map_recursive(REPEATS, block, grid, n, msize, trisize, ddata, dmat1, dmat2, maprec, mapdiag, m, k);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat2);
	cudaFree(ddata);
	cudaFree(dmat1);
	cudaFree(dmat2);
	free(hdata); 
    free(hmat);
    return time*check;
}

#endif
