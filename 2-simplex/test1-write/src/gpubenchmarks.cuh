//////////////////////////////////////////////////////////////////////////////////
//  gpumaps                                                                     //
//  A GPU benchmark of mapping functions                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
//                                                                              //
//  This file is part of gpumaps.                                               //
//  gpumaps is free software: you can redistribute it and/or modify             //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  gpumaps is distributed in the hope that it will be useful,                  //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with gpumaps.  If not, see <http://www.gnu.org/licenses/>.            //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
#ifndef GPUBENCHMARKS_CUH
#define GPUBENCHMARKS_CUH


double bbox(const unsigned long n, const unsigned int REPEATS){
#ifdef DEBUG
    printf("[Bounding Box]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
	unsigned long msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_bbox_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3){
        if(blockIdx.x > blockIdx.y){
            return (uint2){1,0};
        }
        return (uint2){blockIdx.x*blockDim.x + threadIdx.x, blockIdx.y*blockDim.y + threadIdx.y};
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map, 0, 0, 0);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat, grid, block);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    //return time*check;
    return time;
}

double avril(const unsigned long n, const unsigned int REPEATS){
#ifdef DEBUG
    printf("[Avril]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
	unsigned long msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_avril_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3){
        int k = (blockIdx.y*gridDim.x + blockIdx.x)*blockDim.x + threadIdx.x;
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
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map, 0, 0, 0);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat, grid, block);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    return time*check;
}

double lambda_newton(const unsigned long n, const unsigned int REPEATS){
#ifdef DEBUG
    printf("[Lambda (Newton)]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
	unsigned long msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3){
            uint2 p;
            unsigned int bc = blockIdx.x + blockIdx.y*gridDim.x;
            p.y = __fadd_rn(newton_sqrtf(__fmaf_rn(2.0f, (float)bc, 0.25f)), OFFSET);
            p.x = (bc - (p.y*(p.y+1) >> 1));

            p.y = p.y * blockDim.y + threadIdx.y;
            p.x = p.x * blockDim.x + threadIdx.x;
            return p;
    };
    // benchmark
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map, 0, 0, 0);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat, grid, block);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    return time*check;
}

double lambda_standard(const unsigned long n, const unsigned int REPEATS){
#ifdef DEBUG
    printf("[Lambda (standard)]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
	unsigned long msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3){
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
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map, 0, 0, 0);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat, grid, block);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    return time*check;
}

double lambda_inverse(const unsigned long n, const unsigned int REPEATS){
#ifdef DEBUG
    printf("[Lambda (inverse)]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
	unsigned long msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_lambda_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3){
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
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map, 0, 0, 0);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat, grid, block);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    //return time*check;
    return time;
}


double hadouken(const unsigned long n, const unsigned int REPEATS){
#ifdef DEBUG
    printf("[Hadouken]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
    unsigned long msize, trisize;
    dim3 block(BSIZE2D, BSIZE2D);
    init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
#ifdef DEBUG
    printf("gen_hadouken_pspace(%i, ...)\n", n);
#endif
    auto map = [] __device__ (const unsigned long n, const unsigned long msize, const int aux1, const int aux2, const int aux3){
        // trapezoid map 
        const unsigned int h    = WORDLEN - __clz(blockIdx.y+1);
        const unsigned int qb   = (1 << h)*(blockIdx.x >> h);
        const int k = (int)blockIdx.y - aux1 > 0 ? 1 : 0;
        return (uint2){ aux3 + (blockIdx.x + qb + k*gridDim.x)*blockDim.x + threadIdx.x, aux3 + (blockIdx.y - k*aux2 + (qb << 1))*blockDim.y + threadIdx.y };
    };
    // benchmark
    double time = benchmark_map_hadouken(REPEATS, block, n, msize, trisize, ddata, dmat, map);
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat, dim3(1,1,1), block);
    cudaFree(ddata);
    cudaFree(dmat);
    free(hdata); 
    free(hmat);
    //return time*check;
    return time;
}

double rectangle_map(const unsigned long n, const unsigned int REPEATS){
#ifdef DEBUG
    printf("[Rectangle]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
	unsigned long msize, trisize;
    dim3 block, grid;
	init(n, &hdata, &hmat, &ddata, &dmat, &msize, &trisize);	
    gen_rectangle_pspace(n, block, grid);
    // formulate map
    auto map = [] __device__ (const unsigned long n, const unsigned long msize, const unsigned int a1, const unsigned int a2, const unsigned int a3){
        uint2 p;
        p.y = blockIdx.y * blockDim.y + threadIdx.y;
        p.x = blockIdx.x * blockDim.x + threadIdx.x;
        if(p.y >= n+1 || p.x > n/2){
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
    double time = benchmark_map(REPEATS, block, grid, n, msize, trisize, ddata, dmat, map, 0, 0, 0);
    // check result
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat, grid, block);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    //return time*check;
    return time;
}


double recursive_map(const unsigned long n, int recn, const unsigned int REPEATS){
#ifdef DEBUG
    printf("[Recursive]\n");
#endif
    DTYPE *hdata, *ddata;
    MTYPE *hmat, *dmat;
	unsigned long msize, trisize;
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
    auto maprec = [] __device__ (const unsigned long n, const unsigned long msize, const unsigned int bx, const unsigned int by, const unsigned int aux){
        uint2 p;
        // calcula el indice del bloque recursivo, es division entera.
        int rec_index = blockIdx.x/gridDim.y;
        // calcula el offset, el valor x del bloque respecto al recblock que le corresponde.
        int dbx = blockIdx.x % gridDim.y;
        p.x = (bx+(gridDim.y*rec_index*2) + dbx)*blockDim.x + threadIdx.x;
        p.y = (by+(gridDim.y*rec_index*2) + blockIdx.y)*blockDim.y + threadIdx.y;
        return p;
    };
    auto mapdiag = [] __device__ (const unsigned long n, const unsigned long msize, const unsigned int aux1, const unsigned int aux2, const unsigned int aux3){
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
    double check = (float)verify_result(n, msize, hdata, ddata, hmat, dmat, grid, block);
	cudaFree(ddata);
	cudaFree(dmat);
	free(hdata); 
    free(hmat);
    return time*check;
}
#endif
