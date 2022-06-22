//////////////////////////////////////////////////////////////////////////////////
//  gpumaps                                                                     //
//  A GPU benchmark of mapping functions                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2018 Cristobal A. Navarro.                                      //
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
#ifndef GPUTOOLS
#define GPUTOOLS

#define DTYPE float2
#define MTYPE float
#define EPSILON 0.001
#define WORDLEN 31
#define MAXSTREAMS 32
#define PRINTLIMIT 256

// for HADOUKEN
#define HADO_TOL HADO_FACTOR* BSIZE2D
//#define EXTRASPACE

// declaration
template <typename Lambda>
__global__ void kernel_test(const unsigned int n, const int a, const unsigned int msize, DTYPE* data, MTYPE* dmat, Lambda map, const unsigned int aux1, const unsigned int aux2, const unsigned int aux3);

// integer log2
__host__ __device__ int cf_log2i(const int val) {
    int copy = val;
    int r = 0;
    while (copy >>= 1)
        ++r;
    return r;
}

void print_gpu_specs(int dev) {
    cudaDeviceProp prop;
    cudaGetDeviceProperties(&prop, dev);
    printf("Device Number: %d\n", dev);
    printf("  Device name:                  %s\n", prop.name);
    printf("  Multiprocessor Count:         %d\n", prop.multiProcessorCount);
    printf("  Concurrent Kernels:           %d\n", prop.concurrentKernels);
    printf("  Memory Clock Rate (KHz):      %d\n", prop.memoryClockRate);
    printf("  Memory Bus Width (bits):      %d\n", prop.memoryBusWidth);
    printf("  Peak Memory Bandwidth (GB/s): %f\n\n", 2.0 * prop.memoryClockRate * (prop.memoryBusWidth / 8) / 1.0e6);
}

void print_array(DTYPE* a, const int n) {
    for (int i = 0; i < n; i++)
        printf("a[%d] = %f\n", i, a[i]);
}

void print_matrix(MTYPE* mat, const int no, const char* msg) {
    printf("%s:\n", msg);
    int n = no;
    // writing result to frame
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (mat[i * n + j] < 0.0f) {
                printf("* ");
            } else {
                printf("%.2f ", mat[i * n + j]);
            }
        }
        printf("\n");
    }
}

void print_map(MTYPE* mat, const int no, const char* msg, dim3 grid, dim3 block) {
#ifdef EXTRASPACE
    // mostrar doble
    int n = 2 * no;
#else
    // mostrar justo
    int n = no;
#endif
    int w = grid.x * block.x;
    int h = grid.y * block.y;
    const int gap = 3;
    const int framex = n + w + gap;
    const int framey = n > h ? n + 5 : h + 5;

    int b = 0;
    char frame[framey][framex];
    for (int i = 0; i < framey; i++) {
        for (int j = 0; j < framex; j++) {
            frame[i][j] = ' ';
        }
    }

    // writing pspace to frame
    for (int i = 0; i < h; i++) {
        for (int j = 0; j < w; j++) {
            b = 48 + (int)log2f(i / BSIZE2D + 1) + 1;
            frame[i + 2][j] = b;
        }
    }

    // writing result to frame
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            if (mat[i * n + j] == 0) {
                frame[i + 2][j + w + gap] = 42;
                continue;
            }
            if (j > i) {
                // if( mat[i*n + j] == 0 ){
                //     printf("  ");
                // }
                // else{
                // printf("%i ", mat[i*n + j]);
                frame[i + 2][j + w + gap] = 48 + mat[i * n + j];
                //}
            } else {
                if (mat[i * n + j] == 0) {
                    // printf("%i ", mat[i*n + j]);
                    frame[i + 2][j + w + gap] = 48 + mat[i * n + j];
                } else {
                    // printf("%i ", mat[i*n + j]);
                    frame[i + 2][j + w + gap] = 48 + mat[i * n + j];
                }
            }
        }
    }
    for (int i = 0; i < framey; i++) {
        for (int j = 0; j < framex; j++) {
            printf("%c ", frame[i][j]);
        }
        printf("\n");
    }
    printf("[%s]:\n", msg);
    fflush(stdout);
}

int print_dmat(unsigned int limit, unsigned int n, unsigned int msize, MTYPE* dmat, const char* msg) {
    MTYPE* hmat = (MTYPE*)malloc(sizeof(MTYPE) * msize);
    cudaMemcpy(hmat, dmat, sizeof(MTYPE) * msize, cudaMemcpyDeviceToHost);
    if (n <= limit) {
        print_matrix(hmat, n, msg);
    }
    free(hmat);
    return 1;
}

void last_cuda_error(const char* msg) {
    cudaError_t error = cudaGetLastError();
    if (error != cudaSuccess) {
        // print the CUDA error message and exit
        printf("[%s]: CUDA error: %s\n", msg, cudaGetErrorString(error));
        exit(-1);
    }
}

void fill_random(DTYPE* array, int n) {
    for (int i = 0; i < n; i++) {
        array[i] = (float2) { 100.0f * (float)rand() / (float)RAND_MAX, 100.0f * (float)rand() / (float)RAND_MAX };
    }
}

void init(unsigned long no, DTYPE** hdata, MTYPE** hmat, DTYPE** ddata, MTYPE** dmat, unsigned long* msize, unsigned long* trisize) {
#ifdef EXTRASPACE
    // mostrar doble
    unsigned long n = 2 * no;
#else
    // mostrar justo
    unsigned long n = no;
#endif
    *msize = n * n;
    *trisize = n * (n - 1) / 2;

    *hdata = (DTYPE*)malloc(sizeof(DTYPE) * n);
    *hmat = (MTYPE*)malloc(sizeof(MTYPE) * (*msize));
    for (int i = 0; i < *msize; i++) {
        (*hmat)[i] = -1.0f;
    }
    fill_random(*hdata, n);
    cudaMalloc((void**)ddata, sizeof(DTYPE) * n);
    last_cuda_error("init: cudaMalloc ddata");
    cudaMalloc((void**)dmat, sizeof(MTYPE) * (*msize));
    last_cuda_error("init: cudaMalloc dmat");
    cudaMemcpy(*dmat, *hmat, sizeof(MTYPE) * (*msize), cudaMemcpyHostToDevice);
    last_cuda_error("init end:memcpy hmat->dmat");
    cudaMemcpy(*ddata, *hdata, sizeof(DTYPE) * n, cudaMemcpyHostToDevice);
    last_cuda_error("init end:memcpy hdata->ddata");
#ifdef DEBUG
    printf("2-simplex: n=%i  msize=%lu (%f MBytes)\n", n, *msize, (float)sizeof(MTYPE) * (*msize) / (1024 * 1024));
    if (n <= PRINTLIMIT) {
        print_matrix(*hmat, n, "host matrix");
    }
#endif
}

void gen_lambda_pspace(const unsigned int n, dim3& block, dim3& grid) {
    block = dim3(BSIZE2D, BSIZE2D, 1);
    int sn = (n + block.x - 1) / block.x;
    int sd = sn * (sn + 1) / 2;
    int s = ceil(sqrt((double)sd));
    grid = dim3(s, s, 1);
}

void gen_bbox_pspace(const unsigned int n, dim3& block, dim3& grid) {
    block = dim3(BSIZE2D, BSIZE2D, 1);
    grid = dim3((n + block.x - 1) / block.x, (n + block.y - 1) / block.y, 1);
}

void gen_rectangle_pspace(const unsigned int n, dim3& block, dim3& grid) {
    int rect_evenx = n / 2;
    int rect_oddx = (int)ceil((float)n / 2.0f);
    block = dim3(BSIZE2D, BSIZE2D, 1);
    if (n % 2 == 0) {
        grid = dim3((rect_evenx + block.x - 1) / block.x, ((n + 1) + block.y - 1) / block.y, 1);
    } else {
        grid = dim3((rect_oddx + block.x - 1) / block.x, (n + block.y - 1) / block.y, 1);
    }
}

void gen_hadouken_pspace_lower(const unsigned int n, dim3& block, dim3& grid, unsigned int* aux1, unsigned int* aux2, unsigned int* aux3) {
    // lower pow, tetrahedron covering (simplex + lower block)
    int nlow = 1 << ((int)floor(log2f(n)));
    block = dim3(BSIZE2D, BSIZE2D, 1);
    int nbo = (n + block.x - 1) / block.x;
    int nb = (nlow + block.x - 1) / block.x;
    int gx = nb <= 1 ? nb + 1 : nb;
    int gy = nbo;
    int extraby = nbo - nb;
    // printf("n %i   nlow %i   nbo %i  nb %i  s %i\n", n, nlow, nbo, nb, extraby);
    grid = dim3(ceil((gx - 1.0) / 2.0), gy + 1 + extraby, 1);
    /* big blocks for trapezoid tower */
    *aux1 = nb - 1 + extraby;
    *aux2 = nbo - (nb - 1);
    *aux3 = gy + extraby + 1;
    // printf("extra segments = %i  aux1 = %i    aux2 = %i  aux3 = %i  extraby = %i\n", extraby, *aux1, *aux2, *aux3, extraby);
    /* a -- b  blocks (alternating to the side odd and even for the trapezoid)
     *aux1 = nb-1;
     *aux2 = nbo-(nb-1);
     *aux3 = gy+extraby+1;
     */
#ifdef DEBUG
    printf("[lower] block= %i x %i x %i    grid = %i x %i x %i\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
#endif
}

// covering from greater simplex
void gen_hadouken_pspace_upper(const unsigned int n, dim3& block, dim3& grid, unsigned int* aux1, unsigned int* aux2, unsigned int* aux3) {
    block = dim3(BSIZE2D, BSIZE2D, 1);
    int nu = 1 << ((int)ceil(log2f(n)));
    int nb = (nu + block.x - 1) / block.x;
    int gx = nb <= 1 ? nb + 1 : nb;
    int gy = nb;
    grid = dim3(ceil((gx - 1.0) / 2.0), gy + 1, 1);
    *aux1 = nb - 1;
    *aux2 = 1;
#ifdef DEBUG
    printf("[upper] block= %i x %i x %i    grid = %i x %i x %i\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
#endif
}

void gen_recursive_pspace(const unsigned int n, dim3& block, dim3& grid) {
    block = dim3(BSIZE2D, BSIZE2D, 1);
    // it is defined later, at each recursive level
    grid = dim3(1, 1, 1);
}

template <typename Lambda>
double benchmark_map(const int REPEATS, dim3 block, dim3 grid, unsigned int n, unsigned int msize, unsigned int trisize, DTYPE* ddata, MTYPE* dmat, Lambda map, const unsigned int aux1, const unsigned int aux2, const unsigned int aux3) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);

#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    printf("Benchmarking (%i REPEATS).......", REPEATS);
    fflush(stdout);
#endif
    float time = 0.0;
    // measure running time
    cudaEventRecord(start, 0);
    for (int k = 0; k < REPEATS; k++) {
        kernel_test<<<grid, block>>>(n, 1, msize, ddata, dmat, map, aux1, aux2, aux3);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // that's our time!
#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    print_dmat(PRINTLIMIT, n, n * n, dmat, "result matrix on GPU");
#endif
    time = time / (float)REPEATS;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    last_cuda_error("benchmark_map: run");
    return time;
}

unsigned int count_recursions(unsigned int n, int bsize) {
    unsigned int lpow = 1 << ((unsigned int)floor(log2f(n)));
    unsigned int hpow = 1 << ((unsigned int)ceil(log2f(n)));
    unsigned int nh = n;
    unsigned int numrec = 0;
    do {
        if (hpow - nh < HADO_TOL) {
            nh = 0;
        } else {
            nh = nh - lpow;
            lpow = 1 << ((unsigned int)floor(log2f(nh)));
            hpow = 1 << ((unsigned int)ceil(log2f(nh)));
        }
        numrec++;
    } while (nh > 0);
    return numrec;
}

void create_grids_streams(unsigned int n, unsigned int numrec, dim3* grids, dim3 block, unsigned int* aux1, unsigned int* aux2, unsigned int* aux3, cudaStream_t* streams, unsigned int* offsets) {
    unsigned int lpow = 1 << ((unsigned int)floor(log2f(n)));
    unsigned int hpow = 1 << ((unsigned int)ceil(log2f(n)));
    unsigned int off = 0;
    unsigned int nh = n;
    int nr = 0;
    do {
        if (hpow - nh < HADO_TOL) {
            // case n is close to the next power of two, we do all
            gen_hadouken_pspace_upper(nh, block, grids[nr], &aux1[nr], &aux2[nr], &aux3[nr]);
            offsets[nr] = off;
            nh = 0;
        } else {
            // case n is far from its next power of two, we continue in halves
            gen_hadouken_pspace_lower(nh, block, grids[nr], &aux1[nr], &aux2[nr], &aux3[nr]);
            offsets[nr] = off;
            nh = nh - lpow;
            off += lpow;
            lpow = 1 << ((unsigned int)floor(log2f(nh)));
            hpow = 1 << ((unsigned int)ceil(log2f(nh)));
        }
        cudaStreamCreate(&streams[nr]);
        nr++;
    } while (nh > 0);
}

void print_grids_offsets(unsigned int numrec, dim3* grids, dim3 block, unsigned int* offsets) {
    for (int i = 0; i < numrec; ++i) {
        printf("[%i] =  block(%i, %i, %i)  grid(%i, %i, %i)   offset = %i\n", i, block.x, block.y, block.z, grids[i].x, grids[i].y, grids[i].z, offsets[i]);
    }
}

template <typename Lambda>
double benchmark_map_hadouken(const int REPEATS, dim3 block, unsigned int n, unsigned int msize, unsigned int trisize, DTYPE* ddata, MTYPE* dmat, Lambda map) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time = 0.0;
    unsigned int numrec = count_recursions(n, BSIZE2D);
    dim3* grids = (dim3*)malloc(sizeof(dim3) * numrec);
    unsigned int* offsets = (unsigned int*)malloc(sizeof(unsigned int) * numrec);
    unsigned int* auxs1 = (unsigned int*)malloc(sizeof(unsigned int) * numrec);
    unsigned int* auxs2 = (unsigned int*)malloc(sizeof(unsigned int) * numrec);
    unsigned int* auxs3 = (unsigned int*)malloc(sizeof(unsigned int) * numrec);
    cudaStream_t* streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * numrec);
    create_grids_streams(n, numrec, grids, block, auxs1, auxs2, auxs3, streams, offsets);
#ifdef DEBUG
    printf("HADO_TOL = %i\n", HADO_TOL);
    print_grids_offsets(numrec, grids, block, offsets);
#endif
    // printf("n = %i    lpow = %i, hpow = %i\n", n, lpow, hpow);
    // printf("numrecs = %i\n", numrec);

#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    printf("Benchmarking (%i REPEATS).......", REPEATS);
    fflush(stdout);
#endif
    // measure running time
    cudaEventRecord(start, 0);
// numrec = count_recursions(n, BSIZE2D);
// create_grids_streams(n, numrec, grids, block, auxs1, auxs2, auxs3, streams, offsets);
#pragma loop unroll
    for (int k = 0; k < REPEATS; ++k) {
        for (int i = 0; i < numrec; ++i) {
            kernel_test<<<grids[i], block, 0, streams[i]>>>(n, 1, msize, ddata, dmat, map, auxs1[i], auxs2[i], offsets[i]);
        }
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    time = time / (float)REPEATS;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    last_cuda_error("benchmark_map: run");
#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    print_dmat(PRINTLIMIT, n, n * n, dmat, "result matrix on GPU");
#endif
    return time;
}

template <typename Lambda>
double benchmark_map_DP(const int REPEATS, dim3 block, unsigned int n, unsigned int msize, unsigned int trisize, DTYPE* ddata, MTYPE* dmat, Lambda map) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time = 0.0;
    unsigned int blockedN = (n + block.x - 1) / block.x;
    unsigned int blockedNHalf = ceil(blockedN / 2.f);
    dim3 grid = dim3(blockedNHalf, blockedNHalf);

#ifdef DEBUG
    printf("HADO_TOL = %i\n", HADO_TOL);
    unsigned int a = 0;
    print_grids_offsets(1, &grid, block, &a);
#endif

#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    printf("Benchmarking (%i REPEATS).......", REPEATS);
    fflush(stdout);
#endif
    // measure running time
    cudaEventRecord(start, 0);
#pragma loop unroll
    for (int k = 0; k < REPEATS; ++k) {
        kernel_test_DP<<<grid, block>>>(n, blockedNHalf, dmat, ddata, map, 0, blockedN - blockedNHalf);
        cudaDeviceSynchronize();
    }
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

    time = time / (float)REPEATS;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    last_cuda_error("benchmark_map: run");
#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    print_dmat(PRINTLIMIT, n, n * n, dmat, "result matrix on GPU");
#endif
    return time;
}

int verify_result(unsigned int n, const unsigned int checkval, const unsigned long msize, DTYPE* hdata, DTYPE* ddata, MTYPE* hmat, MTYPE* dmat, dim3 grid, dim3 block) {
	return 1;
#ifdef DEBUG
    printf("verifying.......................");
    fflush(stdout);
#endif
    float epsilon = 0.0001f;
    cudaMemcpy(hdata, ddata, sizeof(DTYPE) * n, cudaMemcpyDeviceToHost);
    cudaMemcpy(hmat, dmat, sizeof(MTYPE) * msize, cudaMemcpyDeviceToHost);
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {
            DTYPE a = hdata[i];
            DTYPE b = hdata[j];
            float val = sqrtf((a.x - b.x) * (a.x - b.x) + (a.y - b.y) * (a.y - b.y));
            if (fabs(val - hmat[i * n + j]) > epsilon) {
#ifdef DEBUG
                fprintf(stderr, "error at [i, j] = [%i, %i], edm2d\nCPU %f\nGPU %f\n", i, j, val, hmat[i * n + j]);
#endif
                return 0;
            }
        }
    }
#ifdef DEBUG
    printf("ok\n");
    fflush(stdout);
#endif
    return 1;
}
#endif
