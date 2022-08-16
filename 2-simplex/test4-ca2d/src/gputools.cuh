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

#ifdef MEASURE_POWER
#include "nvmlPower.hpp"
#endif

// for DP
#ifndef DP_DEPTH
#define DP_DEPTH -1
#endif

#define OPT_MINSIZE 2048

#include "kernels.cuh"
#include <string>

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

uint32_t cntsetbits(uint32_t i) {
    // C or C++: use uint32_t
    i = i - ((i >> 1) & 0x55555555);
    i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
    return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

void print_array(DTYPE* a, const int n) {
    for (int i = 0; i < n; i++)
        printf("a[%d] = %f\n", i, a[i]);
}

void set_randconfig(MTYPE* mat, const unsigned long n, double DENSITY) {
    // note: this function does not act on the ghost cells (boundaries)
    for (unsigned int y = 0; y < n; ++y) {
        for (unsigned int x = 0; x < n; ++x) {
            unsigned long i = (y + 1) * (n + 2) + (x + 1);
            mat[i] = x <= y ? (((double)rand() / RAND_MAX) <= DENSITY ? 1 : 0) : 0;
        }
    }
}
void set_ids(MTYPE* mat, const unsigned long n) {
    // note: this function does not act on the ghost cells (boundaries)
    int count = 0;
    for (unsigned int y = 0; y < n; ++y) {
        for (unsigned int x = 0; x < n; ++x) {
            unsigned long i = (y + 1) * (n + 2) + (x + 1);
            if (y >= x) {
                mat[i] = count++;
            }
        }
    }
}

void set_alldead(MTYPE* mat, const unsigned long n) {
    // note: this function does not act on the ghost cells (boundaries)
    for (unsigned int y = 0; y < n; ++y) {
        for (unsigned int x = 0; x < n; ++x) {
            unsigned long i = (y + 1) * (n + 2) + (x + 1);
            mat[i] = 0;
        }
    }
}

void set_everything(MTYPE* mat, const unsigned long n, MTYPE val) {
    // set cells and ghost cells
    for (unsigned int y = 0; y < n + 2; ++y) {
        for (unsigned int x = 0; x < n + 2; ++x) {
            unsigned long i = y * (n + 2) + x;
            mat[i] = val;
        }
    }
}

void set_cell(MTYPE* mat, const unsigned long n, const int x, const int y, MTYPE val) {
    // the boundaries are ghost cells
    unsigned long i = (y + 1) * (n + 2) + (x + 1);
    mat[i] = val;
}

unsigned long count_living_cells(MTYPE* mat, const unsigned long n) {
    unsigned long c = 0;
    // note: this function does not act on the boundaries.
    for (unsigned int y = 0; y < n; ++y) {
        for (unsigned int x = 0; x < n; ++x) {
            if (x <= y) {
                unsigned long i = (y + 1) * (n + 2) + (x + 1);
                c += mat[i];
            }
        }
    }
    return c;
}

void print_ghost_matrix(MTYPE* mat, const int n, const char* msg) {
    printf("[%s]:\n", msg);
    for (int i = 0; i < n + 2; i++) {
        for (int j = 0; j < n + 2; j++) {
            long w = i * (n + 2) + j;
            if (i >= j && i < n + 1 && j > 0) {
                // if(mat[w] == 1){
                //     printf("1 ");
                // }
                // else{
                //     printf("  ");
                // }
                if (mat[w] != 0) {
                    printf("%i ", mat[w]);
                } else {
                    printf("  ", mat[w]);
                }
            } else {
                printf("%i ", mat[w]);
            }
        }
        printf("\n");
    }
}

int print_dmat(int PLIMIT, unsigned int n, unsigned long msize, MTYPE* dmat, const char* msg) {
    if (n <= PLIMIT) {
        MTYPE* hmat = (MTYPE*)malloc(sizeof(MTYPE) * msize);
        cudaMemcpy(hmat, dmat, sizeof(MTYPE) * msize, cudaMemcpyDeviceToHost);
        print_ghost_matrix(hmat, n, msg);
        free(hmat);
    }
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

void init(unsigned int n, DTYPE** hdata, MTYPE** hmat, DTYPE** ddata, MTYPE** dmat1, MTYPE** dmat2, unsigned long* msize, unsigned long* trisize, double DENSITY) {
    // define ghost n, gn, for the (n+2)*(n+2) space for the ghost cells
    *msize = (n + 2) * (n + 2);

    *hmat = (MTYPE*)malloc(sizeof(MTYPE) * (*msize));
    set_everything(*hmat, n, 0);
    set_randconfig(*hmat, n, DENSITY);
    // set_ids(*hmat, n);

    /*
    set_alldead(*hmat, n);
    set_cell(*hmat, n, 3, 6, 1);
    set_cell(*hmat, n, 4, 6, 1);
    set_cell(*hmat, n, 5, 6, 1);

    set_cell(*hmat, n, 3, 7, 1);
    set_cell(*hmat, n, 4, 7, 1);
    set_cell(*hmat, n, 5, 7, 1);

    set_cell(*hmat, n, 3, 8, 1);
    set_cell(*hmat, n, 4, 8, 1);
    set_cell(*hmat, n, 5, 8, 1);
    */

    cudaMalloc((void**)dmat1, sizeof(MTYPE) * (*msize));
    cudaMalloc((void**)dmat2, sizeof(MTYPE) * (*msize));
    last_cuda_error("init: cudaMalloc dmat1 dmat2");
    cudaMemcpy(*dmat1, *hmat, sizeof(MTYPE) * (*msize), cudaMemcpyHostToDevice);
    last_cuda_error("init:memcpy hmat->dmat1");

#ifdef DEBUG
    printf("2-simplex: n=%i  msize=%lu (%f MBytes -> 2 lattices)\n", n, *msize, 2.0f * (float)sizeof(MTYPE) * (*msize) / (1024 * 1024));
    if (n <= PRINTLIMIT) {
        print_ghost_matrix(*hmat, n, "\nhost ghost-matrix initialized");
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
double benchmark_map(const int REPEATS, dim3 block, dim3 grid, unsigned int n,
    unsigned long msize, unsigned int trisize, DTYPE* ddata, MTYPE* dmat1,
    MTYPE* dmat2, Lambda map, unsigned int aux1, unsigned int aux2, unsigned int aux3, char* str) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
#ifdef DEBUG
    printf("block= %i x %i x %i    grid = %i x %i x %i\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
#endif

#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG  dmat2 -> dmat1");
    printf("WARMUP (%i REPEATS).......", REPEATS);
    fflush(stdout);
#endif
    for (int k = 0; k < REPEATS; k++) {
        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        kernel_test<<<grid, block>>>(n, msize, ddata, dmat1, dmat2, map, aux1, aux2, aux3);
        cudaDeviceSynchronize();

        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        kernel_test<<<grid, block>>>(n, msize, ddata, dmat2, dmat1, map, aux1, aux2, aux3);
        cudaDeviceSynchronize();
    }

#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG  dmat2 -> dmat1");
    printf("Benchmarking (%i REPEATS).......", REPEATS);
    fflush(stdout);
#endif
    float time = 0.0;
    // measure running time
    cudaEventRecord(start, 0);
#ifdef MEASURE_POWER
    GPUPowerBegin(n, 100, 0, std::string(str) + std::string("A100"));
#endif
    for (int k = 0; k < REPEATS; k++) {
        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        kernel_test<<<grid, block>>>(n, msize, ddata, dmat1, dmat2, map, aux1, aux2, aux3);
        cudaDeviceSynchronize();

        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        kernel_test<<<grid, block>>>(n, msize, ddata, dmat2, dmat1, map, aux1, aux2, aux3);
        cudaDeviceSynchronize();
    }
#ifdef MEASURE_POWER
    GPUPowerEnd();
#endif

#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
#endif
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // that's our time!
#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG  dmat2 -> dmat1");
    fflush(stdout);
#endif

    last_cuda_error("benchmark-check");
    time = time / (float)REPEATS;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
    return time;
}

template <typename Lambda>
double benchmark_map_rectangle(const int REPEATS, dim3 block, dim3 grid,
    unsigned int n, unsigned long msize, unsigned int trisize, DTYPE* ddata,
    MTYPE* dmat1, MTYPE* dmat2, Lambda map,
    unsigned int aux1, unsigned int aux2, unsigned int aux3, char* str) {
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    // Start record
#ifdef DEBUG
    printf("block= %i x %i x %i    grid = %i x %i x %i\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
#endif

#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG");
    printf("WARMUP (%i REPEATS).......", REPEATS);
    fflush(stdout);
#endif
    for (int k = 0; k < REPEATS; k++) {
        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        kernel_test_rectangle<<<grid, block>>>(n, msize, ddata, dmat1, dmat2, map, aux1, aux2, aux3);
        cudaDeviceSynchronize();

        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        kernel_test_rectangle<<<grid, block>>>(n, msize, ddata, dmat2, dmat1, map, aux1, aux2, aux3);
        cudaDeviceSynchronize();
    }

#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG");
    printf("Benchmarking (%i REPEATS).......", REPEATS);
    fflush(stdout);
#endif
    float time = 0.0;
    // measure running time
    cudaEventRecord(start, 0);
#ifdef MEASURE_POWER
    GPUPowerBegin(n, 100, 0, std::string(str) + std::string("A100"));
#endif
    for (int k = 0; k < REPEATS; k++) {
        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        kernel_test_rectangle<<<grid, block>>>(n, msize, ddata, dmat1, dmat2, map, aux1, aux2, aux3);
        cudaDeviceSynchronize();

        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        kernel_test_rectangle<<<grid, block>>>(n, msize, ddata, dmat2, dmat1, map, aux1, aux2, aux3);
        cudaDeviceSynchronize();
    }
#ifdef MEASURE_POWER
    GPUPowerEnd();
#endif

#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
#endif
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop); // that's our time!
#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG  dmat2 -> dmat1");
    fflush(stdout);
#endif
    last_cuda_error("benchmark-check");
    time = time / (float)REPEATS;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    last_cuda_error("benchmark-check");
    return time;
}

template <typename Lambda>
double benchmark_map_hadouken(const int REPEATS, dim3 block, unsigned int n, unsigned long msize, unsigned int trisize, DTYPE* ddata, MTYPE* dmat1, MTYPE* dmat2, Lambda map, char* str) {
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
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG");
    printf("WARMUP (%i REPEATS).......", REPEATS);
    fflush(stdout);
#endif
#pragma loop unroll
    for (int k = 0; k < REPEATS; ++k) {
        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        for (int i = 0; i < numrec; ++i) {
            // for(int i=numrec-1; i>=0; --i){
            kernel_test<<<grids[i], block, 0, streams[i]>>>(n, msize, ddata, dmat1, dmat2, map, auxs1[i], auxs2[i], offsets[i]);
        }
        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        for (int i = 0; i < numrec; ++i) {
            // for(int i=numrec-1; i>=0; --i){
            kernel_test<<<grids[i], block, 0, streams[i]>>>(n, msize, ddata, dmat2, dmat1, map, auxs1[i], auxs2[i], offsets[i]);
        }
        cudaDeviceSynchronize();
    }
#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG");
    printf("Benchmarking (%i REPEATS).......", REPEATS);
    fflush(stdout);
#endif
    // measure running time
    cudaEventRecord(start, 0);
#ifdef MEASURE_POWER
    GPUPowerBegin(n, 100, 0, std::string(str) + std::string("A100"));
#endif
// numrec = count_recursions(n, BSIZE2D);
// create_grids_streams(n, numrec, grids, block, auxs1, auxs2, auxs3, streams, offsets);
#pragma loop unroll
    for (int k = 0; k < REPEATS; ++k) {
        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        for (int i = 0; i < numrec; ++i) {
            // for(int i=numrec-1; i>=0; --i){
            kernel_test<<<grids[i], block, 0, streams[i]>>>(n, msize, ddata, dmat1, dmat2, map, auxs1[i], auxs2[i], offsets[i]);
        }
        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        for (int i = 0; i < numrec; ++i) {
            // for(int i=numrec-1; i>=0; --i){
            kernel_test<<<grids[i], block, 0, streams[i]>>>(n, msize, ddata, dmat2, dmat1, map, auxs1[i], auxs2[i], offsets[i]);
        }
        cudaDeviceSynchronize();
    }
#ifdef MEASURE_POWER
    GPUPowerEnd();
#endif

#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
#endif
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);

#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG  dmat2 -> dmat1");
    fflush(stdout);
#endif
    time = time / (float)REPEATS;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    last_cuda_error("benchmark_map: run");
    return time;
}

template <typename Lambda>
double benchmark_map_DP(const int REPEATS, dim3 block, unsigned int n, unsigned long msize, unsigned int trisize, DTYPE* ddata, MTYPE* dmat1, MTYPE* dmat2, Lambda map, char* str) {

    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
    float time = 0.0;
    unsigned int blockedN = (n + block.x - 1) / block.x;
    unsigned int blockedNHalf = ceil(blockedN / 2.f);
    dim3 grid = dim3(blockedNHalf, blockedNHalf);

    unsigned int expVal;
    unsigned int minSize;
    if (DP_DEPTH >= 0) {
        expVal = max((int)ceil(log2f(n)) - DP_DEPTH, 0);
        minSize = 1 << expVal;
    } else {
        minSize = OPT_MINSIZE;
    }

#ifdef DEBUG
    printf("DP_DEPTH = %i\n", DP_DEPTH);
    printf("exponent = %i\n", expVal);
    printf("minSize = %i\n", minSize);
    fflush(stdout);
    printf("WARMUP (%i REPEATS).......", REPEATS);
    fflush(stdout);
#endif
#pragma loop unroll
    for (int k = 0; k < REPEATS; ++k) {
        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        // kernel_test_DP<<<grid, block>>>(n, blockedN, ddata, dmat1, dmat2, map, 0, blockedN - blockedNHalf, 1);
        kernelDP_exp<<<1, 1>>>(n, n, dmat1, dmat2, 0, 0, minSize);

        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        // kernel_test_DP<<<grid, block>>>(n, blockedN, ddata, dmat2, dmat1, map, 0, blockedN - blockedNHalf, 1);
        kernelDP_exp<<<1, 1>>>(n, n, dmat2, dmat1, 0, 0, minSize);

        cudaDeviceSynchronize();
    }
#ifdef DEBUG
    printf("DP_DEPTH = %i\n", DP_DEPTH);
    printf("exponent = %i\n", expVal);
    printf("minSize = %i\n", minSize);
    fflush(stdout);
    printf("Benchmarking (%i REPEATS).......", REPEATS);
    fflush(stdout);
#endif // measure running time
    cudaEventRecord(start, 0);
#ifdef MEASURE_POWER
    GPUPowerBegin(n, 100, 0, std::string(str) + std::string("A100"));
#endif
#pragma loop unroll
    for (int k = 0; k < REPEATS; ++k) {
        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat1, dmat1);
        cudaDeviceSynchronize();
        // kernel_test_DP<<<grid, block>>>(n, blockedN, ddata, dmat1, dmat2, map, 0, blockedN - blockedNHalf, 1);
        kernelDP_exp<<<1, 1>>>(n, n, dmat1, dmat2, 0, 0, minSize);

        kernel_update_ghosts<<<(n + BSIZE1D - 1) / BSIZE1D, BSIZE1D>>>(n, msize, dmat2, dmat2);
        cudaDeviceSynchronize();
        // kernel_test_DP<<<grid, block>>>(n, blockedN, ddata, dmat2, dmat1, map, 0, blockedN - blockedNHalf, 1);
        kernelDP_exp<<<1, 1>>>(n, n, dmat2, dmat1, 0, 0, minSize);

        cudaDeviceSynchronize();
    }
#ifdef MEASURE_POWER
    GPUPowerEnd();
#endif

#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
#endif
    cudaEventRecord(stop, 0);
    cudaEventSynchronize(stop);
    cudaEventElapsedTime(&time, start, stop);
#ifdef DEBUG
    printf("done\n");
    fflush(stdout);
    print_dmat(PRINTLIMIT, n, msize, dmat1, "PONG  dmat2 -> dmat1");
    fflush(stdout);
#endif
    time = time / (float)REPEATS;
    cudaEventDestroy(start);
    cudaEventDestroy(stop);
    last_cuda_error("benchmark_map: run");
    return time;
}

//#include <png.h>
//void save_image(const char* filename, int* dwells, uint64_t w, uint64_t h,
//    unsigned int MAX_DWELL, int SAVE_FLAG) {
//    png_bytep row;
//
//    FILE* fp = fopen(filename, "wb");
//    png_structp png_ptr = png_create_write_struct(PNG_LIBPNG_VER_STRING, 0, 0, 0);
//    png_infop info_ptr = png_create_info_struct(png_ptr);
//    // exception handling
//    setjmp(png_jmpbuf(png_ptr));
//    png_init_io(png_ptr, fp);
//    // write header (8 bit colour depth)
//    png_set_IHDR(png_ptr, info_ptr, w, h, 8, PNG_COLOR_TYPE_RGB, PNG_INTERLACE_NONE,
//        PNG_COMPRESSION_TYPE_BASE, PNG_FILTER_TYPE_BASE);
//    // set title
//    png_text title_text;
//    const char* title = "Title";
//    const char* text = "CA, per-pixel";
//
//    title_text.compression = PNG_TEXT_COMPRESSION_NONE;
//
//    title_text.key = (png_charp)title;
//    title_text.text = (png_charp)text;
//
//    png_set_text(png_ptr, info_ptr, &title_text, 1);
//    png_write_info(png_ptr, info_ptr);
//
//    // write image data/
//
//    row = (png_bytep)malloc(3 * w * sizeof(png_byte));
//    for (uint64_t y = 0; y < h; y++) {
//        for (uint64_t x = 0; x < w; x++) {
//            int r, g, b;
//            r = dwells[y * w + x] * 255;
//            row[3 * x + 0] = (png_byte)r;
//            row[3 * x + 1] = (png_byte)r;
//            row[3 * x + 2] = (png_byte)r;
//        }
//        png_write_row(png_ptr, row);
//    }
//    png_write_end(png_ptr, NULL);
//
//    fclose(fp);
//    png_free_data(png_ptr, info_ptr, PNG_FREE_ALL, -1);
//    png_destroy_write_struct(&png_ptr, (png_infopp)NULL);
//    free(row);
//} // save_image

int verify_result(unsigned int n, const unsigned long msize, DTYPE* hdata, DTYPE* ddata, MTYPE* hmat, MTYPE* dmat) {

    // cudaMemcpy(hmat, dmat, sizeof(MTYPE) * msize, cudaMemcpyDeviceToHost);
    // save_image("CA.png", hmat, n + 2, n + 2, 1, 0);
    return 1;
}
#endif
