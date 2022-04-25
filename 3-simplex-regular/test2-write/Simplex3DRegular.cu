#include "Simplex3DRegular.cuh"
#include "StatsCollector.h"
#include "kernels.cuh"
#include "tools.cuh"
#include <cassert>
#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <vector>

#define OFFSET -0.4999f
//#define OFFSET 0.0f
#define REAL float

Simplex3DRegular::Simplex3DRegular(uint32_t deviceId, uint32_t powerOfTwoSize, uint32_t maptype)
    : deviceId(deviceId)
    , powerOfTwoSize(powerOfTwoSize) {

    this->n = 1 << powerOfTwoSize;
    this->nElementsCube = n * n * n;
    this->nElementsSimplex = n * (n + 1) * (n + 2) / 6;

    this->mapType = static_cast<MapType>(mapType);

    this->hasBeenAllocated = false;
}

void Simplex3DRegular::allocateMemory() {
    if (this->hasBeenAllocated) {
        printf("Memory already allocated.\n");
        return
    }
    this->hostData = (MTYPE*)malloc(sizeof(MTYPE) * this->nElementsCube);
    cudaMalloc(&d_outcube, sizeof(char) * Vcube);
    gpuErrchk(cudaPeekAtLastError());
    this->hasBeenAllocated = true;
}

void Simplex3DRegular::freeMemory() {
    // clear
    free(hostData);
    cudaFree(devData);
    gpuErrchk(cudaPeekAtLastError());
    this->hasBeenAllocated = false;
}
bool Simplex3DRegular::isMapTypeImplemented() {
    return mapType < 3;
}

bool Simplex3DRegular::init() {

#ifdef DEBUG
    printf("init(): choosing device %i...", D);
    fflush(stdout);
#endif
    gpuErrchk(cudaSetDevice(D));
#ifdef DEBUG
    printf("ok\n");
    fflush(stdout);
#endif

#ifdef DEBUG
    printf("init(): allocating memory.\n");
#endif
    this->allocateMemory();
#ifdef DEBUG
    printf("init(): memory allocated.\n");
#endif
    if (!isMapTypeImplemented()) {
        printf("ERROR: Map type \"%i\" not implemented! init failed\n", this->mapType);
        this->freeMemory();
        return false;
    }
    if (this->powerOfTwoSize < 1) {
        printf("Power of two cannot be lower than 1, a cube of 1x1x1 is a trivial problem\n");
        return false;
    }

    switch (this->mapType) {
    case MapType::BOUNDING_BOX:
        this->block = dim3(BSIZE3D, BSIZE3D, BSIZE3D);
        this->grid = dim3((n + block.x - 1) / block.x, (n + block.y - 1) / block.y, (n + block.z - 1) / block.z);
        break;
    case MapType::HADOUKEN:
        this->block = dim3(BSIZE3D, BSIZE3D, BSIZE3D);
        this->grid = dim3((n / 2 + block.x - 1) / block.x, (n / 2 + block.y - 1) / block.y, (3 * (n - 1) / 4 + block.z - 1) / block.z);
        break;
    case MapType::DYNAMIC_PARALLELISM:
        this->block = dim3(BSIZE3D, BSIZE3D, BSIZE3D);
        this->grid = dim3((n + block.x - 1) / block.x, (n + block.y - 1) / block.y, (n + block.z - 1) / block.z);
        break;
    }

#ifdef DEBUG
    printf("init(): parallel space: b(%i, %i, %i) g(%i, %i, %i)\n", block.x, block.y, block.z, grid.x, grid.y, grid.z);
    fflush(stdout);
#endif

#ifdef DEBUG
    printf("init(): filling cube with 0.\n");
#endif
    for (size_t i = 0; i < this->nElementsCube; ++i) {
        this->hostData[i] = (MTYPE)0;
    }
#ifdef DEBUG
    printf("init(): done.\n");
#endif
}

void Simplex3DRegular::transferHostToDevice() {
    cudaMemcpy(this->devData, this->hostData, sizeof(MTYPE) * this->nElementsCube, cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
void Simplex3DRegular::transferDeviceToHost() {
    cudaMemcpy(this->hostData, this->devData, sizeof(MTYPE) * this->nElementsCube, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

StatsCollector Simplex3DRegular::doBenchmark() {

#ifdef DEBUG
    printf("doBenchmark(): mapping to simplex of n=%lu   Vcube = %lu   Vsimplex = %lu\n", this->n, this->nElementsCube, this->nElementsSimplex);
    printf("doBenchmark(): Cube size is %f MB\n", (float)this->nElementsCube * sizeof(MTYPE) / (1024.0 * 1024.0f));
#endif

    // begin performance tests
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#ifdef DEBUG
    printf("\x1b[1mdoBenchmark(): kernel (map=%i, r=%i)...\x1b[0m", this->mapType, REPEATS);
    fflush(stdout);
#endif

    cudaEventRecord(start);
    switch (mapType) {
    case MapType::BOUNDING_BOX:
        for (uint32_t i = 0; i < REPEATS; ++i) {
            kernelBoundingBox<<<this->grid, this->block>>>(this->devData, this->n);
        }
        break;
    case MapType::HADOUKEN:
        for (uint32_t i = 0; i < REPEATS; ++i) {
        }
        break;
    case MapType::DYNAMIC_PARALLELISM:
        for (uint32_t i = 0; i < REPEATS; ++i) {
        }
        break;
    }
    cudaEventRecord(stop);
    cudaEventSynchronize(stop);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());

#ifdef DEBUG
    printf("\x1b[1mok\n\x1b[0m");
    fflush(stdout);
#endif

#ifdef DEBUG
    // synchronize GPU/CPU mem
    printf("gpudummy(): synchronizing CPU/GPU mem...");
    fflush(stdout);
#endif
    cudaMemcpy(h_data, d_data, sizeof(float) * N, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    cudaMemcpy(h_outcube, d_outcube, sizeof(char) * Vcube, cudaMemcpyDeviceToHost);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#ifdef DEBUG
    printf("ok\n");
    if (N <= 16) {
        printf("cube:\n");
        printcube(h_outcube, N);
        // printcube_coords(h_outcube, N);
    }
#endif

// verify result
#ifdef VERIFY
#ifdef DEBUG
    printf("gpudummy(): verifying result...");
    fflush(stdout);
#endif
    assert(verify(h_outcube, N, [REPEATS](char d, float x, float y, float z) { return d == 1; }));
#ifdef DEBUG
    printf("ok\n\n");
    fflush(stdout);
#endif
#endif

    // clear
    free(h_data);
    free(h_outcube);
    cudaFree(d_data);
    cudaFree(d_outcube);

    // return computing time
    float msecs = 0;
    cudaEventElapsedTime(&msecs, start, stop);
    return msecs / ((float)REPEATS);
}
