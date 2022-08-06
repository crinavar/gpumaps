#include "Simplex3DRegular.cuh"

#include "GPUKernels.cuh"
#include "GPUTools.cuh"

#define OFFSET -0.4999f
//#define OFFSET 0.0f
#define REAL float

Simplex3DRegular::Simplex3DRegular(uint32_t deviceId, uint32_t powerOfTwoSize, uint32_t maptype)
    : deviceId(deviceId)
    , powerOfTwoSize(powerOfTwoSize) {

    this->n = 1 << powerOfTwoSize;
    this->nElementsCube = n * n * n;
    this->nElementsSimplex = n * (n - 1) * (n + 1) / 6;
    this->mapType = MapType::NOT_IMPLEMENTED;
    this->mapType = MapType::NOT_IMPLEMENTED;

    switch (maptype) {
    case 0:
        this->mapType = MapType::BOUNDING_BOX;
        break;
    case 1:
        this->mapType = MapType::HADOUKEN;
        break;
    case 2:
#ifndef DP
        printf("To enable the Dynamic Parallelism approach, please compile with `make DP=YES`.\n");
        break;
#endif
        this->mapType = MapType::DYNAMIC_PARALLELISM;
        break;
    case 3:
#ifndef DP
        printf("To enable the Dynamic Parallelism approach, please compile with `make DP=YES`.\n");
        break;
#endif
        this->mapType = MapType::DYNAMIC_PARALLELISM2;
        break;
    case 4:
#ifndef DP
        printf("To enable the Dynamic Parallelism approach, please compile with `make DP=YES`.\n");
        break;
#endif
        this->mapType = MapType::DYNAMIC_PARALLELISM3;
        break;
    }
    this->hasBeenAllocated = false;
}

Simplex3DRegular::~Simplex3DRegular() {
    if (this->hasBeenAllocated) {
        freeMemory();
    }
}

void Simplex3DRegular::allocateMemory() {
    if (this->hasBeenAllocated) {
        printf("Memory already allocated.\n");
        return;
    }
    this->hostData = (MTYPE*)malloc(sizeof(MTYPE) * this->nElementsCube);
    cudaMalloc(&devData, sizeof(MTYPE) * nElementsCube);
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
    return mapType != MapType::NOT_IMPLEMENTED;
}

bool Simplex3DRegular::init() {

#ifdef DEBUG
    printf("init(): choosing device %i...", this->deviceId);
    fflush(stdout);
#endif
    gpuErrchk(cudaSetDevice(this->deviceId));
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
        printf("ERROR: Map type not implemented! init failed\n");
        this->freeMemory();
        return false;
    }
#ifdef DEBUG
    printf("init(): Map chosen: %i.\n", this->mapType);
#endif

    if (this->powerOfTwoSize < 1) {
        printf("Power of two cannot be lower than 1, a cube of 1x1x1 is a trivial problem\n");
        return false;
    }

    uint32_t blockedN = ceil(n / (float)BSIZE3DX);

    switch (this->mapType) {
    case MapType::BOUNDING_BOX:
        this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY, BSIZE3DZ);
        this->GPUGrid = dim3(blockedN, blockedN, blockedN);
        break;
    case MapType::HADOUKEN:
        this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY, BSIZE3DZ);
        this->GPUGrid = dim3(ceil(blockedN / (float)2), ceil(blockedN / (float)2), ceil(3 * (blockedN) / (float)4));
        // The +2 in Y is to enable the trick of the concurrent trapezoids in the 2D version
        this->GPUGridAux = dim3(ceil(blockedN / (float)2), blockedN - 1 + 2, 1);

        break;
    case MapType::DYNAMIC_PARALLELISM:
        this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY, BSIZE3DZ);
        this->GPUGrid = dim3((n / 2 + GPUBlock.x - 1) / GPUBlock.x, (n / 2 + GPUBlock.y - 1) / GPUBlock.y, (n / 2 + GPUBlock.z - 1) / GPUBlock.z);
        break;
    case MapType::DYNAMIC_PARALLELISM2:
        this->GPUBlock = dim3(1, 1, 1);
        this->GPUGrid = dim3(1, 1, 1);
        break;
    case MapType::DYNAMIC_PARALLELISM3:
        this->GPUBlock = dim3(1, 1, 1);
        this->GPUGrid = dim3(1, 1, 1);
        break;
    }

#ifdef DEBUG
    printf("init(): parallel space: b(%i, %i, %i) g(%i, %i, %i)\n", GPUBlock.x, GPUBlock.y, GPUBlock.z, GPUGrid.x, GPUGrid.y, GPUGrid.z);
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

#ifdef DEBUG
    printf("init(): Transfering data to device.\n");
#endif
    this->transferHostToDevice();
#ifdef DEBUG
    printf("init(): done.\n");
#endif
    return true;
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

float Simplex3DRegular::doBenchmarkAction(uint32_t nTimes) {

#ifdef DEBUG
    printf("doBenchmark(): mapping to simplex of n=%lu   Vcube = %lu   Vsimplex = %lu\n", this->n, this->nElementsCube, this->nElementsSimplex);
    printf("doBenchmark(): Cube size is %f MB\n", (float)this->nElementsCube * sizeof(MTYPE) / (1024.0 * 1024.0f));
#endif

    // begin performance tests
    cudaEvent_t start, stop;
    cudaEventCreate(&start);
    cudaEventCreate(&stop);
#ifdef DEBUG
    printf("\x1b[1mdoBenchmark(): kernel (map=%i, rep=%i)...\x1b[0m", this->mapType, nTimes);
    fflush(stdout);
#endif
    cudaStream_t* streams;
    if (this->mapType == MapType::HADOUKEN) {
        streams = (cudaStream_t*)malloc(sizeof(cudaStream_t) * 2);
        cudaStreamCreate(&streams[0]);
        cudaStreamCreate(&streams[1]);
    }
    uint32_t blockedN = ceil(n / (float)BSIZE3DX);
    switch (this->mapType) {
    case MapType::BOUNDING_BOX:
        cudaEventRecord(start);
#ifdef MEASURE_POWER
        GPUPowerBegin(this->n, 100, 0, std::string("BB-") + std::to_string(this->deviceId));
#endif
        for (uint32_t i = 0; i < nTimes; ++i) {
            kernelBoundingBox<<<this->GPUGrid, this->GPUBlock>>>(this->devData, this->n, blockedN);
            gpuErrchk(cudaDeviceSynchronize());
        }
        cudaEventRecord(stop);

        break;
    case MapType::HADOUKEN:
        cudaEventRecord(start);
#ifdef MEASURE_POWER
        GPUPowerBegin(this->n, 100, 0, std::string("HAD-") + std::to_string(this->deviceId));
#endif
        for (uint32_t i = 0; i < nTimes; ++i) {
            kernelHadouken<<<this->GPUGrid, this->GPUBlock, 0, streams[0]>>>(this->devData, this->n, blockedN);
            kernelHadoukenStrip<<<this->GPUGridAux, this->GPUBlock, 0, streams[1]>>>(this->devData, this->n, blockedN);
            //  printf("(%i, %i, %i)\n", this->GPUGridAux.x, this->GPUGridAux.y, this->GPUGridAux.z);
            gpuErrchk(cudaDeviceSynchronize());
        }
        cudaEventRecord(stop);

        break;

    case MapType::DYNAMIC_PARALLELISM:
        cudaEventRecord(start);
#ifdef MEASURE_POWER
        GPUPowerBegin(this->n, 100, 0, std::string("DP-") + std::to_string(this->deviceId));
#endif
        for (uint32_t i = 0; i < nTimes; ++i) {
#ifdef DP
            kernelDynamicParallelism<<<this->GPUGrid, this->GPUBlock>>>(this->devData, this->n, 1, n / 2, 0, 0);
#endif
            gpuErrchk(cudaDeviceSynchronize());
        }
        cudaEventRecord(stop);

        break;

    case MapType::DYNAMIC_PARALLELISM2:
        cudaEventRecord(start);
#ifdef MEASURE_POWER
        GPUPowerBegin(this->n, 100, 0, std::string("DP2-") + std::to_string(this->deviceId));
#endif
        for (uint32_t i = 0; i < nTimes; ++i) {
#ifdef DP
            kernelDP_exp<<<this->GPUGrid, this->GPUBlock>>>(this->devData, this->n, 1, this->n, 0, 0, 0);
#endif
            gpuErrchk(cudaDeviceSynchronize());
        }
        cudaEventRecord(stop);

        break;
    case MapType::DYNAMIC_PARALLELISM3:
        cudaEventRecord(start);
#ifdef MEASURE_POWER
        GPUPowerBegin(this->n, 100, 0, std::string("DP3-") + std::to_string(this->deviceId));
#endif
        for (uint32_t i = 0; i < nTimes; ++i) {
#ifdef DP
            kernelDynamicParallelismHYBRID<<<this->GPUGrid, this->GPUBlock>>>(this->devData, this->n, 1, this->n, 0, 0);
#endif
            gpuErrchk(cudaDeviceSynchronize());
        }
        cudaEventRecord(stop);

        break;
    }
    cudaEventSynchronize(stop);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
#ifdef MEASURE_POWER
    GPUPowerEnd();
#endif
#ifdef DEBUG
    printf("\x1b[1mok\n\x1b[0m");
    fflush(stdout);
#endif

    // return computing time
    float msecs = 0;
    cudaEventElapsedTime(&msecs, start, stop);
    return msecs / ((float)nTimes);
}

void Simplex3DRegular::printHostData() {
    // has little use but implemented anyway
    for (int i = 0; i < n; i++) {
        printf("\n[z = %i]\n", i);
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                if ((int)this->hostData[i * n * n + j * n + k] == 0) {
                    printf("  ");
                } else {
                    printf("%i ", (int)this->hostData[i * n * n + j * n + k]);
                }
            }
            printf("\n");
        }
    }
}

void Simplex3DRegular::printDeviceData() {
    transferDeviceToHost();
    printHostData();
}

bool Simplex3DRegular::compare(Simplex3DRegular* a, Simplex3DRegular* b) {
    bool res = true;
    if (a->n != b->n) {
        return false;
    }
    for (size_t z = 0; z < a->n; ++z) {
        for (size_t y = 0; y < a->n; ++y) {
            for (size_t x = 0; x < a->n; ++x) {
                size_t i = z * a->n * a->n + y * a->n + x;
                if (a->hostData[i] != b->hostData[i]) {
                    // printf("a[%lu, %lu, %lu] (%lu) = %i != %i b\n", x, y, z, i, a->hostData[i], b->hostData[i]);
                    res = false;
                }
            }
        }
    }
    return res;
}
