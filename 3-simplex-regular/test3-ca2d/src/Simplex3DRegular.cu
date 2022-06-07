#include "Simplex3DRegular.cuh"

#include "GPUKernels.cuh"
#include "GPUTools.cuh"

#define OFFSET -0.4999f
//#define OFFSET 0.0f
#define REAL float

Simplex3DRegular::Simplex3DRegular(uint32_t deviceId, uint32_t powerOfTwoSize, uint32_t maptype, float density, uint32_t seed)
    : deviceId(deviceId)
    , powerOfTwoSize(powerOfTwoSize)
    , density(density)
    , seed(seed) {

    this->n = 1 << powerOfTwoSize;
    this->nElementsCube = n * n * n;
    this->nElementsSimplex = n * (n - 1) * (n + 1) / 6;
    this->nElementsCubeWithHalo = (n + 2) * (n + 2) * (n + 2);
    this->mapType = MapType::NOT_IMPLEMENTED;
    this->iterationCount = 0;
    if (maptype < 3) {
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
        }
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
    this->hostData = (MTYPE*)malloc(sizeof(MTYPE) * (this->nElementsCubeWithHalo));
    cudaMalloc(&devData, sizeof(MTYPE) * this->nElementsCubeWithHalo);
    cudaMalloc(&devDataPong, sizeof(MTYPE) * this->nElementsCubeWithHalo);
    gpuErrchk(cudaPeekAtLastError());
    this->hasBeenAllocated = true;
}

void Simplex3DRegular::freeMemory() {
    // clear
    free(hostData);
    cudaFree(devData);
    cudaFree(devDataPong);
    gpuErrchk(cudaPeekAtLastError());
    this->hasBeenAllocated = false;
    this->iterationCount = 0;
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
    }

#ifdef DEBUG
    printf("init(): parallel space: b(%i, %i, %i) g(%i, %i, %i)\n", GPUBlock.x, GPUBlock.y, GPUBlock.z, GPUGrid.x, GPUGrid.y, GPUGrid.z);
    fflush(stdout);
#endif

#ifdef DEBUG
    printf("init(): filling cube with 0.\n");
#endif
    for (size_t i = 0; i < this->nElementsCubeWithHalo; ++i) {
        this->hostData[i] = (MTYPE)0;
    }
#ifdef DEBUG
    printf("init(): done.\n");
#endif

    srand(this->seed);
#ifdef DEBUG
    printf("init(): filling simplex initial state with p = %f .\n", this->density);
#endif
    for (size_t z = 0; z < this->n; ++z) {
        for (size_t y = 0; y < this->n; ++y) {
            for (size_t x = 0; x < this->n; ++x) {
                if (x + y + z < n - 1) {
                    size_t i = (z + 1) * (n + 2) * (n + 2) + (y + 1) * (n + 2) + (x + 1);
                    if (rand() / (float)RAND_MAX < this->density) {

                        this->hostData[i] = (MTYPE)1;
                    }
                }
            }
        }
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
    cudaMemcpy(this->devData, this->hostData, sizeof(MTYPE) * this->nElementsCubeWithHalo, cudaMemcpyHostToDevice);
    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}
void Simplex3DRegular::transferDeviceToHost() {
    cudaMemcpy(this->hostData, this->devData, sizeof(MTYPE) * this->nElementsCubeWithHalo, cudaMemcpyDeviceToHost);

    gpuErrchk(cudaPeekAtLastError());
    gpuErrchk(cudaDeviceSynchronize());
}

float Simplex3DRegular::doBenchmarkAction(uint32_t nTimes) {

#ifdef DEBUG
    printf("doBenchmark(): mapping to simplex of n=%lu   Vcube = %lu   Vsimplex = %lu\n", this->n, this->nElementsCube, this->nElementsSimplex);
    printf("doBenchmark(): Cube size is %f MB\n", (float)this->nElementsCube * sizeof(MTYPE) / (1024.0 * 1024.0f));
#endif

    this->iterationCount += nTimes;
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
        gpuErrchk(cudaStreamCreate(&streams[0]));
        gpuErrchk(cudaStreamCreate(&streams[1]));
    }

    MTYPE* devPointerAux;

    uint32_t blockedN = ceil(n / (float)BSIZE3DX);
    switch (this->mapType) {
    case MapType::BOUNDING_BOX:
        cudaEventRecord(start);
#ifdef MEASURE_POWER
        GPUPowerBegin(this->n, 100, 0, std::string("BB-") + std::string(this->deviceId));
#endif
        for (uint32_t i = 0; i < nTimes; ++i) {
            kernelBoundingBox<<<this->GPUGrid, this->GPUBlock>>>(this->devData, this->devDataPong, this->n, blockedN, n + 2);
            gpuErrchk(cudaDeviceSynchronize());
            std::swap(devData, devDataPong);
            // devPointerAux = devData;
            // devData = devDataPong;
            // devDataPong = devPointerAux;
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        break;
    case MapType::HADOUKEN:
        cudaEventRecord(start);
#ifdef MEASURE_POWER
        GPUPowerBegin(this->n, 100, 0, std::string("HAD-") + std::string(this->deviceId));
#endif
        for (uint32_t i = 0; i < nTimes; ++i) {
            kernelHadouken<<<this->GPUGrid, this->GPUBlock, 0, streams[0]>>>(this->devData, this->devDataPong, this->n, blockedN, n + 2);
            kernelHadoukenStrip<<<this->GPUGridAux, this->GPUBlock, 0, streams[1]>>>(this->devData, this->devDataPong, this->n, blockedN, n + 2);
            //  printf("(%i, %i, %i)\n", this->GPUGridAux.x, this->GPUGridAux.y, this->GPUGridAux.z);
            gpuErrchk(cudaDeviceSynchronize());
            std::swap(devData, devDataPong);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        break;

    case MapType::DYNAMIC_PARALLELISM:
        cudaEventRecord(start);
#ifdef MEASURE_POWER
        GPUPowerBegin(this->n, 100, 0, std::string("DP-") + std::string(this->deviceId));
#endif
        for (uint32_t i = 0; i < nTimes; ++i) {
#ifdef DP
            kernelDynamicParallelism<<<this->GPUGrid, this->GPUBlock>>>(this->devData, this->devDataPong, this->n, 1, n / 2, 0, 0, n + 2);
#endif
            gpuErrchk(cudaDeviceSynchronize());
            std::swap(devData, devDataPong);
        }
        cudaEventRecord(stop);
        cudaEventSynchronize(stop);

        break;
    }
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
    for (int i = 0; i < n + 2; i++) {
        printf("\n[z = %i]\n", i);
        for (int j = 0; j < n + 2; j++) {
            for (int k = 0; k < n + 2; k++) {
                size_t index = i * (n + 2) * (n + 2) + j * (n + 2) + k;
                if ((int)this->hostData[index] == 0) {
                    printf("  ");
                } else {
                    printf("%i ", (int)this->hostData[index]);
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
    for (size_t z = 0; z < a->n + 2; ++z) {
        for (size_t y = 0; y < a->n + 2; ++y) {
            for (size_t x = 0; x < a->n + 2; ++x) {
                size_t i = z * (a->n + 2) * (a->n + 2) + y * (a->n + 2) + x;
                if (a->hostData[i] != b->hostData[i]) {
                    // printf("a[%lu, %lu, %lu] (%lu) = %i != %i b\n", x, y, z, i, a->hostData[i], b->hostData[i]);
                    res = false;
                }
            }
        }
    }
    return res;
}
