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
    this->nElementsSimplex = n * (n + 1) * (n + 2) / 6;
    if (maptype < 3) {
        this->mapType = static_cast<MapType>(mapType);
    } else {
        this->mapType = MapType::NOT_IMPLEMENTED;
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
    if (this->powerOfTwoSize < 1) {
        printf("Power of two cannot be lower than 1, a cube of 1x1x1 is a trivial problem\n");
        return false;
    }

    switch (this->mapType) {
    case MapType::BOUNDING_BOX:
        this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY, BSIZE3DZ);
        this->GPUGrid = dim3((n + GPUBlock.x - 1) / GPUBlock.x, (n + GPUBlock.y - 1) / GPUBlock.y, (n + GPUBlock.z - 1) / GPUBlock.z);
        break;
    case MapType::HADOUKEN:
        this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY, BSIZE3DZ);
        this->GPUGrid = dim3((n / 2 + GPUBlock.x - 1) / GPUBlock.x, (n / 2 + GPUBlock.y - 1) / GPUBlock.y, (3 * (n - 1) / 4 + GPUBlock.z - 1) / GPUBlock.z);
        break;
    case MapType::DYNAMIC_PARALLELISM:
        this->GPUBlock = dim3(BSIZE3DX, BSIZE3DY, BSIZE3DZ);
        this->GPUGrid = dim3((n + GPUBlock.x - 1) / GPUBlock.x, (n + GPUBlock.y - 1) / GPUBlock.y, (n + GPUBlock.z - 1) / GPUBlock.z);
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
    printf("\x1b[1mdoBenchmark(): kernel (map=%i, r=%i)...\x1b[0m", this->mapType, nTimes);
    fflush(stdout);
#endif

    cudaEventRecord(start);
    switch (mapType) {
    case MapType::BOUNDING_BOX:
        for (uint32_t i = 0; i < nTimes; ++i) {
            kernelBoundingBox<<<this->GPUGrid, this->GPUBlock>>>(this->devData, this->n);
        }
        break;
    case MapType::HADOUKEN:
        for (uint32_t i = 0; i < nTimes; ++i) {
            kernelHadouken<<<this->GPUGrid, this->GPUBlock>>>(this->devData, this->n);
        }
        break;
    case MapType::DYNAMIC_PARALLELISM:
        for (uint32_t i = 0; i < nTimes; ++i) {
            kernelDynamicParallelism<<<this->GPUGrid, this->GPUBlock>>>(this->devData, this->n);
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

    // return computing time
    float msecs = 0;
    cudaEventElapsedTime(&msecs, start, stop);
    return msecs / ((float)nTimes);
}

void Simplex3DRegular::printHostData() {
    // has little use but implemented anyway
    for (int i = 0; i < n; i++) {
        for (int j = 0; j < n; j++) {
            for (int k = 0; k < n; k++) {
                printf("%i ", (int)this->hostData[i * n * n + j * n + k]);
            }
            printf("\n");
        }
        printf("\n\n");
    }
}

void Simplex3DRegular::printDeviceData() {
    transferDeviceToHost();
    printHostData();
}
