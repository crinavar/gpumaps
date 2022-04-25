#pragma once

#include "StatsCollector.h"
#include <cassert>
#include <cinttypes>
#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <vector>

// Lazy Fix
#define MTYPE uint32_t

#include "GPUKernels.cuh"
#include "GPUTools.cuh"

#define CL 0.4807498567691361274405461035933f
#define CR 0.6933612743506347048433522747859f

enum class MapType {
    BOUNDING_BOX = 0,
    HADOUKEN = 1,
    DYNAMIC_PARALLELISM = 2,
    NOT_IMPLEMENTED = 99
};

class Simplex3DRegular {
    size_t n;
    size_t nElementsCube;
    size_t nElementsSimplex;

    uint32_t deviceId;
    uint32_t powerOfTwoSize;
    MapType mapType;

    dim3 GPUBlock;
    dim3 GPUGrid;

    MTYPE* hostData;
    MTYPE* devData;

    bool hasBeenAllocated;

public:
    Simplex3DRegular(uint32_t deviceId, uint32_t powerOfTwoSize, uint32_t maptype);
    ~Simplex3DRegular();
    bool init();
    void allocateMemory();
    void freeMemory();
    void transferHostToDevice();
    void transferDeviceToHost();
    bool isMapTypeImplemented();

    void printHostData();
    void printDeviceData();

    float doBenchmarkAction(uint32_t nTimes);
};