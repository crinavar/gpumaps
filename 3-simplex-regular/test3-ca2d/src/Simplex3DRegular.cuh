#pragma once

#include <cassert>
#include <cinttypes>
#include <cuda.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>
#include <vector>

// Lazy Fix
#define MTYPE uint32_t

#define CL 0.4807498567691361274405461035933f
#define CR 0.6933612743506347048433522747859f

#ifdef MEASURE_POWER
#include "nvmlPower.hpp"
#endif

enum class MapType {
    BOUNDING_BOX,
    HADOUKEN,
    DYNAMIC_PARALLELISM,
    NOT_IMPLEMENTED
};

class Simplex3DRegular {
public:
    size_t n;
    size_t nElementsCube;
    size_t nElementsSimplex;
    size_t nElementsCubeWithHalo;

    uint32_t iterationCount;
    float density;
    uint32_t seed;

    uint32_t deviceId;
    uint32_t powerOfTwoSize;
    MapType mapType;

    dim3 GPUBlock;
    dim3 GPUGrid;
    dim3 GPUGridAux;

    bool hasBeenAllocated;

    MTYPE* hostData;
    MTYPE* devData;
    MTYPE* devDataPong;

    Simplex3DRegular(uint32_t deviceId, uint32_t powerOfTwoSize, uint32_t maptype, float density, uint32_t seed);
    ~Simplex3DRegular();

    static bool compare(Simplex3DRegular* a, Simplex3DRegular* b);
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
