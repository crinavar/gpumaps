#pragma once

#define BSIZE1D 256
#define BSIZE2DX 32
#define BSIZE2DY 32

#define CL 0.4807498567691361274405461035933f
#define CR 0.6933612743506347048433522747859f

enum class MapType {
    BOUNDING_BOX = 0,
    HADOUKEN = 1,
    DYNAMIC_PARALLELISM = 2
}

class Simplex3DRegular {
    size_t n;
    size_t nElementsCube;
    size_t nElementsSimplex;

    uint32_t deviceId;
    MapType mapType;
    uint32_t powerOfTwoSize;

    dim3 GPUBlock;
    dim3 GPUGrid;

    bool hasBeenAllocated;

    MTYPE hostData;
    MTYPE devData;

    Simplex3DRegular(uint32_t deviceId, uint32_t powerOfTwoSize, uint32_t maptype);
    bool isMapTypeImplemented();

public:
    void init();
    void allocateMemory();
    void freeMemory();
    void transferHostToDevice();
    void transferDeviceToHost();
    StatsCollector doBenchmark(uint32_t repeats);
}
#endif
