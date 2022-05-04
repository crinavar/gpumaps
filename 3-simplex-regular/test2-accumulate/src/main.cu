#include <cinttypes>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

#include "Simplex3DRegular.cuh"
#include "StatsCollector.h"

#define PRINT_LIMIT 6
const uint32_t INNER_REPEATS = 1;

int main(int argc, char** argv) {
    // srand ( time(NULL) );
    if (argc != 5) {
        printf("run as ./prog <deviceId> 2^<N> <repeats> <mapType>\nmapType:\n\t0 = bounding box\n\t1 = Hadouken\n\t2 = Dynamic Parallelism\n");
        exit(1);
    }

    uint32_t deviceId = atoi(argv[1]);
    uint32_t powerOfTwoSize = atoi(argv[2]);
    uint32_t repeats = atoi(argv[3]);
    uint32_t mapType = atoi(argv[4]);

    Simplex3DRegular* benchmark = new Simplex3DRegular(deviceId, powerOfTwoSize, mapType);
    if (!benchmark->init()) {
        exit(1);
    }
    float iterationTime = benchmark->doBenchmarkAction(INNER_REPEATS);
    benchmark->transferDeviceToHost();

#ifdef DEBUG
    if (powerOfTwoSize <= PRINT_LIMIT) {
        benchmark->printDeviceData();
    }
#endif

#ifdef VERIFY
    Simplex3DRegular* reference = new Simplex3DRegular(deviceId, powerOfTwoSize, 0);
    if (!reference->init()) {
        exit(1);
    }
    reference->doBenchmarkAction(INNER_REPEATS);
    reference->transferDeviceToHost();

    if (!Simplex3DRegular::compare(benchmark, reference)) {
        printf("\n[VERIFY] verification FAILED!.\n\n");

        exit(1);
    }

    printf("\n[VERIFY] verification successful.\n\n");

#endif

    // StatsCollector<float> times = prepareAndPerformBenchmark(deviceId, powerOfTwo, repeats, mapType);

#ifdef DEBUG
    printf("maxlong %lu\n", LONG_MAX);
    printf("\x1b[1m");
    fflush(stdout);
    printf("main(): avg kernel time: %f ms\n", iterationTime);
    printf("\x1b[0m");
    fflush(stdout);
#else
    printf("%f\n", iterationTime);
#endif
}
