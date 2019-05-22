#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#include <cuda.h>
#include "gpudummy.cuh"
#include "gputools.cuh"
#include "interface.h"
#include "main.h"

int main(int argc, char **argv){

    checkargs(argc, argv, 6, "./prog <dev> <repeats> <methods> <density> <seed>\nmethods:\n0 bounding box\n1 lambda\n2 lambda-tc\n3 lambda-tc_opt\n");

    unsigned int dev        = atoi(argv[1]);
    unsigned int REPEATS    = atoi(argv[2]);
    unsigned int method     = atoi(argv[3]);
    unsigned int seed       = atof(argv[5]);
    double       density    = atof(argv[4]);

    cudaSetDevice(dev);

#ifdef DEBUG
    print_gpu_specs(dev);
    printf("maxlong %lu\n", LONG_MAX);
#endif
    last_cuda_error("cudaSetDevice");

    srand(seed);

    statistics stat = gpudummy(method, REPEATS, density);
#ifdef DEBUG
    printf("\x1b[1m"); fflush(stdout);
    printf("results: mean=%f[s]   var=%f   stdev=%f   sterr=%f\n", stat.mean, stat.variance, stat.stdev, stat.sterr);
    printf("\x1b[0m"); fflush(stdout);
#endif
}
