#include "main.h"
#include "interface.h"
#include <cinttypes>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h>
#include <time.h>

typedef struct StatsMaker{

    StatsMaker(uint32_t repeats){
        
    }
    vector<double> 
} StatsMaker;

int main(int argc, char** argv) {
    // srand ( time(NULL) );
    if (argc != 5) {
        printf("run as ./prog <deviceId> 2^<N> <repeats> <mapType>\nmapType:\n\t0 = bounding box\n\t2 = Hadouken\n\t3 = Hadouken TC\n");
        exit(1);
    }

    size_t powerOfTwo = atoi(argv[2]);
    uint32_t deviceId = atoi(argv[1]);
    uint32_t repeats = atoi(argv[3]);
    uint32_t mapType = atoi(argv[4]);

    double ktime = gpudummy(deviceId, powerOfTwo, repeats, mapType);
#ifdef DEBUG
    printf("maxlong %lu\n", LONG_MAX);
    printf("\x1b[1m");
    fflush(stdout);
    printf("main(): avg kernel time: %f ms\n", ktime);
    printf("\x1b[0m");
    fflush(stdout);
#else
    printf("%f\n", ktime);
#endif
}
