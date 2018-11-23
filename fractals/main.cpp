#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>
#include <math.h>
#include "interface.h"
#include "main.h"

int main(int argc, char **argv){
	//srand ( time(NULL) );
    checkargs(argc, argv, 4, "./prog <b> <r> <statfile>\n[exponent for blocksize] 2^b\n[fractal level] r = 0, 1, 2, ...\n");

    printf("maxlong %lu\n", LONG_MAX);
    int bpower = atoi(argv[1]);
    unsigned long r = atoi(argv[2]);
    statistics stat = gpudummy(r, bpower);
    write_result(argv[3], stat, bpower, r);
    printf("\x1b[1m"); fflush(stdout);
    printf("bounding-box:mean=%f[s]   var=%f   stdev=%f   sterr=%f\n", stat.mean1, stat.variance1, stat.stdev1, stat.sterr1);
    printf("lambda-map:  mean=%f[s]   var=%f   stdev=%f   sterr=%f\n", stat.mean2, stat.variance2, stat.stdev2, stat.sterr2);
    printf("\x1b[0m"); fflush(stdout);
}
