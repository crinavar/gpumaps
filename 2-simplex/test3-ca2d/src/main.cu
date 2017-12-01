#include <stdio.h>
#include <cuda.h>
#include <time.h>
#include <sys/time.h>
#include "custom_funcs.h"
#include "kernels.cuh"
#include "gputools.cuh"
#include "gpubenchmarks.cuh"

// main routine that executes on the host
int main(int argc, char **argv){
	//srand ( time(NULL) );
	if(argc < 7){
		printf("arguments must be: <dev> <N> <repeats> <method> <density> <seed>\nmethod:\n1 bb\n2 utm\n3 ltmn\n4 ltmx\n5 flatrec\n6 ltmr\n7 rectangle\n8 recursive\n\n");
		exit(1);
	}
    unsigned int dev        = atoi(argv[1]);
    unsigned int n          = atoi(argv[2]);
    unsigned int REPEATS    = atoi(argv[3]);
    unsigned int method     = atoi(argv[4]);
    double        density    = atof(argv[5]);
    unsigned int seed       = atof(argv[6]);
    cudaSetDevice(dev);
    last_cuda_error("cudaSetDevice");
    double time;
    srand(seed);
    switch(method){
        case 1:
	        time = bbox(n, REPEATS, density);
            break;
        case 2:
            time = avril(n, REPEATS, density);
            break;
        case 3:
            time = lambda_newton(n, REPEATS, density);
            break;
        case 4:
            time = lambda_standard(n, REPEATS, density);
            break;
        case 5:
            time = lambda_flatrec(n, REPEATS, density);
            break;
        case 6:
            time = lambda_inverse(n, REPEATS, density);
            break;
        case 7:
            time = rectangle_map(n, REPEATS, density);
            break;
        case 8:
            if(argc != 8){
                fprintf(stderr, "recursive map requires an aditional parameter at the end <recn>\n");
                exit(EXIT_FAILURE);
            }
            time = recursive_map(n, atoi(argv[7]), REPEATS, density);
            break;
    }
    printf("%f\n", time);
}

