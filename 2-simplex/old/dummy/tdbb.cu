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
	if(argc < 4){
		printf("arguments must be: <N> <method> <repeats>\nmethod:\n1 bb\n2 utm\n3 ltmn\n4 ltmx\n5 flatrec\n6 ltmr\n7 rectangle\n8 recursive\n\n");
		exit(1);
	}
    unsigned int n = atoi(argv[1]);
    unsigned int method = atoi(argv[2]);
    unsigned int REPEATS = atoi(argv[3]);
    double time;
    switch(method){
        case 1:
	        time = bbox(n, REPEATS);
        case 2:
            time = avril(n, REPEATS);
        case 3:
            time = lambda_newton(n, REPEATS);
        case 4:
            time = lambda_standard(n, REPEATS);
        case 5:
            time = lambda_flatrec(n, REPEATS);
        case 6:
            time = lambda_inverse(n, REPEATS);
        case 7:
            time = rectangle_map(n, REPEATS);
        case 8:
            time = recursive_map(n, atoi(argv[3]), REPEATS);
    }
    printf("%f\n", time);
    // rectangle map
    /*
    else if(atoi(argv[2])==7){
        if(N%2==0)
            td_computation_rectangle(b_h, sizeb, dgrecteven, block1, a_d, N, b_d, N2);
        else
            td_computation_rectangle(b_h, sizeb, dgrectodd, block1, a_d, N, b_d, N2);

        if(argc >= 4)
            export_result(b_h, N2, argv[3]);
    }
    // recursive map
    else if(atoi(argv[2])==8){
        if(argc != 5){
            fprintf(stderr, "recursive method needs another parameter\n");
            exit(EXIT_FAILURE);
        }
        int n=atoi(argv[3]);
        int m=N/n;
        if( (m % block1.x) != 0 ){
            fprintf(stderr, "error: m=%i, not a multiple of %i\n", m, block1.x);
            exit(1);
        }
        int k=cf_log2i(n);
        printf("N(dim)=%i  n=%i  m=%i\n", N, n, m);
        td_computation_recursive(b_h, sizeb, block1, a_d, N, b_d, N2, m, k);
        if(argc >= 5){
            export_result(b_h, N2, argv[4]);
        }
    }
    */
}

