#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <sys/time.h>
#include <limits.h>
#include "interface.h"
#include "main.h"

int main(int argc, char **argv){
	//srand ( time(NULL) );
	if( argc != 5 ){
		printf("run as ./prog <dev> <N> <repeats> <maptype>\n<maptype>:\n0 = bounding box\n1 = hadouken map\n");
		exit(1);
	}

    unsigned long dev = atoi(argv[1]);
    unsigned long N = atoi(argv[2]);
    unsigned long repeats = atoi(argv[3]);
    unsigned long maptype = atoi(argv[4]);
    double ktime = gpudummy(dev, N, repeats, maptype);
	#ifdef DEBUG
		printf("maxlong %lu\n", LONG_MAX);
		printf("\x1b[1m"); fflush(stdout);
		printf("main(): avg kernel time: %f ms\n", ktime);
		printf("\x1b[0m"); fflush(stdout);
	#else
		printf("%f\n", ktime);
	#endif


}
