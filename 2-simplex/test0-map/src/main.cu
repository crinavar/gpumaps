//////////////////////////////////////////////////////////////////////////////////
//  gpumaps                                                                     //
//  A GPU benchmark of mapping functions                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
//                                                                              //
//  This file is part of gpumaps.                                               //
//  gpumaps is free software: you can redistribute it and/or modify             //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  gpumaps is distributed in the hope that it will be useful,                  //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with gpumaps.  If not, see <http://www.gnu.org/licenses/>.            //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
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
	if(argc < 5){
		printf("arguments must be: <dev> <N> <repeats> <method>\nmethod:\n1 bb\n2 utm\n3 ltmn\n4 ltmx\n5 flatrec\n6 ltmr\n7 rectangle\n8 recursive\n\n");
		exit(1);
	}
    unsigned int dev = atoi(argv[1]);
    unsigned int n = atoi(argv[2]);
    unsigned int REPEATS = atoi(argv[3]);
    unsigned int method = atoi(argv[4]);
    cudaSetDevice(dev);
    last_cuda_error("cudaSetDevice");
    double time;
    switch(method){
        case 1:
	        time = bbox(n, REPEATS);
            break;
        case 2:
            time = avril(n, REPEATS);
            break;
        case 3:
            time = lambda_newton(n, REPEATS);
            break;
        case 4:
            time = lambda_standard(n, REPEATS);
            break;
        case 5:
            time = lambda_flatrec(n, REPEATS);
            break;
        case 6:
            time = lambda_inverse(n, REPEATS);
            break;
        case 7:
            time = rectangle_map(n, REPEATS);
            break;
        case 8:
            if(argc != 6){
                fprintf(stderr, "recursive map requires an aditional parameter <recn>\n");
                exit(EXIT_FAILURE);
            }
            time = recursive_map(n, atoi(argv[5]), REPEATS);
            break;
    }
    printf("%f\n", time);
}

