//////////////////////////////////////////////////////////////////////////////////
//  gpumaps                                                                     //
//  A GPU benchmark of mapping functions                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2018 Cristobal A. Navarro                                       //
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
#include "gputools.cuh"
#include "kernels.cuh"
#include "gpubenchmarks.cuh"

// main routine that executes on the host
int main(int argc, char **argv){
	//srand ( time(NULL) );
	if(argc != 5){
		printf("\nrun as ./prog <dev> <N> <repeats> <map>\nmap:\n1 bbox\n2 lambda\n3 rectangle\n4 hadouken\n5 tensor core-hadouken\n\n");
		exit(EXIT_FAILURE);
	}
    unsigned int dev = atoi(argv[1]);
    unsigned int n = atoi(argv[2]);
    unsigned int REPEATS = atoi(argv[3]);
    unsigned int method = atoi(argv[4]);
    if(method > 5 || method == 0){
		printf("\nrun as ./prog <dev> <N> <repeats> <map>\nmap:\n1 bbox\n2 lambda\n3 rectangle\n4 hadouken\n5 tensor core-hadouken\n\n");
		exit(EXIT_FAILURE);
    }
    cudaSetDevice(dev);
#ifdef DEBUG
    print_gpu_specs(dev);
#endif
    last_cuda_error("cudaSetDevice");
    double time;
    switch(method){
        case 1:
	        time = bbox(n, REPEATS);
            break;
        case 2:
            time = lambda(n, REPEATS);
            break;
        case 3:
            time = rectangle(n, REPEATS);
            break;
        case 4:
            time = hadouken(n, REPEATS);
            break;
        case 5:
            time = tensorCoreHadouken(n, REPEATS);
    }
    printf("%f\n", time);
}
