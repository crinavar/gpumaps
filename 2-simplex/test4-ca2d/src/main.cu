//////////////////////////////////////////////////////////////////////////////////
//  gpumaps                                                                     //
//  A GPU benchmark of mapping functions                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2018 Cristobal A. Navarro.                                      //
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
#define DTYPE char
#define MTYPE char
#define EPSILON 0.001
#define WORDLEN 31
#define MAXSTREAMS 32
#define PRINTLIMIT 256

// for HADOUKEN
#define HADO_TOL HADO_FACTOR* BSIZE2D
//#define EXTRASPACE

#include <cuda.h>
#include <stdint.h>
#include <stdio.h>
#include <sys/time.h>
#include <time.h>

#include "kernels.cuh"

#include "gputools.cuh"

#include "gpubenchmarks.cuh"

int main(int argc, char** argv) {
    // srand ( time(NULL) );
    if (argc < 7) {
        printf("arguments must be: <dev> <N> <repeats> <method> <density> <seed>\nmethod:\n1 bounding box\n2 lambda\n3 rectangle\n4 hadouken\n5 DP\n\n");
        exit(1);
    }
    unsigned int dev = atoi(argv[1]);
    unsigned int n = atoi(argv[2]);
    unsigned int REPEATS = atoi(argv[3]);
    unsigned int method = atoi(argv[4]);
    double density = atof(argv[5]);
    unsigned int seed = atof(argv[6]);
    if (method > 5 || method == 0) {
        printf("\nrun as ./prog <dev> <N> <repeats> <map>\nmap:\n1 bbox\n2 lambda\n3 rectangle\n4 hadouken\n5 DP\n\n");
        exit(EXIT_FAILURE);
    }
#ifndef DP
    if (method == 5) {
        printf("To enable the Dynamic Parallelism approach, please compile with `make DP=YES`.\n");
        exit(-1);
    }
#endif
    cudaSetDevice(dev);
#ifdef DEBUG
    print_gpu_specs(dev);
#endif
    last_cuda_error("cudaSetDevice");
    double time;
    srand(seed);
    switch (method) {
    case 1:
        time = bbox(n, REPEATS, density);
        break;
    case 2:
        time = lambda(n, REPEATS, density);
        break;
    case 3:
        time = rectangle(n, REPEATS, density);
        break;
    case 4:
        time = hadouken(n, REPEATS, density);
        break;
    case 5:
        time = DynamicParallelism(n, REPEATS, density);
        break;
    }
    printf("%f\n", time);
}
