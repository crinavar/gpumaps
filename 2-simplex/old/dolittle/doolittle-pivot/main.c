#include <stdio.h>
#include <stdlib.h>
#include "doolittle_pivot.h"

int main(int argc, char ** argv){
    if(argc != 2){
        fprintf(stderr, "run as ./prog n\n");
        exit(EXIT_FAILURE);
    }
    unsigned int n = atoi(argv[1]);
    double *A = random_matrix(n);
    int *pivot = (int*)malloc(sizeof(int)*n);
    int err = Doolittle_LU_Decomposition_with_Pivoting(A, pivot, n);
    if(err < 0){
        printf(" Matrix A is singular\n");
        exit(EXIT_SUCCESS);
    }
    printf(" The LU decomposition of A is \n");
    printmat(A, n);

}
