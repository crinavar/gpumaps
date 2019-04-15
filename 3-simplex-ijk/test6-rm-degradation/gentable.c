#include <stdio.h>
#include <stdlib.h>
#include <math.h>

int main(int argc, char **argv){
    FILE *fw;
    int i, n, N, *table;
    if(argc < 3){
        printf("error, must run as ./gentable <n> <filename>\n");
        exit(-1);
    }
    n = atoi(argv[1]);
    N = n*(n+1)/2;
    table = (int*)malloc(N*sizeof(int));
    for(i=0; i<N; i++){
        table[i] = (int)(sqrt(0.25 + 2.0*(float)i) - 0.5);
    }
    fw = fopen(argv[2], "wb");
    if(!fw){
        printf("error saving into file '%s'\n", argv[2]);
        exit(-1);
    }
    fwrite(&N, sizeof(int), 1, fw);    
    fwrite(table, sizeof(int), N, fw);
    exit(1);
}
