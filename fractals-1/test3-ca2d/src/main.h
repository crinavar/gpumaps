#ifndef MAIN_H
#define MAIN_H

void write_result(const char *filename, statistics stat, int b, int r){
    FILE *fw = fopen(filename, "a");
    if(!fw){
        fprintf(stderr, "error writing to file %s\n", filename);
        exit(EXIT_FAILURE);
    }
    fprintf(fw, "%i    %i    %i    %g    %g    %g    %g\n", 
            (int)pow(2,r), r, b,
            stat.mean, stat.variance, stat.stdev, stat.sterr);
    fflush(fw);
    fclose(fw);
}

void checkargs(int argc, char **argv, int reqargs, const char* msg){
    if(argc != reqargs){
        fprintf(stderr, "Error: run as %s\n", msg);
        exit(EXIT_FAILURE);
    }
}


void export_result(float *b, int n, const char *filename){
    printf("exporting result vector.......");fflush(stdout);
    FILE *fw = fopen(filename, "w");
    if(!fw){
        fprintf(stderr, "error: cannot write to file %s\n", filename);
        exit(1);
    }
    for(int i=0; i<n; i++)
        fprintf(fw, "%f\n", b[i]);
    printf("ok\n");
}


int* load_table(const char *filename, int *n){
    int *table;
    FILE *fr = fopen(filename, "rb");
    if(!fr){
        printf("error loading table.... does file '%s' exist?\n", filename);
        exit(-1);
    }
    int res = fread(n, sizeof(int), 1, fr);
    table = (int*)malloc((*n)*sizeof(int));
    res = fread(table, sizeof(int), *n, fr);
    return table;
}

void print_results(float *b, int n){
	for(int i=0; i<n; i++)
		printf("b[%d] = %f\n", i, b[i]);
}


#endif
