// custom functions. 
// all have the 'cf_' prefix.
//

#define DTYPE float2
#define MTYPE float

// integer log2
int cf_log2i(int val){
    int copy = val;
    int r = 0;
    while (copy >>= 1) 
        ++r;
    return r;
}

void print_array(DTYPE *a, const int n){
	for(int i=0; i<n; i++)
		printf("a[%d] = %f\n", i, a[i]);
}

void print_matrix(MTYPE *mat, const int n, const char *msg){
    printf("[%s]:\n", msg);
	for(int i=0; i<n; i++){
	    for(int j=0; j<n; j++){
            printf("%i ", mat[i*n + j]);
        }
        printf("\n");
    }
}
