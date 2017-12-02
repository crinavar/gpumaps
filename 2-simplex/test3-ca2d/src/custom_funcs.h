// custom functions. 
// all have the 'cf_' prefix.
//

#define DTYPE char
#define MTYPE char
#define EPSILON 0.001

// integer log2
int cf_log2i(int val){
    int copy = val;
    int r = 0;
    while (copy >>= 1) 
        ++r;
    return r;
}

uint32_t cntsetbits(uint32_t i){
     // C or C++: use uint32_t
     i = i - ((i >> 1) & 0x55555555);
     i = (i & 0x33333333) + ((i >> 2) & 0x33333333);
     return (((i + (i >> 4)) & 0x0F0F0F0F) * 0x01010101) >> 24;
}

void print_array(DTYPE *a, const int n){
	for(int i=0; i<n; i++)
		printf("a[%d] = %f\n", i, a[i]);
}

void print_matrix(MTYPE *mat, const int n, const char *msg){
    printf("[%s]:\n", msg);
	for(int i=0; i<n; i++){
	    for(int j=0; j<n; j++){
            if( i > j ){
                if(mat[i*n + j] == 1){ 
                    printf("1 ");
                }
                else{ 
                    printf("  "); 
                }
            }
            else{
                printf("%i ", mat[i*n + j]);
            }
        }
        printf("\n");
    }
}
