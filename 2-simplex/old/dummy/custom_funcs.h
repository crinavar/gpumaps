// custom functions. 
// all have the 'cf_' prefix.

// integer log2
int cf_log2i(int val){
    int copy = val;
    int r = 0;
    while (copy >>= 1) 
        ++r;
    return r;
}

void print_array(float *b, int n){
	for(int i=0; i<n; i++)
		printf("b[%d] = %f\n", i, b[i]);
}
