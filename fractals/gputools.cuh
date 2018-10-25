#ifndef TOOLS_H
#define TOOLS_H

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) 
            exit(code);
    }
}

template <typename T>
void initmat(T *E, unsigned long n2, T c){
    for(unsigned long i=0; i<n2; ++i){
        E[i] = c;
    }
}

template <typename T>
void printspace(T *E, unsigned long n){
    for(unsigned long i=0; i<n; ++i){
        for(unsigned long j=0; j<n; ++j){
            T val = E[i*n + j];
            if(val > 0){
                printf("%i ", val);
            }
            else{
                printf("  ");
            }
            //printf("%i ", val);
        }
        printf("\n");
    }
}

template <typename T>
int verify(T *E, int n, int c){
    for(unsigned long i=0; i<n; ++i){
        for(unsigned long j=0; j<n; ++j){
            if(E[i*n + j] == c){
                if(((n-1-i)&j) != 0){
                    return 0;
                }
            }
            if(E[i*n + j] != c){
                if(((n-1-i)&j) == 0){
                    return 0;
                }
            }
        }
    }
    return 1;
}

#endif
