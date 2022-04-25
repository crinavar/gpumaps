#pragma once

#define gpuErrchk(ans) \
    { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char* file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        fprintf(stderr, "GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort)
            exit(code);
    }
}

/*
void printcube(float *d, const unsigned long n);

// verifica el tetrahedro inscrito en un cubo de N^3 datos
template<typename Lambda>
bool verify(char *d, const unsigned long n, Lambda ver){
    bool res=true;
    for(unsigned long z=0; z<n; ++z){
        for(unsigned long y=0; y<n; ++y){
            for(unsigned long x=0; x<n; ++x){
                unsigned long i = z*n*n + y*n + x;
                if(x < y && y < z){
                    if(!ver(d[i], x, y, z)){
                        printf("inside data[%lu, %lu, %lu] (%lu) = %i\n", x, y, z, i, d[i]);
                        res=false;
                    }
                }
                else{
                    if(d[i] != 0){
                        printf("outside data[%lu, %lu, %lu] (%lu) = %i\n", x, y, z, i, d[i]);
                        res=false;
                    }
                }
            }
        }
    }
    return res;
}


void printcube(char *d, const unsigned long n){
    for(unsigned int z=0; z<n; ++z){
        printf("layer z=%i:\n", z);
        for(unsigned int y=0; y<n; ++y){
            for(unsigned int x=0; x<n; ++x){
                unsigned long i = z*n*n + y*n + x;
                printf("%i ", d[i]);
            }
            printf("\n");
        }
        printf("\n");
    }
}

void printcube_coords(char *d, const unsigned long n){
    for(unsigned int z=0; z<n; ++z){
        printf("layer z=%i:\n", z);
        for(unsigned int y=0; y<n; ++y){
            for(unsigned int x=0; x<n; ++x){
                unsigned long i = z*n*n + y*n + x;
                printf("%i (%i,%i,%i) [%i]  ", d[i], x, y, z, i);
            }
            printf("\n");
        }
        printf("\n");
    }
}
#endif
*/