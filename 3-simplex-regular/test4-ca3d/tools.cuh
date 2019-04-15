#ifndef TOOLS_CUH
#define TOOLS_CUH

#define gpuErrchk(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort=true){
    if (code != cudaSuccess){
        fprintf(stderr,"GPUassert: %s %s %d\n", cudaGetErrorString(code), file, line);
        if (abort) 
            exit(code);
    }
}


void printcube(MTYPE *d, const unsigned long n);

// verifica el tetrahedro inscrito en un cubo de N^3 datos
template<typename Lambda>
bool verify(MTYPE *d, const unsigned long n, Lambda ver){
    for(unsigned long z=0; z<n; ++z){
        for(unsigned long y=0; y<n; ++y){
            for(unsigned long x=0; x<n; ++x){
                unsigned long i = z*n*n + y*n + x;
                if(x < y && y < z){
                    if(!ver(d[i], x, y, z)){
                        printf("inside data[%lu, %lu, %lu] (%lu) = %i\n", x, y, z, i, d[i]);
                        return false;
                    }
                }
                else{
                    if(d[i] != 0){
                        printf("outside data[%lu, %lu, %lu] (%lu) = %i\n", x, y, z, i, d[i]);
                        return false;
                    }
                }
            }
        }
    }
    return true;
}


void printcube(MTYPE *d, const unsigned long n){
    for(unsigned int z=0; z<n; ++z){
        printf("layer z=%i:\n", z);
        for(unsigned int y=0; y<n; ++y){
            for(unsigned int x=0; x<n; ++x){
                unsigned long i = z*n*n + y*n + x;
                if(x<y && y<z){
                    if(d[i] > 0){ printf("%i ", d[i]); }
                    else{ printf("  "); }
                }
                else{
                    printf("%i ", d[i]);
                }

            }
            printf("\n");
        }
        printf("\n");
    }
}

void printcube_coords(float *d, const unsigned long n){
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

void set_randconfig(MTYPE *cube, const unsigned long n, double DENSITY){
    for(unsigned int z=0; z<n; ++z){
        for(unsigned int y=0; y<n; ++y){
            for(unsigned int x=0; x<n; ++x){
                unsigned long i = z*n*n + y*n + x;
                if(x<y && y<z){
                    cube[i] = ((double)rand()/RAND_MAX) <= DENSITY ? 1 : 0;
                }
                else{
                    cube[i] = 0;
                }
            }
        }
    }
}

void set_alldead(MTYPE *cube, const unsigned long n){
    for(unsigned int z=0; z<n; ++z){
        for(unsigned int y=0; y<n; ++y){
            for(unsigned int x=0; x<n; ++x){
                unsigned long i = z*n*n + y*n + x;
                cube[i] = 0;
            }
        }
    }
}

void set_cell(MTYPE *cube, const unsigned long n, const int x, const int y, const int z, MTYPE val){
    unsigned long i = z*n*n + y*n + x;
    cube[i] = val;
}

unsigned long count_living_cells(MTYPE *cube, const unsigned long n){
    unsigned long c = 0;
    for(unsigned int z=0; z<n; ++z){
        for(unsigned int y=0; y<n; ++y){
            for(unsigned int x=0; x<n; ++x){
                if(x<y && y<z){
                    unsigned long i = z*n*n + y*n + x;
                    c += cube[i];
                }
            }
        }
    }
    return c;
}

void cubestatus(MTYPE *hcube, MTYPE *dcube, unsigned long N, unsigned long Vcube, unsigned int MAXPRINT){
        printf("cube: press enter to print\n");
        //getchar();
        cudaMemcpy(hcube, dcube, sizeof(MTYPE)*Vcube, cudaMemcpyDeviceToHost);
        if(N <= MAXPRINT){
            printcube(hcube, N);
        }
        printf("%lu cells alive\n", count_living_cells(hcube, N)); fflush(stdout);
        getchar();
}
#endif
