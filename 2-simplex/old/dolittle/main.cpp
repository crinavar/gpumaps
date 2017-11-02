#include<iostream>
#include<cstdio>
#define SCALE 10.0f
using namespace std;
void lu(float *a, float *l, float *u, int n);
void matmul(float *a, float *b, float *c, int n);
void print_matrix(float *a, int n, const char *msg);
float* random_matrix(int n);

int main(int argc, char **argv){
    if(argc != 2){
        fprintf(stderr, "run as ./prog n\n");
        exit(EXIT_FAILURE);
    }
    float *a, *l, *u, *b;
    int n = atoi(argv[1]);
    a = random_matrix(n);
    l = (float*)malloc(sizeof(float)*n*n);
    u = (float*)malloc(sizeof(float)*n*n);
    b = (float*)malloc(sizeof(float)*n*n);
    print_matrix(a, n, "A Matrix");

    // LU factorization
    lu(a, l, u, n);


    print_matrix(l, n, "L Matrix");
    print_matrix(u, n, "U Matrix");
    matmul(l, u, b, n); 
    print_matrix(b, n, "B = LU Matrix");
    return 0;
}

void matmul(float *a, float *b, float *c, int n){
    int i, j, k;
    for (i = 0; i < n; ++i){
        for (j = 0; j < n; ++j){
            c[i*n+j] = 0.0f;
            for(k = 0; k < n; ++k)
                c[i*n+j] += a[i*n+k]*b[k*n+j];
        }
    }
}

void lu(float *a, float *l, float *u, int n){
    int i = 0, j = 0, k = 0;
    for (i = 0; i < n; i++){
        // L matrix
        for (j = 0; j < n; j++){
            if (j < i)
                l[j*n+i] = 0;
            else{
                l[j*n+i] = a[j*n+i];
                for (k = 0; k < i; k++){
                    l[j*n+i] = l[j*n+i] - l[j*n+k] * u[k*n+i];
                }
            }
        }
        // U matrix
        for (j = 0; j < n; j++){
            if (j < i)
                u[i*n+j] = 0;
            else if (j == i)
                u[i*n+j] = 1;
            else{
                u[i*n+j] = a[i*n+j] / l[i*n+i];
                for (k = 0; k < i; k++){
                    u[i*n+j] = u[i*n+j] - ((l[i*n+k] * u[k*n+j]) / l[i*n+i]);
                }
            }
        }
    }
}

void print_matrix(float *a, int n, const char *msg){
    printf("%s\n", msg);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            printf("%2.4f ", a[i*n + j]);
        }
        printf("\n");
    }
    printf("\n");
}

float* random_matrix(int n){
    float *a = (float*)malloc(sizeof(float)*n*n);
    for (int i = 0; i < n; i++){
        for (int j = 0; j < n; j++){
            a[i*n + j] = SCALE*(float)rand()/(RAND_MAX);
        }
    }
    return a;
}
