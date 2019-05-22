#ifndef TOOLS_H
#define TOOLS_H

void last_cuda_error(const char *msg);

void print_gpu_specs(int dev);

template <typename T>
void initmat(T *E, unsigned long n2, T c);

template <typename T>
void printspace(T *E, unsigned long n);

template <typename T>
int verify(T *E, int n, int c);

#endif
