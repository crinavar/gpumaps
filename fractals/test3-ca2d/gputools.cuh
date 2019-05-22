#ifndef TOOLS_H
#define TOOLS_H

void last_cuda_error(const char *msg);

void print_gpu_specs(int dev);

template <typename T>
void initmat(T *E, unsigned long n2, T c);

template <typename T>
void printspace(T *E, unsigned long n);

template <typename T>
int verifyWrite(T *E, int n, int c);

template <typename T>
int verifyReduction(T *E, int n);

void init(unsigned long n, MTYPE **hdata, MTYPE **hmat, MTYPE **ddata, MTYPE **dmat1, MTYPE **dmat2, unsigned long *msize, unsigned long *trisize, double DENSITY);

void set_everything(MTYPE *mat, const unsigned long n, MTYPE val);

void set_randconfig(MTYPE *mat, const unsigned long n, double DENSITY);
#endif
