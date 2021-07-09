#ifndef TOOLS_H
#define TOOLS_H

void last_cuda_error(const char *msg);

void print_gpu_specs(int dev);

template <typename T>
void initmat(T *E, size_t n2, T c);

template <typename T>
void printspace(T *E, size_t n);

template <typename T>
int verifyWrite(T *E, int n, int c);

template <typename T>
int verifyReduction(T *E, int n);

void init(size_t n, size_t rb, MTYPE **hdata, MTYPE **hmat, MTYPE **ddata, MTYPE **dmat1, MTYPE **dmat2, size_t *msize, size_t *trisize, size_t &cm, size_t &cn, double DENSITY);

void set_everything(MTYPE *mat, const size_t n, MTYPE val);
int print_dmat(int PLIMIT, int rlevel, size_t nx, size_t ny, MTYPE *mat, const char *msg);
int print_dmat_gpu(int PLIMIT, int rlevel, size_t nx, size_t ny, MTYPE *mat, MTYPE *dmat, const char *msg);
int print_dmat_gpu_comp(int PLIMIT, int rlevel, size_t n, size_t rb, size_t blockSize, size_t nx, size_t ny, MTYPE *mat, MTYPE *dmat, const char *msg);

void set_randconfig(MTYPE *mat, const size_t n, size_t cm, size_t cn, double DENSITY);
#endif
