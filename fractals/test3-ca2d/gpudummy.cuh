#ifndef GPUDUMMY_H
#define GPUDUMMY_H

#define MTYPE char
#define PRINTLIMIT 0xFFFFFFFF
#define INNER_REPEATS 1


#include "interface.h"
#include "running_stat.h"

// cuda side function
statistics gpudummy(unsigned int method, unsigned int repeats, double density);

RunningStat* boundingBox(MTYPE *ddata, MTYPE *dmat1, MTYPE *dmat2, unsigned long n, unsigned long nb, unsigned long rb, unsigned long msize, unsigned long trisize);

RunningStat* lambda(MTYPE *ddata, MTYPE *dmat1, MTYPE *dmat2, unsigned long n, unsigned long nb, unsigned long rb, unsigned long msize, unsigned long trisize);

RunningStat* lambda_tc(MTYPE *ddata, MTYPE *dmat1, MTYPE *dmat2, unsigned long n, unsigned long nb, unsigned long rb, unsigned long msize, unsigned long trisize);

RunningStat* lambda_tc_optimized(MTYPE *ddata, MTYPE *dmat1, MTYPE *dmat2, unsigned long n, unsigned long nb, unsigned long rb, unsigned long msize, unsigned long trisize);

template<typename Lambda>
RunningStat* performLoad( MTYPE *ddata, MTYPE *dmat1, MTYPE *dmat2, unsigned long n, unsigned long nb, unsigned long rb, unsigned long msize, unsigned int trisize, dim3 block, dim3 grid,
                            Lambda map, unsigned int aux1, unsigned int aux2, unsigned int aux3);
#endif
