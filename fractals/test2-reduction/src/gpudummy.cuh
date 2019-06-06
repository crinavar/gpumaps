#ifndef GPUDUMMY_H
#define GPUDUMMY_H

#define MTYPE int

#include "interface.h"
#include "running_stat.h"

// cuda side function
statistics gpudummy(unsigned int method, unsigned int repeats);

RunningStat* boundingBox(MTYPE* M, double* res, unsigned long n, unsigned long nb, unsigned long rb);
RunningStat* lambda(MTYPE* M, double* res, unsigned long n, unsigned long nb, unsigned long rb);
RunningStat* lambda_tc(MTYPE* M, double* res, unsigned long n, unsigned long nb, unsigned long rb);
RunningStat* lambda_tc_optimized(MTYPE* M, double* res, unsigned long n, unsigned long nb, unsigned long rb);

template<typename Lambda>
RunningStat* performLoad(MTYPE* M, double* res, unsigned long n, unsigned long nb, unsigned long rb, dim3 block, dim3 grid, Lambda map); 


#endif
