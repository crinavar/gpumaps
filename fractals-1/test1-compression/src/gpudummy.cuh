#ifndef GPUDUMMY_H
#define GPUDUMMY_H

#include "interface.h"
#include "running_stat.h"

// cuda side function
statistics gpudummy(unsigned int method, unsigned int repeats);

RunningStat* boundingBox(MTYPE* M, unsigned long n, unsigned long nb, unsigned long rb);
RunningStat* lambda(MTYPE* M, unsigned long n, unsigned long nb, unsigned long rb);
RunningStat* lambda_tc(MTYPE* M, unsigned long n, unsigned long nb, unsigned long rb);
RunningStat* lambda_tc2(MTYPE* M, unsigned long n, unsigned long nb, unsigned long rb);
RunningStat* lambda_tc3(MTYPE* M, unsigned long n, unsigned long nb, unsigned long rb);
RunningStat* lambda_tc_optimized(MTYPE* M, unsigned long n, unsigned long nb, unsigned long rb);
RunningStat* lambda_tc_optimized2(MTYPE* M, unsigned long n, unsigned long nb, unsigned long rb);

template<typename Lambda>
RunningStat* performLoad(MTYPE *M, unsigned long n, unsigned long nb, unsigned long rb, dim3 block, dim3 grid, Lambda map, bool isBB); 


#endif
