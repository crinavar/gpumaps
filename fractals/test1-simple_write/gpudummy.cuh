#ifndef GPUDUMMY_H
#define GPUDUMMY_H

#include "interface.h"
#include "running_stat.h"

// cuda side function
statistics gpudummy(unsigned int method, unsigned int repeats);

RunningStat* boundingBox(char* M, unsigned long n, unsigned long nb, unsigned long rb);
RunningStat* lambda(char* M, unsigned long n, unsigned long nb, unsigned long rb);
RunningStat* lambda_tc(char* M, unsigned long n, unsigned long nb, unsigned long rb);
RunningStat* lambda_tc_optimized(char* M, unsigned long n, unsigned long nb, unsigned long rb);

template<typename Lambda>
RunningStat* performLoad(char *M, unsigned long n, unsigned long nb, unsigned long rb, dim3 block, dim3 grid, Lambda map, bool isBB); 


#endif
