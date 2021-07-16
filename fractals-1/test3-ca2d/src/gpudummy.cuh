#ifndef GPUDUMMY_H
#define GPUDUMMY_H

#define MTYPE int
#define PRINTLIMIT 7
#define INNER_REPEATS 1


#include "interface.h"
#include "running_stat.h"

// cuda side function
statistics gpudummy(unsigned int method, unsigned int repeats, double density);

RunningStat* boundingBox(size_t n, size_t nb, size_t rb, double density);

RunningStat* compressed(size_t n, size_t nb, size_t rb, double density);
RunningStat* compressed_tc(size_t n, size_t nb, size_t rb, double density);
RunningStat* lambda(size_t n, size_t nb, size_t rb, double density);

template<typename Lambda, typename Inverse>
RunningStat* performLoad(MTYPE *mat_h, MTYPE *mat1_d, MTYPE *mat2_d, size_t nb, size_t rb, size_t nx, size_t ny, dim3 block, dim3 grid,
                            Lambda map, Inverse inv);
                            
template<typename Lambda, typename Inverse>
RunningStat* performLoadCompressed(MTYPE *mat_h, MTYPE *mat1_d, MTYPE *mat2_d, size_t nb, size_t rb, size_t nx, size_t ny, dim3 block, dim3 grid,
                                Lambda map, Inverse inv);

template<typename Lambda, typename Inverse>
RunningStat* performLoadCompressed_tc(MTYPE *mat_h, MTYPE *mat1_d, MTYPE *mat2_d, size_t nb, size_t rb, size_t nx, size_t ny, dim3 block, dim3 grid,
                                Lambda map, Inverse inv);

template<typename Lambda, typename Inverse>
RunningStat* performLoadLambda(MTYPE *mat_h, MTYPE *mat1_d, MTYPE *mat2_d, size_t nb, size_t rb, size_t nx, size_t ny, dim3 block, dim3 grid,
                                Lambda map, Inverse inv);
                                    
#endif
