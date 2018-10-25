#ifndef INTERFACE_H
#define INTERFACE_H

typedef struct{
    float mean1;
    float variance1;
    float stdev1;
    float sterr1;
    float mean2;
    float variance2;
    float stdev2;
    float sterr2;
} statistics;

// cuda side function
statistics gpudummy(unsigned long r, int BPOWER);

#endif
