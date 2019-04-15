#ifndef DUMMY_CUH
#define DUMMY_CUH

#define BSIZE1D 256
#define BSIZE2DX 32
#define BSIZE2DY 32

#define CL 0.4807498567691361274405461035933f
#define CR 0.6933612743506347048433522747859f

__device__ float carmack_sqrtf(float nb) {
    float nb_half = nb * 0.5F;
    float y = nb;
    long i = * (long *) &y;
    //i = 0x5f3759df - (i >> 1);
    i = 0x5f375a86 - (i >> 1);
    y = * (float *) &i;
    //Repetitions increase accuracy(6)
    y = y * (1.5f - (nb_half * y * y));
    y = y * (1.5f - (nb_half * y * y));
    y = y * (1.5f - (nb_half * y * y));

    return nb * y;
}

__device__ inline float newton_sqrtf(const float number) {
    int i;
    float x,y;
    //const float f = 1.5F;
    x = number * 0.5f;
    i  = * ( int * ) &number;
    i  = 0x5f3759df - ( i >> 1 );
    y  = * ( float * ) &i;
    y  *= (1.5f -  x * y * y);
    y  *= (1.5f -  x * y * y); 
    y  *= (1.5f -  x * y * y); 
    return number * y;
}

__device__ inline float newton1_sqrtf(const float number){

    int i;
    float x,y;
    //const float f = 1.5F;
    x = number * 0.5f;
    i  = * ( int * ) &number;
    i  = 0x5f3759df - ( i >> 1 );
    y  = * ( float * ) &i;
    y *= (1.5f -  x * y * y);
    y *= number; //obteniendo resultado
    //arreglar
    if( (y+1.0f)*(y+1.0f) < number)
        return y;
    else
        return y-0.5f;
}

#endif
