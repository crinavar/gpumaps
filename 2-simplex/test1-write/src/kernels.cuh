//////////////////////////////////////////////////////////////////////////////////
//  gpumaps                                                                     //
//  A GPU benchmark of mapping functions                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro.                                      //
//                                                                              //
//  This file is part of gpumaps.                                               //
//  gpumaps is free software: you can redistribute it and/or modify             //
//  it under the terms of the GNU General Public License as published by        //
//  the Free Software Foundation, either version 3 of the License, or           //
//  (at your option) any later version.                                         //
//                                                                              //
//  gpumaps is distributed in the hope that it will be useful,                  //
//  but WITHOUT ANY WARRANTY; without even the implied warranty of              //
//  MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the               //
//  GNU General Public License for more details.                                //
//                                                                              //
//  You should have received a copy of the GNU General Public License           //
//  along with gpumaps.  If not, see <http://www.gnu.org/licenses/>.            //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
#ifndef KERNELS_CUH
#define KERNELS_CUH

#define OFFSET -0.4999f
//#define OFFSET 0.5f
__device__ void work(DTYPE *data, MTYPE *mat, uint2 p, int n){
    // (1) constant write
    unsigned long i = (unsigned long)p.y*n + (unsigned long)p.x;
    mat[i] += 1;

    // or (2) recursion level write
    //const int b = (int)log2f(blockIdx.y+1) + 1;
    //mat[p.y*n + p.x] = b;
}
// metodo kernel test
template<typename Lambda>
__global__ void kernel_test(const unsigned int n, const unsigned int msize, DTYPE *data, MTYPE* dmat, Lambda map, const unsigned int aux1, const unsigned int aux2, const unsigned int aux3){
    auto p = map(n, msize, aux1, aux2, aux3); 
    /*
    if(p.y > n){
        printf("[ALERT] p(%i, %i) > %i \n", p.x, p.y, n);
    }
    */
    if(p.y >= p.x && p.y < n){
    //if(p.y < n){
        work(data, dmat, p, n);
    }
    //else if(p.y < n/2){
    //    printf("\n[OUT] block (x,y)=(%i, %i) -> (%i, %i)\n", blockIdx.x, blockIdx.y, p.x, p.y);
    //}
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
#endif
