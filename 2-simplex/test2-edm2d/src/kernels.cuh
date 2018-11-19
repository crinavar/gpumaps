//////////////////////////////////////////////////////////////////////////////////
//  gpumaps                                                                     //
//  A GPU benchmark of mapping functions                                        //
//                                                                              //
//////////////////////////////////////////////////////////////////////////////////
//                                                                              //
//  Copyright © 2015 Cristobal A. Navarro, Wei Huang.                           //
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
    DTYPE a = data[p.x];
    DTYPE b = data[p.y];
    mat[p.y*n + p.x] = sqrtf( (a.x-b.x)*(a.x-b.x) + (a.y-b.y)*(a.y-b.y) );
    //printf("%f\n", mat[p.y*n + p.x]);
}

// metodo kernel test
template<typename Lambda>
__global__ void kernel_test(const unsigned int n, const int a, const unsigned int msize, DTYPE *data, MTYPE* dmat, Lambda map, const unsigned int aux1, const unsigned int aux2, const unsigned int aux3){
    auto p = map(n, msize, aux1, aux2, aux3); 
    if(p.y >= p.x && p.y < n){
    //if(p.y < n){
        work(data, dmat, p, n);
    }
}
#endif
