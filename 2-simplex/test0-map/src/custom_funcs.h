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
//
//
//
//
// custom functions. 
// all have the 'cf_' prefix.
//

#define DTYPE float
#define MTYPE char

// integer log2
int cf_log2i(int val){
    int copy = val;
    int r = 0;
    while (copy >>= 1) 
        ++r;
    return r;
}

void print_array(DTYPE *a, const int n){
	for(int i=0; i<n; i++)
		printf("a[%d] = %f\n", i, a[i]);
}

void print_matrix(MTYPE *mat, const int n, const char *msg){
    printf("[%s]:\n", msg);
	for(int i=0; i<n; i++){
	    for(int j=0; j<n; j++){
            printf("%i ", mat[i*n + j]);
        }
        printf("\n");
    }
}
