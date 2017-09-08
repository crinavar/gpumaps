#include <cstdio>
#include <cstdlib>
#include <fstream>
#include <iostream>
#include <sstream>
#include <set>
#include <math.h>
using namespace std;

int main(int argc, char **argv){

    ifstream fr1, fr2;
    if(argc == 3){
        fr1.open(argv[1]);
        fr2.open(argv[2]);
        if( !fr1.is_open() || !fr2.is_open() ){
            fprintf(stderr, "error: one of the files could not be opened\n");
            exit(1);
        }
    }
    else{
        fprintf(stderr, "error: must give valid and existing files\n");
        exit(1);
    }

    string line1, line2;
    int i=0, a, b, c, d;
    while(getline(fr1, line1) && getline(fr2, line2)){
        //printf("%s\n", line.c_str());
        std::stringstream s1(line1), s2(line2);
        s1 >> a;
        s2 >> b;
        
        if( a != b ){
			unsigned int icoord = (unsigned int)(sqrt(0.25 + 2.0*(double)i) - 0.5);
			unsigned int jcoord = i - icoord*(icoord+1)/2;
			fprintf(stderr, "error: %i != %i at line %i! coordinate (%i, %i)\n", a, b, i, icoord, jcoord);
			//exit(1);	    
		}
		i++;
	}
    printf("all ok!\n"); fflush(stdout);
    fr1.close();
    fr2.close();
    return 1;    
}
