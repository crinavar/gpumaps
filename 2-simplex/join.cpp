#include <cstdio>
#include <cstdlib>
#include <fstream>
//#include <string>
#include <sstream>
using namespace std;

int main(int argc, char **argv){

    ifstream bbfr, rsqrtfr, rectanglefr, recfr, avrilfr;
    ofstream fw;
    if(argc == 2){
        string folder(argv[1]);
        string wpath = folder + "results/full_results.dat";
        string rpath1 = folder + "bb_results.dat";
        string rpath2 = folder + "rsqrt_results.dat";
        string rpath3 = folder + "rectangle_results.dat";
        string rpath4 = folder + "recursive_results.dat";
        string rpath5 = folder + "avril_results.dat";
        fw.open(wpath.c_str());
        bbfr.open(rpath1.c_str());
        rsqrtfr.open(rpath2.c_str());
        rectanglefr.open(rpath3.c_str());
        recfr.open(rpath4.c_str());
        avrilfr.open(rpath5.c_str());
        if( !fw.is_open() || !bbfr.is_open() || !rsqrtfr.is_open() || !rectanglefr.is_open() || !recfr.is_open() || !avrilfr.is_open() ){
            fprintf(stderr, "error: one of the files could not be opened\n");
            printf("wpath = '%s'\nrpath1 = '%s'\nrpath2 = '%s'\nrpath3 = '%s'\nrpath4 = '%s'\n", wpath.c_str(), rpath1.c_str(), rpath2.c_str(), rpath3.c_str(), rpath4.c_str());
            exit(1);
        }
    }
    else{
        fprintf(stderr, "error: must give the folder containing the four files\n");
        exit(1);
    }

    string line1, line2, line3, line4, line5, N, time, error;
    fw << "#N	bb	rsqrt	rectangle	recursive	avril\n"<<  std::endl;
    while(getline(bbfr, line1) && getline(rsqrtfr, line2) && getline(rectanglefr, line3) && getline(recfr, line4) && getline(avrilfr, line5)){
       //printf("%s\n", line.c_str());
       std::stringstream s1(line1), s2(line2), s3(line3), s4(line4), s5(line5);
       s1 >> N >> time >> error;
       //printf("N=%s    time=%s    error=%s\n", N.c_str(), time.c_str(), error.c_str());
       fw << N << " " << time << " ";

       s2 >> N >> time >> error;
       fw << time << " ";

       s3 >> N >> time >> error;
       fw << time << " ";

       s4 >> N >> time >> error;
       fw << time << " ";

       s5 >> N >> time >> error;
       fw << time << " ";
       fw << endl;
       
    }

    bbfr.close();
    rsqrtfr.close();
    rectanglefr.close();
    recfr.close();
    avrilfr.close();
    fw.close();

    return 1;    
}
