cd test1-map
make clean
make "ARCH=-arch=sm_75" MEASURE_POWER=MEASURE_POWER
./bin/prog 0 10 1 0
./bin/prog 0 10 1 1
make clean
make "ARCH=-arch=sm_75" DP=YES MEASURE_POWER=MEASURE_POWER
./bin/prog 0 10 1 2
make clean
make "ARCH=-arch=sm_70" MEASURE_POWER=MEASURE_POWER
./bin/prog 1 10 1 0
./bin/prog 1 10 1 1
make clean
make "ARCH=-arch=sm_70" DP=YES MEASURE_POWER=MEASURE_POWER
./bin/prog 1 10 1 2
cd ..
cd test2-accumulate
make clean
make "ARCH=-arch=sm_75" MEASURE_POWER=MEASURE_POWER
./bin/prog 0 10 1 0
./bin/prog 0 10 1 1
make clean
make "ARCH=-arch=sm_75" DP=YES MEASURE_POWER=MEASURE_POWER
./bin/prog 0 10 1 2
make clean
make "ARCH=-arch=sm_70" MEASURE_POWER=MEASURE_POWER
./bin/prog 1 10 1 0
./bin/prog 1 10 1 1
make clean
make "ARCH=-arch=sm_70" DP=YES MEASURE_POWER=MEASURE_POWER
./bin/prog 1 10 1 2
cd ..
cd test3-ca2d
make clean
make "ARCH=-arch=sm_75" MEASURE_POWER=MEASURE_POWER
./bin/prog 0 10 1 0.5 1 0
./bin/prog 0 10 1 0.5 1 1
make clean
make "ARCH=-arch=sm_75" DP=YES MEASURE_POWER=MEASURE_POWER
./bin/prog 0 10 1 0.5 1 2
make clean
make "ARCH=-arch=sm_70" MEASURE_POWER=MEASURE_POWER
./bin/prog 1 10 1 0.5 1 0
./bin/prog 1 10 1 0.5 1 1
make clean
make "ARCH=-arch=sm_70" DP=YES MEASURE_POWER=MEASURE_POWER
./bin/prog 1 10 1 0.5 1 2
cd ..
