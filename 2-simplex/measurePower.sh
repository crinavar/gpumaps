cd test1-map
make clean
make "ARCH=-arch=sm_80" MEASURE_POWER=MEASURE_POWER BSIZE3DX=8 BSIZE3DY=8 BSIZE3DZ=8
./bin/prog 0 11 1 0
./bin/prog 0 11 1 1
make clean
make "ARCH=-arch=sm_80" DP=YES MEASURE_POWER=MEASURE_POWER BSIZE3DX=8 BSIZE3DY=8 BSIZE3DZ=8
./bin/prog 0 11 1 2
./bin/prog 0 11 1 3
./bin/prog 0 11 1 4
cd ..
cd test2-accumulate
make clean
make "ARCH=-arch=sm_80" MEASURE_POWER=MEASURE_POWER BSIZE3DX=8 BSIZE3DY=8 BSIZE3DZ=8
./bin/prog 0 11 1 0
./bin/prog 0 11 1 1
make clean
make "ARCH=-arch=sm_80" DP=YES MEASURE_POWER=MEASURE_POWER BSIZE3DX=8 BSIZE3DY=8 BSIZE3DZ=8
./bin/prog 0 11 1 2
./bin/prog 0 11 1 3
./bin/prog 0 11 1 4
cd ..
cd test3-ca2d
make clean
make "ARCH=-arch=sm_80" MEASURE_POWER=MEASURE_POWER BSIZE3DX=8 BSIZE3DY=8 BSIZE3DZ=8
./bin/prog 0 10 1 0.5 1 0
./bin/prog 0 10 1 0.5 1 1
make clean
make "ARCH=-arch=sm_80" DP=YES MEASURE_POWER=MEASURE_POWER BSIZE3DX=8 BSIZE3DY=8 BSIZE3DZ=8
./bin/prog 0 10 1 0.5 1 2
./bin/prog 0 10 1 0.5 1 3
./bin/prog 0 10 1 0.5 1 4
cd ..
