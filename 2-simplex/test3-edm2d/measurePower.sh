make BSIZE1D=1024 BSIZE2D=32 MEASURE_POWER=MEASURE_POWER ARCH=sm_80
./bin/test3-edm2d 0 99328 500 1
make BSIZE1D=256 BSIZE2D=16 MEASURE_POWER=MEASURE_POWER ARCH=sm_80 LAMBDAFP=LAMBDA_FP64
./bin/test3-edm2d 0 99328 500 2
./bin/test3-edm2d 0 99328 500 3
./bin/test3-edm2d 0 99328 500 4
make BSIZE1D=1024 BSIZE2D=32 MEASURE_POWER=MEASURE_POWER ARCH=sm_80 DP=si
./bin/test3-edm2d 0 99328 500 5