DEBUG=DUMMY
BSIZE1D=256
BSIZE2D=32
HADO_FACTOR=1
ARCH=sm_86
DP=NO
LAMBDAFP=LAMBDA_FP32
MEASURE_POWER := NO
ifeq (${DP}, NO)
PARAMS=-D${LAMBDAFP} -D${MEASURE_POWER} -O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -DHADO_FACTOR=${HADO_FACTOR} -arch ${ARCH} --std=c++11 --expt-extended-lambda -default-stream per-thread -lnvidia-ml
else
PARAMS=-D${LAMBDAFP} -D${MEASURE_POWER} -O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -DHADO_FACTOR=${HADO_FACTOR} -arch ${ARCH} --std=c++11 --expt-extended-lambda -default-stream per-thread -rdc=true -lcudadevrt -DDP  -lnvidia-ml
endif
#PARAMS=-O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -DHADO_FACTOR=${HADO_FACTOR} -arch ${ARCH} --std=c++11 --expt-extended-lambda 
SOURCE=src/Rapl.cpp src/nvmlPower.cu src/main.cu 
BINARY=test3-edm2d
all:
	nvcc ${PARAMS} ${SOURCE} -o bin/${BINARY}

fast:
	nvcc ${PARAMS} -prec-sqrt=false ${SOURCE} -o bin/fast-${BINARY}

ptx:
	nvcc ${PARAMS} -ptx bin/ptx${BINARY}
