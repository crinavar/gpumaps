DEBUG=DUMMY
BSIZE1D=1024
BSIZE2D=32
HADO_FACTOR=1
ARCH=sm_86
ifeq (${DP}, NO)
PARAMS=-O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -DHADO_FACTOR=${HADO_FACTOR} -arch ${ARCH} --std=c++11 --expt-extended-lambda -default-stream per-thread
else
PARAMS=-O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -DHADO_FACTOR=${HADO_FACTOR} -arch ${ARCH} --std=c++11 --expt-extended-lambda -default-stream per-thread -rdc=true -lcudadevrt -DDP
endif#PARAMS=-O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -DHADO_FACTOR=${HADO_FACTOR} -arch ${ARCH} --std=c++11 --expt-extended-lambda 
SOURCE=src/main.cu
BINARY=test4-ca2d
all:
	nvcc ${PARAMS} ${SOURCE} -o bin/${BINARY}

fast:
	nvcc ${PARAMS} -prec-sqrt=false ${SOURCE} -o bin/fast-${BINARY}

ptx:
	nvcc ${PARAMS} -ptx bin/ptx${BINARY}
