DEBUG=DUMMY
BSIZE1D=2
BSIZE2D=4
HADO_FACTOR=1
ARCH=sm_75
PARAMS=-O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -DHADO_FACTOR=${HADO_FACTOR} -arch ${ARCH} --std=c++11 --expt-extended-lambda -default-stream per-thread
#PARAMS=-O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -DHADO_FACTOR=${HADO_FACTOR} -arch ${ARCH} --std=c++11 --expt-extended-lambda 
SOURCE=src/main.cu
BINARY=test1-map
all:
	nvcc ${PARAMS} ${SOURCE} -o bin/${BINARY}

fast:
	nvcc ${PARAMS} -prec-sqrt=false ${SOURCE} -o bin/fast-${BINARY}

ptx:
	nvcc ${PARAMS} -ptx bin/ptx${BINARY}
