DEBUG=DUMMY
BSIZE1D=4
BSIZE2D=2
HADO_FACTOR=1
ARCH=sm_86
DP=NO
MEASURE_POWER := NO
# SUBBLOCK_SIZE only 16 or 8
SUBBLOCK_SIZE=16
SUBBLOCK_EXP=4
ifeq (${DP}, NO)
PARAMS=-D${MEASURE_POWER} -O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -DSUBBLOCK_SIZE=${SUBBLOCK_SIZE} -DSUBBLOCK_EXP=${SUBBLOCK_EXP} -DHADO_FACTOR=${HADO_FACTOR} -arch ${ARCH} --std=c++11 --expt-extended-lambda -default-stream per-thread
else
PARAMS=-D${MEASURE_POWER} -O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -DSUBBLOCK_SIZE=${SUBBLOCK_SIZE} -DSUBBLOCK_EXP=${SUBBLOCK_EXP} -DHADO_FACTOR=${HADO_FACTOR} -arch ${ARCH} --std=c++11 --expt-extended-lambda -default-stream per-thread -rdc=true -lcudadevrt -DDP
endif

#PARAMS=-O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -DHADO_FACTOR=${HADO_FACTOR} -arch ${ARCH} --std=c++11 --expt-extended-lambda 
SOURCE=src/Rapl.cpp src/nvmlPower.cu src/main.cu 
BINARY=test2-accum
all:
	nvcc ${PARAMS} ${SOURCE} -o bin/${BINARY}

fast:
	nvcc ${PARAMS} -prec-sqrt=false ${SOURCE} -o bin/fast-${BINARY}

ptx:
	nvcc ${PARAMS} -ptx bin/ptx${BINARY}
