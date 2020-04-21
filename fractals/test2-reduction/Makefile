DEBUG=DUMMY
BPOWER=1
BSIZE1D=2
BSIZE2D=1# (1 << BPOWER)**2
RLEVEL=14
ARCH=sm_75
DEFINES= -D${DEBUG} -DBPOWER=${BPOWER} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -DRLEVEL=${RLEVEL}
PARAMS=-O3 ${DEFINES} -arch ${ARCH} --std=c++11 --expt-extended-lambda -default-stream per-thread
SOURCE=src/gpudummy.cu src/main.cu
BINARY=prog
all:
	nvcc ${PARAMS} ${SOURCE} -o bin/${BINARY}

fast:
	nvcc ${PARAMS} -prec-sqrt=false ${SOURCE} -o bin/fast-${BINARY}

ptx:
	nvcc ${PARAMS} -ptx bin/ptx${BINARY}

clean:
	rm bin/prog *.o
