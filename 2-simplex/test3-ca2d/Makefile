DEBUG=DUMMY
BSIZE1D=256
BSIZE2D=32
PARAMS=-O3 -D${DEBUG} -DBSIZE1D=${BSIZE1D} -DBSIZE2D=${BSIZE2D} -arch sm_35 -rdc=true -lcudadevrt --std=c++11 --expt-extended-lambda
SOURCE=src/main.cu
BINARY=test3-ca2d
all:
	nvcc ${PARAMS} ${SOURCE} -o bin/${BINARY}

exact:
	nvcc ${PARAMS} ${SOURCE} -o bin/${BINARY}

fast:
	nvcc ${PARAMS} -prec-sqrt=false ${SOURCE} -o bin/fast-${BINARY}

table:
	gcc gentable.c -lm -o gentable

ptx:
	nvcc ${PARAMS} -ptx bin/ptx${BINARY}
