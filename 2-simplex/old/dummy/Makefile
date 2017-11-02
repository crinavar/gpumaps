BSIZE2D=32
PARAMS=-DBSIZE2D=${BSIZE2D} -arch sm_60 -rdc=true -lcudadevrt --std=c++11 --expt-extended-lambda
BINARY=test1-write
all:
	nvcc ${PARAMS} tdbb.cu -o ${BINARY}

exact:
	nvcc ${PARAMS} tdbb.cu -o ${BINARY}

fast:
	nvcc ${PARAMS} -prec-sqrt=false tdbb.cu -o fast-${BINARY}

table:
	gcc gentable.c -lm -o gentable

ptx:
	nvcc ${PARAMS} -ptx tdb${BINARY}
