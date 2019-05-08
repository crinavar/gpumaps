# Makefile,
# Note: to disable the compiler variables DEBUG or VERIFY, set them to =DUMMY. To enable, set them as their name
DEBUG=DUMMY
VERIFY=DUMMY
BSIZE3D=8
COMPILER_DEFINES=-D${DEBUG} -D${VERIFY} -DBSIZE3D=${BSIZE3D}
NVCCFLAGS=${COMPILER_DEFINES} -arch sm_35 --std=c++11 --expt-extended-lambda 
NVCCFLAGS2=-O0 -Xcicc -O0 -Xptxas -O3 -arch sm_35 --std=c++11 --expt-extended-lambda --ptxas-options=-v
PRECFLAGS=-prec-sqrt=false
BIN=bin/test0-map
SRC=src
OBJ=obj
CFLAGS=${COMPILER_DEFINES} --std=c++11 

all: gpudummy.o main.o
	nvcc ${NVCCFLAGS} ${OBJ}/*.o -o ${BIN}
	rm ${OBJ}/main.o ${OBJ}/gpudummy.o

gpudummy.o:
	nvcc -c ${NVCCFLAGS} ${SRC}/gpudummy.cu -o ${OBJ}/gpudummy.o
main.o:
	g++ ${CFLAGS} -c ${SRC}/main.cpp -lm -o ${OBJ}/main.o

ptx:
	nvcc -c -ptx ${NVCCFLAGS} ${SRC}/gpudummy.cu 
	g++ ${CFLAGS} -c ${SRC}/main.cpp -lm
	nvcc -ptx ${NVCCFLAGS} *.o -o ${BIN}
	rm main.o gpudummy.o

clean:
	rm ${BIN}
