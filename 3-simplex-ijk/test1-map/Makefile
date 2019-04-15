# Makefile,
# Note: to disable the compiler variables DEBUG or VERIFY, set them to =DUMMY. To enable, set them as their name
DEBUG=DUMMY
VERIFY=DUMMY
BSIZE3D=8
COMPILER_DEFINES=-D${DEBUG} -D${VERIFY} -DBSIZE3D=${BSIZE3D}
NVCCFLAGS=${COMPILER_DEFINES} -arch sm_35 --std=c++11 --expt-extended-lambda 
NVCCFLAGS2=-O0 -Xcicc -O0 -Xptxas -O3 -arch sm_35 --std=c++11 --expt-extended-lambda --ptxas-options=-v
PRECFLAGS=-prec-sqrt=false
BIN=test0-map
CFLAGS=${COMPILER_DEFINES} --std=c++11 

all: gpudummy.o main.o
	nvcc ${NVCCFLAGS} *.o -o ${BIN}
	rm main.o gpudummy.o

gpudummy.o:
	nvcc -c ${NVCCFLAGS} gpudummy.cu 
main.o:
	g++ ${CFLAGS} -c main.cpp -lm

non-opt:
	nvcc -c ${NVCCFLAGS2} gpudummy.cu 
	g++ ${CFLAGS} -c main.cpp -lm
	nvcc ${NVCCFLAGS2} *.o -o prog
	rm main.o gpudummy.o

non-opt-ptx:
	nvcc -c -ptx ${NVCCFLAGS2} gpudummy.cu 
	g++ ${CFLAGS} -c main.cpp -lm
	nvcc -ptx ${NVCCFLAGS2} *.o -o prog
	rm main.o gpudummy.o

ptx:
	nvcc -c -ptx ${NVCCFLAGS} gpudummy.cu 
	g++ ${CFLAGS} -c main.cpp -lm
	nvcc -ptx ${NVCCFLAGS} *.o -o prog
	rm main.o gpudummy.o

clean:
	rm prog *.o
