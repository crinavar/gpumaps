ARCH=sm_30
NVCCFLAGS=-O0 -arch ${ARCH} --std=c++11 --expt-extended-lambda
CFLAGS=--std=c++11
all: gpudummy.o main.o
	nvcc ${NVCCFLAGS} *.o -o prog
	rm main.o gpudummy.o

gpudummy.o:
	nvcc -c ${NVCCFLAGS} gpudummy.cu

main.o:
	g++ ${CFLAGS} -c main.cpp -lm

clean:
	rm prog *.o
