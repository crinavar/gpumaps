all:
	nvcc -arch sm_30 tdbb.cu -o extd

exact:
	nvcc -arch sm_30 tdbb.cu -o extd

fast:
	nvcc -arch sm_30 -m64 -prec-sqrt=false tdbb.cu -o fastrtd

table:
	gcc gentable.c -lm -o gentable
