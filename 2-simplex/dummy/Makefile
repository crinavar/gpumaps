PARAMS=-arch sm_60
all:
	nvcc ${PARAMS} tdbb.cu -o extd

exact:
	nvcc ${PARAMS} tdbb.cu -o extd

fast:
	nvcc ${PARAMS} -prec-sqrt=false tdbb.cu -o fastrtd

table:
	gcc gentable.c -lm -o gentable

ptx:
	nvcc ${PARAMS} -ptx tdbb.cu
