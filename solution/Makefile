#src = ./src
cus = $(wildcard *.cu)

CU_APPS = $(basename $(cus))
C_APPS = 


all: ${CU_APPS} ${C_APPS}

%: %.cu
	nvcc -arch=sm_35 -o $@ $<

%: %.c
	gcc -std=c99 -o $@ $<

clean:
	rm -rf ${CU_APPS} ${C_APPS}
