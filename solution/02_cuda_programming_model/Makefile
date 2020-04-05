cus = $(wildcard *.cu)

CU_APPS = $(basename $(cus))
C_APPS = 


all: ${CU_APPS} ${C_APPS}

%: %.cu
	nvcc -O2 -arch=sm_35 -o $@ $<

%: %.c
	gcc -O2 -std=c99 -o $@ $<

clean:
	rm -rf ${CU_APPS} ${C_APPS}
