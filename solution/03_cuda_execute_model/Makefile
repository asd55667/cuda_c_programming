#src = ./src
cus = $(wildcard *.cu)

CU_APPS = $(basename $(cus))
C_APPS = 


all: ${C_APPS} ${CU_APPS}

%: %.cu
	nvcc -arch=sm_35 -o $@ $< -lcudadevrt --relocatable-device-code true

%: %.c
	gcc -std=c99 -o $@ $<

clean:
	rm -rf ${CU_APPS} ${C_APPS}
