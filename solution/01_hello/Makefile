#!/bin/bash

CU_APPS = hello
C_APPS = 
all: $(CU_APPS) $(C_APPS)

%: %.cu
	nvcc -arch=sm_35 -o $@ $<

%: %.c
	gcc -02 -std=c99 -o $@ $<

#%.x : %.cu
#    $(CC) $(CFLAGS) $< -o $@

clean:
	rm -f $(CU_APPS) $(C_APPS)
