#include "../common/common_win.h"
#include <cuda_runtime.h>
#include <stdio.h>


__global__ void nested(const int size, int depth)
{
	int tidx = threadIdx.x;
	printf("Recursion: %d, from thread %d block %d\n", depth, tidx, blockIdx.x);

	if (size == 1)
		return ;

	int n = size >> 1;
	if (tidx == 0 && n > 0)
	{
		nested<<<1, n>>>(n, ++depth);
		printf("---------> depth: %d\n", depth);
	}
}

int main(int argc, char **argv)
{
	int size = 8;
	int nblock = 8;
	int ngrid = 1;

	if (argc > 1)
	{
		ngrid = atoi(argv[1]);
		size = ngrid * nblock;
	}

	dim3 block(nblock,1);
	dim3 grid((size + block.x-1)/block.x,1);
	printf("%s Cfg: <<<%d, %d>>>\n", argv[0], grid.x, block.x);
	nested<<<grid, block>>>(block.x, 0);
	
	CHECK(cudaGetLastError());
	CHECK(cudaDeviceReset());
	return 0;
}
