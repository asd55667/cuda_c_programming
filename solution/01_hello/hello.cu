#include <cstdio>
#include <cuda.h>
#include <cuda_runtime.h>

void hello_cpu()
{
	printf("hello world from CPU\n\n");
}

__global__ void hello_gpu()
{
	printf("Hello world from GPU\n");
}


__global__ void hello_gpu_idx()
{
    if (threadIdx.x == 5)
        printf("\nHellow world from GPU %d\n",threadIdx.x);
}

int main()
{
	hello_cpu();
	hello_gpu<<<1, 10>>>();
    hello_gpu_idx<<<1,10>>>();

	//cudaDeviceSynchronize();
    cudaDeviceReset();
}

