#include "../common/common_win.h"
#include<cuda_runtime.h>

__global__ void test1(float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a,b;
    a = b = 0.0f;
    if (i % 2 == 0)
        a = 100.0f;
    else
        b = 200.0f;
    c[i] = a + b;
}

__global__ void test2(float *c)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    float a, b;
    a = b = 0.0f;
    if((i/warpSize)/2==0)
        a = 100.0f;
    else
        b = 200.0f;
    c[i] = a+b;
}

__global__ void test3(float *c)
{
	int i = blockIdx.x *blockDim.x + threadIdx.x;
	float a ,b;
	a = b = 0.0f;
	bool cond = (i % 2 == 0);
	if (cond)
		a = 100.0f;
	else
		b =200.0f;
	c[i] =a+b;
}

__global__ void test4(float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float a, b;
	a = b= 0.0f;

	int j = i >> 5;

	if (j & 0x01 == 0)
		a = 100.0f;
	else
		b = 200.0f;
	c[i] = a+b;
}

__global__ void warm_up(float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	float a, b;
	a = b = 0.0f;

	if ((i/warpSize) %2 ==0)
		a = 100.0f;
	else
		b= 200.0f;
	c[i] = a+b;
}

int main()
{
	int dev = 0;
	cudaDeviceProp devp;
	CHECK(cudaGetDeviceProperties(&devp, dev));
	printf("Device %d: %s\n", dev, devp.name);
	
	int size= 1<<14;
	int x = 512;
	printf("Data size %d\n", size);

	dim3 block(x, 1);
	dim3 grid((size + block.x-1)/block.x,1);
	
	float *dc;
	size_t n = size * sizeof(float);
	CHECK(cudaMalloc((float **)&dc, n));

	size_t start, duration;
	CHECK(cudaDeviceSynchronize());
	start = seconds();
	warm_up<<<grid, block>>>(dc);
	CHECK(cudaDeviceSynchronize());
	duration = seconds() - start;
	printf("warmup <<<%4d, %4d>>> time cost %f ms\n", grid.x,block.x, duration*1000);
	CHECK(cudaGetLastError());
	

	start = seconds();
	test1<<<grid, block>>>(dc);
	CHECK(cudaDeviceSynchronize());
	duration = seconds() - start;
	printf("test1 <<<%4d, %4d>>> time cost %f ms\n", grid.x, block.x, duration*1000);
	CHECK(cudaGetLastError());



	start = seconds();
	test2<<<grid, block>>>(dc);
	CHECK(cudaDeviceSynchronize());
	duration = seconds() - start;
	printf("test2 <<<%4d, %4d>>> time cost %f ms\n", grid.x, block.x, duration*1000);
	CHECK(cudaGetLastError());



	start = seconds();
	test3<<<grid, block>>>(dc);
	CHECK(cudaDeviceSynchronize());
	duration = seconds() - start;
	printf("test3 <<<%4d, %4d>>> time cost %f ms\n", grid.x, block.x, duration*1000);
	CHECK(cudaGetLastError());



	start = seconds();
	test4<<<grid, block>>>(dc);
	CHECK(cudaDeviceSynchronize());
	duration = seconds() - start;
	printf("test4 <<<%4d, %4d>>> time cost %f ms\n", grid.x, block.x, duration*1000);
	CHECK(cudaGetLastError());


	CHECK(cudaFree(dc));
	CHECK(cudaDeviceReset());
	int c = getchar();
	return EXIT_SUCCESS;
}






