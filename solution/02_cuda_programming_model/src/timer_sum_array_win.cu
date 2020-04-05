#include <cuda_runtime.h>
#include <stdio.h>
#include <Windows.h>
#include <time.h>
#include <stdint.h> 

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		printf("Error: %s:%d, ", __FILE__, __LINE__);\
		printf("code: %d, reason: %s\n", error, cudaGetErrorString(error));\
		exit(1);\
	}\
}\


int gettimeofday(struct timeval * tp, struct timezone *tzp)
{
	static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

	SYSTEMTIME system_time;
	FILETIME file_time;
	uint64_t time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}

__global__ void sum_array_gpu(float *a, float *b, float *c)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	printf("%d\t%d\t%d\n", blockDim.x, blockIdx.x, threadIdx.x);
	c[i] = a[i] + b[i];
}

void sum_array_cpu(float *a, float *b, float *c, int n)
{
	for (int i = 0; i < n; i++)
	{
		c[i] = a[i] + b[i];
	}
}

void initData(float *data,int n)
{
	time_t t;
	srand((unsigned) time(&t));

	for (int i = 0; i < n; i++)
		data[i] = (float)(rand() & 0xFF) / 10.0f;
}

double cpu_sec()
{
	timeval tp;
	gettimeofday(&tp, NULL);\
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void check_sum(float *c, float *g, int n)
{
	double epsilon = 1.0E-8;
	int match = 1;
	for (int i = 0; i < n; i++)
	{
		if (abs(c[i] - g[i]) > epsilon)
		{
			match = 0;
			printf("Don't match!\n");
			printf("host %5.2f device %5.2f at current %d\n", c[i], g[i], i);
			break;
		}
	}
	if (match)
		printf("Array match\n\n");
	return;
}

int main()
{
	int dev = 0;
	cudaDeviceProp deviceProp;
	CHECK(cudaGetDeviceProperties(&deviceProp, dev));
	printf("Using Device %d: %s\n", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));

	int nelem = 1 << 24;
	printf("Vector size %d\n", nelem);

	size_t nbytes = nelem * sizeof(float);

	float *h_a, *h_b, *cpuref, *gpuref;
	h_a = (float *)malloc(nbytes);
	h_b = (float *)malloc(nbytes);
	cpuref = (float *)malloc(nbytes);
	gpuref = (float *)malloc(nbytes);

	double istart, ielaps;
	
	initData(h_a, nelem);
	initData(h_b, nelem);


	memset(cpuref, 0, nbytes);
	memset(gpuref, 0, nbytes);

	istart = cpu_sec();
	sum_array_cpu(h_a, h_b, cpuref, nelem);
	ielaps = cpu_sec() - istart;
	printf("sum cpu time cost %f ms\n", ielaps*1000);

	float *da, *db, *dc;
	cudaMalloc((float**)&da, nbytes);
	cudaMalloc((float**)&db, nbytes);
	cudaMalloc((float**)&dc, nbytes);

	cudaMemcpy(da, h_a, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(db, h_b, nbytes, cudaMemcpyHostToDevice);

	int len = 1024;
	dim3 block(len);
	dim3 grid((nelem+block.x-1)/block.x);

	istart = cpu_sec();
	sum_array_gpu<<<grid, block>>>(da,db,dc);
	cudaDeviceSynchronize();
	ielaps = cpu_sec() - istart;
	printf("sum gpu <<<%d,%d>>> time cost %f ms\n", grid.x, block.x, ielaps*1000);

	cudaMemcpy(gpuref, dc, nbytes, cudaMemcpyDeviceToHost);

	check_sum(cpuref, gpuref, nelem);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	free(h_a);
	free(h_b);
	free(cpuref);
	free(gpuref);

	int c = getchar();
	return 0;			
}
