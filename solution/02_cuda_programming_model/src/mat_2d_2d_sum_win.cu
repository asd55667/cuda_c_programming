#include<cuda_runtime.h>
#include<stdio.h>
#include<time.h>
#include<Windows.h>
#include<stdint.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		printf("Error: %s, %d\n", __FILE__, __LINE__);\
		printf("Code: %d, reason: %s\n", error, cudaGetErrorString(error));\
		exit(0);\
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

void mat_sum(float *a, float *b, float *c, const int x, const int y)
{
	float *aa = a;
	float *bb = b, *cc = c;
	for (int i=0; i< y; i++)
	{
		for(int j=0; j<x; j++)
			cc[j] = aa[j] + bb[j];
		aa += x;
		bb += x;
		cc += x;
	}
}

__global__ void mat_sum_g(float *a, float *b, float *c, int x, int y)
{
	unsigned int i = blockDim.x * blockIdx.x + threadIdx.x;
	unsigned int j = blockDim.y * blockIdx.y + threadIdx.y;
	unsigned int idx = i + j * x;
	
	if (i < x && j < y)
		c[idx] = a[idx] + b[idx];
}


void init_data(float *inp, int n)
{
	time_t t;
	srand((unsigned) time(&t));
	for(int i=0; i<n; i++)
		inp[i] = (float)(rand() & 0xFF) / 10.f;
}


double cpu_sec()
{
	timeval tp;
	gettimeofday(&tp, NULL);\
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void check_mat(float *c, float *g, int n)
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
	cudaDeviceProp devp;
	CHECK(cudaGetDeviceProperties(&devp, dev));
	printf("Device %d: %s\n", dev, devp.name);
	CHECK(cudaSetDevice(dev));

	int nx = 1 << 14, ny = 1 << 14;

	int n = nx * ny, nbytes = n * sizeof(float);
	printf("Mat Size: x %d, y %d\n", nx, ny);

	float *ha, *hb, *cpu, *gpu;
	ha = (float *)malloc(nbytes);
	hb = (float *)malloc(nbytes);
	cpu = (float *)malloc(nbytes);
	gpu = (float *)malloc(nbytes);

	init_data(ha, n);
	init_data(hb, n);

	memset(cpu, 0, nbytes);
	memset(gpu, 0, nbytes);

	float *da, *db, *dc;
	cudaMalloc((float **)&da, nbytes);
	cudaMalloc((float **)&db, nbytes);
	cudaMalloc((float **)&dc, nbytes);

	cudaMemcpy(da, ha, nbytes, cudaMemcpyHostToDevice);
	cudaMemcpy(db, hb, nbytes, cudaMemcpyHostToDevice);

	double start, duration;
	start = cpu_sec();
	mat_sum(ha, hb, cpu, nx, ny);
	duration = cpu_sec() - start;
	printf("Mat sum cpu time cost %f ms\n", duration*1000);


	dim3 block(32,32);
	dim3 grid((nx+31)/32, (ny+31)/32);

	start = cpu_sec();
	mat_sum_g<<<grid, block>>>(da, db, dc,nx, ny);
	cudaDeviceSynchronize();
	duration = cpu_sec() - start;
	printf("Mat sum GPU<<<(%d,%d), (%d,%d)>>> time cost %f ms\n",
			grid.x, grid.y, block.x, block.y, duration*1000);

	cudaMemcpy(gpu, dc, nbytes, cudaMemcpyDeviceToHost);
	check_mat(cpu, gpu, n);

	cudaFree(da);
	cudaFree(db);
	cudaFree(dc);

	free(ha);
	free(hb);
	free(cpu);
	free(gpu);

	cudaDeviceReset();

	int c = getchar();
	return 0;
}
	
