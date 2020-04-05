#include <cuda_runtime.h>
#include <stdio.h>
#include <time.h>

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
	


void sumArrayOnHost(float *A, float *B,float *C, const int N)
{
	for (int i = 0; i < N; i++)
		C[i] = A[i] + B[i];
}
// CHECK(cudaMemcpy(d_C, gpuRes, bBytes, cudaMemcpyHostToDevice));

__global__ void sumArrayOnDevice(float *A, float *B, float *C)
{
	int i = threadIdx.x;
	C[i] = A[i] + B[i];
}


void checkResult(float *host, float *device, const int N)
{
	double epsilon = 1.0E-8;
	int match = 1;
	for (int i = 0; i < N; i++)
	{
		if (abs(host[i] - device[i]) > epsilon)
		{
			match = 0;
			printf("Don't match!\n");
			printf("host %5.2f device %5.2f at current %d\n", host[i], device[i], i);
			break;
		}
	}
	if (match)
		printf("Array match\n\n");
	return;
}
	

void initData(float *inp, int size)
{
	time_t t;
	srand((unsigned) time(&t));

	for (int i = 0; i < size; i++)
		inp[i] = (float)(rand() & 0xFF) / 10.0f;
}

int main(int argc, char **argv)
{
	int dev = 0;
	cudaSetDevice(dev);
	
	int nElem = 32;
	printf("Inp Size %d\n", nElem);

	size_t nBytes = nElem *sizeof(float);
	
	float *h_A, *h_B, *host, *gpu;
	h_A = (float *)malloc(nBytes);	
	h_B = (float *)malloc(nBytes);
	host = (float *)malloc(nBytes);
	gpu = (float *)malloc(nBytes);

	initData(h_A, nElem);
	initData(h_B, nElem);
	memset(host, 0, nBytes);
	memset(gpu, 0, nBytes);

	float *d_A, *d_B, *d_C;
	cudaMalloc((float**)&d_A, nBytes);
	cudaMalloc((float**)&d_B, nBytes);
	cudaMalloc((float**)&d_C, nBytes);

	cudaMemcpy(d_A, h_A, nBytes, cudaMemcpyHostToDevice);
	cudaMemcpy(d_B, h_B, nBytes, cudaMemcpyHostToDevice);

	dim3 block(nElem);
	dim3 grid(nElem/block.x);

	sumArrayOnDevice<<<grid, block>>>(d_A, d_B, d_C);
	printf("Execution Cfg <<<%d, %d>>>\n", grid.x, block.x);

	cudaMemcpy(gpu, d_C, nBytes, cudaMemcpyDeviceToHost);

	sumArrayOnHost(h_A, h_B, host, nElem);

	checkResult(host, gpu, nElem);

	cudaFree(d_A);
	cudaFree(d_B);
	cudaFree(d_C);

	free(h_A);
	free(h_B);
	free(host);
	free(gpu);
	int c = getchar();
	return 0;
}


