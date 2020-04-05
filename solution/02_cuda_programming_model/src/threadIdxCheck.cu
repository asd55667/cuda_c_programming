#include <cuda_runtime.h>
#include <stdio.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if (error != cudaSuccess)\
	{\
		printf("Error %s, %s\n", __FILE__, __LINE__);\
		printf("code: %s, reason: %s\n", error, cudaGetErrorString(error));\
		exit(-10 * error);\
	}\
}\

void init_data(int *inp, int n)
{
	for (int i = 0; i < n; i++)
		inp[i] = i;
}

void print_matrix(int *mat, const int x, const int y)
{
	int *m = mat;
	printf("\nMatrix: (%d,%d)\n", x, y);
	for (int j=0;j<y;j++)
	{
		for (int i=0;i<x;i++)
			printf("%3d",m[i]);
		m += x;
		printf("\n");
	}
	printf("\n");
}

__global__ void print_thread_idx(int *a, const int x, const int y)
{
	int i = threadIdx.x + blockIdx.x * blockDim.x;
	int j = threadIdx.y + blockIdx.y * blockDim.y;
	unsigned int idx = j * x + i;

	printf("thread_id (%d, %d) block_id (%d,%d) coord (%d, %d) "
			"global index %2d ival %2d\n", threadIdx.x, threadIdx.y, blockIdx.x,
			blockIdx.y, i, j, idx, a[idx]);
}


int main()
{
	int dev = 0;
	cudaDeviceProp devp;
	CHECK(cudaGetDeviceProperties(&devp, dev));
	printf("Device %d: %s\n", dev, devp.name);
	CHECK(cudaSetDevice(dev));

	int x = 8, y = 6;
	int n = 48;
	int nbytes = n * sizeof(float);

	int *ha;
	ha = (int *)malloc(nbytes);
	init_data(ha, n);
	print_matrix(ha, x, y);

	int *da;
	cudaMalloc((void **)&da, nbytes);
	cudaMemcpy(da, ha, nbytes, cudaMemcpyHostToDevice);

	dim3 block(4,2);
	dim3 grid((x+block.x-1)/block.x, (y+block.y-1)/block.y);

	print_thread_idx<<<grid, block>>>(da, x,y);
	cudaDeviceSynchronize();

	cudaFree(da);

	free(ha);
	
	cudaDeviceReset();
	
	int c = getchar();
	return 0;
}

