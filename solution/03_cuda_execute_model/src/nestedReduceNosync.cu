#include "../common/common.h"
#include <cuda_runtime.h>
#include <stdio.h>

int cpu_recursive_reduce(int *data, const int n)
{
	if (n == 1)
		return data[0];

	const int stride = n / 2;

	for (int i = 0; i < stride; i++)
		data[i] += data[i + stride];

	return cpu_recursive_reduce(data, stride);
}


__global__ void nested_neighbored(int *gi, int *go, unsigned int n)
{
	unsigned int tidx = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;
	
	int *data = gi + blockIdx.x * blockDim.x;

	if (idx > n)
		return ;

	for (int i = 1; i < blockDim.x; i*=2)
	{
		if (tidx % (i * 2) == 0)
			data[tidx] += data[tidx + i];

		__syncthreads();
	}
	if (tidx == 0)
		go[blockIdx.x] = data[0];
}

__global__ void nested_reduce(int *gi, int *go, unsigned int n)
{
	unsigned int tidx = threadIdx.x;

	int *idata = gi + blockIdx.x * blockDim.x;
	int *odata = &go[blockIdx.x];

	if (tidx == 0 && n == 2)
	{
		go[blockIdx.x] = idata[0] + idata[1];
		return ;
	}

	unsigned int stride = n >> 1;

	if (stride > 1 && tidx < stride)
		idata[tidx] += idata[tidx + stride];
	__syncthreads();

	if (tidx == 0)
	{	
		nested_reduce<<<1, stride>>>(idata, odata, stride);
	cudaDeviceSynchronize();
	}
	__syncthreads();
}


__global__ void nested_reduce_no_sync(int *gi, int *go, unsigned int n)
{
	unsigned int tidx = threadIdx.x;
	
	int *idata = gi + blockDim.x * blockIdx.x;
	int *odata = &go[blockIdx.x];

	if (n == 2 && tidx == 0)
	{
		go[blockIdx.x] = idata[0] + idata[1];
		return ;
	}

	int stride = n >> 1;

	if (stride > 1 && tidx < stride)
	{
		idata[tidx] += idata[tidx + stride];
		if (tidx == 0)
			nested_reduce_no_sync<<<1, stride>>>(idata, odata, stride);
	}
}



int main(int argc, char **argv)
{
	int dev = 0;
	cudaDeviceProp devp;
	CHECK(cudaSetDevice(dev));
	CHECK(cudaGetDeviceProperties(&devp, dev));
	printf("Device: %d, info: %s\n", dev, devp.name);
	printf("%s starting reduction at", argv[0]);

	int nblock = 2048;
	int nthread = 512;
	
	if (argc > 1)
		nblock = atoi(argv[1]);
	if (argc > 2)
		nthread = atoi(argv[2]);

	const int n = nblock * nthread;
	int bytes = n * sizeof(int);
	
	dim3 block(nthread, 1);
	dim3 grid((n + block.x-1)/block.x, 1);
	printf("grid %d, block %d\n", grid.x, block.x);

    int *hi, *ho, *tmp;
	hi = (int *)malloc(bytes);
	ho = (int *)malloc(bytes);
	tmp = (int *)malloc(bytes);

    for(int i=0;i<n;i++)
    {
        hi[i] = (int)(rand()&0xFF);
        hi[i] =1 ;
    }
	memcpy(tmp, hi, bytes);

	double start, elaps;
	start = seconds();
	int cpu_sum = cpu_recursive_reduce(tmp, n);
	elaps = seconds() - start;
	printf("cpu recursive reduce: %f ms, sum %d\n", elaps, cpu_sum);

	int *d_idata = NULL;
	int *d_odata = NULL;
	int gpu_sum = 0;
	CHECK(cudaMalloc((void **) &d_idata, bytes));
	CHECK(cudaMalloc((void **) &d_odata, grid.x*sizeof(int)));

	CHECK(cudaMemcpy(d_idata, hi, bytes, cudaMemcpyHostToDevice));
	
	start = seconds();
	nested_neighbored<<<grid, block>>>(d_idata, d_odata, n);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	elaps = seconds() - start;

	CHECK(cudaMemcpy(ho, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
	for(int i=0;i<grid.x;i++)
		gpu_sum += ho[i];

	printf("reduce neighbored: %fms sum: %d\n", elaps, cpu_sum);








	CHECK(cudaMemcpy(d_idata, hi, bytes, cudaMemcpyHostToDevice));
	start = seconds();
	nested_reduce<<<grid, block>>>(d_idata, d_odata, block.x);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	elaps = seconds() - start;

	CHECK(cudaMemcpy(ho, d_odata, grid.x*sizeof(int), cudaMemcpyDeviceToHost));
	gpu_sum = 0;
	for(int i=0;i<grid.x;i++)
		gpu_sum += ho[i];
	printf("recursive reduce: %fms sum: %d\n",elaps, gpu_sum);






	CHECK(cudaMemcpy(d_idata, hi, bytes, cudaMemcpyHostToDevice));
	start = seconds();
	nested_reduce_no_sync<<<grid, block>>>(d_idata, d_odata, block.x);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	elaps = seconds() - start;

	CHECK(cudaMemcpy(ho, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
	gpu_sum = 0 ;
	for (int i= 0;i<grid.x;i++)
		gpu_sum += ho[i];
	printf("recursive reduce nosync: %fms sum: %d\n",elaps, gpu_sum);

	free(hi);
	free(ho);
	free(tmp);

	CHECK(cudaFree(d_idata));
	CHECK(cudaFree(d_odata));

	CHECK(cudaDeviceReset());

	bool res;
	res = (gpu_sum == cpu_sum);
	if (!res)
		printf("Don't Match!\n");
	return EXIT_SUCCESS;
}
