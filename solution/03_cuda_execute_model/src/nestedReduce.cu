#include "../common/common_win.h"
#include <cuda_runtime.h>
#include <stdio.h>

int cpu_recursive_reduce(int *data, const int n)
{
    if(n==1) return data[0];
    
    const int stride = n / 2;

    for (int i=0;i<stride; i++)
	{
        data[i] += data[i+stride];
	}
    return cpu_recursive_reduce(data, stride);
}



__global__ void reduce_neighbored(int *g_idata, int *g_odata, unsigned int n)
{
    unsigned int tidx = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x + threadIdx.x;

    int *idata = g_idata + blockIdx.x * blockDim.x;

    if (idx >= n)
        return ;

    for (int i = 1; i < blockDim.x; i*=2)
    {
        if ((tidx%(2*i)) == 0)
            idata[tidx] += idata[tidx + i];
        __syncthreads();
    }
    if (tidx == 0)
        g_odata[blockIdx.x] = idata[0];
}

__global__ void recursive_reduce(int *g_idata, int *g_odata,unsigned int n)
{
	unsigned int tidx = threadIdx.x;

	int *idata = g_idata + blockDim.x*blockIdx.x;
	int *odata = &g_odata[blockIdx.x];

	if (n == 2 && tidx == 0)
	{
		g_odata[blockIdx.x] = idata[0] + idata[1];
		return ;
	}

	int stride = n >> 1;

	if (stride > 1 && tidx < stride)
		idata[tidx] += idata[tidx + stride];

	__syncthreads();

	if (tidx == 0)
	{
		recursive_reduce<<<1, stride>>>(idata, odata, stride);
		cudaDeviceSynchronize();
	}

	__syncthreads();
}


int main(int argc,char **argv)
{
	int dev = 0;
	cudaDeviceProp devp;
	CHECK(cudaGetDeviceProperties(&devp, dev));
	printf("%s starting reduction at ", argv[0]);
	printf("device %d: %s ", dev, devp.name);
	CHECK(cudaSetDevice(dev));

	bool res = false;

	int nblock = 2048;
	int nthread = 512;
	if (argc >1)
		nblock = atoi(argv[1]);

	if (argc >2)
		nthread = atoi(argv[2]);
	
	int n = nblock * nthread;

	dim3 block(nthread,1);
	dim3 grid((n+block.x-1)/block.x,1);
	printf("array %d cfg <<<%d, %d>>>\n", n, grid.x, block.x);

	size_t bytes = n * sizeof(int);
	int *h_idata = (int *)malloc(bytes);
	int *h_odata = (int *)malloc(bytes);
	int *tmp = (int *)malloc(bytes);

	for (int i = 0; i < n; i++)
	{
		h_idata[i] = (int)(rand()&0xFF);
		h_idata[i] = 1;
	}

	memcpy(tmp, h_idata, bytes);

	int *d_idata = NULL;
	int *d_odata = NULL;
	CHECK(cudaMalloc((void **) &d_idata, bytes));
	CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

	double start, elaps;
	start = seconds();
	int cpu_sum = cpu_recursive_reduce(tmp, n);
	elaps = seconds() - start;
	printf("cpu reduce\t\t %f ms, sum: %d\n",elaps, cpu_sum);
	
	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	start = seconds();
	reduce_neighbored<<<grid, block>>>(d_idata, d_odata, n);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	elaps = seconds() - start;
	CHECK(cudaMemcpy(h_odata, d_idata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));

	int gpu_sum = 0;
	for(int i = 0; i < grid.x; i++)
		gpu_sum += h_odata[i];
	printf("gpu neighbored\t\t %f ms,sum: %d<<<%d,%d>>>\n",elaps, gpu_sum, grid.x, block.x);
	
	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
	start = seconds();
	recursive_reduce<<<grid, block>>>(d_idata, d_odata, block.x);
	CHECK(cudaDeviceSynchronize());
	CHECK(cudaGetLastError());
	elaps = seconds() - start;
	CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int), cudaMemcpyDeviceToHost));
	gpu_sum = 0;
	for (int i = 0; i < grid.x; i++)
		gpu_sum += h_odata[i];
	printf("gpu nested\t\t %f ms gpu_sum %d <<<%d, %d>>>\n",elaps, gpu_sum, grid.x, block.x);

	free(h_idata);
	free(h_odata);

	CHECK(cudaFree(d_idata));
	CHECK(cudaFree(d_odata));
	CHECK(cudaDeviceReset());

	
	res = (gpu_sum == cpu_sum);
	if (!res) 
		printf("Test Failed\n");
	return EXIT_SUCCESS;
}
