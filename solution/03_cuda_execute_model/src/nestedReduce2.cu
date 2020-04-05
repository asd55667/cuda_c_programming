#include "../common/common_win.h"
#include <cuda_runtime.h>
#include <stdio.h>

int cpu_reduce(int *data, const int n)
{
	if(n==1)
		return data[0];

	const int stride = n /2 ;
	for (int i=0;i<stride;i++)
		data[i] += data[i+stride];

	return cpu_reduce(data, stride);
}



__global__ void nested_neighbored(int *gi, int *go,unsigned int n)
{
	unsigned int tidx = threadIdx.x;
	unsigned int idx = blockIdx.x*blockDim.x + threadIdx.x;

	int *data = gi + blockDim.x * blockIdx.x;
	if (idx > n)
		return ;

	for (int i=1; i < blockDim.x; i*=2)
	{
		if (tidx % (2 * i) == 0)
			data[tidx] += data[tidx + i];
			
		__syncthreads();
	}

	if (tidx == 0)
		go[blockIdx.x] = data[0];
}

__global__ void nested_reduce(int *gi, int *go, unsigned n)
{
	unsigned int tidx = threadIdx.x;

	int *idata = gi + blockDim.x * blockIdx.x;
	int *odata = &go[blockIdx.x];

	
	if (n == 2 && tidx == 0)
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
	
	unsigned int stride = n >> 1;
	if (stride > 1 && tidx< stride)
	{
		idata[tidx] += idata[tidx + stride];
		if(tidx == 0)
			nested_reduce_no_sync<<<1, stride>>>(idata, odata,stride);
	}
}


__global__ void nested_reduce_2(int *gi, int *go, int stride, const int dim)
{
	unsigned int tidx = threadIdx.x;
	int *idata = gi + dim * blockIdx.x;
	
	if (stride == 1&& tidx == 0)
	{
		go[blockIdx.x] = idata[0] + idata[1];
		return ;
	}

	idata[tidx] += idata[tidx + stride];
	if (tidx == 0 && blockIdx.x == 0)
		nested_reduce_2<<<gridDim.x, stride / 2>>>(gi, go, stride /2, dim);
}


int main(int argc, char **argv)
{
	int dev = 0, gpu_sum;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
	CHECK(cudaSetDevice(dev));
	

	int nblock  = 2048;
	int nthread = 512;
	if(argc > 1)
    {
        nblock = atoi(argv[1]);   // block size from command line argument
    }

    if(argc > 2)
    {
        nthread = atoi(argv[2]);   // block size from command line argument
	}
	
	int size = nblock * nthread;

	dim3 block (nthread, 1);
	dim3 grid  ((size + block.x - 1) / block.x, 1);
	
	size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
	int *tmp     = (int *) malloc(bytes);
	
	for (int i = 0; i < size; i++)
    {
        h_idata[i] = (int)( rand() & 0xFF );
        h_idata[i] = 1;
	}
	
	memcpy (tmp, h_idata, bytes);

	int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

	double iStart, iElaps;
	
	//cpu
	iStart = seconds();
    int cpu_sum = cpu_reduce(tmp, size);
    iElaps = seconds() - iStart;
	printf("\ncpu reduce\t\telapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);
	

    // nested_neighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    nested_neighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored\t\telapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);


	// nested_reduce
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    nested_reduce<<<grid, block>>>(d_idata, d_odata, block.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu nested\t\telapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n",
           iElaps, gpu_sum, grid.x, block.x);


	// nested_reduce_no_sync
	CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    nested_reduce_no_sync<<<grid, block>>>(d_idata, d_odata, block.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu nestedNosyn\t\telapsed %f sec gpu_sum: %d <<<grid %d block "
		   "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);		   
		   

	// nested_reduce_2
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    iStart = seconds();
    nested_reduce_2<<<grid, block.x / 2>>>(d_idata, d_odata, block.x / 2,
            block.x);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu nested2\t\telapsed %f sec gpu_sum: %d <<<grid %d block %d>>>\n",
		   iElaps, gpu_sum, grid.x, block.x);		   
		   

    free(h_idata);
	free(h_odata);
	
    CHECK(cudaFree(d_idata));
	CHECK(cudaFree(d_odata));
	
	CHECK(cudaDeviceReset());

	bool res;
	res = (gpu_sum == cpu_sum);
    if(!res) printf("Test failed!\n");

    return EXIT_SUCCESS;
}
