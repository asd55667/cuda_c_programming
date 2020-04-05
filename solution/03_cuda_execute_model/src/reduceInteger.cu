#include "../common/common_win.h"
#include <cuda_runtime.h>
#include <stdio.h>

int cpu_reduce(int *data, const int n)
{
	if (n==1)
		return data[0];

	const int stride = n / 2;
	for (int i=0;i<stride;i++)
		data[i] += data[i+stride];

	return cpu_reduce(data, stride);
}

__global__ void nested_neighbored(int *gi, int *go,unsigned int n)
{
	unsigned int tidx = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;

	int *data = gi + blockIdx.x * blockDim.x;
	
	if (idx > n)
		return ;

	for (int i=1; i<blockDim.x;i*=2)
	{
		if(tidx % (2*i) == 0)
			data[tidx] += data[tidx + i];

		__syncthreads();
	}

	if(tidx == 0)
		go[blockIdx.x] = data[0];
}

__global__ void nested_neighbored_less(int *gi, int *go, unsigned int n)
{
	unsigned int tidx = threadIdx.x;
	unsigned int bidx = blockIdx.x;
	unsigned int idx = blockDim.x * bidx + tidx;

	int *data = gi + bidx * blockDim.x;
	if(idx > n)
		return ;

	for (int i=1; i<blockDim.x; i*=2)
	{
		int j = 2 * i * tidx;
		if (j < blockDim.x)
			data[j] += data[i+j];

		__syncthreads();
	}
	if(tidx == 0)
		go[bidx] = data[0];
}


__global__ void reduce_leaved(int *gi, int *go, unsigned int n)
{
	unsigned int tidx = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x + tidx;

	int *data = gi + blockIdx.x * blockDim.x;
	if (idx > n)
		return ;

	for (int i = blockDim.x / 2; i > 0; i >>= 1)
	{
		if(tidx < i)
			data[tidx] += data[tidx + i];

		__syncthreads();
	}

	if(tidx==0)
		go[blockIdx.x] = data[0];
}

__global__ void reduce_unrolling2(int *gi, int *go, unsigned int n)
{
	unsigned int tidx = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 2 + tidx;

	int *data = gi + blockIdx.x * blockDim.x *2;

	if(idx > n)
		return ;
	for (int i = blockDim.x /2 ; i > 0; i>>=1)
	{
		if(tidx < i)
			data[tidx] += data[tidx + i];

		__syncthreads();
	}
	if (tidx == 0)
		go[blockIdx.x] = data[0];
}

__global__ void reduce_unrolling4(int *gi, int *go, unsigned int n)
{
	unsigned int tidx = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x * 4 + tidx;

	int *data = gi + blockIdx.x * blockDim.x * 4;

	if (idx + 3*blockDim.x < n)
	{
		int a1 = gi[idx];
		int a2 = gi[idx + blockDim.x];
		int a3 = gi[idx + blockDim.x * 2];
		int a4 = gi[idx + blockDim.x * 3];
		gi[idx] = a1 + a2 + a3 + a4;
	}
	
	__syncthreads();

	for (int i = blockDim.x / 2; i > 0; i>>=1)
	{
		if (tidx < i)
			data[tidx] += data[tidx + i];

		__syncthreads();
	}

	if (tidx == 0)
		go[blockIdx.x] = data[0];
}

__global__ void reduce_unrolling8(int *gi, int *go, unsigned int n)
{
	unsigned int tidx = threadIdx.x;
	unsigned int idx = blockIdx.x * blockDim.x * 8 + tidx;
	int *data = gi + blockDim.x * blockIdx.x * 8;

	if(idx + 7 * blockDim.x < n)
	{
		int a1 = gi[idx];
		int a2 = gi[idx + blockDim.x];
		int a3 = gi[idx + blockDim.x * 2];
		int a4 = gi[idx + blockDim.x * 3];
		int b1 = gi[idx + blockDim.x * 4];
		int b2 = gi[idx + blockDim.x * 5];
		int b3 = gi[idx + blockDim.x * 6];
		int b4 = gi[idx + blockDim.x * 7];
		gi[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 +b4;
	}

	__syncthreads();

	for (int i = blockDim.x / 2; i > 0; i>>=1)
	{
		if(tidx < i)
			data[tidx] += data[tidx + i];

		__syncthreads();
	}

	if (tidx == 0)
		go[blockIdx.x] = data[0];
}

__global__ void reduce_unrolling_warps8(int *gi, int *go, unsigned int n)
{
	unsigned int tidx = threadIdx.x;
	unsigned int idx = blockDim.x * blockIdx.x * 8 + tidx;

	int *data = gi + blockIdx.x * blockDim.x * 8;

	if (idx + blockDim.x * 7 < n)
	{
		int a1 = gi[idx];
        int a2 = gi[idx + blockDim.x];
        int a3 = gi[idx + 2 * blockDim.x];
        int a4 = gi[idx + 3 * blockDim.x];
        int b1 = gi[idx + 4 * blockDim.x];
        int b2 = gi[idx + 5 * blockDim.x];
        int b3 = gi[idx + 6 * blockDim.x];
        int b4 = gi[idx + 7 * blockDim.x];
        gi[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
	}
	__syncthreads();

	for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
	{
		if (tidx < stride)
		{
			data[tidx] += data[tidx + stride];
		}
		__syncthreads();
	}

	if (tidx < 32)
	{
		volatile int *vmem = data;
		vmem[tidx] = vmem[tidx + 32];
		vmem[tidx] = vmem[tidx + 16];
		vmem[tidx] = vmem[tidx + 8 ];
		vmem[tidx] = vmem[tidx + 4 ];
		vmem[tidx] = vmem[tidx + 2 ];
		vmem[tidx] = vmem[tidx + 1 ];
	}

	if (tidx == 0)
	{
		go[blockIdx.x] = data[0];
	}

}


__global__ void reduce_complete_unrolling_warps8(int *gi, int *go, unsigned int n)
{
	unsigned int tidx = threadIdx.x;
	unsigned idx = blockIdx.x * blockDim.x * 8 + tidx;

	int *data = gi + blockDim.x * blockIdx.x * 8;

	if (idx + blockDim.x * 7 < n)
	{
		int a1 = gi[idx];
        int a2 = gi[idx + blockDim.x];
        int a3 = gi[idx + 2 * blockDim.x];
        int a4 = gi[idx + 3 * blockDim.x];
        int b1 = gi[idx + 4 * blockDim.x];
        int b2 = gi[idx + 5 * blockDim.x];
        int b3 = gi[idx + 6 * blockDim.x];
        int b4 = gi[idx + 7 * blockDim.x];
        gi[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
	}
	__syncthreads();


	if (blockDim.x >= 1024 && tidx < 512)	data[tidx] += data[tidx + 512];
	__syncthreads();

	if (blockDim.x >= 512 && tidx < 256) 	data[tidx] += data[tidx + 256];
	__syncthreads();
	
	if (blockDim.x >= 256 && tidx < 128)	data[tidx] += data[tidx + 128];
	__syncthreads();
	
	if (blockDim.x > 128 && tidx < 64) 		data[tidx] += data[tidx + 64];
	__syncthreads();
	
	if (tidx < 32)
    {
        volatile int *vsmem = data;
        vsmem[tidx] += vsmem[tidx + 32];
        vsmem[tidx] += vsmem[tidx + 16];
        vsmem[tidx] += vsmem[tidx +  8];
        vsmem[tidx] += vsmem[tidx +  4];
        vsmem[tidx] += vsmem[tidx +  2];
        vsmem[tidx] += vsmem[tidx +  1];
    }
 
	if (tidx == 0) go[blockIdx.x] = data[0];
}

	
template <unsigned int iBlockSize>
__global__ void reudce_complete_unroll(int *gi, int *go, unsigned int n)
{
	unsigned int tid = threadIdx.x;
	unsigned int idx = blockDim.x*blockIdx.x*8 + tid;

	int *data = gi + blockIdx.x*blockDim.x*8;

	if (idx+7*blockDim.x <n)
	{
		int a1 = gi[idx];
        int a2 = gi[idx + blockDim.x];
        int a3 = gi[idx + 2 * blockDim.x];
        int a4 = gi[idx + 3 * blockDim.x];
        int b1 = gi[idx + 4 * blockDim.x];
        int b2 = gi[idx + 5 * blockDim.x];
        int b3 = gi[idx + 6 * blockDim.x];
        int b4 = gi[idx + 7 * blockDim.x];
        gi[idx] = a1 + a2 + a3 + a4 + b1 + b2 + b3 + b4;
	}
	__syncthreads();
	
    if (iBlockSize >= 1024 && tid < 512) data[tid] += data[tid + 512];

    __syncthreads();

    if (iBlockSize >= 512 && tid < 256)  data[tid] += data[tid + 256];

    __syncthreads();

    if (iBlockSize >= 256 && tid < 128)  data[tid] += data[tid + 128];

    __syncthreads();

    if (iBlockSize >= 128 && tid < 64)   data[tid] += data[tid + 64];

	__syncthreads();
	
	if(tid< 32)
	{
		volatile int *vsmem = data;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
	}

	if (tid == 0) go[blockIdx.x] = data[0];


}

__global__ void reduceUnrollWarps (int *g_idata, int *g_odata, unsigned int n)
{
    // set thread ID
    unsigned int tid = threadIdx.x;
    unsigned int idx = blockIdx.x * blockDim.x * 2 + threadIdx.x;

    // convert global data pointer to the local pointer of this block
    int *idata = g_idata + blockIdx.x * blockDim.x * 2;

    // unrolling 2
    if (idx + blockDim.x < n) g_idata[idx] += g_idata[idx + blockDim.x];

    __syncthreads();

    // in-place reduction in global memory
    for (int stride = blockDim.x / 2; stride > 32; stride >>= 1)
    {
        if (tid < stride)
        {
            idata[tid] += idata[tid + stride];
        }

        // synchronize within threadblock
        __syncthreads();
    }

    // unrolling last warp
    if (tid < 32)
    {
        volatile int *vsmem = idata;
        vsmem[tid] += vsmem[tid + 32];
        vsmem[tid] += vsmem[tid + 16];
        vsmem[tid] += vsmem[tid +  8];
        vsmem[tid] += vsmem[tid +  4];
        vsmem[tid] += vsmem[tid +  2];
        vsmem[tid] += vsmem[tid +  1];
    }

    if (tid == 0) g_odata[blockIdx.x] = idata[0];
}


int main(int argc, char **argv)
{
    // set up device
    int dev = 0;
    cudaDeviceProp deviceProp;
    CHECK(cudaGetDeviceProperties(&deviceProp, dev));
    printf("%s starting reduction at ", argv[0]);
    printf("device %d: %s ", dev, deviceProp.name);
    CHECK(cudaSetDevice(dev));

    bool bResult = false;

    // initialization
    int size = 1 << 24; // total number of elements to reduce
    printf("    with array size %d  ", size);

    // execution configuration
    int blocksize = 512;   // initial block size

    if(argc > 1)
    {
        blocksize = atoi(argv[1]);   // block size from command line argument
    }

    dim3 block (blocksize, 1);
    dim3 grid  ((size + block.x - 1) / block.x, 1);
    printf("grid %d block %d\n", grid.x, block.x);

    // allocate host memory
    size_t bytes = size * sizeof(int);
    int *h_idata = (int *) malloc(bytes);
    int *h_odata = (int *) malloc(grid.x * sizeof(int));
    int *tmp     = (int *) malloc(bytes);

    // initialize the array
    for (int i = 0; i < size; i++)
    {
        // mask off high 2 bytes to force max number to 255
        h_idata[i] = (int)( rand() & 0xFF );
    }

    memcpy (tmp, h_idata, bytes);

    double iStart, iElaps;
    int gpu_sum = 0;

    // allocate device memory
    int *d_idata = NULL;
    int *d_odata = NULL;
    CHECK(cudaMalloc((void **) &d_idata, bytes));
    CHECK(cudaMalloc((void **) &d_odata, grid.x * sizeof(int)));

    // cpu reduction
    iStart = seconds();
    int cpu_sum = cpu_reduce (tmp, size);
    iElaps = seconds() - iStart;
    printf("cpu reduce  elapsed %f sec cpu_sum: %d\n", iElaps, cpu_sum);


////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
////////////////////////////////////////////////////////////////////////////////////////
    // kernel 1: reduceNeighbored
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    nested_neighbored<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 2: reduceNeighbored with less divergence
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    nested_neighbored_less<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Neighbored2 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 3: reduceInterleaved
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduce_leaved<<<grid, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x; i++) gpu_sum += h_odata[i];

    printf("gpu Interleaved elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x, block.x);

    // kernel 4: reduceUnrolling2
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduce_unrolling2<<<grid.x / 2, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 2 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 2; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling2  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 2, block.x);

    // kernel 5: reduceUnrolling4
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduce_unrolling4<<<grid.x / 4, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 4 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 4; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling4  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 4, block.x);

    // kernel 6: reduceUnrolling8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduce_unrolling8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Unrolling8  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    for (int i = 0; i < grid.x / 16; i++) gpu_sum += h_odata[i];

    // kernel 8: reduceUnrollWarps8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduce_unrolling_warps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu UnrollWarp8 elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);


    // kernel 9: reudce_complete_unrollWarsp8
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
    iStart = seconds();
    reduce_complete_unrolling_warps8<<<grid.x / 8, block>>>(d_idata, d_odata, size);
    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));
    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll8  elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // kernel 9: reudce_complete_unroll
    CHECK(cudaMemcpy(d_idata, h_idata, bytes, cudaMemcpyHostToDevice));
    CHECK(cudaDeviceSynchronize());
	iStart = seconds();

	switch (blocksize)
	{
		case 1024:
		reudce_complete_unroll<1024><<<grid.x / 8, block>>>(d_idata, d_odata,
			size);
		break;

		case 512:
        reudce_complete_unroll<512><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 256:
        reudce_complete_unroll<256><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 128:
        reudce_complete_unroll<128><<<grid.x / 8, block>>>(d_idata, d_odata,
                size);
        break;

    case 64:
        reudce_complete_unroll<64><<<grid.x / 8, block>>>(d_idata, d_odata, size);
        break;
	}


    CHECK(cudaDeviceSynchronize());
    iElaps = seconds() - iStart;
    CHECK(cudaMemcpy(h_odata, d_odata, grid.x / 8 * sizeof(int),
                     cudaMemcpyDeviceToHost));

    gpu_sum = 0;

    for (int i = 0; i < grid.x / 8; i++) gpu_sum += h_odata[i];

    printf("gpu Cmptnroll   elapsed %f sec gpu_sum: %d <<<grid %d block "
           "%d>>>\n", iElaps, gpu_sum, grid.x / 8, block.x);

    // free host memory
    free(h_idata);
    free(h_odata);

    // free device memory
    CHECK(cudaFree(d_idata));
    CHECK(cudaFree(d_odata));

    // reset device
    CHECK(cudaDeviceReset());

    // check the results
    bResult = (gpu_sum == cpu_sum);

    if(!bResult) printf("Test failed!\n");

    return EXIT_SUCCESS;	

	
}