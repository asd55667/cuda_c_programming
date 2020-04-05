#include<cuda_runtime.h>
#include<stdio.h>
#include "../common/common_win.h"

void cpu_sum(float *a, float *b, float *c, const int n)
{
    for (int i=0; i< n; i++)
    {
        c[i] = a[i] + b[i];
    }
}

__global__ void gpu_sum(float *a, float *b, float *c, unsigned int n)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}

__global__ void gpu_sum_zero_cpy(float *a, float *b, float *c, unsigned int n)
{
    unsigned int idx = blockDim.x * blockIdx.x + threadIdx.x;
    if (idx < n)
        c[idx] = a[idx] + b[idx];
}


int main(int argc, char **argv)
{
    int dev = 0;
    cudaDeviceProp devp;
    CHECK(cudaSetDevice(dev));
    CHECK(cudaGetDeviceProperties(&devp, dev));
    printf("Dev %d, name %s\n", dev, devp.name);

    if (!devp.canMapHostMemory)
    {
        printf("Device %d does not support mapping CPU host memory!\n", dev);
        CHECK(cudaDeviceReset());
        exit(EXIT_SUCCESS);
    }

    int pow = 22;
    // if (argc > 1) pow = atoi(argv[1]);
        
    unsigned int size = 1 << pow;
    unsigned int nbytes = size * sizeof(float);
    printf("Vec size %5.2fMB\n", nbytes/1024.f/1024.f);

    // cpu alloc
    float *ha, *hb, *hc, *hd;
    ha = (float *)malloc(nbytes);
    hb = (float *)malloc(nbytes);
    hc = (float *)malloc(nbytes);
    hd = (float *)malloc(nbytes);

    init_float(ha, size);
    init_float(hb, size);
    memset(hc, 0, nbytes);
    memset(hd, 0, nbytes);

    float *da, *db, *dc;
    CHECK(cudaMalloc((void **)&da, nbytes));
    CHECK(cudaMalloc((void **)&db, nbytes));
    CHECK(cudaMalloc((void **)&dc, nbytes));
    
    CHECK(cudaMemcpy(da, ha, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(db, hb, nbytes, cudaMemcpyHostToDevice));
    
    int nthread = 512;
    dim3 block(nthread);
    dim3 grid((size+block.x-1)/block.x);
    printf("Cfg <<<%d,%d>>>\n",grid.x, block.x);
    
    cpu_sum(ha, hb, hc, size);
    gpu_sum<<<grid, block>>>(da, db, dc, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());
    
    CHECK(cudaMemcpy(hd, dc, nbytes, cudaMemcpyDeviceToHost));
    check_float(hd, hc, size);

    CHECK(cudaFree(da));
    CHECK(cudaFree(db));
    free(ha);
    free(hb);

    // zero-cpy
    printf("---------zero-copy mem------------")
    CHECK(cudaHostAlloc((void **) &ha, nbytes, cudaHostAllocMapped));
    CHECK(cudaHostAlloc((void **) &hb, nbytes, cudaHostAllocMapped));
    init_float(ha, size);
    init_float(hb, size);
    memset(hc, 0, nbytes);
    memset(hd, 0, nbytes);

    CHECK(cudaHostGetDevicePointer((void **)&da, (void *)ha, 0));
    CHECK(cudaHostGetDevicePointer((void **)&db, (void *)hb, 0));
    
    
    cpu_sum(ha, hb, hc, size);
    gpu_sum_zero_cpy<<<grid, block>>>(da, db, dc, size);
    CHECK(cudaDeviceSynchronize());
    CHECK(cudaGetLastError());

    CHECK(cudaMemcpy(hd, dc, nbytes, cudaMemcpyDeviceToHost))   ;
    //printf("Result of GPU sum zero-copy %f\n",hd);
    
    check_float(hc, hd, size);

    CHECK(cudaFreeHost(da));
    CHECK(cudaFreeHost(db));
    CHECK(cudaFree(dc));
    free(hc);
    free(hd);

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}