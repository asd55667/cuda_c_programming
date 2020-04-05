#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common_win.h"

int main()
{
    int dev = 0;
    CHECK(cudaSetDevice(dev));

    unsigned int size = 1 << 24;
    unsigned int nbytes = size * sizeof(float);
    cudaDeviceProp devp;
    CHECK(cudaGetDeviceProperties(&devp, dev));
    printf("Dev: %d, name %s, size %5.2fMB\n",dev, devp.name, nbytes/1024.f/1024.f);

    float *h = (float *)malloc(nbytes);
    init_float(h, size);

    float *d;
    CHECK(cudaMalloc((void **)&d, nbytes));

    CHECK(cudaMemcpy(d, h, nbytes, cudaMemcpyHostToDevice));
    CHECK(cudaMemcpy(h, d, nbytes, cudaMemcpyDeviceToHost));


    CHECK(cudaFree(d));
    free(h);

    

    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;
}
