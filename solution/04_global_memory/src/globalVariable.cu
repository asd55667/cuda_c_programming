#include <cuda_runtime.h>
#include <stdio.h>
#include "../common/common_win.h"

__device__ float devData;


__global__ void d_global()
{
    devData += 2.0f;
}


int main()
{
    float val = 3.2f;
    printf("raw data %f\n", val);
    
    CHECK(cudaMemcpyToSymbol(devData, &val, sizeof(float)));
    
    d_global<<<1,1>>>();
    
    CHECK(cudaMemcpyFromSymbol(&val, devData, sizeof(float)));
    
    printf("device global data after modified %f\n", val);
    CHECK(cudaDeviceReset());
    return EXIT_SUCCESS;

}