#include<cuda_runtime.h>
#include<stdio.h>

int main()
{
    int dev_count = 0;
    cudaError_t error_id = cudaGetDeviceCount(&dev_count);
    if(error_id != cudaSuccess)
    {
        printf("cudaGetDeviceCount returned %d\n->%s\n", int(error_id), cudaGetErrorString(error_id));
        printf("Result = FAIL\n");
        exit(EXIT_FAILURE);
    }

    if(dev_count == 0)
        printf("There are no avaiable device(s) that support CUDA\n");
    else
        printf("Detected %d Cuda capable device(s)\n", dev_count);

    int dev = 0,driver_version = 0,runtime_version = 0;
    cudaSetDevice(dev);
    cudaDeviceProp devp;
    cudaGetDeviceProperties(&devp, dev);
    printf("Device %d: \"%s\"\n", dev, devp.name);

    cudaDriverGetVersion(&driver_version);
    cudaRuntimeGetVersion(&runtime_version);
    printf("CUDA driver version / runtime version %d.%d / %d.%d\n",
        driver_version / 1000, (driver_version%100)/10,
        runtime_version / 1000, (runtime_version%100)/10);
    printf("CUDA capabnility major/minor version number: %d.%d\n", devp.major, devp.minor);
    printf(" Total amount of global memory: %.2f MBytes (%llu bytes)\n",
             (float)devp.totalGlobalMem/(pow(1024.0,3)), 
              (unsigned long long) devp.totalGlobalMem);
    printf(" GPU Clock rate: %.0f MHz (%0.2f GHz)\n", 
            devp.clockRate * 1e-3f, devp.clockRate * 1e-6f);
    printf(" Memory Clock rate: %.0f Mhz\n", devp.memoryClockRate * 1e-3f);

    printf(" Memory Bus Width: %d-bit\n",    devp.memoryBusWidth);
    if (devp.l2CacheSize)
        printf(" L2 Cache Size: %d bytes\n",  devp.l2CacheSize);


    printf(" Max Texture Dimension Size (x,y,z) " 
        " 1D=(%d), 2D=(%d,%d), 3D=(%d,%d,%d)\n",
            devp.maxTexture1D , devp.maxTexture2D[0], 
            devp.maxTexture2D[1],
            devp.maxTexture3D[0], devp.maxTexture3D[1], 
            devp.maxTexture3D[2]);
    printf(" Max Layered Texture Size (dim) x layers 1D=(%d) x %d, 2D=(%d,%d) x %d\n",
           devp.maxTexture1DLayered[0], devp.maxTexture1DLayered[1],
           devp.maxTexture2DLayered[0], devp.maxTexture2DLayered[1], 
           devp.maxTexture2DLayered[2]);

    printf(" Total amount of constant memory: %lu bytes\n",
            devp.totalConstMem);
    printf(" Total amount of shared memory per block: %lu bytes\n",
              devp.sharedMemPerBlock);
    printf(" Total number of registers available per block: %d\n",
       devp.regsPerBlock);
    printf(" Warp size: %d\n", devp.warpSize);
    printf(" Maximum number of threads per multiprocessor: %d\n",
        devp.maxThreadsPerMultiProcessor);
    printf(" Maximum number of threads per block: %d\n",
          devp.maxThreadsPerBlock);
    printf(" Maximum sizes of each dimension of a block: %d x %d x %d\n",
           devp.maxThreadsDim[0],
            devp.maxThreadsDim[1],
             devp.maxThreadsDim[2]);
    printf(" Maximum sizes of each dimension of a grid: %d x %d x %d\n",
            devp.maxGridSize[0],
             devp.maxGridSize[1],
              devp.maxGridSize[2]);
    printf(" Maximum memory pitch: %lu bytes\n", devp.
            memPitch);
    exit(EXIT_SUCCESS);
}


