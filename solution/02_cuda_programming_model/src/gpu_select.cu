#include<cuda_runtime.h>
#include<stdio.h>

int main()
{
    int num = 0;
    int maxdev = 0;
    cudaGetDeviceCount(&num);
    if (num > 1)
    {
        int maxp = 0;
        for (int i = 0; i < num; i++)
        {
            cudaDeviceProp props;
            cudaGetDeviceProperties(&props, i);
            if (maxp < props.multiProcessorCount)
            {
                maxp = props.multiProcessorCount;
                maxdev = i;
            }
        }
        cudaSetDevice(maxdev);
    }
    printf("dev idx: %d\n", maxdev);
}
