#include<cuda_runtime.h>

#include<sys/time.h>
#include<stdint.h>
#include<stdio.h>

#define CHECK(call)\
{\
	const cudaError_t error = call;\
	if(error != cudaSuccess)\
	{\
		printf("Error: %s, %d\n", __FILE__, __LINE__);\
		printf("Code: %d, reason: %s\n", error, cudaGetErrorString(error));\
		exit(0);\
	}\
}\



double cpu_sec(void)
{
    struct timeval tp;
	gettimeofday(&tp, NULL);
	return ((double)tp.tv_sec + (double)tp.tv_usec*1.e-6);
}

void check_num(float *c, float *g, int n)
{
	double epsilon = 1.0E-8;
	int match = 1;
	for (int i = 0; i < n; i++)
	{
		if (abs(c[i] - g[i]) > epsilon)
		{
			match = 0;
			printf("Don't match!\n");
			printf("host %5.2f device %5.2f at current %d\n", c[i], g[i], i);
			break;
		}
	}
	if (match)
		printf("Array match\n\n");
	return;
}

void init_data(float *inp, int n)
{
    time_t t;
    srand((unsigned) time(&t));

    for (int i = 0; i< n; i++)
        inp[i] = (float)(rand() & 0xFF) /10.0f;
}


void mat_sum(float *a, float *b, float *c, int x, int y)
{
    float *aa = a;
    float *bb = b;
    float *cc = c;
    for(int j = 0; j < y; j++)
    {
        for(int i = 0; i < x; i++)
            cc[i] = aa[i] + bb[i];
        aa += x;
        bb += x;
        cc += x;
    }
}


__global__ void mat_sum_g(float *a, float *b, float *c, int x, int y)
{
	unsigned int i = threadIdx.x + blockIdx.x * blockDim.x;
	if(i < x){
		for(int j =0; j < y; j++)
		{
			int idx = i + j * x;
			c[idx] = a[idx] + b[idx];
		}
	}
}



int main()
{
	int x = 1 << 6, y = 1 << 6;
    int n = x * y;
	size_t num = n * sizeof(float);

	float *ha, *hb, *cpu, *gpu;
	ha = (float *)malloc(num);
    hb = (float *)malloc(num);
    gpu = (float *)malloc(num);
    cpu = (float *)malloc(num);

    init_data(ha, n);
    init_data(hb, n);

    memset(gpu, 0, num);
    memset(cpu, 0, num);

    double start, duration;
    start = cpu_sec();
    mat_sum(ha, hb, cpu, x, y);
    duration = cpu_sec() - start;
    printf("Mat sum cpu time cost %f ms\n", duration*1000);
    
    float *da, *db, *dc;
    cudaMalloc((float **)&da, num);
    cudaMalloc((float **)&db, num);
    cudaMalloc((float **)&dc, num);

    cudaMemcpy(da, ha, num, cudaMemcpyHostToDevice);
    cudaMemcpy(db, hb, num, cudaMemcpyHostToDevice);

    dim3 block(32, 1);
    dim3 grid((x + block.x - 1) / block.x, 1); 
    
    start = cpu_sec();
    mat_sum_g<<<grid, block>>>(da, db, dc, x, y);
    cudaDeviceSynchronize();
    duration = cpu_sec() - start;
    printf("Mat sum GPU <<<(%d, %d), (%d,%d)>>> time cost %f ms\n",
            grid.x, grid.y, block.x, block.y, duration * 1000);


    cudaMemcpy(gpu, dc, num, cudaMemcpyDeviceToHost);

    check_num(cpu, gpu, n);

    cudaFree(da);
    cudaFree(db);
    cudaFree(dc);

    free(ha);
    free(hb);
    free(cpu);
    free(gpu);

    cudaDeviceReset();
    int c = getchar();

    return 0;
}

