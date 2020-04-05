#include <stdio.h>
#include <time.h>
#include <stdint.h>
#include <Windows.h>

#ifndef _UTILS_H
#define _UTILS_H

#define CHECK(call)\
{\
    const cudaError_t err = call;\
    if(err != cudaSuccess)\
    {\
        printf("Error: %s, %d\n", __FILE__, __LINE__);\
        printf("Code: %d, reason %s\n", err, cudaGetErrorString(err));\
        exit(0);\
    }\
}



int gettimeofday(struct timeval * tp, struct timezone *tzp)
{
	static const uint64_t EPOCH = ((uint64_t) 116444736000000000ULL);

	SYSTEMTIME system_time;
	FILETIME file_time;
	uint64_t time;

	GetSystemTime(&system_time);
	SystemTimeToFileTime(&system_time, &file_time);
	time = ((uint64_t)file_time.dwLowDateTime);
	time += ((uint64_t)file_time.dwHighDateTime) << 32;

	tp->tv_sec = (long)((time - EPOCH) / 10000000L);
	tp->tv_usec = (long)(system_time.wMilliseconds * 1000);
	return 0;
}



inline double seconds()
{
    struct timeval tp;
    int i = gettimeofday(&tp, NULL);
    return ((double)tp.tv_sec + (double)tp.tv_usec * 1.e-6);
}

void init_int(int *inp, const int n)
{
    for(int i = 0; i < n; i++)
        inp[i] = (int)(rand() & 0xFF);
}


void init_float(float *inp, const int n)
{
    for(int i = 0; i < n; i++)
        inp[i] = (float)(rand() & 0xFF) / 10.0f;
}


void check_float(float *c, float *g, const int n)
{
    int epsilon = 1.e-8;
    int match = 1;
    for(int i = 0; i < n; i++)
    {
        if(abs(c[i] - g[i]) > epsilon)
        {
            match = 0;
            printf("cpu %f, gpu %f\n", c[i], g[i]);
            break;
        }
    }
    if(match)
        printf("Match\n");
    else
        printf("Don't Match\n");
}

void check_int(int *c, int *g, const int n)
{
    int match = 1;
    for (int i = 0; i <n; i++)
    {
        if (c[i] != g[i])
        {
            match = 0;
            printf("cpu %d, gpu %d", c[i], g[i]);
            break;
        }
    }
    if (match)
        printf("Match\n");
    else
        printf("Don't match\b");
}


#endif // _UTILS_H
