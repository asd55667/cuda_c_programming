#include <stdlib.h>
#include <string.h>
#include <time.h>

void sumArrayOnHost(float *A, float *B, float *C, const int N)
{
	for (int i = 0; i < N; i++)
	{
		C[i] = A[i] + B[i];
	}
}

void initData(float *ip, int size)
{
	time_t t;
	srand((unsigned int) time(&t));

	for (int i = 0; i< size; i++)
	{
		ip[i] = (float)( rand() & 0xFF) / 10.0f;
	}
}

int main(int argc, char **argv)
{
	int nElem = 1024;
	size_t nBytes = nElem * sizeof(float);

	float *h_A, *h_B, *h_C;
	h_A = (float *)malloc(nBytes);
	h_B = (float *)malloc(nBytes);
	h_C = (float *)malloc(nBytes);

	initData(h_A, nElem);
	initData(h_B, nElem);

	sumArrayOnHost(h_A, h_B, h_C, nElem);

	free(h_A);
	free(h_B);
	free(h_C);

	int c = getchar();
	return 0;
}
