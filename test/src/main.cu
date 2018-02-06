#include <stdio.h>
#include <iostream>

using namespace std;

#define N 8
#define K 8

__global__ void test(float* d_in, float* d_out)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x;
	int j = __shfl_sync(0xffffffff, d_in[i], 5, 8);
	d_out[i] = j;
}
int main(int argc, char **argv)
{
	float *in = (float *) malloc(N * sizeof(float));
	float *out = (float *) malloc(N * sizeof(float));
	float *gold = (float *) malloc(N * sizeof(float));

	for (int i= 0; i < N; i++)
	{
		in[i] = i + 8;
	}

	float *d_in, *d_out;

	cudaMalloc(&d_in, N * sizeof(float));
	cudaMalloc(&d_out, N * sizeof(float));
	cudaMemcpy(d_in, in, N * sizeof(float), cudaMemcpyHostToDevice);

	dim3 blocks(N/K);
	dim3 threads(K);

	test<<<blocks,threads>>>(d_in, d_out);
	cudaMemcpy(out, d_out, N * sizeof(float), cudaMemcpyDeviceToHost);
	cudaMemcpy(in, d_in, N * sizeof(float), cudaMemcpyDeviceToHost);

	for (int i = 0; i < N; i++)
	{
		cout << i << " : " << out[i] << endl;
	}

	cout << endl;

	for (int i = 0; i < N; i++)
	{
		cout << i << " : " << in[i] << endl;
	}

	cudaFree(d_in);
	cudaFree(d_out);
	return 0;
}
