#include <stdio.h>
#include <memory.h>

#include <Matrix.cu>


int main()
{
	int size = 1024;
	int size2 = 1024;
	float data[size * size2];

	//int k[] = { 2, 2, 3, 4 };
	for (int i = 0; i < size2; i++)
	{
		for (int j = 0; j < size; j++)
		{
			int x = size * i + j;
			data[x] = i;
		}
	}

	Matrix mat1(data, size, size2);

	//int t[] = { 5, 6, 7, 8 };
	for (int i = 0; i < size2; i++)
	{
		for (int j = 0; j < size; j++)
		{
			int x = size * i + j;
			data[x] = i;
		}
	}

	Matrix mat2(data, size, size2);
	Matrix mat3(size, size2);

	cudaEvent_t start, stop;
	cudaEventCreate(&start);
	cudaEventCreate(&stop);
	cudaEventRecord(start, 0);

	mat1.inverse();

	

	cudaEventRecord(stop, 0);
	cudaEventSynchronize(stop);
	float elapse;
	cudaEventElapsedTime(&elapse, start, stop);
	printf("%.6f ms\n", elapse);
	cudaEventDestroy(start);
	cudaEventDestroy(stop);


	//mat1.printMatrix();
	//mat2.printMatrix();
	//mat3.printMatrix();
	

	return 0;
}
