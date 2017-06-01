#include <stdio.h>
#include <memory.h>

#include <PrintLog.h>

#include <Matrix.cuh>
#include <Timer.cuh>

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

	Timer tt;

	tt.start();
	mat1 = mat1 * mat2;
	printf("%f ms\n", tt.stop());

	//mat1.printMatrix();
	//mat2.printMatrix();
	//mat3.printMatrix();

	DEBUG(DEBUG_ERR, "test %d, %d\n", 1, 2);
	DEBUG(DEBUG_WARN, "test %d, %d\n", 1, 2);
	DEBUG(DEBUG_INFO, "test %d, %d\n", 1, 2);
	DEBUG(DEBUG_VERB, "test %d, %d\n", 1, 2);
	DEBUG(4, "test %d, %d\n", 1, 2);

	return 0;
}
