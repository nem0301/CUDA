#include <stdio.h>
#include <memory.h>

#include <PrintLog.h>

#include <Matrix.cuh>
#include <Timer.cuh>

int main()
{
	int size = 2;
	int size2 = 2;
	float data[size * size2];

	int k[] = { 2, 2, 3, 4 };
	for (int i = 0; i < size2; i++)
	{
		for (int j = 0; j < size; j++)
		{
			int x = size * i + j;
			data[x] = k[size * i + j];
		}
	}

	Matrix mat1(data, size, size2);

	int t[] = { 5, 6, 7, 8 };
	for (int i = 0; i < size2; i++)
	{
		for (int j = 0; j < size; j++)
		{
			int x = size * i + j;
			data[x] = t[size * i + j];
		}
	}
	mat1.printMatrix();

	/*
	for (int i = 0; i < size2; i++)
	{
		for (int j = 0; j < size; j++)
		{
			int x = size * i + j;
			data[x] = i;
		}
	}
	Matrix mat1(data, size, size2);
	*/

	Matrix mat2(data, size, size2);
	Matrix mat3(size, size2);

	Timer tt;

	tt.start();
	mat3 = mat1.inverse();
	DEBUG(DEBUG_INFO, "%f ms\n", tt.stop());
	mat3.printMatrix();

	tt.start();
	mat3 = mat1.GaussJordanInverse();
	DEBUG(DEBUG_INFO, "%f ms\n", tt.stop());
	mat3.printMatrix();



	//mat1.printMatrix();
	//mat2.printMatrix();
	//mat3.printMatrix();

	return 0;
}
