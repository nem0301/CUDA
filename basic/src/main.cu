#include <stdio.h>
#include <time.h>

#define ROW 10
#define COL 10
float ref[ROW][COL];

void initRef()
{
	float *tmp = (float*)ref;
	srand((unsigned int) time (NULL));
	for (int i = 0; i < ROW * COL; i++)
	{
		tmp[i] = (float)rand() / RAND_MAX;
	}
}

void printData(float *data)
{
	printf("printData\n");
	for (int i = 0; i < ROW; i++)
	{
		for(int j = 0; j < COL; j++)
		{
			float tmp = data[i * COL + j];
			printf("%8.6f ", tmp);
		}
		printf("\n");
	}
}
void printRef()
{
	printf("printRef\n");
	for (int i = 0; i < ROW; i++)
	{
		for(int j = 0; j < COL; j++)
		{
			printf("%8.6f ", ref[i][j]);
		}
		printf("\n");
	}
}

__global__ void map(float d_data)
{
}

void map()
{
}

void transpose()
{
}

void gather()
{
}

void scatter()
{
}

void stencil()
{
}

void reduce()
{
}

void scan()
{
}

void sort()
{
}


int main()
{
	initRef();


	return 0;
}

