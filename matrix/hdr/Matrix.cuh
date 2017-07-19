#ifndef __MATRIX_H__
#define __MATRIX_H__

#include <random>
#include <algorithm>
#include <math.h>
#include <stdlib.h>

#include <Common.h>
#include <PrintLog.h>

#define E_PI 3.1415926535897932384626433832795028841971693993751058209749445923078164062

#define htod(dst, src, size)\
	    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)
#define dtoh(dst, src, size)\
	    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)
#define dtod(dst, src, size)\
	    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)

__global__ void MatAdd(float *C, float *A, float *B)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	C[index] = A[index] + B[index];
	//printf("C[%d]=%f, A[%d]=%f, B[%d]=%f\n",
	//	index, C[index], index, A[index], index, B[index]);
}

__global__ void MatSub(float *C, float *A, float *B)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	C[index] = A[index] - B[index];
}

__global__ void MatMul(float *C, float *A, float *B, int width, int height)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int size = gridDim.x * blockDim.x;
	int rows = size / height;
	int i = index % rows;
	int j = index / rows;
	float cValue = 0;

	for (int e = 0; e < width; e++)
	{
		int a = width * j + e;
		int b = height * e + i;
		//printf("C[%d], A[%d]=%f, B[%d]=%f\n", index, a, A[a], b, B[b]);
		cValue += A[a] * B[b];
	}
	//printf("C[%d]=%f\n", index, cValue);

	C[index] = cValue;
}

__global__ void MatAddC(float *C, float *A, float f)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	C[index] = A[index] + f;
	//printf("C[%d]=%f, A[%d]=%f, f=%f\n", index, C[index], index, A[index], f);
}

__global__ void MatSubC(float *C, float *A, float f)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	C[index] = A[index] - f;
}

__global__ void MatMulC(float *C, float *A, float f)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	C[index] = A[index] * f;
}

__global__ void MatArgu(float *C, float *A, float *B, int width, int height, int AWidth)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int size = gridDim.x * blockDim.x;
	int rows = size / height;
	int x = index % rows;	// colId
	int y = index / rows;	// rowId

	if (x < AWidth)
	{
		C[width * y + x] = A[AWidth * y + x];
	}
	else
	{
		C[width * y + x] = B[(width - AWidth) * y + x - AWidth];
	}
}

__global__ void MatTrans(float *C, float *A, int width, int height)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int size = gridDim.x * blockDim.x;
	int rows = size / height;
	int i = index % rows;
	int j = index / rows;

	int a = width * j + i;
	int c = height * i + j;
	C[c] = A[a];
}

__global__ void checkValidInverse(float *d_AI, int width, int height)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int size = gridDim.x * blockDim.x;
	int rows = size / height;
	int x = index % rows;	// colId
	int y = index / rows;	// rowId
}

__global__ void GaussJordan(float *d_AI, int width, int height, int rowId)
{
	int index = blockIdx.x * blockDim.x + threadIdx.x;
	int size = gridDim.x * blockDim.x;
	int rows = size / height;
	int x = index % rows;	// colId
	int y = index / rows;	// rowId
	float Aii = d_AI[rowId * width + rowId];
	if (Aii == 0)
		return;

	int ii = width * y + x;
	d_AI[ii] /= Aii;

	//__syncthreads();

	/*
	for (int i = 0; i < height; i++)
	{
		int tmp = width * i + x;
		if (i == rowId || d_AI[tmp] == 0)
			continue;

		for (int j = 0; j < width / 2 + 1; j++)
		{
			int tmp2 = width * i + j;
			d_AI[tmp2] -= d_AI[width * rowId + j];
		}
	}
	*/
}

class Matrix
{
	float *h_value = NULL;
	float *d_value = NULL;
	int size;
	int grid;
	int block;
	unsigned int dimx;
	unsigned int dimy;
public:
	unsigned int getDimX() { return this->dimx; }
	unsigned int getDimY() { return this->dimy; }
	float* getMatrix() { return this->h_value; }

	~Matrix()
	{
		if (this->h_value != NULL)
			free(this->h_value);

		if (this->d_value != NULL)
			cudaFree(this->d_value);
	}

	void setGridBlock()
	{
		int min = this->size / sizeof(float);
		int num = min;
		for (this->block = 0; this->block <= sqrt(num);)
		{
			this->block++;
			if (num % this->block == 0)
			{
				if ( abs(num / this->block - this->block) < min)
				{
					this->grid = num / this->block;
					min = abs(this->grid - this->block);
				}
			}
		}
		this->block = num / this->grid;
	}

	void initMatrix(float *value, int dimx, int dimy)
	{
		this->size = dimx * dimy * sizeof(float);
		
		if (this->h_value != NULL)
			free(this->h_value);

		if (this->d_value != NULL)
			cudaFree(this->d_value);

		if (this->size == 0)
		{
			this->dimx = 0;
			this->dimy = 0;
			return;
		}

		setGridBlock();

		this->dimx = dimx;
		this->dimy = dimy;
		this->h_value = (float*) malloc (this->size); 
		cudaMalloc(&this->d_value, this->size);

		if (value == NULL)
			zero();
		else
			memcpy(this->h_value, value, this->size);

		htod(this->d_value, this->h_value, this->size);
	}

	Matrix()
	{
		initMatrix(NULL, 0, 0);
	}

	Matrix(int dim)
	{
		initMatrix(NULL, dim, dim);
	}

	Matrix(int dimx, int dimy)
	{
		initMatrix(NULL, dimx, dimy);
	}
	
	
	Matrix(float *value, int dimx, int dimy)
	{
		initMatrix(value, dimx, dimy);
	}

	Matrix(Matrix *mat)
	{
		initMatrix(mat->h_value, mat->dimx, mat->dimy);
	}

	Matrix(Matrix &mat)
	{
		initMatrix(mat.h_value, mat.dimx, mat.dimy);
	}

	Matrix zero()
	{
		if (dimx < 1 || dimy < 1)
			DEBUG(DEBUG_ERR, "0 invalid size\n");
		else
		{
			for (int i = 0; i < dimy; i++)
			{
				for (int j = 0; j < dimx; j++)
				{
					int x = i * dimx + j;
					this->h_value[x] = 0;
				}
			}
			htod(this->d_value, this->h_value, this->size);
		}
		return *this;
	}
	
	Matrix identity()
	{
		if (dimx < 1 || dimy < 1)
			DEBUG(DEBUG_ERR, "I invalid size\n");
		else
		{
			for (int i = 0; i < dimy; i++)
			{
				for (int j = 0; j < dimx; j++)
				{
					int x = i * dimx + j;
					if (i == j)
						this->h_value[x] = 1;
					else
						this->h_value[x] = 0;
				}
			}
			htod(this->d_value, this->h_value, this->size);
		}
		return *this;
	}

	Matrix operator=(const Matrix &m)
	{
		if (m.dimx != this->dimx ||
			m.dimy != this->dimy)
		{
			DEBUG(DEBUG_ERR, "= invalid size %d %d %d %d\n",
					this->dimx, this->dimy, m.dimx, m.dimy);
		}
		else
		{
			dtod(this->d_value, m.d_value, this->size);
		}
		return *this;
	}

	Matrix operator+(const Matrix &m)
	{
		unsigned int x1 = this->getDimX();
		unsigned int y1 = this->getDimY();
		unsigned int x2 = m.dimx;
		unsigned int y2 = m.dimy;


		if (x1 != x2 ||	y1 != y2)
		{
			DEBUG(DEBUG_ERR, "+ invalid size %d %d %d %d\n",
					x1, y1, x2, y2);
			return *this;
		}
		else
		{
			Matrix ret(this->dimx, this->dimy);
			MatAdd<<<ret.grid, ret.block>>>
				(ret.d_value, this->d_value, m.d_value);
			dtoh(ret.h_value, ret.d_value, ret.size);
			return ret;
		}
	}
	
	Matrix operator-(const Matrix &m)
	{
		unsigned int x1 = this->getDimX();
		unsigned int y1 = this->getDimY();
		unsigned int x2 = m.dimx;
		unsigned int y2 = m.dimy;


		if (x1 != x2 ||	y1 != y2)
		{
			DEBUG(DEBUG_ERR, "- invalid size %d %d %d %d\n",
					x1, y1, x2, y2);
			return *this;
		}
		else
		{
			Matrix ret(x1, y1);
			MatSub<<<ret.grid, ret.block>>>(ret.d_value, this->d_value, m.d_value);
			dtoh(ret.h_value, ret.d_value, ret.size);
			return ret;
		}
	}
	
	Matrix operator*(const Matrix &mat)
	{
		unsigned int x1 = this->getDimX();
		unsigned int y1 = this->getDimY();
		unsigned int x2 = mat.dimx;
		unsigned int y2 = mat.dimy;

		if (x1 != y2)
		{
			DEBUG(DEBUG_ERR, "* invalid size %d %d %d %d\n",
					x1, y1, x2, y2);

			return *this;
		}
		else
		{
			Matrix ret(x2, x2);
			MatMul<<<ret.grid, ret.block>>>
				(ret.d_value, this->d_value, mat.d_value, this->dimx, this->dimy);
			dtoh(ret.h_value, ret.d_value, ret.size);
			return ret;
		}
	}
	Matrix operator+(const float f)
	{
		Matrix ret(this->dimx, this->dimy);
		MatAddC<<<ret.grid, ret.block>>>
			(ret.d_value, this->d_value, f);

		return ret;
	}

	Matrix operator-(const float f)
	{
		Matrix ret(this->dimx, this->dimy);
		MatSubC<<<ret.grid, ret.block>>>
			(ret.d_value, this->d_value, f);

		return ret;
	}

	Matrix operator*(const float f)
	{
		Matrix ret(this->dimx, this->dimy);
		MatMulC<<<ret.grid, ret.block>>>
			(ret.d_value, this->d_value, f);

		return ret;
	}

	Matrix operator|(const Matrix &m)
	{
		unsigned int x1 = this->getDimX();
		unsigned int y1 = this->getDimY();
		unsigned int x2 = m.dimx;
		unsigned int y2 = m.dimy;


		if (y1 != y2)
		{
			DEBUG(DEBUG_ERR, "| invalid size %d %d %d %d\n",
					x1, y1, x2, y2);
			return *this;
		}
		else
		{
			Matrix ret(x1 + x2, y1);
			MatArgu<<<ret.grid, ret.block>>>
				(ret.d_value, this->d_value, m.d_value, ret.dimx, ret.dimy, this->dimx);
			dtoh(ret.h_value, ret.d_value, ret.size);
			return ret;
		}
	}


	Matrix transpose()
	{
		Matrix ret(this->dimy, this->dimx);
		
		MatTrans<<<this->grid, this->block>>>
			(ret.d_value, this->d_value, this->dimx, this->dimy);

		return ret;
	}

	Matrix GaussJordanInverse()
	{
		Matrix ret(this->dimx, this->dimy);

		// argument matrix A|I
		Matrix tmp(this->dimx * 2, this->dimy);
		Matrix I(this->dimx, this->dimy);
		I = I.identity();
		tmp = *this | I;

		for (int i = 0; i < tmp.dimx; i++)
		{
			GaussJordan<<<tmp.grid, tmp.block>>>
				(tmp.d_value, tmp.dimx, tmp.dimy, i);
		}

		return ret;
	}


	/*
	Matrix inverse()
	{
		Matrix ret(this->dimy, this->dimx);
		
		for (int i = 0; i < this->dimx; i++)
		{
			GaussJordan<<<this->grid, this->block, this->dimx>>>
				(ret.d_value, this->d_value, this->dimx, this->dimy, i);
		}

		dtoh(ret.h_value, ret.d_value, ret.size);
		return ret;
	}
	*/


	Matrix Cholesky(float ztol = 1.0e-5)
	{
		Matrix ret(this->dimx, this->dimy);

		int x;
		for (int i = 0; i < this->dimx; i++)
		{
			float s = 0;	
			for (int k = 0; k < i; k++)
			{
				x = ret.dimx * k + i;
				s += pow(ret.h_value[x], 2);
			}
			x = ret.dimx * i + i;
			float d = this->h_value[x] - s;

			if (abs(d) < ztol)
			{
				x = ret.dimx * i + i;
				ret.h_value[x] = 0;
			}
			else
			{
				if (d < 0.0)
				{
					DEBUG(DEBUG_INFO, "matrix not positive-definite\n");
					return ret;
				}
				x = ret.dimx * i + i;
				ret.h_value[x] = sqrt(d);
			}

			for (int j = i+1; j < this->dimx; j++)
			{
				float s = 0;
				for (int k = 0; k < i; k++)
				{
					x = ret.dimx * k + i;
					int x2 = ret.dimx * k + j;
					s += ret.h_value[x] * ret.h_value[x2];
				}
				if (abs(s) < ztol)
					s = 0;

				x = ret.dimx * i + i;
				int x2 = ret.dimx * i + j;
				ret.h_value[x2] = (this->h_value[x2] - s) / ret.h_value[x];
			}
		}
		htod(ret.d_value, ret.h_value, ret.size);
		return ret;
	}
	
	Matrix CholeskyInverse()
	{
		Matrix ret(this->dimx, this->dimy);
		int x;
		
		for (int j = this->dimx - 1;  j >= 0; j--)
		{
			x = ret.dimx * j + j;
			float tjj = this->h_value[x];
			float s = 0;
			for (int k = j+1; k < this->dimx; k++)
			{
				x = ret.dimx * j + k;
				s += this->h_value[x] * ret.h_value[x];
			}
			x = ret.dimx * j + j;
			ret.h_value[x] = 1.0/pow(tjj, 2) - s / tjj;

			for (int i = j - 1;  i >= 0; i--)
			{
				float s = 0; 
				for (int k = i+1; k < this->dimx; k++)
				{
					x = ret.dimx * i + k;
					int x2 = ret.dimx * k + j;
					s += this->h_value[x] * ret.h_value[x2];
				}
				x = ret.dimx * j + i;
				int x2 = ret.dimx * i + j;
				int x3 = ret.dimx * i + i;
				ret.h_value[x] = ret.h_value[x2] = -s / this->h_value[x3];
			}
		}
		htod(ret.d_value, ret.h_value, ret.size);
		return ret;
	}

	Matrix inverse()
	{
		dtoh(this->h_value, this->d_value, this->size);
		Matrix ret(this->dimx, this->dimy);
		Matrix aux(this->dimx, this->dimy);
		aux = this->Cholesky();
		aux.dtohSync();
		ret = aux.CholeskyInverse();
		ret.dtohSync();
		return ret;
	}

	void dtohSync()
	{
		dtoh(this->h_value, this->d_value, this->size);
	}

	void htodSync()
	{
		htod(this->d_value, this->h_value, this->size);
	}

	void printMatrix()
	{
		dtoh(this->h_value, this->d_value, this->size);
		DEBUG(DEBUG_INFO, "[%d X %d]\n", this->grid, this->block);
		for (int i = 0; i < this->dimy; i++)
		{
			for (int j = 0; j < this->dimx; j++)
			{
				int x = i * this->dimx + j;
				DEBUG(DEBUG_INFO, "%10.5f ", this->h_value[x]);
			}
			DEBUG(DEBUG_INFO, "\n");
		}
		DEBUG(DEBUG_INFO, "\n");
	}
};

#endif
