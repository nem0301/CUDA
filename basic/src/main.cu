#include <stdio.h>
#include <memory.h>
#include <time.h>

#define htod(dst, src, size)\
	    cudaMemcpy(dst, src, size, cudaMemcpyHostToDevice)
#define dtoh(dst, src, size)\
	    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToHost)
#define dtod(dst, src, size)\
	    cudaMemcpy(dst, src, size, cudaMemcpyDeviceToDevice)

#define ROW 8
#define COL 8
float ref[ROW][COL];

void initRef()
{
	float *tmp = (float*)ref;
	srand((unsigned int) time (NULL));
	for (int i = 0; i < ROW * COL; i++)
	{
		tmp[i] = i;//(float)rand() / RAND_MAX;
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
			printf("%9.3f ", tmp);
		}
		printf("\n");
	}
}

void printData2(float *data, int r, int c)
{
	printf("printData\n");
	for (int i = 0; i < r; i++)
	{
		for(int j = 0; j < c; j++)
		{
			float tmp = data[i * c + j];
			printf("%9.6f ", tmp);
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

__global__ void map(float *d_data, float *d_func)
{
	int width = gridDim.x * blockDim.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	d_data[y * width + x] += *d_func;
}

void map()
{
	float *d_mapped, *d_func;
	float *h_mapped, h_func;
	float tmp[ROW][COL];
	size_t size = sizeof(ref);
	memcpy(tmp, ref, size);
	h_mapped = (float*)tmp;

	printf("before : ");
	printData(h_mapped);

	cudaMalloc(&d_mapped, size);
	cudaMalloc(&d_func, sizeof(float));

	h_func = 1.5;
	htod(d_func, &h_func, 1);
	htod(d_mapped, h_mapped, size);

	
	int c = 2;
	dim3 blockSize = dim3(c, c);
	dim3 gridSize = dim3(COL / c, ROW / c);
	map<<<gridSize, blockSize>>>(d_mapped, d_func);

	dtoh(h_mapped, d_mapped, size);

	printf("after : ");
	printData(h_mapped);

	cudaFree(d_mapped);
}

__global__ void transpose(float *d_transposed, float *d_ref)
{
	int width = gridDim.x * blockDim.x;
	int height = gridDim.y * blockDim.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	

	d_transposed[x * height + y] = d_ref[y * width + x];
}

void transpose()
{
	float *d_transposed, *d_ref;
	float *h_transposed, *h_ref;
	float tmp[ROW][COL];
	size_t size = sizeof(ref);
	memcpy(tmp, ref, size);
	h_ref = (float*)tmp;
	h_transposed = (float*)tmp;

	printf("before : ");
	printData(h_ref);

	cudaMalloc(&d_transposed, size);
	cudaMalloc(&d_ref, size);

	htod(d_ref, h_ref, size);

	
	int c = 2;
	dim3 blockSize = dim3(c, c);
	dim3 gridSize = dim3(COL / c, ROW / c);
	transpose<<<gridSize, blockSize>>>(d_transposed, d_ref);

	dtoh(h_transposed, d_transposed, size);

	printf("after : ");
	printData(h_transposed);

	cudaFree(d_transposed);
	cudaFree(d_ref);
}

__global__ void gather(float *d_gathered, float *d_ref)
{
	int width = gridDim.x * blockDim.x;
	int height = gridDim.y * blockDim.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int cnt = 0;
	
	for (int yy = y - 1; yy < y + 2; yy++)
	{
		for (int xx = x - 1; xx < x + 2; xx++)
		{
			if (xx >= 0 && xx < width && yy >= 0 && yy < height)
			{
				d_gathered[y * width + x] += d_ref[yy * width + xx];
				cnt++;
			}
		}
	}
	d_gathered[y * width + x] /= cnt;
}

void gather()
{
	float *d_gathered, *d_ref;
	float *h_gathered, *h_ref;
	float tmp[ROW][COL];
	size_t size = sizeof(ref);
	memcpy(tmp, ref, size);
	h_ref = (float*)tmp;
	h_gathered = (float*)tmp;

	printf("before : ");
	printData(h_ref);

	cudaMalloc(&d_ref, size);
	cudaMalloc(&d_gathered, size);

	htod(d_ref, h_ref, size);
	cudaMemset(d_gathered, 0, size);
	
	int c = 2;
	dim3 blockSize = dim3(c, c);
	dim3 gridSize = dim3(COL / c, ROW / c);
	printf("run \n");
	gather<<<gridSize, blockSize>>>(d_gathered, d_ref);

	dtoh(h_gathered, d_gathered, size);

	printf("after : ");
	printData(h_gathered);

	cudaFree(d_gathered);
	cudaFree(d_ref);
}

__global__ void scatter(float *d_scattered, float *d_ref)
{
	int width = gridDim.x * blockDim.x;
	int height = gridDim.y * blockDim.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int cnt = 0;
	
	for (int yy = y - 1; yy < y + 2; yy++)
	{
		for (int xx = x - 1; xx < x + 2; xx++)
		{
			if (xx >= 0 && xx < width && yy >= 0 && yy < height)
			{
				d_scattered[yy * width + xx] += d_ref[y * width + x];
				cnt++;
			}
		}
	}
	d_scattered[y * width + x] /= cnt;
}

void scatter()
{
	float *d_scattered, *d_ref;
	float *h_scattered, *h_ref;
	float tmp[ROW][COL];
	size_t size = sizeof(ref);
	memcpy(tmp, ref, size);
	h_ref = (float*)tmp;
	h_scattered = (float*)tmp;

	printf("before : ");
	printData(h_ref);

	cudaMalloc(&d_ref, size);
	cudaMalloc(&d_scattered, size);

	htod(d_ref, h_ref, size);
	cudaMemset(d_scattered, 0, size);
	
	int c = 2;
	dim3 blockSize = dim3(c, c);
	dim3 gridSize = dim3(COL / c, ROW / c);
	printf("run \n");
	scatter<<<gridSize, blockSize>>>(d_scattered, d_ref);

	dtoh(h_scattered, d_scattered, size);

	printf("after : ");
	printData(h_scattered);

	cudaFree(d_scattered);
	cudaFree(d_ref);
}

__global__ void stencil(float *d_stenciled, float *d_ref, float *d_window)
{
	int width = gridDim.x * blockDim.x;
	int height = gridDim.y * blockDim.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	float cnt = 0;
	
	for (int yy = y - 1, yyy = 0; yy < y + 2; yy++, yyy++)
	{
		for (int xx = x - 1, xxx = 0; xx < x + 2; xx++, xxx++)
		{
			if (xx >= 0 && xx < width && yy >= 0 && yy < height)
			{
				d_stenciled[y * width + x] += 
					d_ref[yy * width + xx] *
					d_window[yyy * width + xxx];

				cnt += 1 * d_window[yyy * width + xxx];
			}
		}
	}
	d_stenciled[y * width + x] /= cnt;
}

void stencil()
{
	float *d_stenciled, *d_ref, *d_window;
	float *h_stenciled, *h_ref, *h_window;
	float tmp[ROW][COL];
	float window[3][3];
	size_t size = sizeof(ref);
	memcpy(tmp, ref, size);
	h_ref = (float*)tmp;
	h_stenciled = (float*)tmp;
	h_window = (float*)window;

	window[0][1] = 0.2;
	window[1][0] = 0.2;
	window[1][1] = 0.2;
	window[1][2] = 0.2;
	window[2][1] = 0.2;

	printf("before : ");
	printData(h_ref);

	cudaMalloc(&d_ref, size);
	cudaMalloc(&d_stenciled, size);
	cudaMalloc(&d_window, sizeof(window));

	htod(d_ref, h_ref, size);
	htod(d_window, h_window, sizeof(window));
	cudaMemset(d_stenciled, 0, size);
	
	int c = 2;
	dim3 blockSize = dim3(c, c);
	dim3 gridSize = dim3(COL / c, ROW / c);
	printf("run \n");
	stencil<<<gridSize, blockSize>>>(d_stenciled, d_ref, d_window);

	dtoh(h_stenciled, d_stenciled, size);

	printf("after : ");
	printData(h_stenciled);

	cudaFree(d_stenciled);
	cudaFree(d_ref);
}

__global__ void reduce(float *d_reduced, float *d_ref)
{
	int width = gridDim.x * blockDim.x;
	//int height = gridDim.y * blockDim.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int xx = blockIdx.x * blockDim.x;
	int yy = blockIdx.y * blockDim.y;
	__shared__ float data[2][2];
	
	data[threadIdx.y][threadIdx.x] = d_ref[y * width + x];

	if (yy == y && xx == x)
	{
		for (int i = 0; i < 4; i++)
		{
			d_reduced[yy/2 * (width/2) + xx/2] += ((float*)data)[i];
		}
	}
}

void reduce()
{
	float *d_reduced, *d_ref;
	float *h_reduced, *h_ref;
	float tmp[ROW][COL];
	size_t size = sizeof(ref);
	memcpy(tmp, ref, size);
	h_ref = (float*)tmp;
	h_reduced = (float*) malloc (sizeof(float) * size);

	printf("before : ");
	printData(h_ref);

	cudaMalloc(&d_ref, size);
	cudaMalloc(&d_reduced, size/4);

	htod(d_ref, h_ref, size);
	cudaMemset(d_reduced, 0, size/4);

	for (int i = ROW/2; i > 0; i /= 2)
	{
		dim3 blockSize = dim3(2, 2);
		dim3 gridSize = dim3(i, i);
		reduce<<<gridSize, blockSize>>>(d_reduced, d_ref);

		printf("before : ");
		free(h_reduced);
		h_reduced = (float*) malloc (size/4);
		dtoh(h_reduced, d_reduced, size/4);
		printData2(h_reduced, i, i);

		if (i == 1) break;

		size /= 4;
		cudaFree(d_ref);
		cudaMalloc(&d_ref, size);
		dtod(d_ref, d_reduced, size);

		cudaFree(d_reduced);
		cudaMalloc(&d_reduced, size/4);
		cudaMemset(d_reduced, 0, size/4);
	}

	dtoh(h_reduced, d_reduced, 1);

	printf("%f\n", *h_reduced);

	free(h_reduced);
	cudaFree(d_reduced);
	cudaFree(d_ref);

}

__global__ void hills_steele(float *d_scaned, float *d_ref, int step)
{
	int width = gridDim.x * blockDim.x;
	//int height = gridDim.y * blockDim.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y * width + x;

	if (index < step)
	{
		d_scaned[index] = d_ref[index];
	}
	else
	{
		d_scaned[index] = d_ref[index] + d_ref[index - step];
	}
}

// step efficiency, inclusive
void Hills_Steele_scan()
{
	float *d_scaned, *d_ref;
	float *h_scaned, *h_ref;
	float tmp[ROW][COL];
	size_t size = sizeof(ref);
	memcpy(tmp, ref, size);
	h_ref = (float*)tmp;
	h_scaned = (float*) malloc (sizeof(float) * size);

	printf("before : \n");
	printData(h_ref);

	cudaMalloc(&d_ref, size);
	cudaMalloc(&d_scaned, size);

	htod(d_ref, h_ref, size);
	cudaMemset(d_scaned, 0, size);

	for (int i = 1; i < size/2; i *= 2)
	{
		dim3 blockSize = dim3(2, 2);
		dim3 gridSize = dim3(ROW/2, COL/2);

		hills_steele<<<gridSize, blockSize>>>(d_scaned, d_ref, i);
		dtod(d_ref, d_scaned, size);
	}
	dtoh(h_scaned, d_scaned, size);
	printData(h_scaned);

	cudaFree(d_ref);
	cudaFree(d_scaned);
	free(h_scaned);
}

__global__ void blelloch(float *d_scaned, float *d_ref, int step)
{
	int width = gridDim.x * blockDim.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y * width + x;

	if ((index % step) == (step - 1))
	{
		d_scaned[index] = d_ref[index] + d_ref[index - (step/2)];
	}
	else
	{
		d_scaned[index] = d_ref[index];
	}
}

__global__ void blelloch_post(float *d_scaned, float *d_ref, int step)
{
	int width = gridDim.x * blockDim.x;
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	int index = y * width + x;

	if ((index % step) == (step - 1))
	{
		d_scaned[index] = d_ref[index] + d_ref[index - (step/2)];
		d_scaned[index - (step/2)] = d_ref[index];
	}
}

// work efficiency, exclusive
void Blelloch_scan()
{
	float *d_scaned, *d_ref;
	float *h_scaned, *h_ref;
	float tmp[ROW][COL];
	size_t size = sizeof(ref);
	memcpy(tmp, ref, size);
	h_ref = (float*)tmp;
	h_scaned = (float*) malloc (sizeof(float) * size);

	printData(h_ref);

	cudaMalloc(&d_ref, size);
	cudaMalloc(&d_scaned, size);

	htod(d_ref, h_ref, size);
	cudaMemset(d_scaned, 0, size);

	dim3 blockSize = dim3(2, 2);
	dim3 gridSize = dim3(ROW/2, COL/2);

	for (int i = 2; i < ROW * COL; i *= 2)
	{
		blelloch<<<gridSize, blockSize>>>(d_scaned, d_ref, i);

		dtoh(h_scaned, d_scaned, size);
		printData(h_scaned);
		dtod(d_ref, d_scaned, size);
	}

	dtoh(h_ref, d_scaned, size);
	h_ref[(size/sizeof(float))-1] = 0;
	htod(d_ref, h_ref, size);
	dtod(d_scaned, d_ref, size);

	for (int i = ROW * COL; i > 1; i /= 2)
	{
		blelloch_post<<<gridSize, blockSize>>>(d_scaned, d_ref, i);

		dtoh(h_scaned, d_scaned, size);
		printData(h_scaned);
		dtod(d_ref, d_scaned, size);
	}


	cudaFree(d_ref);
	cudaFree(d_scaned);
	free(h_scaned);
}

__global__ void mapBits(
		unsigned int* d_ref,
		unsigned int* d_inputVals,
		unsigned int bits,
		unsigned int val)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if ((d_inputVals[index] & bits) != 0)
		d_ref[index] = val;
	else 
		d_ref[index] = 0;
}

__global__ void mapBitsRev(
		unsigned int* d_ref,
		unsigned int* d_inputVals,
		unsigned int bits,
		unsigned int val)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
	if ((d_inputVals[index] & bits) != 0)
		d_ref[index] = 0;
	else 
		d_ref[index] = val;
}

__global__ void sumScan(
		unsigned int *d_scanned,
		unsigned int *d_ref, 
		int step)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;
		

	if (index < step)
	{
		d_scanned[index] = d_ref[index];
	}
	else
	{
		d_scanned[index] = d_ref[index] + d_ref[index - step];
	}
}
__global__ void blelloch(
		unsigned int *d_scanned,
		unsigned int *d_ref, 
		int step)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if ((index % step) == (step - 1))
		d_scanned[index] = d_ref[index] + d_ref[index - (step/2)];
	else
		d_scanned[index] = d_ref[index];

}

__global__ void blellochPost(
		unsigned int *d_scanned,
		unsigned int *d_ref, 
		int step)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if ( (index % step) == (step - 1))
	{
		d_scanned[index] = d_ref[index] + d_ref[index - (step/2)];
		d_scanned[index - (step/2)] = d_ref[index];
	}
}

__global__ void getOffset(
		unsigned int *d_offset,
		unsigned int *d_0sum,
		unsigned int *d_1sum,
		unsigned int *d_map)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	if (d_map[index] == 1)
		d_offset[index] = d_1sum[index];
	else
		d_offset[index] = d_0sum[index];
}

__global__ void getNewIndices(
		unsigned int *d_newIndices,
		unsigned int *d_offset,
		unsigned int *d_histPSum)
{
	int index = blockDim.x * blockIdx.x + threadIdx.x;

	d_newIndices[index] = d_offset[index] + d_histPSum[index];
}



#define numberOfSort 16
#define num 14
void radixSort()
{
	const unsigned int h_input[16] = {
		5, 10, 0, 123, 12, 3, 11, 99, 11, 20, 2, 133, 27, 67, 0, 0
	};

	unsigned int* d_ref;
	unsigned int* d_map;
	unsigned int* d_sumScaned;
	unsigned int* d_histPSum;
	unsigned int* d_0sum;
	unsigned int* d_1sum;
	unsigned int* d_offset;
	unsigned int* d_newIndices;
	cudaMalloc(&d_ref, sizeof(unsigned int) * numberOfSort);
	cudaMalloc(&d_map, sizeof(unsigned int) * numberOfSort);
	cudaMalloc(&d_sumScaned, sizeof(unsigned int) * numberOfSort);
	cudaMalloc(&d_histPSum, sizeof(unsigned int) * numberOfSort);
	cudaMalloc(&d_0sum, sizeof(unsigned int) * numberOfSort);
	cudaMalloc(&d_1sum, sizeof(unsigned int) * numberOfSort);
	cudaMalloc(&d_offset, sizeof(unsigned int) * numberOfSort);
	cudaMalloc(&d_newIndices, sizeof(unsigned int) * numberOfSort);

	unsigned int* h_ref = (unsigned int*) malloc (sizeof(unsigned int) * numberOfSort);
	unsigned int* h_map = (unsigned int*) malloc (sizeof(unsigned int) * numberOfSort);
	unsigned int* h_sumScaned = (unsigned int*) malloc (sizeof(unsigned int) * numberOfSort);
	unsigned int* h_histPSum = (unsigned int*) malloc (sizeof(unsigned int) * numberOfSort);
	unsigned int* h_0sum = (unsigned int*) malloc (sizeof(unsigned int) * numberOfSort);
	unsigned int* h_1sum = (unsigned int*) malloc (sizeof(unsigned int) * numberOfSort);
	unsigned int* h_offset = (unsigned int*) malloc (sizeof(unsigned int) * numberOfSort);
	unsigned int* h_newIndices = (unsigned int*) malloc (sizeof(unsigned int) * numberOfSort);

	dim3 gridSize = dim3(4);
	dim3 blockSize = dim3(4);


	for (unsigned int bits = 0; bits < 32; bits++)
	{
		memcpy(h_ref, h_input, sizeof(unsigned int) * numberOfSort);
		cudaMemcpy(d_ref, h_ref, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyHostToDevice);

		mapBits<<<gridSize, blockSize>>>(d_map, d_ref, 1 << bits, 1);

		cudaMemcpy(h_sumScaned, d_map, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToHost);

		for (int i = 0; i < numberOfSort; i++)
		{
			printf("%u\n", h_sumScaned[i]);
		}

		cudaMemcpy(d_ref, d_map, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToDevice);

		// sum scan
		for (unsigned int i = 1; i <= numberOfSort; i*=2)
		{
			sumScan<<<gridSize, blockSize>>>(d_sumScaned, d_ref, i);
			cudaMemcpy(d_ref, d_sumScaned, 
					sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToDevice);
		}

		cudaMemcpy(h_sumScaned, d_sumScaned, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToHost);

		cudaMemcpy(d_ref, h_ref, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyHostToDevice);
		mapBits<<<gridSize, blockSize>>>(
				d_histPSum, d_ref, 1 << bits, numberOfSort - h_sumScaned[num-1]);
		cudaMemcpy(h_histPSum, d_histPSum,
				sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToHost);


		// 0 sum
		memcpy(h_ref, h_input, sizeof(unsigned int) * numberOfSort);
		cudaMemcpy(d_ref, h_ref, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyHostToDevice);
		mapBitsRev<<<gridSize, blockSize>>>(d_ref, d_ref, (1<<bits), 1);
		cudaMemset(d_0sum, 0, sizeof(unsigned int) * numberOfSort);
		for (int i = 2; i < numberOfSort ; i *= 2)
		{
			blelloch<<<gridSize, blockSize>>>(d_0sum, d_ref, i);
			cudaMemcpy(d_ref, d_0sum, 
					sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToDevice);
		}

		cudaMemcpy(h_ref, d_0sum,
				sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToHost);
		h_ref[numberOfSort-1] = 0;
		cudaMemcpy(d_ref, h_ref,
				sizeof(unsigned int) * numberOfSort, cudaMemcpyHostToDevice);
		cudaMemcpy(d_0sum, d_ref, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToDevice);

		for (int i = numberOfSort; i > 1; i /= 2)
		{
			blellochPost<<<gridSize, blockSize>>>(d_0sum, d_ref, i);
			cudaMemcpy(d_ref, d_0sum, 
					sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToDevice);
		}
		cudaMemcpy(h_0sum, d_0sum, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToHost);

		// 1 sum
		memcpy(h_ref, h_input, sizeof(unsigned int) * numberOfSort);
		cudaMemcpy(d_ref, h_ref, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyHostToDevice);
		mapBits<<<gridSize, blockSize>>>(d_ref, d_ref, (1<<bits), 1);
		cudaMemset(d_1sum, 0, sizeof(unsigned int) * numberOfSort);
		for (int i = 2; i < numberOfSort ; i *= 2)
		{
			blelloch<<<gridSize, blockSize>>>(d_1sum, d_ref, i);
			cudaMemcpy(d_ref, d_1sum, 
					sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToDevice);
		}

		cudaMemcpy(h_ref, d_1sum,
				sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToHost);
		h_ref[numberOfSort-1] = 0;
		cudaMemcpy(d_ref, h_ref,
				sizeof(unsigned int) * numberOfSort, cudaMemcpyHostToDevice);
		cudaMemcpy(d_1sum, d_ref, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToDevice);

		for (int i = numberOfSort; i > 1; i /= 2)
		{
			blellochPost<<<gridSize, blockSize>>>(d_1sum, d_ref, i);
			cudaMemcpy(d_ref, d_1sum, 
					sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToDevice);
		}
		cudaMemcpy(h_1sum, d_1sum, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToHost);
		
		// offset
		getOffset<<<gridSize, blockSize>>>(d_offset, d_0sum, d_1sum, d_map);
	
		cudaMemcpy(h_offset, d_offset, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToHost);

		// new Indices
		getNewIndices<<<gridSize, blockSize>>>(d_newIndices, d_offset, d_histPSum);
		cudaMemcpy(h_newIndices, d_newIndices, 
				sizeof(unsigned int) * numberOfSort, cudaMemcpyDeviceToHost);

		printf("\n");
		for (int i = 0; i < numberOfSort; i++)
		{
			printf("%u %u %u\n",
					h_offset[i], h_histPSum[i], h_newIndices[i]);
		}
		break;
	}

}

int main()
{
	//initRef();
	//map();
	//transpose();
	//gather();
	//scatter();
	//stencil();
	//reduce();
	//Hills_Steele_scan();
	//Blelloch_scan();
	radixSort();

	radixSort();
	radixSort();

	return 0;
}

