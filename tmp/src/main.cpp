#include <stdio.h>
#include <stdlib.h>

#include <cublas_v2.h>
#include <cuda_runtime.h>

#include <Matrix.h>

#define n 6

int main()
{
	cudaError_t cudaStat ; // cudaMalloc status
	cublasStatus_t stat ; // CUBLAS functions status
	cublasHandle_t handle ; // CUBLAS context
	int j; // index of elements
	float * x; // n- vector on the host
	x=( float *) malloc (n* sizeof (*x)); // host memory alloc
	for(j=0;j<n;j++)
		x[j]=( float )j; 
	printf ("x: ");
	for(j=0;j<n;j++)
		printf (" %4.0f,",x[j]); // print x
	printf ("\n");

	// on the device
	float * d_x; // d_x - x on the device
	cudaStat = cudaMalloc (( void **)& d_x ,n* sizeof (*x)); // device
	// memory alloc for x
	stat = cublasCreate (& handle ); // initialize CUBLAS context
	stat = cublasSetVector (n, sizeof (*x),x ,1,d_x ,1); // cp x ->d_x
	int result ; // index of the maximal / minimal element

	stat=cublasIsamax(handle,n,d_x,1,&result);
	printf ("max |x[i]|: %4.0f\n",fabs(x[result -1])); // print

	stat=cublasIsamin(handle,n,d_x,1,&result);
	printf ("min |x[i]|: %4.0f\n",fabs(x[result -1])); // print

	float ret;
	stat=cublasSasum(handle,n,d_x,1,&ret);
	printf ("sum |x[i]|: %4.0f\n",ret); // print

	cudaFree (d_x ); // free device memory
	cublasDestroy ( handle ); // destroy CUBLAS context
	free (x); // free host memory

	return EXIT_SUCCESS ;
}
