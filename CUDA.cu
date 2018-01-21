#include<stdio.h>
#include<stdlib.h>
#include<sys/types.h>
#include<sys/stat.h>
#include<fcntl.h>
#include<unistd.h>
#include<errno.h>
#include<cuda.h>
#include<cuda_runtime.h>



#define NXPROB      288                /* x dimension of problem grid */
#define NYPROB      288                 /* y dimension of problem grid */
#define STEPS      	100              /* number of time steps */
#define MAXWORKER   16                  /* maximum number of worker tasks */
#define MINWORKER   1                  /* minimum number of worker tasks */



__global__ void updateU(float * u, int time)
{

	/*i - row and j - column*/
	int i = (blockIdx.x*blockDim.x + threadIdx.x);
	int j = (blockIdx.y*blockDim.y + threadIdx.y);
	
	if (i >= NXPROB || j >= NYPROB) return;
	
	if (i%NXPROB == 0 || j%NYPROB == 0 || j%NYPROB == NYPROB - 1 || i%NXPROB == NXPROB - 1) return;

	
	int offsetw = (time % 2 == 0) ? NXPROB*NYPROB : 0;
	int offsetr = (offsetw == 0) ? NYPROB*NXPROB : 0;

	u[offsetw + i*NYPROB + j] = u[offsetr + i*NYPROB + j] +

		0.1 * (u[offsetr + i*NYPROB + NYPROB + j] + u[offsetr + i*NYPROB - NYPROB + j] - 2 * u[offsetr + i*NYPROB + j])

		+ 0.1 * (u[offsetr + i*NYPROB + j + NYPROB] + u[offsetr + i*NYPROB + j - NYPROB] - 2 * u[offsetr + i*NYPROB + j]);

}


int main(int argc, char* argv[])
{

	printf("Cuda-Start\n");

	cudaDeviceProp deviceProp;
	cudaError error;
	cudaGetDeviceProperties(&deviceProp, 0);
	int sum_mem = 2 * NYPROB*NXPROB;
	cudaEvent_t start, stop;
	float time = 0;
	int i;


	float * u = (float *)malloc(sizeof(float)*sum_mem);
	float * device_u;
	

	for (i = 0; i<sum_mem / 2; i++){
		if ((i%NXPROB == 0) || (i%NYPROB == NYPROB - 1)){
			u[i] = 0;
			continue;
		}
		if (i < NXPROB){
			u[i] = 0;
			continue;
		}
		/*if (i == NXPROB){
			u[i] == 0.0;
			continue;
		}*/
		if ((i<NYPROB*NXPROB) && (i>(NXPROB-1)*NYPROB)){
			u[i] = 0;
			continue;
		}
		//printf("WILL WRITE TO %d\n", i);
		u[i] = rand() % 100;
	}

	for (i = sum_mem / 2; i < sum_mem; i++){
		u[i] = 0;
	}

	error = cudaMalloc(&device_u, sum_mem*sizeof(float));
	
	if (error != cudaSuccess){ fprintf(stderr, "Failed to allocate memory for matrix  %s\n", cudaGetErrorString(error)); return -4; }
	
	error = cudaMemcpy(device_u, u, sum_mem*sizeof(float), cudaMemcpyHostToDevice);

	if (error != cudaSuccess){ fprintf(stderr, "Failed to copy matrix to device: %s\n", cudaGetErrorString(error)); fflush(stderr); return -7; }

	int root;
	for (root = 2; root*root <= deviceProp.maxThreadsPerBlock; root++)
		if (root*root == deviceProp.maxThreadsPerBlock) break;

	if (root*root>deviceProp.maxThreadsPerBlock) root--;

	error = cudaEventCreate(&start);
	if (error != cudaSuccess){ fprintf(stderr, "Failure(time): %s\n", cudaGetErrorString(error));fflush(stderr); return-17; }

	error = cudaEventCreate(&stop);
	if (error != cudaSuccess){ fprintf(stderr, "Failure(time): %s\n", cudaGetErrorString(error));fflush(stderr); return-17; }

	dim3 threadsPerBlock(root, root);
	int blockDimX = (NXPROB%root==0) ? (NXPROB / root) : (NXPROB / root + 1);
	int blockDimY = (NYPROB%root==0) ? (NYPROB / root) : (NYPROB / root + 1);
	dim3 numOfBlocks(blockDimX, blockDimY, 1);//pixsize
	cudaEventRecord(start);

	for (i = 0; i<STEPS; i++){
		updateU <<< numOfBlocks, threadsPerBlock>>>(device_u,i);//3o shared
		error = cudaGetLastError();
		if (error != cudaSuccess){ fprintf(stderr, "Error in steps call %s\n", cudaGetErrorString(error));fflush(stderr); return -11; }
		
	}

	cudaEventRecord(stop);
	error = cudaMemcpy(u, device_u, sum_mem*sizeof(float), cudaMemcpyDeviceToHost);
	if (error != cudaSuccess){ fprintf(stderr, "Failed to copy matrix(to host): %s\n", cudaGetErrorString(error));fflush(stderr); return -8; }

	cudaEventElapsedTime(&time, start, stop);
	printf("Time %f \n", time);
	
	return 0;
}
