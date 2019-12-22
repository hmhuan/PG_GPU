#include "myScan.h"

using namespace std;

// kernel for exclusive scan
__global__ void reductionPhase(uint32_t *d_in, uint32_t *d_scan, int n)
{
	int i = threadIdx.x;
	for(int step = 1; step < blockDim.x + 1; step*=2)
	{
		int stride = step*2;
		int add = step - 1;
		int ind = i*stride + add + blockDim.x*blockIdx.x;
		if(ind >= blockIdx.x*blockDim.x && ind + step < (blockIdx.x+1)*blockDim.x)
		{
			d_in[ind + step] += d_in[ind];
		}
		__syncthreads();
	}
	i = blockDim.x*blockIdx.x + threadIdx.x;
	d_scan[i] = d_in[i];
}

__global__ void postReductionPhase(uint32_t *d_in, uint32_t *d_out, int n)
{
	int i = threadIdx.x;
	for(int stride = blockDim.x/2; stride > 1;stride /= 2)
	{
		int step = stride/2;
		int tmp = blockDim.x*blockIdx.x;
		int add = stride - 1;
		int ind = stride*i + add + tmp;

		if(ind + step < (blockIdx.x+1)*blockDim.x && ind >= blockIdx.x*blockDim.x)
		{
			d_in[ind + step] += d_in[ind];
		}
		__syncthreads();
	}
	i += blockDim.x*blockIdx.x;
	d_out[i] = d_in[i];
}

__global__ void addPhase(uint32_t *d_in, uint32_t *d_out, int n)
{
	int i = threadIdx.x + blockDim.x*blockIdx.x;
	if(blockIdx.x != 0 && i < n)
	{
		d_out[i] = d_in[i] + d_in[blockIdx.x*blockDim.x - 1];
	}
	if(blockIdx.x == 0)
		d_out[i] = d_in[i];
}

void MyScan(const uint32_t *in, uint32_t* res,int n, int sizeBlock)
{
	dim3 blockSize(sizeBlock);
	dim3 gridSize((n - 1)/blockSize.x + 1);
	uint32_t *d_in, *d_scan, *d_out, *d_res;
	cudaMalloc(&d_in, n*sizeof(uint32_t));
	cudaMalloc(&d_scan, n*sizeof(uint32_t));
	cudaMalloc(&d_out, n*sizeof(uint32_t));
	cudaMalloc(&d_res, n*sizeof(uint32_t));
	
	cudaMemset(d_scan, 0, n*sizeof(uint32_t));
	cudaMemset(d_out, 0, n*sizeof(uint32_t));

	cudaMemcpy(d_in, in, sizeof(uint32_t)*n, cudaMemcpyHostToDevice);
	reductionPhase<<<gridSize, blockSize>>>(d_in, d_scan, n);
	
	postReductionPhase<<<gridSize, blockSize>>>(d_scan, d_out, n);
	
	addPhase<<<gridSize, blockSize>>>(d_out, d_res, n);
	cudaMemcpy(res, d_res, sizeof(uint32_t)*n, cudaMemcpyDeviceToHost);

	cudaFree(d_in), cudaFree(d_out), cudaFree(d_scan);
}