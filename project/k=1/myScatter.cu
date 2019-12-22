#include "myScatter.h"
#include <iostream>
using namespace std;

__global__ void myScatterKernel(uint32_t* in, uint32_t* scan, uint32_t* out, int n)
{
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if(i < n)
    {
        if(in[i] == 0)
        {
            out[i - scan[i]] = in[i];
            out[i - scan[i]] = 1;
        }
        // num0 = n - scan[n - 1] - in[n-1]
        else
        {
            //out[n - scan[n - 1] - in[n - 1] + scan[i]] = in[i];
            out[n - scan[n - 1] - in[n - 1] + scan[i]] = 1;
        }
    }
}

void MyScatter(const uint32_t* in, uint32_t *scan, uint32_t *out, int n, int blockSizeX)
{
    dim3 blockSize(blockSizeX), gridSize((n - 1)/blockSize.x + 1);

    uint32_t *d_in, *d_scan, *d_out;

    cudaMalloc(&d_in, n*sizeof(uint32_t));
    cudaMalloc(&d_scan, n*sizeof(uint32_t));
    cudaMalloc(&d_out, n*sizeof(uint32_t));

    cudaMemcpy(d_in, in, n*sizeof(uint32_t), cudaMemcpyHostToDevice);
    cudaMemcpy(d_scan, scan, n*sizeof(uint32_t), cudaMemcpyHostToDevice);

    myScatterKernel<<<gridSize, blockSize>>>(d_in, d_scan, d_out, n);

    cudaMemcpy(out, d_out, n*sizeof(uint32_t), cudaMemcpyDeviceToHost);

    cout << "out ";
    for(int i = 0;i < n;i++)
        cout << out[i] << ' ';
    cout << endl;

    cudaFree(d_scan), cudaFree(d_out), cudaFree(d_in);
}