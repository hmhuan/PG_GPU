// Last update: 2018/12/01
#include <stdio.h>

#define CHECK(call)                                                            \
{                                                                               \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(1);                                                               \
    }                                                                          \
}

struct GpuTimer
{
    cudaEvent_t start;
    cudaEvent_t stop;

    GpuTimer()
    {
        cudaEventCreate(&start);
        cudaEventCreate(&stop);
    }

    ~GpuTimer()
    {
        cudaEventDestroy(start);
        cudaEventDestroy(stop);
    }

    void Start()
    {
        cudaEventRecord(start, 0);
    }

    void Stop()
    {
        cudaEventRecord(stop, 0);
    }

    float Elapsed()
    {
        float elapsed;
        cudaEventSynchronize(stop);
        cudaEventElapsedTime(&elapsed, start, stop);
        return elapsed;
    }
};

void scanByHost(int * in, int * out, int n)
{   
    GpuTimer timer; 
    timer.Start();

    out[0] = in[0];
    for (int i = 1; i < n; i++)
    {
    	out[i] = out[i - 1] + in[i];
    }

    timer.Stop();
    printf("Time of scanByHost: %.3f ms\n\n", timer.Elapsed());
}

/*
Scan within each block's data (work-inefficient), write results to "out", and 
write each block's sum to "blkSums" if "blkSums" is not NULL.
*/
__global__ void scanBlks1(int * in, int * out, int n, int * blkSums)
{   
    // TODO
    extern __shared__ int s_data[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
        s_data[threadIdx.x] = in[i];
    else
        s_data[threadIdx.x] = 0;
    __syncthreads();
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        if (threadIdx.x >= stride){
            int neededVal = s_data[threadIdx.x - stride];
            __syncthreads();
            s_data[threadIdx.x] += neededVal;
        }        
        __syncthreads();
    } 
    if (i < n)
        out[i] = s_data[threadIdx.x];
    if (blkSums != NULL)
        blkSums[blockIdx.x] = s_data[blockDim.x - 1];
}

/*
Scan within each block's data (work-efficient), write results to "out", and 
write each block's sum to "blkSums" if "blkSums" is not NULL.
*/
__global__ void scanBlks2(int * in, int * out, int n, int * blkSums)
{
    // TODO
	// 1. Each block loads data from GMEM to SMEM
    extern __shared__ int s_data[];
    int i1 = blockIdx.x * 2 * blockDim.x + threadIdx.x;
    int i2 = i1 + blockDim.x;
    if (i1 < n)
        s_data[threadIdx.x] = in[i1];
    if (i2 < n)
        s_data[threadIdx.x + blockDim.x] = in[i2];
    __syncthreads();

    // 2. Each block does scan with data on SMEM
    // 2.1. Reduction phase
    for (int stride = 1; stride < 2 * blockDim.x; stride *= 2)
    {
        int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1; // To avoid warp divergence
        if (s_dataIdx < 2 * blockDim.x)
            s_data[s_dataIdx] += s_data[s_dataIdx - stride];
        __syncthreads();
    }
    // 2.2. Post-reduction phase
    for (int stride = blockDim.x / 2; stride > 0; stride /= 2)
    {
        int s_dataIdx = (threadIdx.x + 1) * 2 * stride - 1 + stride; // Wow
        if (s_dataIdx < 2 * blockDim.x)
            s_data[s_dataIdx] += s_data[s_dataIdx - stride];
        __syncthreads();
    }

    // 3. Each block writes results from SMEM to GMEM
    if (i1 < n)
        out[i1] = s_data[threadIdx.x];
    if (i2 < n)
        out[i2] = s_data[threadIdx.x + blockDim.x];

    if (blkSums != NULL && threadIdx.x == 0)
        blkSums[blockIdx.x] = s_data[2 * blockDim.x - 1];
}

__global__ void addPrevSum(int * blkSumsScan, int * blkScans, int n)
{
	int i = blockIdx.x * blockDim.x + threadIdx.x + blockDim.x;
	if (i < n)
	{
		blkScans[i] += blkSumsScan[blockIdx.x];
	}
}

void scanByDevice(int * in, int * out, int n, int kernelType, int blkSize)
{
    GpuTimer timer; 
    timer.Start();

    // Allocate device memories
    int *d_in, *d_out;
    size_t bytes = n * sizeof(int);
    CHECK(cudaMalloc(&d_in, bytes));
    CHECK(cudaMalloc(&d_out, bytes));
    int blkDataSize;
    if (kernelType == 1)
        blkDataSize = blkSize;
    else
        blkDataSize = 2 * blkSize;
    int * d_blkSums;
    int numBlks = (n - 1) / blkDataSize + 1;
    CHECK(cudaMalloc(&d_blkSums, numBlks * sizeof(int)));
    
    // Copy data to device memories
    CHECK(cudaMemcpy(d_in, in, bytes, cudaMemcpyHostToDevice));

    // Call kernel to scan within each block's input data
    if (kernelType == 1)
        scanBlks1<<<numBlks, blkSize, blkDataSize * sizeof(int)>>>(d_in, d_out, n, d_blkSums);
    else // KernelType == 2
        scanBlks2<<<numBlks, blkSize, blkDataSize * sizeof(int)>>>(d_in, d_out, n, d_blkSums);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());
    
    // Scan "d_blkSums" (by host)
    int * blkSums;
    blkSums = (int *)malloc(numBlks * sizeof(int));
    CHECK(cudaMemcpy(blkSums, d_blkSums, numBlks * sizeof(int), cudaMemcpyDeviceToHost));
    for (int i = 1; i < numBlks; i++)
        blkSums[i] += blkSums[i-1];
    CHECK(cudaMemcpy(d_blkSums, blkSums, numBlks * sizeof(int), cudaMemcpyHostToDevice));
    free(blkSums);
    
    // Call kernel to add block's previous sum to block's scan result
    addPrevSum<<<numBlks - 1, blkDataSize>>>(d_blkSums, d_out, n);
    cudaDeviceSynchronize();
    CHECK(cudaGetLastError());

    // Copy result from device memories
    CHECK(cudaMemcpy(out, d_out, bytes, cudaMemcpyDeviceToHost));
        
    // Free device memories
    CHECK(cudaFree(d_in));
    CHECK(cudaFree(d_out));
    CHECK(cudaFree(d_blkSums));

    timer.Stop();
    printf("Time of scanByDevice (kernelType=%d): %.3f ms\n\n", kernelType, timer.Elapsed());
}

void printDeviceInfo()
{
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n\n");
}

bool checkCorrectness(int * out, int * correctOut, int n)
{
    for (int i = 0; i < n; i++)
        if (out[i] != correctOut[i])
            return false;
    return true;
}

int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n = (1 << 24) + 1;
    printf("Input size: %d\n\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(int);
    int * in = (int *)malloc(bytes);
    int * out = (int *)malloc(bytes); // Device result
    int * correctOut = (int *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = (int)(rand() & 0xFF) - 127; // random int in [-127, 128]

    // DETERMINE BLOCK SIZE
    int blockSize1 = 512; // Default for "scanBlks1"
    int blockSize2 = 512; // Default for "scanBlks2"
    if (argc == 2)
    {
        blockSize1 = blockSize2 = atoi(argv[1]);
    }
    else if (argc == 3)
    {
        blockSize1 = atoi(argv[1]);
        blockSize2 = atoi(argv[2]);
    }

    // SCAN BY HOST
    scanByHost(in, correctOut, n);
    
    // SCAN BY DEVICE, KERNEL 1
    int kernelType = 1;
    scanByDevice(in, out, n, kernelType, blockSize1);
    if (checkCorrectness(out, correctOut, n) == false)
        printf("scanByDevice (kernelType=%d) is INCORRECT!\n\n", kernelType);

    // SCAN BY DEVICE, KERNEL 2
    memset(out, 0, bytes); // Reset output
    kernelType = 2;
    scanByDevice(in, out, n, kernelType, blockSize2);
    if (checkCorrectness(out, correctOut, n) == false)
        printf("scanByDevice (kernelType=%d) is INCORRECT!\n\n", kernelType);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}