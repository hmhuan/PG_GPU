#include <stdio.h>

#define CHECK(call)                                                            \
{                                                                              \
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
        cudaEventSynchronize(start); 
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

/*
Scan within each block's data (work-inefficient), write results to "out", and 
write each block's sum to "blkSums" if "blkSums" is not NULL.
*/
// scan kernel
__global__ void scanBlkKernel(int * in, int n, int * out, int * blkSums)
{   
    // TODO
    extern __shared__ int s_data[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < n)
        s_data[threadIdx.x] = in[i - 1];
    else
        s_data[threadIdx.x] = 0;
    __syncthreads();
    
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int val = 0;
        if (threadIdx.x >= stride)
            val = s_data[threadIdx.x - stride];
        __syncthreads();
        s_data[threadIdx.x] += val;
        __syncthreads();
    }
    if (i < n)
        out[i] = s_data[threadIdx.x];
    if (blkSums != NULL)
        blkSums[blockIdx.x] = s_data[blockDim.x - 1];
}
void printArray(int * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}
__global__ void scanBlkKernel_1(int * in, int n, int * out, int * blkSums)
{   
    // TODO
    extern __shared__ int s_data[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < n)
        s_data[threadIdx.x] = in[i - 1];
    else
        s_data[threadIdx.x] = 0;
    __syncthreads();
    int stride = 1;
    // TODO: Reduction phase
    while (stride < blockDim.x)
    {
        if (threadIdx.x >= stride && (threadIdx.x + 1) % (stride * 2) == 0)
            s_data[threadIdx.x] += s_data[threadIdx.x - stride]; 
        __syncthreads();
        stride *= 2;
    }
    //__syncthreads();
    // TODO: Post-reduction phase
    while(stride > 0)
    {
        if (threadIdx.x + stride < blockDim.x && (threadIdx.x + 1) % (stride * 2) == 0)
            s_data[threadIdx.x + stride] += s_data[threadIdx.x];
        __syncthreads();
        stride /= 2;
    }
    //__syncthreads();
    // TODO: blockSums save
    if (i < n)
        out[i] = s_data[threadIdx.x];
    if (blkSums != NULL)
        blkSums[blockIdx.x] = s_data[blockDim.x - 1];
}

// TODO: You can define necessary functions here
__global__ void addBlkSums(int * in, int n, int* blkSums)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && blockIdx.x > 0)
        in[i] += blkSums[blockIdx.x - 1];
}
void scan(int * in, int n, int * out,  
        bool useDevice=false, dim3 blkSize=dim3(1))
{
    GpuTimer timer; 
    timer.Start();
    if (useDevice == false)
    {
    	printf("\nScan by host\n");
		out[0] = 0;
	    for (int i = 1; i < n; i++)
	    {
	    	out[i] = out[i - 1] + in[i - 1];
	    }
    }
    else // Use device
    {
    	printf("\nScan by device\n");
        // TODO
        int * d_in, *d_out, *d_blkSums;
        dim3 gridSize((n - 1) / blkSize.x + 1);
        int * blkSums;
        blkSums = (int*)malloc( gridSize.x * sizeof(int));
        
        CHECK(cudaMalloc(&d_in, n * sizeof(int)));
		CHECK(cudaMalloc(&d_out, n * sizeof(int)));
        CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(int)));

        CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));

        size_t sMemSize = blkSize.x * sizeof(int);
        scanBlkKernel_1<<<gridSize, blkSize, sMemSize>>>(d_in, n, d_out, d_blkSums);
        cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

        CHECK(cudaMemcpy(blkSums, d_blkSums, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 1; i < gridSize.x; i++)
        {
            blkSums[i] += blkSums[i-1];
        }
        CHECK(cudaMemcpy(d_blkSums, blkSums, gridSize.x * sizeof(int), cudaMemcpyHostToDevice));
        addBlkSums<<<gridSize, blkSize>>>(d_out, n, d_blkSums);
        cudaDeviceSynchronize();
		CHECK(cudaGetLastError());
        
        CHECK(cudaMemcpy(out, d_out, n * sizeof(int), cudaMemcpyDeviceToHost));

        CHECK(cudaFree(d_blkSums));
        CHECK(cudaFree(d_in));
		CHECK(cudaFree(d_out));
        free(blkSums);
	}
    timer.Stop();
    printf("Processing time: %.3f ms\n", timer.Elapsed());
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
    printf("****************************\n");
}

void checkCorrectness(int * out, int * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("%d with %d != %d\n", i, out[i], correctOut[i]);
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}
int main(int argc, char ** argv)
{
    // PRINT OUT DEVICE INFO
    printDeviceInfo();

    // SET UP INPUT SIZE
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(int);
    int * in = (int *)malloc(bytes);
    //int in[11] = {1, 2, 3, 4, 5, 6, 7, 8, 9 , 10 , 11}; n = 11;
    int * out = (int *)malloc(bytes); // Device result
    int * correctOut = (int *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = rand();

    // DETERMINE BLOCK SIZE
    dim3 blockSize(512); 
    if (argc == 2)
    {
        blockSize.x = atoi(argv[1]);
    }

    // SCAN BY HOST
    scan(in, n, correctOut);
    // SCAN BY DEVICE
    scan(in, n, out, true, blockSize);
    //printArray(correctOut, n);
    //printArray(out, n);

    checkCorrectness(out, correctOut, n);

    // FREE MEMORIES
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
