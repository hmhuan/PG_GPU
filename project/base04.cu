#include <stdio.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>

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

// Sequential radix sort
// Assume: nBits (k in slides) in {1, 2, 4, 8, 16}
void sortByHost(const uint32_t * in, int n,
                uint32_t * out,
                int nBits)
{
    int nBins = 1 << nBits; // 2^nBits
    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));

    // In each counting sort, we sort data in "src" and write result to "dst"
    // Then, we swap these 2 pointers and go to the next counting sort
    // At first, we assign "src = in" and "dest = out"
    // However, the data pointed by "in" is read-only 
    // --> we create a copy of this data and assign "src" to the address of this copy
    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;

    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of nBits bits)
	// In each loop, sort elements according to the current digit 
	// (using STABLE counting sort)
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
    	// TODO: Compute "hist" of the current digit
        memset(hist, 0, nBins * sizeof(int));
        for (int i = 0; i < n; i++)
        {
            int bin = (src[i] >> bit) & (nBins - 1);
            hist[bin]++;
        }

    	// TODO: Scan "hist" (exclusively) and save the result to "histScan"
        histScan[0] = 0;
        for (int bin = 1; bin < nBins; bin++)
            histScan[bin] = histScan[bin - 1] + hist[bin - 1];

    	// TODO: From "histScan", scatter elements in "src" to correct locations in "dst"
        for (int i = 0; i < n; i++)
        {
            int bin = (src[i] >> bit) & (nBins - 1);
            dst[histScan[bin]] = src[i];
            histScan[bin]++;
        }    

    	// TODO: Swap "src" and "dst"
        uint32_t * temp = src;
        src = dst;
        dst = temp; 
    }

    // TODO: Copy result to "out"
    memcpy(out, src, n * sizeof(uint32_t));
    // Free memories
    free(hist);
    free(histScan);
    free(originalSrc);
}


// histogram kernel
__global__ void computeHistKernel(uint32_t * in, int n, int * hist, int nBins, int bit)
{
    // TODO
    // Each block computes its local hist using atomic on SMEM
    //extern __shared__ int s_bin[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    //int delta = (nBins - 1) / blockDim.x + 1;
    //for (int i = 0; i < delta; i++)
    //{
    //    int id = threadIdx.x + i * blockDim.x;
    //    if (id < nBins)
    //        s_bin[id] = 0;
    //}
    //__syncthreads();
    if (i < n)
    {
        int bin = (in[i] >> bit) & (nBins - 1);
        atomicAdd(&hist[bin], 1);
    }
    //__syncthreads();
    // Each block adds its local hist to global hist using atomic on GMEM
    //for (int i = 0; i < delta; i++)
    //{
    //    int id = threadIdx.x + i * blockDim.x;
    //    if (id < nBins)
    //        atomicAdd(&hist[id], s_bin[id]);
    //}

    
}

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

void sortByDevice_1(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits, int * blockSizes)
{

    dim3 blkSize1(blockSizes[0]); // block size for histogram kernel
    dim3 blkSize2(blockSizes[1]); // block size for scan kernel
    dim3 gridSize((n - 1) / blkSize1.x + 1); // grid size for histogram kernel 
    // TODO
    int nBins = 1 << nBits; // 2^nBits
    int * hist = (int *)malloc(nBins * gridSize.x * sizeof(int));
    //int * histScan = (int *)malloc(nBins * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
    	// TODO: Compute "hist" of the current digit
        memset(hist, 0, nBins * gridSize.x * sizeof(int));
        for (int i = 0; i < n; i++)
        {
            int bin = (src[i] >> bit) & (nBins - 1);
            hist[i / blkSize1.x + bin]++;
        }
        for (int i = 0; i < gridSize.x; i++)
        {
            for (int j = 0; j <nBins; j++)
                printf("%d ", hist[i * blkSize1.x + j]);
            printf("\n");
        }
        printf("\n");
    }
    
    // Free memories
    //free(blkSums);
    free(hist);
    //free(histScan);
    free(originalSrc);
}
// (Partially) Parallel radix sort: implement parallel histogram and parallel scan in counting sort
// Assume: nBits (k in slides) in {1, 2, 4, 8, 16}
// Why "int * blockSizes"? 
// Because we may want different block sizes for diffrent kernels:
//   blockSizes[0] for the histogram kernel
//   blockSizes[1] for the scan kernel
void sortByDevice(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits, int * blockSizes)
{
    // TODO
    int nBins = 1 << nBits; // 2^nBits
    int * hist = (int *)malloc(nBins * blockSizes[0] * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;

    dim3 blkSize1(blockSizes[0]); // block size for histogram kernel
    dim3 blkSize2(blockSizes[1]); // block size for scan kernel
    dim3 gridSize((n - 1) / blkSize1.x + 1); // grid size for histogram kernel 
    dim3 gridSize2((nBins - 1)/ blkSize2.x + 1); // grid size for scan kernel
    
    size_t smemSize = nBins * sizeof(int); // shared memory size for histogram kernel
    int * d_hist, *d_histScan, * d_blkSums;
    uint32_t *d_src;

    int * blkSums;
    blkSums = (int*)malloc(gridSize2.x * sizeof(int));
    size_t sMemSize = blkSize2.x * sizeof(int); // shared memory size for scan kernel

    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
	CHECK(cudaMalloc(&d_hist, nBins * blkSize1.x * sizeof(int)));
	CHECK(cudaMalloc(&d_histScan, nBins * sizeof(int)));
    CHECK(cudaMalloc(&d_blkSums, gridSize2.x * sizeof(int)));
    
    CHECK(cudaMemcpy(d_src, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    //printf("nBins: %d\n", nBins);
    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of nBits bits)
	// In each loop, sort elements according to the current digit 
	// (using STABLE counting sort)
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
    	// TODO: compute hist by Device
        CHECK(cudaMemset(d_hist, 0, nBins * sizeof(int)));
        
	    computeHistKernel<<<gridSize, blkSize1>>>(d_src, n, d_hist, nBins, bit);
        cudaDeviceSynchronize();
	    CHECK(cudaGetLastError());
	    CHECK(cudaMemcpy(hist, d_hist, nBins * sizeof(int), cudaMemcpyDeviceToHost));

    	// TODO: exclusice scan
        scanBlkKernel<<<gridSize2, blkSize2, sMemSize>>>(d_hist, nBins, d_histScan, d_blkSums);
        cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

        //CHECK(cudaMemcpy(histScan, d_histScan, nBins * sizeof(int), cudaMemcpyDeviceToHost));
        //CHECK(cudaMemcpy(blkSums, d_blkSums, gridSize2.x * sizeof(int), cudaMemcpyDeviceToHost));
        //for (int i = 1; i < gridSize2.x; i++)
        //    blkSums[i] += blkSums[i-1];
        //for (int i = blkSize2.x; i < nBins; i++)
        //    histScan[i] += blkSums[(i - 1) / blkSize2.x];

        //addBlkSums<<<gridSize2, blkSize2>>>(d_histScan, nBins, d_blkSums);
        //cudaDeviceSynchronize();
		//CHECK(cudaGetLastError());
        //CHECK(cudaMemcpy(histScan, d_histScan, nBins * sizeof(int), cudaMemcpyDeviceToHost));

    	// TODO: From "histScan", scatter elements in "src" to correct locations in "dst"
    	for (int i = 0; i < n; i++)
        {
            int bin = (src[i] >> bit) & (nBins - 1);
            dst[histScan[bin]] = src[i];
            histScan[bin]++;
        }
    	// TODO: Swap "src" and "dst"
        uint32_t * temp = src;
        src = dst;
        dst = temp;
    }

    CHECK(cudaFree(d_src));
	CHECK(cudaFree(d_hist));
    CHECK(cudaFree(d_blkSums));
    CHECK(cudaFree(d_histScan));
    // TODO: Copy result to "out"
    memcpy(out, src, n * sizeof(uint32_t));
    // Free memories
    free(blkSums);
    free(hist);
    free(histScan);
    free(originalSrc);
}

// Radix sort
void sort(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits,
        bool useDevice=false, int * blockSizes=NULL)
{
    GpuTimer timer; 
    timer.Start();

    if (useDevice == false)
    {
    	printf("\nRadix sort by host\n");
        sortByHost(in, n, out, nBits);
    }
    else // use device
    {
    	printf("\nRadix sort by device\n");
        sortByDevice_1(in, n, out, nBits, blockSizes);
    }

    timer.Stop();
    printf("Time: %.3f ms\n", timer.Elapsed());
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

void checkCorrectness(uint32_t * out, uint32_t * correctOut, int n)
{
    for (int i = 0; i < n; i++)
    {
        if (out[i] != correctOut[i])
        {
            printf("INCORRECT :(\n");
            return;
        }
    }
    printf("CORRECT :)\n");
}

void printArray(uint32_t * a, int n)
{
    for (int i = 0; i < n; i++)
        printf("%i ", a[i]);
    printf("\n");
}

int main(int argc, char ** argv)
{
	// PRINT OUT DEVICE INFO
    printDeviceInfo();

	// SET UP INPUT SIZE
    int nBits = 1;
    int n = 32; //(1 << 24) + 1;
    //n = 10;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = rand() % 100 + 1;
    printArray(in, n);
	
	// DETERMINE BLOCK SIZES
    int blockSizes[2] = {8, 8};//{512, 512}; // One for histogram, one for scan
    if (argc == 4)
    {
        blockSizes[0] = atoi(argv[2]);
        blockSizes[1] = atoi(argv[3]);
    }
    printf("\nHist block size: %d, scan block size: %d\n", blockSizes[0], blockSizes[1]);

    // SORT BY HOST
    sort(in, n, correctOut, nBits);
    printArray(correctOut, n);
    
    // SORT BY DEVICE
    sort(in, n, out, nBits, true, blockSizes);
    //checkCorrectness(out, correctOut, n);
	
	// FREE MEMORIES 
    free(in);
    free(out);
    free(correctOut);
    
    return EXIT_SUCCESS;
}