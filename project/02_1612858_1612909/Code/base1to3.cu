/*
22/12/2019
hmhuan-1612858
nnkhai-1612909
*/
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
    	// TODO: Compute "hist" of the current digit  // histogram cua mang in xet tren digit hien tai
        memset(hist, 0, nBins * sizeof(int));
        for (int i = 0; i < n; i++)
    	{
    		int bin = (src[i] >> bit) & (nBins - 1);
    		hist[bin]++;
    	}
    	// TODO: Scan "hist" (exclusively) and save the result to "histScan"
        histScan[0] = 0;
        for (int i = 1; i < nBins; i++)
            histScan[i] = histScan[i - 1] + hist[i - 1];
    	// TODO: From "histScan", scatter elements in "src" to correct locations in "dst"
    	for (int i = 0; i < n; i++)
        {
            int bin = (src[i] >> bit) & (nBins - 1);
            dst[histScan[bin]] = src[i];
            histScan[bin]++;  // (neu cung bin thi ghi ben canh)
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
    extern __shared__ int s_bin[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int delta = (nBins - 1) / blockDim.x + 1;
    for (int i = 0; i < delta; i++)
    {
        int id = threadIdx.x + i * blockDim.x;
        if (id < nBins)
            s_bin[id] = 0;
    }
    __syncthreads();
    if (i < n)
    {
        int bin = (in[i] >> bit) & (nBins - 1);
        atomicAdd(&s_bin[bin], 1);
    }
    __syncthreads();
    // Each block adds its local hist to global hist using atomic on GMEM
    for (int i = 0; i < delta; i++)
    {
        int id = threadIdx.x + i * blockDim.x;
        if (id < nBins)
            atomicAdd(&hist[id], s_bin[id]);
    }
}
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
    if (threadIdx.x == 0 && blkSums != NULL)
        blkSums[blockIdx.x] = s_data[blockDim.x - 1];
}

// TODO: You can define necessary functions here
__global__ void addBlkSums(int * in, int n, int* blkSums)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && blockIdx.x > 0)
        in[i] += blkSums[blockIdx.x - 1];
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
    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));
    
    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;

    dim3 blkSize1(blockSizes[0]); // block size for histogram kernel
    dim3 blkSize2(blockSizes[1]); // block size for scan kernel
    dim3 gridSize1((n - 1) / blkSize1.x + 1); // grid size for histogram kernel 
    dim3 gridSize2((nBins - 1)/ blkSize2.x + 1); // grid size for scan kernel
    
    size_t smemSize = nBins * sizeof(int); // shared memory size for histogram kernel
    int * d_hist, *d_histScan, * d_blkSums;
    uint32_t *d_src;


    int * blkSums;
    blkSums = (int*)malloc(gridSize2.x * sizeof(int));
    size_t sMemSize = blkSize2.x * sizeof(int); // shared memory size for scan kernel

    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
	CHECK(cudaMalloc(&d_hist, nBins * sizeof(int)));
	CHECK(cudaMalloc(&d_histScan, nBins * sizeof(int)));
    CHECK(cudaMalloc(&d_blkSums, gridSize2.x * sizeof(int)));
    
    CHECK(cudaMemcpy(d_src, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    // Loop from LSD (Least Significant Digit) to MSD (Most Significant Digit)
    // (Each digit consists of nBits bits)
	// In each loop, sort elements according to the current digit 
	// (using STABLE counting sort)
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
    	// TODO: compute hist by Device
        CHECK(cudaMemset(d_hist, 0, nBins * sizeof(int)));
        
	    computeHistKernel<<<gridSize1, blkSize1, smemSize>>>(d_src, n, d_hist, nBins, bit);
        cudaDeviceSynchronize();
	    CHECK(cudaGetLastError());
	    CHECK(cudaMemcpy(hist, d_hist, nBins * sizeof(int), cudaMemcpyDeviceToHost));

    	// TODO: exclusice scan
        scanBlkKernel<<<gridSize2, blkSize2, sMemSize>>>(d_hist, nBins, d_histScan, d_blkSums);
        cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

        //CHECK(cudaMemcpy(histScan, d_histScan, nBins * sizeof(int), cudaMemcpyDeviceToHost));
        CHECK(cudaMemcpy(blkSums, d_blkSums, gridSize2.x * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 1; i < gridSize2.x; i++)
            blkSums[i] += blkSums[i-1];
        //for (int i = blkSize2.x; i < nBins; i++)
        //    histScan[i] += blkSums[(i - 1) / blkSize2.x];
        CHECK(cudaMemcpy(d_blkSums, blkSums, gridSize2.x * sizeof(int), cudaMemcpyHostToDevice));
        addBlkSums<<<gridSize2, blkSize2>>>(d_histScan, nBins, d_blkSums);
        
        cudaDeviceSynchronize();
		CHECK(cudaGetLastError());
        CHECK(cudaMemcpy(histScan, d_histScan, nBins * sizeof(int), cudaMemcpyDeviceToHost));

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

__global__ void scanBlkKernel_1(uint32_t *in, int n, int bit, int *out, int * blkSums)
{   
    // TODO: compute bits
    extern __shared__ int s_data[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < n)
    {
        s_data[threadIdx.x] = (in[i - 1] >> bit) & 1;
    }
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
    if (threadIdx.x == 0 && blkSums != NULL)
        blkSums[blockIdx.x] = s_data[blockDim.x - 1];
}

__global__ void scatter(uint32_t * in, int bit, int *inScan, int n, uint32_t *out)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n)
    {
        int nZeros = n - inScan[n - 1] - ((in[n - 1] >> bit) & 1);
        int inBit = (in[i] >> bit) & 1;
        int rank = 0;
        if (inBit == 0)
            rank = i - inScan[i];
        else
            rank = nZeros + inScan[i];
        out[rank] = in[i];
    }
}
void printArray(uint32_t * a, int n);

void sortByDevice_base03(const uint32_t * in, int n, 
        uint32_t * out, int * blockSizes)
{
    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    uint32_t * dst = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later

    dim3 blkSize(blockSizes[0]); // block size for histogram kernel
    dim3 gridSize((n - 1) / blkSize.x + 1); // grid size for histogram kernel 
    int *d_bitsScan, * d_bits, * d_blkSums;
    uint32_t *d_src, *d_dst;
    size_t sMemSize = blkSize.x * sizeof(int); // shared memory size for scan kernel

    int * blkSums = (int *)malloc(gridSize.x * sizeof(int));
    int * bitsScan = (int *)malloc(n * sizeof(int));
    int * bits = (int *)malloc(n * sizeof(int));

    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));
	CHECK(cudaMalloc(&d_bitsScan, n * sizeof(int)));
	CHECK(cudaMalloc(&d_bits, n * sizeof(int)));
    CHECK(cudaMalloc(&d_blkSums, gridSize.x * sizeof(int)));
    
    CHECK(cudaMemcpy(d_src, src, n * sizeof(uint32_t), cudaMemcpyHostToDevice));
    
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit++)
    {
    	// TODO: compute bits [0 1 1 . ..] and exclusice scan
        scanBlkKernel_1<<<gridSize, blkSize, sMemSize>>>(d_src, n, bit, d_bitsScan, d_blkSums);
        cudaDeviceSynchronize();
		CHECK(cudaGetLastError());
        
        CHECK(cudaMemcpy(blkSums, d_blkSums, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 1; i < gridSize.x; i++)
            blkSums[i] += blkSums[i-1];
        CHECK(cudaMemcpy(d_blkSums, blkSums, gridSize.x * sizeof(int), cudaMemcpyHostToDevice));

        addBlkSums<<<gridSize, blkSize>>>(d_bitsScan, n, d_blkSums);
        cudaDeviceSynchronize();
		CHECK(cudaGetLastError());
    	
        // TODO: scatter
        scatter<<<gridSize, blkSize>>>(d_src, bit, d_bitsScan, n, d_dst);
        cudaDeviceSynchronize();
		CHECK(cudaGetLastError());
        
    	// TODO: Swap "src" and "dst"
        uint32_t * d_temp = d_src;
        d_src = d_dst;
        d_dst = d_temp;
    }
    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    //free Cuda
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_bits));
    CHECK(cudaFree(d_bitsScan));
    CHECK(cudaFree(d_blkSums));    
    // Free memories
    free(originalSrc);
    free(dst);
    free(blkSums);
    free(bitsScan);
    free(bits);
}

void sortByDevice_thrust(const uint32_t * in, int n, uint32_t * out)
{
    // TODO
	thrust::device_vector<uint32_t> dv_out(in, in + n);
	thrust::sort(dv_out.begin(), dv_out.end());
	thrust::copy(dv_out.begin(), dv_out.end(), out);
}

// Radix sort
float sort(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits,
        int useDevice=0, int * blockSizes=NULL)
{
    GpuTimer timer; 
    timer.Start();
    if (useDevice == 0)
    {
    	printf("\nRadix sort by host\n");
        sortByHost(in, n, out, nBits);
    }
    else if (useDevice == 1)// use device
    {
        
    	printf("\nRadix sort by device\n");
        sortByDevice(in, n, out, nBits, blockSizes);
        
    }
    else if (useDevice == 2)
    {
        sortByDevice_base03(in, n, out, blockSizes);
    }
    else
    {
        printf("\nSort by thrust\n");
        sortByDevice_thrust(in, n, out);
    }
    timer.Stop();
    float time = timer.Elapsed();
    if (useDevice != 2)
        printf("Time: %.3f ms\n", time);
    return time;
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
            printf("%d, %d != %d\n", i, out[i], correctOut[i]);
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
    int n = (1 << 24) + 1;
    printf("\nInput size: %d\n", n);

    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out = (uint32_t *)malloc(bytes); // Device result
    uint32_t * out_base03 = (uint32_t *)malloc(bytes); // Device result base03
    uint32_t * out_thrust = (uint32_t *)malloc(bytes); // result by Thrust
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
        in[i] = rand();
    // SET UP NBITS
    int nBits = 8;
    if (argc > 1)
        nBits = atoi(argv[1]);
    printf("\nNum bits per digit: %d\n", nBits);

    // DETERMINE BLOCK SIZES
    int blockSizes[2] = {512, 512}; // One for histogram, one for scan
    if (argc == 4)
    {
        blockSizes[0] = atoi(argv[2]);
        blockSizes[1] = atoi(argv[3]);
    }
    printf("\nHist block size: %d, scan block size: %d\n", blockSizes[0], blockSizes[1]);

    // SORT BY HOST
    sort(in, n, correctOut, nBits, 0);

    // SORT BY DEVICE
    sort(in, n, out, nBits, 1, blockSizes);
    checkCorrectness(out, correctOut, n);

    // SORT base 03
    printf("\nRadix sort by device by base03\n");
    float avg_time = 0;
    int loop = 16;
    for (int i = 0; i < loop; i++)
    {
        float time = sort(in, n, out_base03, 1, 2, blockSizes);
        avg_time += time;
    }
    printf("Avg Time: %.3f ms\n", avg_time / loop);    
    checkCorrectness(out_base03, correctOut, n);
    
    // SORT BY DEVICE by thrust
    sort(in, n, out_thrust, nBits, 3, blockSizes);
    checkCorrectness(out_thrust, correctOut, n);

    // FREE MEMORIES 
    free(in);
    free(out);
    free(out_base03);
    free(out_thrust);
    free(correctOut);
    
    return EXIT_SUCCESS;
}
