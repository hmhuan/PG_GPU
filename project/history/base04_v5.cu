/*
29/12/2019
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

void sortByHost(const uint32_t * in, int n,
                uint32_t * out,
                int nBits)
{
    int nBins = 1 << nBits; // 2^nBits
    int * hist = (int *)malloc(nBins * sizeof(int));
    int * histScan = (int *)malloc(nBins * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;

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

void sortRadixBase04(const uint32_t * in, int n, uint32_t * out,  int nBits, int * blockSizes)
{
    dim3 blkSize1(blockSizes[0]); // block size for histogram kernel
    dim3 blkSize2(blockSizes[1]); // block size for scan kernel
    dim3 gridSize((n - 1) / blkSize1.x + 1); // grid size for histogram kernel 
    // TODO
    int nBins = 1 << nBits; // 2^nBits
    int * hist = (int *)malloc(nBins * gridSize.x * sizeof(int));
    int *histScan = (int * )malloc(nBins * gridSize.x * sizeof(int));

    uint32_t * src = (uint32_t *)malloc(n * sizeof(uint32_t));
    memcpy(src, in, n * sizeof(uint32_t));
    uint32_t * originalSrc = src; // Use originalSrc to free memory later
    uint32_t * dst = out;
    int nHist = nBins * gridSize.x;

    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
        // TODO: Compute "hist" of the current digit
        memset(hist, 0, nHist * sizeof(int));
        for (int i = 0; i < n; i++)
        {
            int bin = (src[i] >> bit) & (nBins - 1);
            hist[bin * gridSize.x + i / blkSize1.x]++;
        }
        // TODO: Exclusive scan
        histScan[0] = 0;
        for (int i = 1; i < nHist; i++)
            histScan[i] = histScan[i - 1] + hist[i - 1];
        
        // TODO: Scatter
        for (int i = 0; i < n ; i++)
        {
            int bin = i / blkSize1.x + ((src[i] >> bit) & (nBins - 1)) * gridSize.x;
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
    // Each block computes its local hist using atomic on SMEM
    extern __shared__ int s_bin[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    int delta = (nBins - 1) / blockDim.x + 1;

    for (int j = 0; j < delta; j++)
    {
        int id = threadIdx.x + j * blockDim.x;
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
    for (int j = 0; j < delta; j++)
    {
        int id = threadIdx.x + j * blockDim.x;
        if (id < nBins)
            hist[id * gridDim.x + blockIdx.x] += s_bin[id];
    }
}
       
__global__ void scanBlkKernel(int * in, int n, int * out, int * blkSums, int mode = 1)
{   
    extern __shared__ int s_data[];
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i > 0 && i < n)
        s_data[blockDim.x - 1 - threadIdx.x] = in[i - 1];
    else
        s_data[blockDim.x - 1 - threadIdx.x] = 0;
    __syncthreads();
    for (int stride = 1; stride < blockDim.x; stride *= 2)
    {
        int val = 0;
        if (threadIdx.x < blockDim.x - stride)
            val = s_data[threadIdx.x + stride];
        __syncthreads();
        s_data[threadIdx.x] += val;
        __syncthreads();
    }
    if (i < n)
        out[i] = s_data[blockDim.x - 1 - threadIdx.x];
    if (blkSums != NULL)
        blkSums[blockIdx.x] = s_data[0];
}

__global__ void addBlkSums(int * in, int n, int* blkSums)
{
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && blockIdx.x > 0)
        in[i] += blkSums[blockIdx.x - 1];
}

__global__ void Scatter(uint32_t * in, int n, int nBits, int bit, int nBins, int *histScan, uint32_t * out)
{
    extern __shared__ int s_data[];
    int * s_in = s_data;
    int * s_hist = (int *)&s_in[blockDim.x];
    int *dst = (int *)&s_hist[blockDim.x];
    int *dst_ori = (int *)&dst[blockDim.x];
    int *startIndex = (int *)&dst_ori[blockDim.x]; // Cấp phát nBins
    int * scan = (int *)&startIndex[nBins]; 
    int * hist = (int *)&scan[blockDim.x];

    int id = blockIdx.x * blockDim.x + threadIdx.x;
    
    if (id < n)
    {
        s_in[threadIdx.x] = in[id];
        s_hist[threadIdx.x] = (s_in[threadIdx.x] >> bit) & (nBins - 1);
    }
    else 
        s_hist[threadIdx.x] = nBins - 1;
    scan[threadIdx.x] = 0;
    __syncthreads();
    // TODO: B1 - sort radix with k = 1
    for (int b = 0; b < nBits; b++)
    {
        // compute hist
        int _hist = s_hist[threadIdx.x];
        int _in = s_in[threadIdx.x];
        int _bin = (_hist >> b) & 1;
        hist[threadIdx.x] = _bin;
        if (threadIdx.x < blockDim.x - 1)
            scan[threadIdx.x + 1] = _bin;
        __syncthreads();
        int _last_hist = hist[blockDim.x - 1];
        for (int stride = 1; stride < blockDim.x; stride *= 2)
        {
            int val = 0;
            if (threadIdx.x >= stride)
                val = scan[threadIdx.x - stride];
            __syncthreads();
            scan[threadIdx.x] += val;
            __syncthreads();
        }
        __syncthreads();
        // scatter
        int scan_ = scan[threadIdx.x];
        int nZeros = blockDim.x - scan[blockDim.x - 1] - _last_hist;//hist[blockDim.x - 1];
        int rank = 0;
        if (_bin == 0)
            rank = threadIdx.x - scan_;//scan[threadIdx.x];
        else
            rank = nZeros + scan_;//scan[threadIdx.x];
        dst[rank] = _hist;//s_hist[threadIdx.x];
        dst_ori[rank] = _in;//s_in[threadIdx.x];
        __syncthreads();        
        // copy or swap
        s_hist[threadIdx.x] = dst[threadIdx.x];
        s_in[threadIdx.x] = dst_ori[threadIdx.x];
    }
    int _hist = s_hist[threadIdx.x];
    int _in = s_in[threadIdx.x];
    __syncthreads();
    // TODO: B2 + B3
    if (threadIdx.x == 0)
        startIndex[_hist] = 0;
    else
    {
        if (_hist != s_hist[threadIdx.x - 1])
            startIndex[_hist] = threadIdx.x;
    }
    __syncthreads();
    // TODO: B4 real scatter
    if (id < n)
    {
        int preRank = threadIdx.x - startIndex[_hist];
        int bin = ((_in >> bit) & (nBins - 1));
        int scan = histScan[bin * gridDim.x + blockIdx.x];
        out[scan + preRank] = _in;
    }
}

void sortRadixBase04_device(const uint32_t * in, int n,  uint32_t * out, int nBits, int * blockSizes)
{
    int nBins = 1 << nBits;
    dim3 blkSize1(blockSizes[0]); // block size for histogram kernel
    dim3 blkSize2(blockSizes[1]); // block size for scan kernel
    dim3 gridSize1((n - 1) / blkSize1.x + 1); // grid size for histogram kernel 
    dim3 gridSize2((nBins * gridSize1.x - 1) / blkSize2.x + 1);

    int * blkSums = (int *)malloc(gridSize2.x * sizeof(int));
    uint32_t * d_src, *d_dst;
    int *d_scan, *d_blkSums;

    CHECK(cudaMalloc(&d_src, n * sizeof(uint32_t)));
    CHECK(cudaMalloc(&d_dst, n * sizeof(uint32_t)));
	CHECK(cudaMalloc(&d_scan, nBins * gridSize1.x * sizeof(int)));
	CHECK(cudaMalloc(&d_blkSums, gridSize2.x * sizeof(int)));

    CHECK(cudaMemcpy(d_src, in, n * sizeof(uint32_t), cudaMemcpyHostToDevice));

    size_t sMemSize2 = blkSize2.x * sizeof(int);
    
    for (int bit = 0; bit < sizeof(uint32_t) * 8; bit += nBits)
    {
    	// TODO: Compute "hist" of the current digit
        CHECK(cudaMemset(d_scan, 0, nBins * gridSize1.x * sizeof(int)));
        computeHistKernel<<<gridSize1, blkSize1, nBins * sizeof(int)>>>(d_src, n, d_scan, nBins, bit);
        cudaDeviceSynchronize();
	    CHECK(cudaGetLastError());

        // TODO: Scan
        scanBlkKernel<<<gridSize2, blkSize2, sMemSize2>>>(d_scan, nBins * gridSize1.x, d_scan, d_blkSums);
        cudaDeviceSynchronize();
	    CHECK(cudaGetLastError());

        CHECK(cudaMemcpy(blkSums, d_blkSums, gridSize2.x * sizeof(int), cudaMemcpyDeviceToHost));
        for (int i = 1; i < gridSize2.x; i++)
            blkSums[i] += blkSums[i - 1];
        CHECK(cudaMemcpy(d_blkSums, blkSums, gridSize2.x * sizeof(int), cudaMemcpyHostToDevice));
        
        addBlkSums<<<gridSize2, blkSize2>>>(d_scan, nBins * gridSize1.x, d_blkSums);
        cudaDeviceSynchronize();
	    CHECK(cudaGetLastError());
        // TODO: Scatter
        Scatter<<<gridSize1, blkSize1, (blkSize1.x * 6 +  nBins) * sizeof(int)>>>(d_src, n, nBits, bit, nBins, d_scan, d_dst);
        cudaDeviceSynchronize();
	    CHECK(cudaGetLastError());
        // TODO: Swap "src" and "dst"
        uint32_t * temp = d_src;
        d_src = d_dst;
        d_dst = temp; 
    }
    // TODO: Copy result to "out"
    CHECK(cudaMemcpy(out, d_src, n * sizeof(uint32_t), cudaMemcpyDeviceToHost));

    // Free memories
    free(blkSums);
    CHECK(cudaFree(d_src));
    CHECK(cudaFree(d_dst));
    CHECK(cudaFree(d_scan));
    CHECK(cudaFree(d_blkSums));    
}

void sortByDevice_thrust(const uint32_t * in, int n, uint32_t * out)
{
    // TODO
	thrust::device_vector<uint32_t> dv_out(in, in + n);
	thrust::sort(dv_out.begin(), dv_out.end());
	thrust::copy(dv_out.begin(), dv_out.end(), out);
}
GpuTimer timer;
float sort(const uint32_t * in, int n, 
        uint32_t * out, 
        int nBits,
        int useDevice=0, int * blockSizes=NULL)
{
     
    timer.Start();
    if (useDevice == 0)
    {
    	printf("\nRadix sort by host\n");
        sortByHost(in, n, out, nBits);
    }
    else if (useDevice == 1)
    {
    	printf("\nRadix sort by host level 1\n");
        sortRadixBase04(in, n, out, nBits, blockSizes); //use default 8
    }
    else if (useDevice == 2)
    {
        sortRadixBase04_device(in, n, out, nBits, blockSizes);
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
            printf("%d with %d != %d\n", i, out[i], correctOut[i]);
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
    int nBits = 8;
    int n = (1 << 24) + 1;
    if (argc > 1)
        nBits = atoi(argv[1]);
    printf("\nInput size: %d\n", n);
    printf("nBits: %d\n", nBits);
    // ALLOCATE MEMORIES
    size_t bytes = n * sizeof(uint32_t);
    uint32_t * in = (uint32_t *)malloc(bytes);
    uint32_t * out_0 = (uint32_t *)malloc(bytes); // base 4 Host result
    uint32_t * out_1 = (uint32_t *)malloc(bytes); // base 4 Device result
    uint32_t * out_thrust = (uint32_t *)malloc(bytes); // result by Thrust
    uint32_t * correctOut = (uint32_t *)malloc(bytes); // Host result

    // SET UP INPUT DATA
    for (int i = 0; i < n; i++)
       in[i] = rand();
	
	// DETERMINE BLOCK SIZES
    int blockSizes[2] = {512, 512}; // One for histogram, one for scan
    if (argc == 4)
    {
        blockSizes[0] = atoi(argv[2]);
        blockSizes[1] = atoi(argv[3]);
    }
    printf("\nHist block size: %d, scan block size: %d\n", blockSizes[0], blockSizes[1]);

    // SORT BY HOST
    sort(in, n, correctOut, nBits);

	sort(in, n, out_0, nBits, 1, blockSizes);
	checkCorrectness(out_0, correctOut, n);
    
    float avg_time = 0;
    int loop = 16;
    printf("\nRadix sort by device level 2\n");
    for (int i = 0; i < loop; i++)
    {
        float time = sort(in, n, out_1, nBits, 2, blockSizes);
        avg_time += time;
    }
    printf("Avg Time: %.3f ms\n", avg_time / loop);
    checkCorrectness(out_1, correctOut, n);
        
    sort(in, n, out_thrust, nBits, 3, blockSizes);
    checkCorrectness(out_thrust, out_1, n);

    // FREE MEMORIES 
    free(in);
    free(out_0);
    free(out_thrust);
    free(out_1);
    free(correctOut);
    
    return EXIT_SUCCESS;
}