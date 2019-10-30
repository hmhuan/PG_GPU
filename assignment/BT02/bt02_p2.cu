#include <stdio.h>

#define CHECK(call)                                                            \
{                                                                              \
    const cudaError_t error = call;                                            \
    if (error != cudaSuccess)                                                  \
    {                                                                          \
        fprintf(stderr, "Error: %s:%d, ", __FILE__, __LINE__);                 \
        fprintf(stderr, "code: %d, reason: %s\n", error,                       \
                cudaGetErrorString(error));                                    \
        exit(EXIT_FAILURE);                                                    \
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

__global__ void reduceBlksKernel(int * in, int n, int * out)
{
	// TODO: Copy-paste the best kernel you have implemented in "bt02.cu" 
    int i = blockIdx.x * blockDim.x * 2 + threadIdx.x;
    for (int stride = blockDim.x; stride > 0; stride /= 2){
        if (threadIdx.x < stride)
            if (i + stride < n)
                in[i] += in[i + stride];
        __syncthreads(); // Synchronize within each block
    }
    if (threadIdx.x == 0)
        out[blockIdx.x] = in[blockIdx.x * blockDim.x * 2];
}

int reduce(int const * in, int n,
        int useDevice=0, dim3 blockSize=dim3(1))
{
    GpuTimer timer;
    timer.Start();

	int result = 0; // Init
	if (useDevice == 0)
	{
		result = in[0];
		for (int i = 1; i < n; i++)
		{
			result += in[i];
		}
	}
	else if (useDevice == 1)// Use device partly
	{
		// Allocate device memories
		int * d_in, * d_out;
		dim3 gridSize((n - 1) / (2 * blockSize.x) + 1); // TODO: Compute gridSize from n and blockSize
		CHECK(cudaMalloc(&d_in, n * sizeof(int)));
		CHECK(cudaMalloc(&d_out, gridSize.x * sizeof(int)));

		// Copy data to device memory
		CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));

		// Call kernel
        reduceBlksKernel<<<gridSize, blockSize>>>(d_in, n, d_out);
		cudaDeviceSynchronize();
		CHECK(cudaGetLastError());

		// Copy result from device memory
		int * out = (int *)malloc(gridSize.x * sizeof(int));
		CHECK(cudaMemcpy(out, d_out, gridSize.x * sizeof(int), cudaMemcpyDeviceToHost));

		// Free device memories
		CHECK(cudaFree(d_in));
		CHECK(cudaFree(d_out));

		// Host do the rest of the work
		result = out[0];
		for (int i = 1; i < gridSize.x; i++)
			result += out[i];

		// Free memory
		free(out);
	}
    else // Use device fully
    {
        // TODO
        // Allocate device memories
		int * d_in, * d_out;
		dim3 gridSize(1); // TODO: Compute gridSize from n and blockSize
        gridSize.x = ((n - 1) / (2 * blockSize.x) + 1);
		CHECK(cudaMalloc(&d_in, n * sizeof(int)));
		CHECK(cudaMalloc(&d_out, gridSize.x * sizeof(int)));

		// Copy data to device memory
		CHECK(cudaMemcpy(d_in, in, n * sizeof(int), cudaMemcpyHostToDevice));
        int m = n;
		// Call kernel
        while (m > 1)
        {
            gridSize.x = ((m - 1) / (2 * blockSize.x) + 1);
            reduceBlksKernel<<<gridSize, blockSize>>>(d_in, m, d_out);
		    cudaDeviceSynchronize();
		    CHECK(cudaGetLastError());
            m = gridSize.x;
            CHECK(cudaMemcpy(d_in, d_out, m * sizeof(int), cudaMemcpyDeviceToDevice));
        }

		// Copy result from device memory
		int * out = (int *)malloc(m * sizeof(int));
        CHECK(cudaMemcpy(out, d_out, m * sizeof(int), cudaMemcpyDeviceToHost));

		// Free device memories
		CHECK(cudaFree(d_in));
		CHECK(cudaFree(d_out));

		// Host do the rest of the work
		result = out[0];
		for (int i = 1; i < gridSize.x; i++)
			result += out[i];

		// Free memory
		free(out);
    }

	timer.Stop();
	float time = timer.Elapsed();
    if (useDevice == 0)
        printf("\nProcessing time (use host): %f ms\n", time); 
    else if (useDevice == 1)
        printf("\nProcessing time (use device partly): %f ms\n", time); 
    else
        printf("\nProcessing time (use device fully): %f ms\n", time); 

	return result;
}

void checkCorrectness(int r1, int r2)
{
	if (r1 == r2)
		printf("CORRECT :)\n");
	else
		printf("INCORRECT :(\n");
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
    printf("GMEM: %lu bytes\n", devProv.totalGlobalMem);
    printf("****************************\n");
}

int main(int argc, char ** argv)
{
	printDeviceInfo();

	// Set up input size
    int n = (1 << 25) + 1;
    printf("\nInput size: %d\n", n);

    // Set up input data
    int * in = (int *) malloc(n * sizeof(int));
    for (int i = 0; i < n; i++)
    {
        // Generate a random integer in [0, 255]
        in[i] = (int)(rand() & 0xFF);
    }

    // Reduce NOT using device
    int correctResult = reduce(in, n);

    // Reduce using device partly
    dim3 blockSize(512); // Default
    if (argc == 2)
    	blockSize.x = atoi(argv[1]);
    int result1 = reduce(in, n, 1, blockSize);
    checkCorrectness(result1, correctResult);

    // Reduce using device fully
    int result2 = reduce(in, n, 2, blockSize);
    checkCorrectness(result2, correctResult);

    // Free memories
    free(in);
}
