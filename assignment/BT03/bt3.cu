#include <stdio.h>
#include <stdint.h>

#define FILTER_WIDTH 9
__constant__ float dc_filter[FILTER_WIDTH * FILTER_WIDTH];

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

void readPnm(char * fileName, int &width, int &height, uchar3 * &pixels)
{
	FILE * f = fopen(fileName, "r");
	if (f == NULL)
	{
		printf("Cannot read %s\n", fileName);
		exit(EXIT_FAILURE);
	}

	char type[3];
	fscanf(f, "%s", type);
	
	if (strcmp(type, "P3") != 0)
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); // In this exercise, we don't touch other types
		exit(EXIT_FAILURE); 
	}

	fscanf(f, "%i", &width);
	fscanf(f, "%i", &height);
	
	uint8_t max_val;
	fscanf(f, "%hhu", &max_val);
	if (max_val > 255)
	{
		fclose(f);
		printf("Cannot read %s\n", fileName); // In this exercise, we assume 1 byte per value
		exit(EXIT_FAILURE); 
	}

	pixels = (uchar3 *)malloc(width * height * sizeof(uchar3));
	for (int i = 0; i < width * height; i++)
		fscanf(f, "%hhu%hhu%hhu", &pixels[i].x, &pixels[i].y, &pixels[i].z);

	fclose(f);
}

void writePnm(char * fileName, int width, int height, uchar3 * pixels)
{
	FILE * f = fopen(fileName, "w");
	if (f == NULL)
	{
		printf("Cannot write %s\n", fileName);
		exit(EXIT_FAILURE);
	}	

	fprintf(f, "P3\n%i\n%i\n255\n", width, height); 

	for (int i = 0; i < width * height; i++)
		fprintf(f, "%hhu\n%hhu\n%hhu\n", pixels[i].x, pixels[i].y, pixels[i].z);
	
	fclose(f);
}

void compare2Pnms(char * fileName1, char * fileName2)
{
	int width1, height1;
	uchar3 * pixels1;
	readPnm(fileName1, width1, height1, pixels1);

	int width2, height2;
	uchar3 * pixels2;
	readPnm(fileName2, width2, height2, pixels2);

	if (width1 != width2)
	{
		printf("'%s' is DIFFERENT from '%s' (width: %i vs %i)\n\n", fileName1, fileName2, width1, width2);
		return;
	}
	if (height1 != height2)
	{
		printf("'%s' is DIFFERENT from '%s' (width: %i vs %i)\n\n", fileName1, fileName2, height1, height2);
		return;
	}
	float mae = 0;
	for (int i = 0; i < width1 * height1; i++)
	{
		mae += abs((int)pixels1[i].x - (int)pixels2[i].x);
		mae += abs((int)pixels1[i].y - (int)pixels2[i].y);
		mae += abs((int)pixels1[i].z - (int)pixels2[i].z);
	}
	mae /= (width1 * height1 * 3);
	printf("The average pixel difference between '%s' and '%s': %f\n\n", fileName1, fileName2, mae);
}

void blurImgByHost(uchar3 * inPixels, uchar3 * outPixels, int width, int height, 
					float * filter, int filterWidth)
{
	GpuTimer timer;
	timer.Start();
	for (int outPixelsR = 0; outPixelsR < height; outPixelsR++)
	{
		for (int outPixelsC = 0; outPixelsC < width; outPixelsC++)
		{
			float3 outPixel = make_float3(0, 0, 0);
			for (int filterR = 0; filterR < filterWidth; filterR++)
			{
				for (int filterC = 0; filterC < filterWidth; filterC++)
				{
					float filterVal = filter[filterR * filterWidth + filterC];

					int inPixelsR = (outPixelsR - filterWidth/2) + filterR;
					int inPixelsC = (outPixelsC - filterWidth/2) + filterC;
					inPixelsR = min(height - 1, max(0, inPixelsR)); // Chỉnh lại chỉ số dòng nếu < 0 hoặc > heigh-1
					inPixelsC = min(width - 1, max(0, inPixelsC));  // Chỉnh lại chỉ số cột nếu < 0 hoặc > width-1
 					uchar3 inPixel = inPixels[inPixelsR * width + inPixelsC];
					
					outPixel.x += filterVal * inPixel.x;
					outPixel.y += filterVal * inPixel.y;
					outPixel.z += filterVal * inPixel.z;
				}
			}
			outPixels[outPixelsR * width + outPixelsC] = make_uchar3(outPixel.x, outPixel.y, outPixel.z);
		}
	}
	timer.Stop();
	printf("Time of blurImgByHost:    %.3f ms\n", timer.Elapsed());
}

/*
This kernel function is similar to "blurImgByDevice" in BT2. The minor change is 
the type of "inPixels" and "outPixels": uchar3* in stead of uint8_t*. uchar3 is a
struct in CUDA with 3 unsigned char fields: x, y, z (unsigned char is equivalent 
to uint8_t). An uchar3 element corresponds to a pixel; and x, y, z of this uchar3 
element correspond to red, green, blue of the corresponding pixel. This change 
will make accessing pixels easier.
*/
__global__ void blurImgByDevice1(uchar3 * inPixels, uchar3 * outPixels, 
								int width, int height, 
								float * filter, int filterWidth)
{
	// TODO
	int ix = blockIdx.x * blockDim.x + threadIdx.x;
	int iy = blockIdx.y * blockDim.y + threadIdx.y;
	if (ix < width && iy < height)
	{
		int i = iy * width + ix;
		int half = filterWidth / 2;
		float3 G;
		int idx = 0, dy = 0;
		G.x = G.y = G.z = 0.0f;
		for (int filterR = 0; filterR < filterWidth; filterR++)
		{
			for (int filterC = 0; filterC < filterWidth; filterC++)
			{ 
				idx = iy + filterR - half;
				dy = ix + filterC - half;
				if (idx < 0)
					idx = 0;
				if (idx >= height)
					idx = height - 1;
				idx *= width;
				if (dy < 0)
					dy = 0;
				if (dy >= width)
					dy = width - 1;
				idx += dy;
				G.x += filter[filterR * filterWidth + filterC] * inPixels[idx].x; 
				G.y += filter[filterR * filterWidth + filterC] * inPixels[idx].y;
				G.z += filter[filterR * filterWidth + filterC] * inPixels[idx].z;
			}
		}
		outPixels[i].x = G.x;
		outPixels[i].y = G.y;
		outPixels[i].z = G.z;
	}
}

/*
This kernel is the improved version of "blurImgByDevice1". In this kernel, we use
SMEM to "cache" input image of each block.
*/
__global__ void blurImgByDevice2(uchar3 * inPixels, uchar3 * outPixels, 
								int width, int height, 
								float * filter, int filterWidth)
{
	// TODO
	
}

/*
This kernel is the improved version of "blurImgByDevice2". In this kernel, we use
SMEM to "cache" input image of each block + CMEM to store filter
*/
__global__ void blurImgByDevice3(uchar3 * inPixels, uchar3 * outPixels, 
								int width, int height, 
								int filterWidth)
{
	// TODO
	
}

void blurImgByDevice(uchar3 * inPixels, uchar3 * outPixels, int width, int height, 
					float * filter, int filterWidth,
					int kernelType, dim3 blockSize)
{
	if (kernelType < 1 || kernelType > 3)
	{
		fprintf(stderr, "Kernel type is invalid!\n");
		return;
	}

	// Allocate device memories
	uchar3 * d_inPixels, * d_outPixels;
	float * d_filter;
	CHECK(cudaMalloc(&d_inPixels, width * height * sizeof(uchar3)));
	CHECK(cudaMalloc(&d_outPixels, width * height * sizeof(uchar3)));
	if (kernelType != 3)
	{
		CHECK(cudaMalloc(&d_filter, filterWidth * filterWidth * sizeof(float)));
	}

	// Copy data to device memories
	CHECK(cudaMemcpy(d_inPixels, inPixels, width * height * sizeof(uchar3), 
					 cudaMemcpyHostToDevice));
	if (kernelType != 3)
	{
		CHECK(cudaMemcpy(d_filter, filter, filterWidth * filterWidth * sizeof(float), 
			             cudaMemcpyHostToDevice));
	}
	else // kernelType == 3
	{
		// TODO: copy data from filter to dc_filter

	}

	// Call kernel
	GpuTimer timer;
	timer.Start();
	dim3 gridSize((width - 1) / blockSize.x + 1, (height - 1) / blockSize.y + 1);
	if (kernelType == 1)
	{	
		// TODO: Call "blurImgByDevice1" kernel
		blurImgByDevice1<<<gridSize, blockSize>>>(d_inPixels, d_outPixels, width, height, d_filter, filterWidth);
	}
	else if (kernelType == 2)
	{
		// TODO: Call "blurImgByDevice2" kernel
		
	}
	else // kernelType == 3
	{
		// TODO: Call "blurImgByDevice3" kernel
		
	}
	timer.Stop(); 
	printf("Time of blurImgByDevice%d: %.3f ms\n", kernelType, timer.Elapsed());
	cudaDeviceSynchronize();
	CHECK(cudaGetLastError());

	// Copy result from device memories
	CHECK(cudaMemcpy(outPixels, d_outPixels, width * height * sizeof(uchar3), cudaMemcpyDeviceToHost));

	// Free device memories
	CHECK(cudaFree(d_inPixels));
	CHECK(cudaFree(d_outPixels));
	if (kernelType != 3)
	{
		CHECK(cudaFree(d_filter));
	}
}

int main(int argc, char ** argv)
{
	// CHECK ...
	if (argc < 4 || argc > 6)
	{
		printf("The number of arguments is invalid\n");
		return EXIT_FAILURE;
	}

	// PRINT OUT DEVICE INFO
    cudaDeviceProp devProv;
    CHECK(cudaGetDeviceProperties(&devProv, 0));
    printf("**********GPU info**********\n");
    printf("Name: %s\n", devProv.name);
    printf("Compute capability: %d.%d\n", devProv.major, devProv.minor);
    printf("Num SMs: %d\n", devProv.multiProcessorCount);
    printf("Max num threads per SM: %d\n", devProv.maxThreadsPerMultiProcessor); 
    printf("Max num warps per SM: %d\n", devProv.maxThreadsPerMultiProcessor / devProv.warpSize);
    printf("GMEM: %zu byte\n", devProv.totalGlobalMem);
    printf("CMEM: %zu byte\n", devProv.totalConstMem);
    printf("SMEM per SM: %zu byte\n", devProv.sharedMemPerMultiprocessor);
    printf("SMEM per block: %zu byte\n", devProv.sharedMemPerBlock);
    printf("****************************\n\n");

	// READ INPUT IMAGE FILE
	int width, height;
	uchar3 * inPixels;
	readPnm(argv[1], width, height, inPixels);
	printf("Image size (width x height): %i x %i\n\n", width, height);

	// SET UP A SIMPLE FILTER WITH BLURRING EFFECT 
	int filterWidth = FILTER_WIDTH;
	float * filter = (float *)malloc(filterWidth * filterWidth * sizeof(float));
	for (int filterR = 0; filterR < filterWidth; filterR++)
	{
		for (int filterC = 0; filterC < filterWidth; filterC++)
		{
			filter[filterR * filterWidth + filterC] = 1. / (filterWidth * filterWidth);
		}
	}

	// DETERMINE BLOCK SIZE
	dim3 blockSize(32, 32); // Default
	if (argc == 6)
	{
		blockSize.x = atoi(argv[4]);
		blockSize.y = atoi(argv[5]);
	}

	// PREPARE FOR OUTPUT IMAGE
	uchar3 * outPixels= (uchar3 *)malloc(width * height * sizeof(uchar3));
	char * outFileNameBase = strtok(argv[2], "."); // Get rid of extension
	char * outFileName = (char *)malloc(strlen(outFileNameBase) + strlen("_devicex.pnm") + 1);

	// BLUR INPUT IMAGE BY HOST 
	blurImgByHost(inPixels, outPixels, width, height, filter, filterWidth);
	sprintf(outFileName, "%s_host.pnm", outFileNameBase);
	writePnm(outFileName, width, height, outPixels);
	compare2Pnms(outFileName, argv[3]);
	
    // BLUR INPUT IMAGE BY DEVICE, KERNEL 1 (NOT USE SMEM)
    memset(outPixels, 0, width * height * sizeof(uchar3)); // Reset outPixels
	blurImgByDevice(inPixels, outPixels, width, height, filter, filterWidth, 1, blockSize);
    sprintf(outFileName, "%s_device1.pnm", outFileNameBase);
	writePnm(outFileName, width, height, outPixels);
	compare2Pnms(outFileName, argv[3]);

    // BLUR INPUT IMAGE BY DEVICE, KERNEL 2 (USE SMEM TO "CACHE" INPIXELS)
    memset(outPixels, 0, width * height * sizeof(uchar3)); // Reset outPixels
	blurImgByDevice(inPixels, outPixels, width, height, filter, filterWidth, 2, blockSize);
    sprintf(outFileName, "%s_device2.pnm", outFileNameBase);
	writePnm(outFileName, width, height, outPixels);
	compare2Pnms(outFileName, argv[3]);

	// BLUR INPUT IMAGE BY DEVICE, KERNEL 3 (USE SMEM TO "CACHE" INPIXELS
	//										 + CMEM TO STORE FILTER)
    memset(outPixels, 0, width * height * sizeof(uchar3)); // Reset outPixels
	blurImgByDevice(inPixels, outPixels, width, height, filter, filterWidth, 3, blockSize);
    sprintf(outFileName, "%s_device3.pnm", outFileNameBase);
	writePnm(outFileName, width, height, outPixels);
	compare2Pnms(outFileName, argv[3]);

	// CLEAN MEMORIES
	free(inPixels);
	free(outPixels);
	free(filter);
	free(outFileName); 
}