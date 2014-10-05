

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

// STUDENTS: be sure to set the single define at the top of this file,
// depending on which machines you are running on.
#include "im1.h"


using namespace std;
// handy error macro:
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
static void gpuCheckError( cudaError_t err,
                          const char *file,
                          int line ) {
    if (err != cudaSuccess) {
        printf( "%s in %s at line %d\n", cudaGetErrorString( err ),
               file, line );
        exit( EXIT_FAILURE );
    }
}

int getLargestPowerOf2LessThan(double num)
{
	int i = -1;
	for (int num1 = (int)num; num1 != 0; num1 >>= 1)
		i++;
	return 1 << i;

}

int getMaxThreadsPerBlock()
{
	int count;
	GPU_CHECKERROR( cudaGetDeviceCount(&count) );
	int maxThreads = 0;
	int maxThreadDevId = 0;
	cudaDeviceProp devProp;
	for (int i = 0; i < count; i++)
	{
		GPU_CHECKERROR( cudaGetDeviceProperties(&devProp, i) );
		if (devProp.maxThreadsPerBlock > maxThreads)
		{
			maxThreadDevId = i;
			maxThreads = devProp.maxThreadsPerBlock;
		}
	}
	GPU_CHECKERROR( cudaSetDevice (maxThreadDevId) );
	return maxThreads;
}

__global__ void rgbToGrayParallel(float *dev_imageArray, unsigned int offset, unsigned int size)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	if (x >= size)
		return;
	unsigned int idx = (offset + x) * 3;

	float L = 0.2126f*dev_imageArray[idx] +
		  0.7152f*dev_imageArray[idx+1] +
		  0.0722f*dev_imageArray[idx+2];

	dev_imageArray[idx] = L;
	dev_imageArray[idx+1] = L;
	dev_imageArray[idx+2] = L;
}



int main (int argc, char *argv[])
{

	if (argc < 3 || argc > 3)
	{
		cout<<"\n Wrong number of arguments!!!";
		return -1;
	}
    printf("reading openEXR file %s\n", argv[1]);
    int numChunks  = strtol(argv[2], NULL, 10);

    int w, h;   // the width & height of the image, used frequently!

    // First, convert the openEXR file into a form we can use on the CPU
    // and the GPU: a flat array of floats:
    // This makes an array h*w*sizeof(float)*3, with sequential r/g/b indices
    // don't forget to free it at the end


    float *h_imageArray, *h_imageArrayCpy;
    readOpenEXRFile (argv[1], &h_imageArray, w, h);
    unsigned int DATA_SIZE = w * h;
    GPU_CHECKERROR( cudaHostAlloc ( (void **) &h_imageArrayCpy,
    					sizeof(float) * 3 * DATA_SIZE, cudaHostAllocDefault) );
    memcpy((void*) h_imageArrayCpy , (void *) h_imageArray , sizeof(float) * 3 * DATA_SIZE);
    free(h_imageArray);


    float *dev_imageArray;
    unsigned int chunkSize = ceil( (double) DATA_SIZE / numChunks);
    GPU_CHECKERROR( cudaMalloc((void**) &dev_imageArray, 3 * DATA_SIZE * sizeof(float)) );

    //Create Streams
    cudaStream_t streams[numChunks];
    for (int i = 0; i < numChunks; i++)
    	GPU_CHECKERROR( cudaStreamCreate(&streams[i]) );

    cudaEvent_t start, stop;
    GPU_CHECKERROR( cudaEventCreate ( &start ) );
    GPU_CHECKERROR( cudaEventCreate ( &stop ) );

    GPU_CHECKERROR( cudaEventRecord(start, 0) );
    //Copy image to device in chunks, each chunk in a separate stream
    for (int i = 0; i < numChunks; i++)
    {
    	unsigned int offset = i * chunkSize;
    	if (offset + chunkSize < DATA_SIZE)
			GPU_CHECKERROR( cudaMemcpyAsync((void *) (dev_imageArray + offset * 3),
											(void *) (h_imageArrayCpy + offset * 3),
											3 * chunkSize * sizeof(float),
											cudaMemcpyHostToDevice,
											streams[i]) );
    	else
    	{
    		GPU_CHECKERROR( cudaMemcpyAsync((void *) (dev_imageArray + offset * 3),
											(void *) (h_imageArrayCpy + offset * 3),
											3 * (DATA_SIZE - offset) * sizeof(float),
											cudaMemcpyHostToDevice,
											streams[i]) );
    		break;
    	}
    }

    //Invoke kernels in separate streams
    for (int i = 0; i < numChunks; i++)
    {
    	int maxThreads = getMaxThreadsPerBlock();
    	//int xDim = getLargestPowerOf2LessThan(ceil(sqrt(maxThreads)));
    	//dim3 threadsPerBlock(xDim, xDim, 1);
    	//dim3 numBlocks(ceil((double)w / threadsPerBlock.x), ceil((double)h / threadsPerBlock.y));
    	int threadsPerBlock = maxThreads;
    	unsigned int offset = i * chunkSize;
    	if (offset + chunkSize < DATA_SIZE)
    	{
    		unsigned int numBlocks = ceil((double)chunkSize / maxThreads);
    		rgbToGrayParallel<<<numBlocks, threadsPerBlock, 0, streams[i]>>>(dev_imageArray, offset, chunkSize);
    	}
    	else
    	{
    		unsigned int numBlocks = ceil((double) (DATA_SIZE - offset) / maxThreads);
    		rgbToGrayParallel<<<numBlocks , threadsPerBlock, 0, streams[i]>>>(dev_imageArray, offset, DATA_SIZE - offset);
    		break;
    	}
    }

    for (int i = 0; i < numChunks; i++)
	{
		unsigned int offset = i * chunkSize;
		if (offset + chunkSize < DATA_SIZE)
			GPU_CHECKERROR( cudaMemcpyAsync((void *) (h_imageArrayCpy + offset * 3),
											(void *) (dev_imageArray + offset * 3),
											3 * chunkSize * sizeof(float),
											cudaMemcpyDeviceToHost,
											streams[i]) );
		else
		{
			GPU_CHECKERROR( cudaMemcpyAsync((void *) (h_imageArrayCpy + offset * 3),
											(void *) (dev_imageArray + offset * 3),
											3 * (DATA_SIZE - offset) * sizeof(float),
											cudaMemcpyDeviceToHost,
											streams[i]) );
			break;
		}
	}

    //Wait for streams to finish
    for (int i = 0; i < numChunks; i++)
    	GPU_CHECKERROR( cudaStreamSynchronize( streams[i] ) );
    
    GPU_CHECKERROR( cudaEventRecord(stop, 0) );
    GPU_CHECKERROR( cudaEventSynchronize (stop) );
    float time_delta;
    GPU_CHECKERROR( cudaEventElapsedTime(&time_delta, start, stop) );
    cout<<endl<<"Time taken to gray the image: "<<time_delta<<" ms.";

    for (int i = 0; i < numChunks; i++)
        GPU_CHECKERROR( cudaStreamDestroy(streams[i]) );


    printf("writing output image hw2_gpu.exr\n");
    writeOpenEXRFile ("hw2_gpu.exr", h_imageArrayCpy, w, h);
    GPU_CHECKERROR( cudaFreeHost(h_imageArrayCpy) );
    GPU_CHECKERROR( cudaFree(dev_imageArray) );

    printf("done.\n");

    return 0;
}


