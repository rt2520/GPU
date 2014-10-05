

#include <iostream>
#include <stdio.h>
#include <cmath>
#include <cuda.h>
#include <sys/time.h>

// STUDENTS: be sure to set the single define at the top of this file, 
// depending on which machines you are running on.
#include "im1.h"

using namespace std;

#define _USE_MATH_DEFINES
// handy error macro:
#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
#define RADIUS_THRESH 200

//__constant__ float *dev_gaussian;

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

__global__ void rgbToGrayParallel(float *dev_imageArray, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;
	int idx = ((y * width) + x) * 3;

	float L = 0.2126f*dev_imageArray[idx] +
			0.7152f*dev_imageArray[idx+1] +
			0.0722f*dev_imageArray[idx+2];

	dev_imageArray[idx] = L;
	dev_imageArray[idx+1] = L;
	dev_imageArray[idx+2] = L;
}

__global__ void separateRGBParallel(float *dev_monoImagesRed, float *dev_monoImagesGreen, float *dev_monoImagesBlue, float *dev_imageArray, int width, int height)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	if (x >= width || y >= height)
		return;
	int idx = ((y * width) + x) * 3;
	dev_monoImagesRed[y * width + x] = dev_imageArray[idx];
	dev_monoImagesGreen[y * width + x] = dev_imageArray[idx + 1];
	dev_monoImagesBlue[y * width + x] = dev_imageArray[idx + 2];

}

__global__ void gaussianBlurParallel(float *dev_blurredArray, float *dev_imageArray,
										float *gaussian, int width, int height, int radius)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;
	//if (x >= width || y >= height)
	//	return;
	//dev_blurredArray[(y * width + x) * 3 + color] = 1.0;
	//return;

	if (x >= (width + radius) || y >= (height + radius))
		return;
	int sharedWidth = blockDim.x + 2 * radius;
	
	extern __shared__ float workingMemory[];
	//float workingMemory[10][10];
	int w2 = width + 2 * radius;
	int h2 = height + 2 * radius;
	int y2 = y + radius;
	int x2 = x + radius;
	int writeIdx = ((threadIdx.y + radius) * sharedWidth + threadIdx.x + radius) * 3;
	int readIdx = (y2 * w2 + x2) * 3;

	workingMemory[writeIdx] = dev_imageArray[readIdx];
	workingMemory[writeIdx + 1] = dev_imageArray[readIdx + 1];
	workingMemory[writeIdx + 2] = dev_imageArray[readIdx + 2];
	if (threadIdx.x < radius)
	{
		writeIdx = ((threadIdx.y + radius) * sharedWidth + threadIdx.x) * 3;
		readIdx = (y2 * w2 + (x2 - radius)) * 3;
		workingMemory[writeIdx] = dev_imageArray[readIdx];
		workingMemory[writeIdx + 1] = dev_imageArray[readIdx + 1];
	        workingMemory[writeIdx + 2] = dev_imageArray[readIdx + 2];

	}
	if (threadIdx.y < radius)
	{
		writeIdx = ((threadIdx.y) * sharedWidth + threadIdx.x + radius) * 3;
		readIdx = ((y2 - radius) * w2 + x2) * 3;
		workingMemory[writeIdx] = dev_imageArray[readIdx];
		workingMemory[writeIdx + 1] = dev_imageArray[readIdx + 1];
		workingMemory[writeIdx + 2] = dev_imageArray[readIdx + 2];
	}
	if (threadIdx.x < radius && threadIdx.y < radius)
	{
		writeIdx = ((threadIdx.y) * sharedWidth + threadIdx.x) * 3;
		readIdx = ((y2 - radius) * w2 + (x2 - radius)) * 3;
		workingMemory[writeIdx] = dev_imageArray[readIdx];
		workingMemory[writeIdx + 1] = dev_imageArray[readIdx + 1];
		workingMemory[writeIdx + 2] = dev_imageArray[readIdx + 2];
	}
	if (blockIdx.x * blockDim.x + blockDim.x < width)
	{
		if (threadIdx.x >= blockDim.x - radius)
		{
			writeIdx = ((threadIdx.y + radius) * sharedWidth + threadIdx.x + 2 * radius) * 3;
			readIdx = (y2 * w2 + (x2 + radius)) * 3;
			workingMemory[writeIdx] = dev_imageArray[readIdx];
			workingMemory[writeIdx + 1] = dev_imageArray[readIdx + 1];
			workingMemory[writeIdx + 2] = dev_imageArray[readIdx + 2];
		}
		if (threadIdx.x < radius && threadIdx.y < radius)
		{
			writeIdx = ((threadIdx.y) * sharedWidth + radius + threadIdx.x + blockDim.x) * 3;
			readIdx = ((y2 - radius) * w2 + (x2 + blockDim.x)) * 3;
			workingMemory[writeIdx] = dev_imageArray[readIdx];
			workingMemory[writeIdx + 1] = dev_imageArray[readIdx + 1];
			workingMemory[writeIdx + 2] = dev_imageArray[readIdx + 2];
		}
	}
	if (blockIdx.y * blockDim.y + blockDim.y < height)
	{
		if (threadIdx.y >= blockDim.y - radius)
		{
			writeIdx = ((threadIdx.y + 2 * radius) * sharedWidth + threadIdx.x + radius) * 3;
			readIdx = ((y2 + radius) * w2 + x2) * 3;
			workingMemory[writeIdx] = dev_imageArray[readIdx];
			workingMemory[writeIdx + 1] = dev_imageArray[readIdx + 1];
			workingMemory[writeIdx + 2] = dev_imageArray[readIdx + 2];
		}
		if (threadIdx.x < radius && threadIdx.y < radius)
		{
			writeIdx = ((radius + threadIdx.y + blockDim.y) * sharedWidth + threadIdx.x) * 3;
			readIdx = ((y2 + blockDim.y) * w2 + (x2 - radius)) * 3;
			workingMemory[writeIdx] = dev_imageArray[readIdx];
			workingMemory[writeIdx + 1] = dev_imageArray[readIdx + 1];
			workingMemory[writeIdx + 2] = dev_imageArray[readIdx + 2];
		}
	}
	if (blockIdx.x * blockDim.x + blockDim.x < width && blockIdx.y * blockDim.y + blockDim.y < height)
	{
		if (threadIdx.x < radius && threadIdx.y < radius)
		{
			writeIdx = ((radius + threadIdx.y  + blockDim.y) * sharedWidth + radius + threadIdx.x + blockDim.x) * 3;
			readIdx = ((y2 + blockDim.y) * w2 + (x2 + blockDim.x)) * 3;
			workingMemory[writeIdx] = dev_imageArray[readIdx];
			workingMemory[writeIdx + 1] = dev_imageArray[readIdx + 1];
			workingMemory[writeIdx + 2] = dev_imageArray[readIdx + 2];
		}
	}
	__syncthreads();

	if (x >= width || y >= height)
		return;
	float result1 = 0.0;
	float result2 = 0.0;
	float result3 = 0.0;
	int kernelSize = 2 * radius + 1;
	/*for (int i = -radius; i <= radius; i++)
		for (int j = -radius; j <= radius; j++)
		{
			int idx1 = (radius + i) * (2 * radius + 1) + j + radius;
			int idx2 = ((threadIdx.y + radius + i) * sharedWidth + threadIdx.x + radius + j) * 3;
			result1 += gaussian[idx1] * workingMemory[idx2];
			result2 += gaussian[idx1] * workingMemory[idx2 + 1];
			result3 += gaussian[idx1] * workingMemory[idx2 + 2];
		}*/
	//Reducing the number of multiplications using the symmetry of the Gaussian kernel
        for (int i = -radius; i <= 0; i++)
                for (int j = -radius; j <= 0; j++)
                {
                        int y1 = radius + i;
                        int x1 = radius + j;
                        int y2 = radius - i;
                        int x2 = radius - j;
                        float tmpR = workingMemory[((threadIdx.y + y1) * sharedWidth + threadIdx.x + x1) * 3];
                        float tmpG = workingMemory[((threadIdx.y + y1) * sharedWidth + threadIdx.x + x1) * 3 + 1];
                        float tmpB = workingMemory[((threadIdx.y + y1) * sharedWidth + threadIdx.x + x1) * 3 + 1];
                        if (i != 0)
                        {
                                tmpR += workingMemory[((threadIdx.y + y2) * sharedWidth + threadIdx.x + x1) * 3];
                                tmpG += workingMemory[((threadIdx.y + y2) * sharedWidth + threadIdx.x + x1) * 3 + 1];
                                tmpB += workingMemory[((threadIdx.y + y2) * sharedWidth + threadIdx.x + x1) * 3 + 2];
                        }
                        if (j != 0)
                        {
                                tmpR += workingMemory[((threadIdx.y + y1) * sharedWidth + threadIdx.x + x2) * 3];
                                tmpG += workingMemory[((threadIdx.y + y1) * sharedWidth + threadIdx.x + x2) * 3 + 1];
                                tmpB += workingMemory[((threadIdx.y + y1) * sharedWidth + threadIdx.x + x2) * 3 + 2];
                                if (i != 0)
                                {
                                        tmpR += workingMemory[((threadIdx.y + y2) * sharedWidth + threadIdx.x + x2) * 3];
                                        tmpG += workingMemory[((threadIdx.y + y2) * sharedWidth + threadIdx.x + x2) * 3 + 1];
                                        tmpB += workingMemory[((threadIdx.y + y2) * sharedWidth + threadIdx.x + x2) * 3 + 2];
                                }
                        }
                        result1 += gaussian[y1 * kernelSize + x1] * tmpR;
                        result2 += gaussian[y1 * kernelSize + x1] * tmpG;
                        result3 += gaussian[y1 * kernelSize + x1] * tmpB;
                }
	//result1 = workingMemory[((threadIdx.y + radius) * sharedWidth + threadIdx.x + radius) * 3];
	//result2 = workingMemory[((threadIdx.y + radius) * sharedWidth + threadIdx.x + radius) * 3 + 1];
	//result3 = workingMemory[((threadIdx.y + radius) * sharedWidth + threadIdx.x + radius) * 3 + 2];
	dev_blurredArray[(y * width + x) * 3] = result1;
	dev_blurredArray[(y * width + x) * 3 + 1] = result2;
	dev_blurredArray[(y * width + x) * 3 + 2] = result3;
}

__global__ void gaussianBlurSpecialBlockParallel(float *dev_blurredArray, float *dev_imageArray,
										float *gaussian, int width, int height, int radius)
{
	int x = blockIdx.x * blockDim.x + threadIdx.x;
	int y = blockIdx.y * blockDim.y + threadIdx.y;

	//This is for making contiguous reads possible because now
	//each thread can read 4 contiguous memory locations
	//Block size is 2 times radius
	int sharedWidth = blockDim.x + 2 * radius;
	int sharedHeight = blockDim.y + 2 * radius;

	extern __shared__ float workingMemory[];
	//float workingMemory[10][10];
	int w2 = width + 2 * radius;
	int h2 = height + 2 * radius;
	//int y2 = y + radius;
	//int x2 = x + radius;
	
	//int threadNum = threadIdx.y * blockDim.x + threadIdx.x;
	//int imageReadIdx = ((threadNum / radius) + blockIdx.y * blockDim.y) * w2 + 4 * (threadNum % radius) + blockIdx.x * blockDim.x;//blockIdx.y * blockDim.y * w2 + blockIdx.x * blockDim.x;


	//Now read 4 pixels from the imageRead start + offset into working memory
	//(store starting from offset in the working memory)
	/*for (int i = 0; i < 12; i++)
	{
		workingMemory[threadNum * 12 + i] = dev_imageArray[imageReadIdx * 3 + i];
	}*/

	//threadNum *= 12;
	//imageReadIdx *= 3;

	/*workingMemory[threadNum] = dev_imageArray[imageReadIdx];
        workingMemory[threadNum + 1] = dev_imageArray[imageReadIdx + 1];
        workingMemory[threadNum + 2] = dev_imageArray[imageReadIdx + 2];
        workingMemory[threadNum + 3] = dev_imageArray[imageReadIdx + 3];
        workingMemory[threadNum + 4] = dev_imageArray[imageReadIdx + 4];
        workingMemory[threadNum + 5] = dev_imageArray[imageReadIdx + 5];
        workingMemory[threadNum + 6] = dev_imageArray[imageReadIdx + 6];
        workingMemory[threadNum + 7] = dev_imageArray[imageReadIdx + 7];
        workingMemory[threadNum + 8] = dev_imageArray[imageReadIdx + 8];
        workingMemory[threadNum + 9] = dev_imageArray[imageReadIdx + 9];
        workingMemory[threadNum + 10] = dev_imageArray[imageReadIdx + 10];
        workingMemory[threadNum + 11] = dev_imageArray[imageReadIdx + 11]; */
	//Now read 4 pixels from the imageRead start + offset into working memory
        //(store starting from offset in the working memory)
        //for (int i = 0; i < 12; i++)
        //{
	//	if (imageReadIdx < 3 * w2 * h2)
        //        workingMemory[threadNum + i] = dev_imageArray[imageReadIdx + i];
        //}
	//__syncthreads();
	int maxi = (blockDim.y + 2 * radius) / blockDim.y;	
	int maxj = (blockDim.x + 2 * radius) / blockDim.x;
	if ((blockDim.y + 2 * radius) % blockDim.y > 0)
		maxi++;
	if ((blockDim.x + 2 * radius) % blockDim.x > 0) 
                maxj++;	
	for (int i = 0; i < maxi; i++)
		for (int j = 0; j < maxj; j++)
		{
			int threadNum = (i * blockDim.y + threadIdx.y) * sharedWidth + (j * blockDim.x + threadIdx.x);
        		int imageReadIdx = (i * blockDim.y + threadIdx.y + blockIdx.y * blockDim.y) * w2 + j * blockDim.x + threadIdx.x + blockIdx.x * blockDim.x;//blockIdx.y * blockDim.y * w2 + blockIdx.x * blockDim.x;
			threadNum *= 3;
		        imageReadIdx *= 3;
			if (imageReadIdx <= 3 * w2 * h2 - 3 && threadNum <= 3 * sharedWidth * sharedHeight - 3)
			{	
	        	        workingMemory[threadNum] = dev_imageArray[imageReadIdx];
	        	        workingMemory[threadNum + 1] = dev_imageArray[imageReadIdx + 1];
	        	        workingMemory[threadNum + 2] = dev_imageArray[imageReadIdx + 2];
			}		
			__syncthreads();
		}
	
	

	if (x >= width || y >= height)
		return;
	float result1 = 0.0;
	float result2 = 0.0;
	float result3 = 0.0;
	int kernelSize = 2 * radius + 1;
	/*for (int i = -radius; i <= radius; i++)
		for (int j = -radius; j <= radius; j++)
		{
			int idx1 = (radius + i) * (2 * radius + 1) + j + radius;
			int idx2 = ((threadIdx.y + radius + i) * sharedWidth + threadIdx.x + radius + j) * 3;
			result1 += gaussian[idx1] * workingMemory[idx2];
			result2 += gaussian[idx1] * workingMemory[idx2 + 1];
			result3 += gaussian[idx1] * workingMemory[idx2 + 2];
		}*/
	//Reducing the number of multiplications using the symmetry of the Gaussian kernel
	for (int i = -radius; i <= 0; i++)
                for (int j = -radius; j <= 0; j++)
                {
                        int y1 = radius + i;
                        int x1 = radius + j;
                        int y2 = radius - i;
                        int x2 = radius - j;
			float tmpR = workingMemory[((threadIdx.y + y1) * sharedWidth + threadIdx.x + x1) * 3];
			float tmpG = workingMemory[((threadIdx.y + y1) * sharedWidth + threadIdx.x + x1) * 3 + 1];
			float tmpB = workingMemory[((threadIdx.y + y1) * sharedWidth + threadIdx.x + x1) * 3 + 1];
                        if (i != 0)
                        {
				tmpR += workingMemory[((threadIdx.y + y2) * sharedWidth + threadIdx.x + x1) * 3];
				tmpG += workingMemory[((threadIdx.y + y2) * sharedWidth + threadIdx.x + x1) * 3 + 1];
				tmpB += workingMemory[((threadIdx.y + y2) * sharedWidth + threadIdx.x + x1) * 3 + 2];
                        }
                        if (j != 0)
                        {
				tmpR += workingMemory[((threadIdx.y + y1) * sharedWidth + threadIdx.x + x2) * 3];
				tmpG += workingMemory[((threadIdx.y + y1) * sharedWidth + threadIdx.x + x2) * 3 + 1];
				tmpB += workingMemory[((threadIdx.y + y1) * sharedWidth + threadIdx.x + x2) * 3 + 2];
                                if (i != 0)
				{
					tmpR += workingMemory[((threadIdx.y + y2) * sharedWidth + threadIdx.x + x2) * 3];
					tmpG += workingMemory[((threadIdx.y + y2) * sharedWidth + threadIdx.x + x2) * 3 + 1];
					tmpB += workingMemory[((threadIdx.y + y2) * sharedWidth + threadIdx.x + x2) * 3 + 2];
				}
                        }
			result1 += gaussian[y1 * kernelSize + x1] * tmpR;
			result2 += gaussian[y1 * kernelSize + x1] * tmpG;
			result3 += gaussian[y1 * kernelSize + x1] * tmpB;
                }
	//result1 = workingMemory[((threadIdx.y + radius) * sharedWidth + threadIdx.x + radius) * 3];
	//result2 = workingMemory[((threadIdx.y + radius) * sharedWidth + threadIdx.x + radius) * 3 + 1];
	//result3 = workingMemory[((threadIdx.y + radius) * sharedWidth + threadIdx.x + radius) * 3 + 2];
	dev_blurredArray[(y * width + x) * 3] = result1;
	dev_blurredArray[(y * width + x) * 3 + 1] = result2;
	dev_blurredArray[(y * width + x) * 3 + 2] = result3;
}


void populateImageBlock(float *arr, int startx, int starty, int width, int w, int h, float val1, float val2, float val3)
{
	//	for (int i = startx; i < startx + width; i++)
	//		for (int j = starty; j < starty + width; j++)
	if (startx + width > w || starty + width > h)
		return;
	for (int i = 0; i < width; i++)
		for (int j = 0; j < width; j++)
		{
			int index = (((startx + i) * w) + (starty + j)) * 3;
			*(arr + index) = val1;
			*(arr + index + 1) = val2;
			*(arr + index + 2) = val3;
		}
}


int main (int argc, char *argv[])
{

	if (argc != 3)
	{
		cout<<"Wrong arguments!! Exiting!!";
		return -1;
	}


	cout<<"Trying to blur: "<<argv[1]<<" with a 2D filter of radius: "<<argv[2];

	int w, h;   // the width & height of the image, used frequently!


	// First, convert the openEXR file into a form we can use on the CPU
	// and the GPU: a flat array of floats:
	// This makes an array h*w*sizeof(float)*3, with sequential r/g/b indices
	// don't forget to free it at the end


	float *h_imageArray, *h_imageArrayCpy;
	readOpenEXRFile (argv[1], &h_imageArray, w, h);
	struct timeval t0, t1;
	gettimeofday(&t0, NULL);
	int radius  = strtol(argv[2], NULL, 10);
	int kernelSize = 2 * radius + 1;
	float *gaussian = new float[kernelSize * kernelSize];

	//
	// serial code: saves the image in "hw1_serial.exr"
	//

	// for every pixel in p, get it's Rgba structure, and convert the
	// red/green/blue values there to luminance L, effectively converting
	// it to greyscale:

	/*for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {

            unsigned int idx = ((y * w) + x) * 3;

            float L = 0.2126f*h_imageArray[idx] + 
                      0.7152f*h_imageArray[idx+1] + 
                      0.0722f*h_imageArray[idx+2];

            h_imageArray[idx] = L;
            h_imageArray[idx+1] = L;
            h_imageArray[idx+2] = L;

       }
    }*/
	int w2 = w + 2 * radius;
	int h2 = h + 2 * radius;
	//TODO make it pinned memory
	//h_imageArrayCpy  = new float[3 * (w2) * (h2)];
	GPU_CHECKERROR( cudaHostAlloc ( (void **) &h_imageArrayCpy, sizeof(float) * 3 * w2 * h2, cudaHostAllocDefault) );

	populateImageBlock(h_imageArrayCpy, 0, 0, radius,
			w2, h2, h_imageArray[0], h_imageArray[1], h_imageArray[2]);
	populateImageBlock(h_imageArrayCpy, 0, w + radius, radius,
			w2, h2, h_imageArray[3 * (w - 1)], h_imageArray[3 * (w - 1) + 1], h_imageArray[3 * (w - 1) + 2]);
	populateImageBlock(h_imageArrayCpy, h + radius, 0, radius,
			w2, h2, h_imageArray[3 * (h - 1) * w], h_imageArray[3 * (h - 1) * w + 1], h_imageArray[3 * (h - 1) * w + 2]);
	populateImageBlock(h_imageArrayCpy, h + radius, w + radius, radius,
			w2, h2, h_imageArray[3 * (w * h -1)], h_imageArray[3 * (w * h -1) + 1], h_imageArray[3 * (w * h -1) + 2]);
	for (int i = 0; i < radius; i++)
		memcpy((void *)(h_imageArrayCpy + 3 * (i * w2 + radius)), (void*) h_imageArray, sizeof(float) * 3 * w);
	for (int i = 0; i < h; i++)
		memcpy((void *)(h_imageArrayCpy + 3 * ((i+radius) * w2 + radius)),(void*) (h_imageArray + 3 * i * w), sizeof(float) * 3 * w);
	for (int i = h + radius; i < h2; i++)
		memcpy((void *)(h_imageArrayCpy + 3 * (i * w2 + radius)), (void*) (h_imageArray + 3 * (h-1) * w), sizeof(float) * 3 * w);

	for (int i = radius; i < h + radius; i++)
	{
		for (int j = radius - 1; j >= 0; j--)
		{
			int idx = ((i * w2) + j) * 3;
			h_imageArrayCpy[idx] = h_imageArrayCpy[idx + 3];
			h_imageArrayCpy[idx + 1] = h_imageArrayCpy[idx + 4];
			h_imageArrayCpy[idx + 2] = h_imageArrayCpy[idx + 5];
		}
		for (int j = w + radius; j < w2; j++)
		{
			int idx = ((i * w2) + j) * 3;
			h_imageArrayCpy[idx] = h_imageArrayCpy[idx - 3];
			h_imageArrayCpy[idx + 1] = h_imageArrayCpy[idx - 2];
			h_imageArrayCpy[idx + 2] = h_imageArrayCpy[idx - 1];
		}

	}
	//writeOpenEXRFile ("padded.exr", h_imageArrayCpy, w2, h2);

	//memcpy((void*) h_imageArrayCpy, (void*) h_imageArray);
	//Calculating Gaussian kernel
	float sigma = (float)radius / 3;
	float sum = 0;
	/*for (int i = -radius; i <= radius; i++)
		for (int j = -radius; j <= radius; j++)
		{
			int y = i + radius;
			int x = j + radius;
			gaussian[y * kernelSize + x] = exp (-1 * (i * i + j * j) / (2 * sigma * sigma)) / (2 * sigma * sigma * M_PI);
			sum += gaussian[y * kernelSize + x];
		}*/
	for (int i = -radius; i <= 0; i++)
                for (int j = -radius; j <= 0; j++)
                {
                        int y1 = radius + i;
                        int x1 = radius + j;
                        int y2 = radius - i;
                        int x2 = radius - j;
			int pow = 0;
                        gaussian[y1 * kernelSize + x1] = exp (-1 * (i * i + j * j) / (2 * sigma * sigma)) / (2 * sigma * sigma * M_PI);
			if (i != 0)
			{
				pow++;
				gaussian[y2 * kernelSize + x1] = gaussian[y1 * kernelSize + x1];
			}
			if (j != 0)
			{
                                pow++;
                                gaussian[y1 * kernelSize + x2] = gaussian[y1 * kernelSize + x1];
				if (i != 0)
                                	gaussian[y2 * kernelSize + x2] = gaussian[y1 * kernelSize + x1];
                        }

                        sum += (gaussian[y1 * kernelSize + x1] * (1 << pow));
                }
	//Normalize
	for (int i = 0; i < kernelSize; i++)
		for (int j = 0; j < kernelSize; j++)
		{
			gaussian[i * kernelSize + j] /= sum;
		}



	/*for (int y = 0; y < h; ++y) {
		for (int x = 0; x < w; ++x) {
			float tmpR = 0.0;
			float tmpG = 0.0;
			float tmpB = 0.0;
			unsigned int idx = ((y * w) + x) * 3;
			unsigned int idx1 = (((y+radius) * w2) + x + radius) * 3;
			for (int i = -radius; i <= radius; i++
			)
				for (int j = -radius; j <= radius; j++)
				{
					int offset = (i * w2 + j) * 3;
					tmpR += gaussian[(i + radius) * kernelSize + j + radius]
					                 * h_imageArrayCpy[idx1 + offset];
					tmpG += gaussian[(i + radius) * kernelSize + j + radius]
					                 * h_imageArrayCpy[idx1 + offset + 1];
					tmpB += gaussian[(i + radius) * kernelSize + j + radius]
					                 * h_imageArrayCpy[idx1 + offset + 2];
				}
			h_imageArray[idx] = tmpR;//h_imageArrayCpy[idx1];
			h_imageArray[idx + 1] = tmpG;//h_imageArrayCpy[idx1];
			h_imageArray[idx + 2] = tmpB;//h_imageArrayCpy[idx1];
		}
	}*/
	//memcpy((void*) h_imageArray, (void*) h_imageArrayCpy);

	//printf("writing output image hw1_serial.exr\n");
	//writeOpenEXRFile ("hw1_serial.exr", h_imageArray, w, h);
	//	writeOpenEXRFile ("hw1_serial.exr", h_imageArrayCpy, w2, h2);
	//writeOpenEXRFile ("hw1_serial.exr", h_imageArrayCpy, w, h);
	//free(h_imageArray); // make sure you free it: if you use this variable
	// again, readOpenEXRFile will allocate more memory
	//delete[] h_imageArrayCpy;

	//
	// Now the GPU version: it will save whatever is in h_imageArray
	// to the file "hw1_gpu.exr"
	//

	//    // read the file again - the file read allocates memory for h_imageArray:
	//readOpenEXRFile (argv[1], &h_imageArray, w, h);


	float *dev_imageArray;
	GPU_CHECKERROR(cudaMalloc((void**) &dev_imageArray, 3 * w2 * h2 * sizeof(float)));
	GPU_CHECKERROR( cudaMemcpy((void*) dev_imageArray,
				(void*) h_imageArrayCpy, 3 * w2 * h2 * sizeof(float), cudaMemcpyHostToDevice) );
	//TODO constant gaussian memory
	float *dev_gaussian;
	GPU_CHECKERROR(cudaMalloc((void**) &dev_gaussian, kernelSize * kernelSize * sizeof(float)));
	GPU_CHECKERROR( cudaMemcpy((void*) dev_gaussian,
				(void*) gaussian, kernelSize * kernelSize * sizeof(float), cudaMemcpyHostToDevice) );


	int maxThreads = getMaxThreadsPerBlock();
	int xDim = 0;
	dim3 threadsPerBlock;
	dim3 numBlocks;


	float *dev_blurredImage;
	GPU_CHECKERROR(cudaMalloc((void**) &dev_blurredImage, 3 * w * h * sizeof(float)));

	//if (radius > RADIUS_THRESH)
	//{
	//	cout<< "\n NEW";
	//	//xDim = 2 * radius;
	//	xDim = getLargestPowerOf2LessThan(ceil(sqrt(maxThreads)));
	//	threadsPerBlock = dim3(xDim, xDim, 1);
	//	numBlocks = dim3(ceil((double)w / threadsPerBlock.x), ceil((double)h / threadsPerBlock.y));
	//	size_t shared_mem_size = 3 * sizeof(float) * (xDim + 2 * radius) * (xDim + 2 * radius);

	//	gaussianBlurSpecialBlockParallel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(dev_blurredImage, dev_imageArray, dev_gaussian,
	//																w, h, radius);
	//	GPU_CHECKERROR( cudaDeviceSynchronize() );
	//}
	//else
	//{
		xDim = getLargestPowerOf2LessThan(ceil(sqrt(maxThreads)));
		threadsPerBlock = dim3(xDim, xDim, 1);
		numBlocks = dim3(ceil((double)w / threadsPerBlock.x), ceil((double)h / threadsPerBlock.y));
		size_t shared_mem_size = 3 * sizeof(float) * (xDim + 2 * radius) * (xDim + 2 * radius);

		gaussianBlurParallel<<<numBlocks, threadsPerBlock, shared_mem_size>>>(dev_blurredImage, dev_imageArray, dev_gaussian,
																	w, h, radius);
        	GPU_CHECKERROR( cudaDeviceSynchronize() );
	//}

	GPU_CHECKERROR( cudaMemcpy((void*) h_imageArray,
					(void*) dev_blurredImage, 3 * w * h * sizeof(float), cudaMemcpyDeviceToHost) );
	gettimeofday(&t1, NULL);
	float timdiff1 = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000000.0;
	std::cout<<std::endl<<"Time taken by Parallel Version: "<<timdiff1<<" s\n";

	printf("writing output image hw1b.exr\n");
	writeOpenEXRFile ("hw1b.exr", h_imageArray, w, h);
	free (h_imageArray);
	//delete[] h_imageArrayCpy;
	GPU_CHECKERROR( cudaFreeHost(h_imageArrayCpy) );
	delete[] gaussian;
	GPU_CHECKERROR( cudaFree(dev_blurredImage) );
	GPU_CHECKERROR( cudaFree(dev_imageArray) );
	GPU_CHECKERROR( cudaFree(dev_gaussian) );
	
	printf("done.\n");

	return 0;
}
