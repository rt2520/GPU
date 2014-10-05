

#include <iostream>
#include <stdio.h>
#include <cuda.h>
#include <sys/time.h>

// STUDENTS: be sure to set the single define at the top of this file, 
// depending on which machines you are running on.
#include "im1.h"



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



int main (int argc, char *argv[])
{
 

    printf("reading openEXR file %s\n", argv[1]);
        
    int w, h;   // the width & height of the image, used frequently!


    // First, convert the openEXR file into a form we can use on the CPU
    // and the GPU: a flat array of floats:
    // This makes an array h*w*sizeof(float)*3, with sequential r/g/b indices
    // don't forget to free it at the end


    float *h_imageArray;
    readOpenEXRFile (argv[1], &h_imageArray, w, h);

    // 
    // serial code: saves the image in "hw1_serial.exr"
    //

    // for every pixel in p, get it's Rgba structure, and convert the
    // red/green/blue values there to luminance L, effectively converting
    // it to greyscale:

    for (int y = 0; y < h; ++y) {
        for (int x = 0; x < w; ++x) {
            
            unsigned int idx = ((y * w) + x) * 3;
            
            float L = 0.2126f*h_imageArray[idx] + 
                      0.7152f*h_imageArray[idx+1] + 
                      0.0722f*h_imageArray[idx+2];

            h_imageArray[idx] = L;
            h_imageArray[idx+1] = L;
            h_imageArray[idx+2] = L;

       }
    }
    
    printf("writing output image hw1_serial.exr\n");
    writeOpenEXRFile ("hw1_serial.exr", h_imageArray, w, h);
    free(h_imageArray); // make sure you free it: if you use this variable
                        // again, readOpenEXRFile will allocate more memory


    //
    // Now the GPU version: it will save whatever is in h_imageArray
    // to the file "hw1_gpu.exr"
    //
    
    // read the file again - the file read allocates memory for h_imageArray:
    readOpenEXRFile (argv[1], &h_imageArray, w, h);



    // at this point, h_imageArray has sequenial floats for red, green , and
    // blue for each pixel: r,g,b,r,g,b,r,g,b,r,g,b. You need to copy
    // this array to GPU global memory, and have one thread per pixel compute
    // the luminance value, with which you will overwrite each r,g,b, triple.
    float *dev_imageArray;
    GPU_CHECKERROR(cudaMalloc((void**) &dev_imageArray, 3 * w * h * sizeof(float)));
	GPU_CHECKERROR( cudaMemcpy((void*) dev_imageArray,
				(void*) h_imageArray, 3 * w * h * sizeof(float), cudaMemcpyHostToDevice) );

	int maxThreads = getMaxThreadsPerBlock();
	int xDim = getLargestPowerOf2LessThan(ceil(sqrt(maxThreads)));
	dim3 threadsPerBlock(xDim, xDim, 1);
	dim3 numBlocks(ceil((double)w / threadsPerBlock.x), ceil((double)h / threadsPerBlock.y));
	struct timeval t0, t1;
	gettimeofday(&t0, NULL);
	rgbToGrayParallel<<<numBlocks , threadsPerBlock>>>(dev_imageArray , w , h);
	GPU_CHECKERROR( cudaDeviceSynchronize() );
	gettimeofday(&t1, NULL);

	GPU_CHECKERROR( cudaMemcpy((void*) h_imageArray,
					(void*) dev_imageArray, 3 * w * h * sizeof(float), cudaMemcpyDeviceToHost) );
	float timdiff1 = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000000.0;
	std::cout<<std::endl<<"Time taken by Parallel Version: "<<timdiff1<<" s\n";

    //
    // process it on the GPU: 1) copy it to device memory, 2) process
    // it with a 2d grid of 2d blocks, with each thread assigned to a 
    // pixel. then 3) copy it back.
    //





    //
    // Your memory copy, & kernel launch code goes here:
    //




    // All your work is done. Here we assume that you have copied the 
    // processed image data back, frmm the device to the host, into the
    // original host array h_imageArray. You can do it some other way,
    // this is just a suggestion
    
    printf("writing output image hw1_gpu.exr\n");
    writeOpenEXRFile ("hw1_gpu.exr", h_imageArray, w, h);
    free (h_imageArray);

    printf("done.\n");

    return 0;
}


