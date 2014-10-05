#include <iostream>
#include <fstream>
#include <vector>
#include <cstdlib>
#include <time.h>
#include <math.h>
#include <stdio.h>
#include <sys/time.h>

#define GPU_CHECKERROR( err ) (gpuCheckError( err, __FILE__, __LINE__ ))
using namespace std;

static void gpuCheckError(cudaError_t err, const char *file, int line)
{
	if (err != cudaSuccess)
	{
		cout<<endl<<cudaGetErrorString(err)<<" in "<<file<< " at"<<line<<".";
	}
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

unsigned int gcd(unsigned int a, unsigned int b)
{
	while (b != 0)
	{
		int tmp = b;
		b = a % b;
		a = tmp;
	}
	return a;
}

__global__ void parallelCoprimeCountAtomicAdd(int *dev_A, int *dev_B, unsigned int *dev_count, unsigned int size)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;
	int a = dev_A[i];
	int b = dev_B[i];
	while (b != 0)
	{
		int tmp = b;
		b = a % b;
		a = tmp;
	}

	if (a == 1)
		atomicAdd(dev_count, 1);
}

__global__ void reduceAdd(int *dev_A, unsigned int size)
{
	unsigned int tid = threadIdx.x;
	unsigned int index = blockDim.x * blockIdx.x + tid;
	if (index >= size)
		return;
	for (unsigned int s = blockDim.x/2; s > 0; s /= 2)
	{
		if (tid < s && (index + s) < size)
			dev_A[index] += dev_A[index + s];
		__syncthreads();
	}
}

__global__ void parallelCoprimeCount(int *dev_A, int *dev_B, unsigned int size)
{
	unsigned int i = blockIdx.x * blockDim.x + threadIdx.x;
	if (i >= size)
		return;
	int a = dev_A[i];
	int b = dev_B[i];
	while (b != 0)
	{
		int tmp = b;
		b = a % b;
		a = tmp;
	}

	if (a == 1)
		dev_A[i] = 1;
	else
		dev_A[i] = 0;
}


int main (int argc, char *argv[])
{
	int *A, *B;
	unsigned int count = 0;
	unsigned int size = 0;
	timeval t0, t1, t2;

	if (argc < 2 || argc > 3)
		return -1;
	if (argc == 2)
	{
		srand(time(NULL));
		size = strtol(argv[1], NULL, 10);
		if (size < 1)
		{
			cout<<endl<<"Size of the arrays should be greater than 0."<<endl;
			return -1;
		}
		A = new int[size];
		B = new int[size];
		for (int i = 0; i < size; i++)
		{
			A[i] = rand()%10000000 + 1;
			B[i] = rand()%10000000 + 1;
		}
	}
	else if (argc == 3)
	{
		const char *filename = argv[2];
		std::ifstream input_file(filename);
		if (!input_file.is_open())
		{
			cout<<endl<<"Error opening file!! Exiting!!";
			return -1;
		}
		int x;
		bool a = true;
		while(input_file >> x)
			size++;
		size /= 2;
		if (size < 1)
		{
			cout<<endl<<"Size of the arrays should be greater than 0."<<endl;
			return -1;
		}
		A = new int[size];
		B = new int[size];

		input_file.clear();
		input_file.seekg(0, input_file.beg);
		unsigned int i = 0;
		while(input_file >> x)
		{
			if (a)
				A[i] = x;
			else
				B[i++] = x;
			a = !a;
		}
		input_file.close();
	}


	/*int a[size], b[size];
	std::copy(A.begin(), A.end(), a);
	std::copy(B.begin(), B.end(), b);*/

	int *device_A;
	GPU_CHECKERROR(cudaMalloc((void**) &device_A, size * sizeof(int)));
	//check if allocated
	GPU_CHECKERROR( cudaMemcpy((void*) device_A,
				(void*) A, size * sizeof(int), cudaMemcpyHostToDevice) );
	//cudaMemcpy((void*) device_A,
					//(void*) a, size * sizeof(int), cudaMemcpyHostToDevice);

	int *device_B;
	GPU_CHECKERROR( cudaMalloc((void**) &device_B, size * sizeof(int)) );
	//check if allocated
	GPU_CHECKERROR( cudaMemcpy((void*) device_B,
				(void*) B, size * sizeof(int), cudaMemcpyHostToDevice) );
	//cudaMemcpy((void*) device_B,
					//(void*) b, size * sizeof(int), cudaMemcpyHostToDevice);

	unsigned int *device_count;
	GPU_CHECKERROR( cudaMalloc((void**) &device_count, sizeof(unsigned int)) );
	GPU_CHECKERROR( cudaMemset((void*) device_count, 0, sizeof(unsigned int)) );

	cout<<endl<<"Beginning Serial Version...";
	gettimeofday(&t0, NULL);
	for (unsigned int i = 0; i < size; i++)
		if (gcd(A[i], B[i]) == 1)
			count++;
	gettimeofday(&t1, NULL);
	float timdiff1 = (1000000.0*(t1.tv_sec - t0.tv_sec)
						+ (t1.tv_usec - t0.tv_usec)) / 1000000.0;
	cout<<endl<<"Serial Version ended in "<<timdiff1<<" s";
	cout<<endl<<"Serial Version says "<<count<<" pairs are co-prime.";

	//int threadsPerBlock = 512;
	int threadsPerBlock = getMaxThreadsPerBlock();
	unsigned int numBlocks = ceil((double)size/(double)threadsPerBlock);

	cout<<endl<<"\nBeginning Parallel Version using AtomicAdd...";
	gettimeofday(&t0, NULL);
	parallelCoprimeCountAtomicAdd<<<numBlocks, threadsPerBlock>>>(device_A, device_B, device_count, size);
	GPU_CHECKERROR( cudaDeviceSynchronize() );

	unsigned int parallel_count = 0;
	GPU_CHECKERROR( cudaMemcpy((void *) &parallel_count,(void *) device_count, sizeof(unsigned int), cudaMemcpyDeviceToHost) );
	gettimeofday(&t2, NULL);
	float timdiff2 = (1000000.0*(t2.tv_sec - t0.tv_sec)
								+ (t2.tv_usec - t0.tv_usec)) / 1000000.0;
	cout<<endl<<"Parallel Version with AtomicAdd ended in "<<timdiff2<<" s";
	cout<<endl<<"Parallel Version with AtomicAdd says "<<parallel_count<<" pairs are co-prime."<<endl;

	cout<<endl<<"Beginning Parallel Version using Reduce...";
	gettimeofday(&t0, NULL);
	parallelCoprimeCount<<<numBlocks, threadsPerBlock>>>(device_A, device_B, size);
	GPU_CHECKERROR( cudaDeviceSynchronize() );

	parallel_count = 0;

	//Please note that I am destroying the original array A to save space but if we need to
	//preserve the input we can use a dedicated output array
	reduceAdd<<<numBlocks, threadsPerBlock>>>(device_A, size);
	GPU_CHECKERROR( cudaDeviceSynchronize() );
	GPU_CHECKERROR( cudaMemcpy((void*) A,
				(void*) device_A, size * sizeof(int), cudaMemcpyDeviceToHost) );
	for (unsigned int i = 0; i < size; i += threadsPerBlock)
	{
		parallel_count += A[i];
	}

	gettimeofday(&t2, NULL);
	float timdiff3 = (1000000.0*(t2.tv_sec - t0.tv_sec)
						+ (t2.tv_usec - t0.tv_usec)) / 1000000.0;
	cout<<endl<<"Parallel Version with Reduce ended in "<<timdiff3<<" s";
	cout<<endl<<"Parallel Version with Reduce says "<<parallel_count<<" pairs are co-prime."<<endl;

	GPU_CHECKERROR( cudaFree(device_A) );
	GPU_CHECKERROR( cudaFree(device_B) );
	GPU_CHECKERROR( cudaFree(device_count) );
	delete[] A;
	delete[] B;

	return 0;
}
