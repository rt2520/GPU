#include <fcntl.h>
#include <stdio.h>
#include <iostream>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <unistd.h>
#include <sys/types.h>
#include <sys/stat.h>
#include <CL/opencl.h>
#include <fstream>
#include <sys/time.h>

const char *KernelSource = "\n" \
			    "__kernel void parallelCoprimeCount(__global int *dev_A," \
			    "__global int *dev_B, __global unsigned int *dev_count, unsigned int size)" \
			    "{\n" \
			    "   unsigned int i = get_global_id(0);    \n" \
			    "   //if (i == 0)    \n" \
			    "   //      *dev_count = 0;    \n" \
			    "   if (i < size)    \n" \
			    "   {    \n" \
			    "           int a = dev_A[i];    \n" \
			    "           int b = dev_B[i];    \n" \
			    "           while (a != b)    \n" \
			    "                   if (a > b)    \n" \
			    "                           a -= b;    \n" \
			    "                   else    \n" \
			    "                           b -= a;    \n" \
			    "           if (a == 1)    \n" \
			    "                   atomic_add(dev_count, 1);    \n" \
			    "   }    \n" \
			    "}    \n";

int main(int argc, char** argv)
{
	int err;                            // error code returned from api calls

	int *h_A, *h_B;              // original data set given to device
	unsigned int h_result;           // results returned from device
	unsigned int count = 0;
	timeval t0, t1, t2;

	if (argc < 2 || argc > 3)
	{
		std::cout<<std::endl<<"Insufficient arguments!!\n";
		return -1;
	}
	if (argc == 2)
	{
		srand(time(NULL));
		count = strtol(argv[1], NULL, 10);
		if (count < 1)
		{
			std::cout<<std::endl<<"Size of the arrays should be greater than 0."<<std::endl;
			return -1;
		}
		h_A = new int[count];
		h_B = new int[count];
		for (int i = 0; i < count; i++)
		{
			h_A[i] = rand()%10000000 + 1;
			h_B[i] = rand()%10000000 + 1;
		}
	}
	else if (argc == 3)
	{
		const char *filename = argv[2];
		std::ifstream input_file(filename);
		if (!input_file.is_open())
		{
			std::cout<<std::endl<<"Error opening file!! Exiting!!";
			return -1;
		}
		int x;
		bool a = true;
		while(input_file >> x)
			count++;
		count /= 2;
		if (count < 1)
		{
			std::cout<<std::endl<<"Size of the arrays should be greater than 0."<<std::endl;
			return -1;
		}
		h_A = new int[count];
		h_B = new int[count];

		input_file.clear();
		input_file.seekg(0, input_file.beg);
		unsigned int i = 0;
		while(input_file >> x)
		{
			if (a)
				h_A[i] = x;
			else
				h_B[i++] = x;
			a = !a;
		}
		input_file.close();
	}

	size_t global;                      // global domain size for our calculation
	size_t local;                       // local domain size for our calculation

	cl_device_id device_ids[32];             // compute device id
	cl_device_id device_id;             // compute device id
	cl_context context;                 // compute context
	cl_command_queue commands;          // compute command queue
	cl_program program;                 // compute program
	cl_kernel kernel;                   // compute kernel

	cl_mem dev_A;                       // device memory used for the input array
	cl_mem dev_B;                       // device memory used for the input array
	cl_mem dev_result;                      // device memory used for the output array
	cl_platform_id platforms[32];
	cl_uint num_platforms;
	clGetPlatformIDs (32, platforms, &num_platforms);

	if (num_platforms == 0) {
		printf("Error: cant find any platforms!\n");
		return EXIT_FAILURE;
	}
	for (int j = 0; j < 2; j++)
	{

		if (j == 0)
			std::cout<<"\nFinding GPU:\n";
		else
			std::cout<<"\nFinding CPU:\n";
		for (int i = 0; i < num_platforms; i++)
		{
			err = clGetDeviceIDs(platforms[i], (j == 0) ? CL_DEVICE_TYPE_GPU : CL_DEVICE_TYPE_CPU, 1, &device_id, NULL);
			if (err != CL_SUCCESS)
			{
				continue;
				//printf("Error: Failed to create a device group!\n");
				//return EXIT_FAILURE;
			}

			context = clCreateContext(0, 1, &device_id, NULL, NULL, &err);
			if (!context)
			{
				printf("Error: Failed to create a compute context!\n");
				return EXIT_FAILURE;
			}


			commands = clCreateCommandQueue(context, device_id, 0, &err);
			if (!commands)
			{
				printf("Error: Failed to create a command commands!\n");
				return EXIT_FAILURE;
			}


			program = clCreateProgramWithSource(context, 1, (const char **) & KernelSource, NULL, &err);
			if (!program)
			{
				printf("Error: Failed to create compute program!\n");
				return EXIT_FAILURE;
			}


			err = clBuildProgram(program, 0, NULL, NULL, NULL, NULL);
			if (err != CL_SUCCESS)
			{
				size_t len;
				char buffer[2048];

				printf("Error: Failed to build program executable!\n");
				clGetProgramBuildInfo(program, device_id, CL_PROGRAM_BUILD_LOG, sizeof(buffer), buffer, &len);
				printf("%s\n", buffer);
				exit(1);
			}


			kernel = clCreateKernel(program, "parallelCoprimeCount", &err);
			if (!kernel || err != CL_SUCCESS)
			{
				printf("Error: Failed to create compute kernel!\n");
				exit(1);
			}


			dev_A = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(int) * count, NULL, NULL);
			dev_B = clCreateBuffer(context,  CL_MEM_READ_ONLY,  sizeof(int) * count, NULL, NULL);
			dev_result = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(unsigned int), NULL, NULL);
			if (!dev_A || !dev_result || !dev_B)
			{
				printf("Error: Failed to allocate device memory!\n");
				exit(1);
			}

			err = clEnqueueWriteBuffer(commands, dev_A, CL_TRUE, 0, sizeof(int) * count,(void *) h_A, 0, NULL, NULL);
			if (err != CL_SUCCESS)
			{
				printf("Error: Failed to write to source array!\n");
				exit(1);
			}

			err = clEnqueueWriteBuffer(commands, dev_B, CL_TRUE, 0, sizeof(int) * count, (void *)h_B, 0, NULL, NULL);
			if (err != CL_SUCCESS)
			{
				printf("Error: Failed to write to source array!\n");
				exit(1);
			}

			h_result = 0;
			err = clEnqueueWriteBuffer(commands, dev_result, CL_TRUE, 0, sizeof(unsigned int), &h_result, 0, NULL, NULL);
			if (err != CL_SUCCESS)
			{
				printf("Error: Failed to write to source array!\n");
				exit(1);
			}


			err = 0;
			err  = clSetKernelArg(kernel, 0, sizeof(cl_mem), &dev_A);
			err |= clSetKernelArg(kernel, 1, sizeof(cl_mem), &dev_B);
			err |= clSetKernelArg(kernel, 2, sizeof(cl_mem), &dev_result);
			err |= clSetKernelArg(kernel, 3, sizeof(unsigned int), &count);
			if (err != CL_SUCCESS)
			{
				printf("Error: Failed to set kernel arguments! %d\n", err);
				exit(1);
			}


			err = clGetKernelWorkGroupInfo(kernel, device_id, CL_KERNEL_WORK_GROUP_SIZE, sizeof(local), &local, NULL);
			if (err != CL_SUCCESS)
			{
				printf("Error: Failed to retrieve kernel work group info! %d\n", err);
				exit(1);
			}


			global = ceilf((float) count/(float) local) * local;
			gettimeofday(&t0, NULL);
			err = clEnqueueNDRangeKernel(commands, kernel, 1, NULL, &global, &local, 0, NULL, NULL);
			if (err)
			{
				printf("Error: Failed to execute kernel!\n");
				return EXIT_FAILURE;
			}


			clFinish(commands);
			gettimeofday(&t1, NULL);

			err = clEnqueueReadBuffer( commands, dev_result, CL_TRUE, 0, sizeof(unsigned int), &h_result, 0, NULL, NULL );
			if (err != CL_SUCCESS)
			{
				printf("Error: Failed to read output array! %d\n", err);
				exit(1);
			}


			std::cout<<"Number of coprimes:"<<h_result<<std::endl;
			float timdiff1 = (1000000.0*(t1.tv_sec - t0.tv_sec) + (t1.tv_usec - t0.tv_usec)) / 1000000.0;
			std::cout<<std::endl<<"Time taken by Kernel: "<<timdiff1<<" s\n";
			clReleaseMemObject(dev_A);
			clReleaseMemObject(dev_B);
			clReleaseMemObject(dev_result);
			clReleaseProgram(program);
			clReleaseKernel(kernel);
			clReleaseCommandQueue(commands);
			clReleaseContext(context);
			break;
		}
	}

	delete[] h_A;
	delete[] h_B;

	return 0;
}

