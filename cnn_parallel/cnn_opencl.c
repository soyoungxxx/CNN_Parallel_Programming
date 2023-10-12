#include <CL/cl.h>
#include "cnn.h"
#pragma warning(disable : 4996)
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <errno.h>
#include <windows.h>
#include <math.h>
#include <direct.h>

extern const char* CLASS_NAME[];

int batchsize = 24;


static void softmax(float* input, int N) {
	int i;
	float max = input[0];
	for (i = 1; i < N; i++) {
		if (max < input[i]) max = input[i];
	}
	float sum = 0;
	for (i = 0; i < N; i++) {
		sum += exp(input[i] - max);
	}
	for (i = 0; i < N; i++) {
		input[i] = exp(input[i] - max) / (sum + 1e-7);
	}
}

static int find_max(float* input, int classNum) {
	int i;
	int maxIndex = 0;
	float max = 0;
	for (i = 0; i < classNum; i++) {
		if (max < input[i]) {
			max = input[i];
			maxIndex = i;
		}
	}
	return maxIndex;
}


const int INPUT_DIM[] = {
   3, 64,
   64,

   64,128,
   128,

   128, 256, 256,
   256,

   256, 512, 512,
   512,

   512, 512, 512,
   512,

   512,
   512,
   512
};

const int OUTPUT_DIM[] = {
   64, 64,
   64,

   128, 128,
   128,

   256, 256, 256,
   256,

   512, 512, 512,
   512,

   512, 512, 512,
   512,

   512,
   512,
   10
};

const int NBYN[] = {
   32, 32,
   16,

   16, 16,
   8,

   8, 8, 8,
   4,

   4, 4, 4,
   2,

   2, 2, 2,
   1,

   1,
   1,
   1
};
#define ReLU(x) (((x)>0)?(x):0)

#define CHECK_ERROR(err) \
    if(err != CL_SUCCESS) { \
        printf("[%s:%d] OpenCL error %d\n", __FILE__, __LINE__, err); \
        exit(EXIT_FAILURE); \
    }

char* get_source_code(const char* file_name, size_t* len);

float* w[21];
float* b[21];

cl_platform_id platform;
cl_device_id device;
cl_context context;
cl_command_queue queue;
cl_program program;
cl_kernel kernel_conv;
cl_kernel kernel_pool;
cl_kernel kernel_fc;
cl_kernel kernel_test;
cl_int err;

char* kernel_source;
size_t kernel_source_size;

cl_mem bufInput, bufOutput, bufWeights, bufBiases;




static void fc_layer(float* input, float* output, float* weights, float* biases, int inDim, int outDim) {

	size_t global_size[2] = { batchsize, outDim };

	cl_mem fcInput, fcOutput, fcWeights, fcBiases;

	fcInput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * inDim * batchsize, NULL, &err);
	fcOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDim * batchsize, NULL, &err);
	fcWeights = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDim * inDim + inDim, NULL, &err);
	fcBiases = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDim, NULL, &err);


	err = clEnqueueWriteBuffer(queue, fcInput, CL_TRUE, 0, sizeof(float) * inDim * batchsize, input, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, fcWeights, CL_TRUE, 0, sizeof(float) * outDim * inDim + inDim, weights, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, fcBiases, CL_TRUE, 0, sizeof(float) * outDim, biases, 0, NULL, NULL);
	CHECK_ERROR(err);


	err = clSetKernelArg(kernel_fc, 0, sizeof(cl_mem), &fcInput);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 1, sizeof(cl_mem), &fcOutput);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 2, sizeof(cl_mem), &fcWeights);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 3, sizeof(cl_mem), &fcBiases);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 4, sizeof(int), &inDim);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_fc, 5, sizeof(int), &outDim);
	CHECK_ERROR(err);

	clEnqueueNDRangeKernel(queue, kernel_fc, 2, NULL, &global_size, NULL, 0, NULL, NULL);

	clFinish(queue);

	err = clEnqueueReadBuffer(queue, fcOutput, CL_TRUE, 0, sizeof(float) * outDim * batchsize, output, 0, NULL, NULL);
	CHECK_ERROR(err);

	clReleaseMemObject(fcInput);
	clReleaseMemObject(fcOutput);
	clReleaseMemObject(fcWeights);
	clReleaseMemObject(fcBiases);

}

static void max_pooling(float* inputs, float* outputs, int D, int N) {

	//size_t global_size[3] = { D,N,N };
	size_t global_size[3] = { batchsize, D, N * N };

	err = clEnqueueWriteBuffer(queue, bufInput, CL_TRUE, 0, sizeof(float) * D * N * N * 4 * batchsize, inputs, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_pool, 0, sizeof(cl_mem), &bufInput);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_pool, 1, sizeof(cl_mem), &bufOutput);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_pool, 2, sizeof(int), &D);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_pool, 3, sizeof(int), &N);

	clEnqueueNDRangeKernel(queue, kernel_pool, 3, NULL, &global_size, NULL, 0, NULL, NULL);

	err = clEnqueueReadBuffer(queue, bufOutput, CL_TRUE, 0, sizeof(float) * D * N * N * batchsize, outputs, 0, NULL, NULL);
	CHECK_ERROR(err);

	clFinish(queue);

}

static void convolution(float* inputs, float* outputs, float* filters, float* biases, int inDim, int outDim, int N) {

	int index = N * N;
	int x = outDim / N;

	size_t global_size[3] = { batchsize, outDim, index };
	//size_t local_size[3] = { batchsize , N, N };

	memset(outputs, 0, sizeof(float) * N * N * outDim * batchsize);

	cl_mem convInput, convOutput, convWeights, convBiases;

	convInput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * inDim * N * N * batchsize, NULL, &err);
	convOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDim * N * N * batchsize, NULL, &err);
	convWeights = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * inDim * outDim * 3 * 3, NULL, &err);
	convBiases = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * outDim, NULL, &err);


	err = clEnqueueWriteBuffer(queue, convInput, CL_TRUE, 0, sizeof(float) * N * N * inDim * batchsize, inputs, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, convWeights, CL_TRUE, 0, sizeof(float) * inDim * outDim * 3 * 3, filters, 0, NULL, NULL);
	CHECK_ERROR(err);
	err = clEnqueueWriteBuffer(queue, convBiases, CL_TRUE, 0, sizeof(float) * outDim, biases, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clSetKernelArg(kernel_conv, 0, sizeof(cl_mem), &convInput);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_conv, 1, sizeof(cl_mem), &convOutput);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_conv, 2, sizeof(cl_mem), &convWeights);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_conv, 3, sizeof(cl_mem), &convBiases);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_conv, 4, sizeof(int), &outDim);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_conv, 5, sizeof(int), &inDim);
	CHECK_ERROR(err);
	err = clSetKernelArg(kernel_conv, 6, sizeof(int), &N);
	CHECK_ERROR(err);

	err = clEnqueueNDRangeKernel(queue, kernel_conv, 3, NULL, &global_size, NULL, 0, NULL, NULL);
	CHECK_ERROR(err);

	err = clEnqueueReadBuffer(queue, convOutput, CL_TRUE, 0, sizeof(float) * outDim * N * N * batchsize, outputs, 0, NULL, NULL);
	CHECK_ERROR(err);

	clFinish(queue);


	clReleaseMemObject(convInput);
	clReleaseMemObject(convOutput);
	clReleaseMemObject(convWeights);
	clReleaseMemObject(convBiases);


}



void cnn_init() {

	err = clGetPlatformIDs(1, &platform, NULL);
	CHECK_ERROR(err);
	err = clGetDeviceIDs(platform, CL_DEVICE_TYPE_GPU, 1, &device, NULL);
	CHECK_ERROR(err);
	context = clCreateContext(NULL, 1, &device, NULL, NULL, &err);
	CHECK_ERROR(err);
	queue = clCreateCommandQueue(context, device, 0, &err);
	CHECK_ERROR(err);

	kernel_source = get_source_code("kernel.cl", &kernel_source_size);
	program = clCreateProgramWithSource(context, 1, (const char**)&kernel_source, &kernel_source_size, &err);
	CHECK_ERROR(err);
	err = clBuildProgram(program, 1, &device, "", NULL, NULL);
	CHECK_ERROR(err);

	kernel_conv = clCreateKernel(program, "conv", &err);
	CHECK_ERROR(err);
	kernel_pool = clCreateKernel(program, "pool", &err);
	CHECK_ERROR(err);
	kernel_fc = clCreateKernel(program, "fc", &err);
	CHECK_ERROR(err);

	bufInput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * 512 * 32 * 32, NULL, &err);
	bufOutput = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * 512 * 32 * 32, NULL, &err);
	bufWeights = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512 * 512 * 3 * 3, NULL, &err);
	bufBiases = clCreateBuffer(context, CL_MEM_READ_WRITE, sizeof(float) * 512, NULL, &err);




}

void cnn(float* images, float** network, int* labels, float* confidences, int num_of_image) {

	int offset = 0;
	// link weights and biases to network
	for (int i = 0; i < 17; ++i) {
		if (i == 2 || i == 5 || i == 9 || i == 13) i++;   // pooling layer has no weights and biases
		w[i] = network + offset;
		offset += 3 * 3 * INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}
	for (int i = 18; i < 21; ++i) {
		w[i] = network + offset;
		offset += INPUT_DIM[i] * OUTPUT_DIM[i];
		b[i] = network + offset;
		offset += OUTPUT_DIM[i];
	}


	// allocate memory for layer
	float* layer[21];
	for (int i = 0; i < 21; ++i) {
		layer[i] = (float*)malloc(sizeof(float) * batchsize * OUTPUT_DIM[i] * NBYN[i] * NBYN[i]);
		if (layer[i] == NULL) {
			perror("malloc error");
		}
	}


	for (int i = 0; i < num_of_image / batchsize; ++i) {
		for (int j = 0; j < 32; j++) {
			//if (j % 32 == 0) printf("%f\n", images[j]);
		}

		convolution(images, layer[0], w[0], b[0], INPUT_DIM[0], OUTPUT_DIM[0], NBYN[0]);
		for (int j = 0; j < 32 * 32 * 3 * 2; j++) {
			//if (j % 32 == 31) printf("%f ", layer[0][j]);
		}
		convolution(layer[0], layer[1], w[1], b[1], INPUT_DIM[1], OUTPUT_DIM[1], NBYN[1]);
		max_pooling(layer[1], layer[2], INPUT_DIM[2], NBYN[2]);

		convolution(layer[2], layer[3], w[3], b[3], INPUT_DIM[3], OUTPUT_DIM[3], NBYN[3]);
		convolution(layer[3], layer[4], w[4], b[4], INPUT_DIM[4], OUTPUT_DIM[4], NBYN[4]);
		max_pooling(layer[4], layer[5], INPUT_DIM[5], NBYN[5]);

		convolution(layer[5], layer[6], w[6], b[6], INPUT_DIM[6], OUTPUT_DIM[6], NBYN[6]);
		convolution(layer[6], layer[7], w[7], b[7], INPUT_DIM[7], OUTPUT_DIM[7], NBYN[7]);
		convolution(layer[7], layer[8], w[8], b[8], INPUT_DIM[8], OUTPUT_DIM[8], NBYN[8]);
		max_pooling(layer[8], layer[9], INPUT_DIM[9], NBYN[9]);

		convolution(layer[9], layer[10], w[10], b[10], INPUT_DIM[10], OUTPUT_DIM[10], NBYN[10]);
		convolution(layer[10], layer[11], w[11], b[11], INPUT_DIM[11], OUTPUT_DIM[11], NBYN[11]);
		convolution(layer[11], layer[12], w[12], b[12], INPUT_DIM[12], OUTPUT_DIM[12], NBYN[12]);
		max_pooling(layer[12], layer[13], INPUT_DIM[13], NBYN[13]);

		convolution(layer[13], layer[14], w[14], b[14], INPUT_DIM[14], OUTPUT_DIM[14], NBYN[14]);
		convolution(layer[14], layer[15], w[15], b[15], INPUT_DIM[15], OUTPUT_DIM[15], NBYN[15]);
		convolution(layer[15], layer[16], w[16], b[16], INPUT_DIM[16], OUTPUT_DIM[16], NBYN[16]);
		max_pooling(layer[16], layer[17], INPUT_DIM[17], NBYN[17]);

		fc_layer(layer[17], layer[18], w[18], b[18], INPUT_DIM[18], OUTPUT_DIM[18]);
		fc_layer(layer[18], layer[19], w[19], b[19], INPUT_DIM[19], OUTPUT_DIM[19]);
		fc_layer(layer[19], layer[20], w[20], b[20], INPUT_DIM[20], OUTPUT_DIM[20]);



		for (int j = 0; j < batchsize; j++)
		{
			softmax(layer[20] + j * 10, 10);
			labels[i * batchsize + j] = find_max(layer[20] + j * 10, 10);
			confidences[i * batchsize + j] = layer[20][labels[i * batchsize + j] + j * 10];
		}

		images += 32 * 32 * 3 * batchsize;
	}


	for (int i = 0; i < 21; ++i) {
		free(layer[i]);
	}

}

char* get_source_code(const char* file_name, size_t* len) {
	FILE* file = fopen(file_name, "rb");
	if (file == NULL) {
		printf("[%s:%d] Failed to open %s\n", __FILE__, __LINE__, file_name);
		exit(EXIT_FAILURE);
	}

	fseek(file, 0, SEEK_END);
	size_t length = (size_t)ftell(file);
	rewind(file);

	char* source_code = (char*)malloc(length + 1);
	fread(source_code, length, 1, file);
	source_code[length] = '\0';
	fclose(file);
	*len = length;

	return source_code;
}