#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>
#define N 40  // matrix length, 40 * 40
#define BLOCK_SIZE 8 // change in {1, 2, 4, 8, 10, 20}

char* loadProgSource(const char* filename, const char* preamble, size_t *sz) {
  FILE* fptr = NULL;
  size_t szSource, szPreamble, howmany;
  char* sourceString;

  // Open the OpenCL source code file
  fptr = fopen(filename, "r");
  szPreamble = strlen(preamble);

  // Get the length of the source code
  fseek(fptr, 0, SEEK_END);
  szSource = ftell(fptr);
  fseek(fptr, 0, SEEK_SET);

  // Allocate a buffer for the source code string and read it in
  sourceString = (char *) calloc(szSource + szPreamble+1, sizeof(char));
  howmany = fread((sourceString) + szPreamble, szSource, 1, fptr);
  fclose(fptr);
  *sz = szSource + szPreamble;
  sourceString[szSource + szPreamble] = '\0';
  return sourceString;
}

int main(){
  cl_float *inputMatrix1;
  cl_float *inputMatrix2;
  cl_float *results;
  cl_uint width = N;

  // Profiling variables
  cl_event event;
  cl_ulong start, end;

  cl_platform_id platform_id;
  cl_uint num_of_platforms = 0;
  cl_uint num_of_devices = 0;
  cl_device_id device_id;
  cl_context_properties properties[3];
  cl_int err;
  cl_context context;
  cl_command_queue command_queue;
  char *kernelSource;
  size_t kernelSize;
  cl_program program;
  cl_kernel kernel;
  cl_mem A, B, C; 
  size_t global[2] = {width, width};
  size_t local[2] = {BLOCK_SIZE, BLOCK_SIZE}; 

  int x, y;
  int data = 0;
  inputMatrix1 = (cl_float*) malloc(sizeof(cl_float) * width * width);
  inputMatrix2 = (cl_float*) malloc(sizeof(cl_float) * width * width);
  results = (cl_float*) malloc(sizeof(cl_float) * width * width);
  
  for(y = 0; y < width; y++){
    for(x = 0; x < width; x++){
      inputMatrix1[y * width + x] = data;
      inputMatrix2[y * width + x] = data;
      results[y * width + x] = 0;
      data++;
    }
  }

  // Retrives a list of platforms available
  if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS) {
    printf("Unable to get platform_id\n");
    return 1;
  }

  // Get a supported GPU device
  if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
     &num_of_devices) != CL_SUCCESS) {
     printf("Unable to get device_id\n");
     return 1;
  }

  // Context properties list (must be terminated with 0)
  properties[0] = CL_CONTEXT_PLATFORM;
  properties[1] = (cl_context_properties) platform_id;
  properties[2] = 0;

  // Create a context with the GPU device
  context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);

  // Create a command queue using the context and device
  command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);

  // Load kernel file, prepend static info, and return total kernel size
  kernelSource = loadProgSource("matmul.cl", "", &kernelSize);

  // Create a program from the kernel source code
  program = clCreateProgramWithSource(context, 1, (const char **)
            &kernelSource, NULL, &err);

  // Compile the program
  if (clBuildProgram(program, 0, NULL, NULL, NULL, NULL) != CL_SUCCESS) {
     printf("Error building program\n");

     char buffer[4096];
     size_t length;
     clGetProgramBuildInfo(
      program,
      device_id,
      CL_PROGRAM_BUILD_LOG,
      sizeof(buffer),
      buffer,
      &length
     );
     printf("%s\n", buffer);
     return 1;
  }

  // Specify which kernel from the program to execute
  kernel = clCreateKernel(program, "matmul", &err);

  // Create buffers for the input and output
  A = clCreateBuffer(context, CL_MEM_READ_ONLY,
          sizeof(float) * width * width, NULL, NULL);
  B = clCreateBuffer(context, CL_MEM_READ_ONLY,
          sizeof(float) * width * width, NULL, NULL);
  C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
          sizeof(float) * width * width, NULL, NULL);

  // Load data into the input buffer
  clEnqueueWriteBuffer(command_queue, A, CL_TRUE, 0,
                       sizeof(float) * width * width, inputMatrix1, 0, NULL, NULL);
  clEnqueueWriteBuffer(command_queue, B, CL_TRUE, 0,
                       sizeof(float) * width * width, inputMatrix2, 0, NULL, NULL);

// Set the argument list for the kernel command
  clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);

  // Enqueue the kernel command for execution
  clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global,
                         local, 0, NULL, &event);
  clWaitForEvents(1, &event);
  clFinish(command_queue);

  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

  cl_double miliSeconds = (cl_double)(end - start) * (cl_double)(1e-06);
  printf("The kernel took %.4lf ms\n", miliSeconds);

  // Copy the results from out of the output buffer
  clEnqueueReadBuffer(command_queue, C, CL_TRUE, 0,
                      sizeof(float) * width * width, results, 0, NULL, NULL);

  // Cleanup (release OpenCL resources)
  clReleaseContext(context);
  clReleaseCommandQueue(command_queue);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseMemObject(A);
  clReleaseMemObject(B);
  clReleaseMemObject(C);

}
