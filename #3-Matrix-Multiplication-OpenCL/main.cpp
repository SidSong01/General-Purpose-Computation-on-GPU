#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <CL/cl.h>
#define N 40
#define BLOCK_SIZE 10

char* loadProgSource(const char* filename, const char* preamble, size_t *sz) {
  FILE* fptr = NULL;
  size_t szSource, szPreamble, howmany;
  char* sourceString;


  fptr = fopen(filename, "r");
  szPreamble = strlen(preamble);


  fseek(fptr, 0, SEEK_END);
  szSource = ftell(fptr);
  fseek(fptr, 0, SEEK_SET);


  sourceString = (char *) calloc(szSource + szPreamble+1, sizeof(char));
  howmany = fread((sourceString) + szPreamble, szSource, 1, fptr);
  fclose(fptr);
  *sz = szSource + szPreamble;
  sourceString[szSource + szPreamble] = '\0';
  return sourceString;
}

void print(float *A){
  size_t width = N, y, x;
  for(y = 0; y < width; y++){
    for(x = 0; x < width; x++){
      printf("%.2f ", A[y * width + x]);
    }
    printf("\n");
  }
}


int main(){
  cl_float *inputMatrix1;
  cl_float *inputMatrix2;
  cl_float *results;
  cl_uint width = N;

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


  if (clGetPlatformIDs(1, &platform_id, &num_of_platforms) != CL_SUCCESS) {
    printf("Unable to get platform_id\n");
    return 1;
  }


  if (clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_GPU, 1, &device_id,
     &num_of_devices) != CL_SUCCESS) {
     printf("Unable to get device_id\n");
     return 1;
  }


  properties[0] = CL_CONTEXT_PLATFORM;
  properties[1] = (cl_context_properties) platform_id;
  properties[2] = 0;


  context = clCreateContext(properties, 1, &device_id, NULL, NULL, &err);


  command_queue = clCreateCommandQueue(context, device_id, CL_QUEUE_PROFILING_ENABLE, &err);


  kernelSource = loadProgSource("matmul.cl", "", &kernelSize);


  program = clCreateProgramWithSource(context, 1, (const char **)
            &kernelSource, NULL, &err);


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


  kernel = clCreateKernel(program, "matmul", &err);


  A = clCreateBuffer(context, CL_MEM_READ_ONLY,
          sizeof(float) * width * width, NULL, NULL);
  B = clCreateBuffer(context, CL_MEM_READ_ONLY,
          sizeof(float) * width * width, NULL, NULL);
  C = clCreateBuffer(context, CL_MEM_WRITE_ONLY,
          sizeof(float) * width * width, NULL, NULL);


  clEnqueueWriteBuffer(command_queue, A, CL_TRUE, 0,
                       sizeof(float) * width * width, inputMatrix1, 0, NULL, NULL);
  clEnqueueWriteBuffer(command_queue, B, CL_TRUE, 0,
                       sizeof(float) * width * width, inputMatrix2, 0, NULL, NULL);

  clSetKernelArg(kernel, 0, sizeof(cl_mem), &A);
  clSetKernelArg(kernel, 1, sizeof(cl_mem), &B);
  clSetKernelArg(kernel, 2, sizeof(cl_mem), &C);


  clEnqueueNDRangeKernel(command_queue, kernel, 2, NULL, global,
                         local, 0, NULL, &event);
  clWaitForEvents(1, &event);
  clFinish(command_queue);

  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_START, sizeof(start), &start, NULL);
  clGetEventProfilingInfo(event, CL_PROFILING_COMMAND_END, sizeof(end), &end, NULL);

  cl_double miliSeconds = (cl_double)(end - start) * (cl_double)(1e-06);
  printf("The kernel took %.4lf ms\n", miliSeconds);


  clEnqueueReadBuffer(command_queue, C, CL_TRUE, 0,
                      sizeof(float) * width * width, results, 0, NULL, NULL);


  clReleaseContext(context);
  clReleaseCommandQueue(command_queue);
  clReleaseProgram(program);
  clReleaseKernel(kernel);
  clReleaseMemObject(A);
  clReleaseMemObject(B);
  clReleaseMemObject(C);

}
