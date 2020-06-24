#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#define DIM 512  // threads per block
#define N (1 << 24) // vector length 2^24

float *get_vec(unsigned int n) {
  float *vector = (float *)malloc(sizeof(float) * n);
  srand((unsigned int)time(NULL));
  float a = rand() % 100 + 1;
  for (int i = 0; i < n; i++) {
  vector[i] =(((float) rand() /(float)(RAND_MAX)) * a);     
  }
  return vector;
}

__global__ void kernel1(float* output, float *vector_1, float *vector_2, unsigned int n){
  __shared__ float smem[DIM];

  int idx = blockIdx.x * blockDim.x + threadIdx.x;
  if(idx >= n) return;


  int tid = threadIdx.x;
  smem[tid] = vector_1[idx] * vector_2[idx];
  __syncthreads();


  for(int stride = blockDim.x / 2; stride > 0; stride >>= 1){
    if(tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
  }

  if(tid == 0) output[blockIdx.x] = smem[0];
}


__global__ void kernel2(float* result, float *vector_1, float *vector_2, int n) {
 
  __shared__ float smem[DIM];
  int idx = blockIdx.x * blockDim.x + threadIdx.x;

  if(idx >= n) return;

  int tid = threadIdx.x;
  smem[tid] = vector_1[idx] * vector_2[idx];
  __syncthreads();

  for(int stride = blockDim.x / 2; stride > 0; stride >>= 1) {
    if(tid < stride) smem[tid] += smem[tid + stride];
    __syncthreads();
  }
  if(tid == 0) atomicAdd(result, smem[0]);
}

float gpu_dot_product_1(float *vector_1, float *vector_2, int n) {
  cudaEvent_t start, stop;
  float elapsed;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  unsigned int n_blocks = (n + DIM - 1) / DIM;
  float *d_output, *d_vector_1, *d_vector_2, *output;
  unsigned int output_size = n_blocks * sizeof(float),
               input_size = n * sizeof(float);
  float result = 0;

  cudaMalloc(&d_output, output_size);
  cudaMalloc(&d_vector_1, input_size);
  cudaMalloc(&d_vector_2, input_size);

  cudaMemcpy(d_vector_1, vector_1, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector_2, vector_2, input_size, cudaMemcpyHostToDevice);

  
  cudaEventRecord(start);
  kernel1<<<n_blocks, DIM>>>(d_output, d_vector_1, d_vector_2, n);
  cudaEventRecord(stop); 
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  printf("     Kernel1 execution time: %fms\n", elapsed);
  

  output = (float*)malloc(output_size);
  cudaMemcpy(output, d_output, output_size, cudaMemcpyDeviceToHost);

  cudaFree(d_output);
  cudaFree(d_vector_1);
  cudaFree(d_vector_2);

  for(unsigned int i = 0; i < n_blocks; i++) {
    result += output[i];
  }

  free(output);
  return result;

}


float gpu_dot_product_2(float *vector_1, float *vector_2, int n) {
  cudaEvent_t start, stop;
  float elapsed;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  unsigned int n_blocks = (n + DIM - 1) / DIM;
  float *d_result, *d_vector_1, *d_vector_2, *result;
  unsigned int output_size = sizeof(float),
               input_size = n * sizeof(float);

  cudaMalloc(&d_result, output_size);
  cudaMalloc(&d_vector_1, input_size);
  cudaMalloc(&d_vector_2, input_size);

  cudaMemcpy(d_vector_1, vector_1, input_size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_vector_2, vector_2, input_size, cudaMemcpyHostToDevice);


  cudaEventRecord(start);
  kernel2<<<n_blocks, DIM>>>(d_result, d_vector_1, d_vector_2, n);
  cudaEventRecord(stop); 
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  printf("     Kernel2 execution time: %fms\n", elapsed);
  

  result = (float*)malloc(output_size);
  cudaMemcpy(result, d_result, output_size, cudaMemcpyDeviceToHost);

  cudaFree(d_result);
  cudaFree(d_vector_1);
  cudaFree(d_vector_2);

  return *result;
}

int main(){
  float *a = get_vec(N);
  float *b = get_vec(N);

  cudaEvent_t start, stop;
  float elapsed;
  cudaEventCreate(&start);
  cudaEventCreate(&stop);
  
  printf("===================Kernel1=================\n");
  cudaEventRecord(start);
  float r1 = gpu_dot_product_1(a, b, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  printf("     Result : %f\n", r1);
  printf("     Total time: %fms\n\n", elapsed);

  printf("===================Kernel2=================\n");
  cudaEventRecord(start);
  float r2 = gpu_dot_product_2(a, b, N);
  cudaEventRecord(stop);
  cudaEventSynchronize(stop);
  cudaEventElapsedTime(&elapsed, start, stop);
  printf("     Result : %f\n", r2);
  printf("     Total time: %fms\n\n", elapsed);

  free(a);
  free(b);
}
