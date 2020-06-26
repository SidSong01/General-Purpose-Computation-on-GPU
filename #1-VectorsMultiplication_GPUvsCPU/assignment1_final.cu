#include <stdio.h>
#include <stdlib.h>
#include <sys/time.h> 

#define THREADS_PER_BLOCK 512   // threads number per block

/**
*****************************            necessary functions          **********************************
*/

long long start_timer(); // timing start
long long stop_timer(long long start_time, char *name); // timing end
__global__ void multiply(float *a, float *b, float *c, int n);  // multiplication operation for GPU
float CPU_big_dot(float *A,float *B,int n);  // CPU computation function
float GPU_big_dot(float *A,float *B,int n);  // GPU computation function
void random_vec(float *x, int size);  // get random random single precision floating point vectors

/**
*****************************            main execution          **********************************
*/

int main(void) {
  
  int N = 1 << 20; // vector size 1024 * 1024
  int size = N * sizeof(float);
  float *a, *b;
  float cpu_out;
  float gpu_out;
  float cputime;
  float gputime;
  long long s_cpu;
  long long s_gpu;

  a = (float *) malloc(size);
  b = (float *) malloc(size);
  random_vec(a, N);
  random_vec(b, N);

  char t_cpu[50] = "Total computation time for CPU";
  char t_gpu[50] = "Total computation time for GPU";
 
  printf("\n=========================== CPU Time Details ===========================\n");
  s_cpu = start_timer(); 
  cpu_out = CPU_big_dot(a,b,N);
  cputime = float(stop_timer(s_cpu, t_cpu)) / 1e-6;
  printf("========================================================================\n\n");

  printf("=========================== GPU Time Details ===========================\n");
  s_gpu = start_timer();
  gpu_out = GPU_big_dot(a,b,N);
  gputime = float(stop_timer(s_gpu, t_gpu)) / 1e-6;
  printf("=======================================================================\n\n");

  float error = fabs((gpu_out - cpu_out) / cpu_out);
  printf("========================= Computation Results =========================\n");
  printf("                  CPU result = %.4f\n", cpu_out);
  printf("                  GPU result = %.4f\n", gpu_out);
  printf("=======================================================================\n\n");

  printf("============================= Comparision =============================\n");
  printf("      Check correctness = (GPU_out-CPU_out)/CPU_out:%.6f\n", error);
  printf("             Speedup = CPU_time/GPU_time:%.6f\n", cputime/gputime);
  printf("=======================================================================\n\n");
  
  return 0;
}

/**
*****************************            helper functions define          **********************************
*/

long long start_timer() {
  struct timeval tv;
  gettimeofday(&tv, NULL);
  return tv.tv_sec * 1000000 + tv.tv_usec;
}

long long stop_timer(long long start_time, char *name) { 
  struct timeval tv;
  gettimeofday(&tv, NULL);
  long long end_time = tv.tv_sec * 1000000 + tv.tv_usec;
  printf("  %s: %.5f sec\n", name, ((float) (end_time -start_time)) / (1000 * 1000));
  return end_time -start_time; 
}

__global__ void multiply(float *a, float *b, float *c, int n) {
  int index = threadIdx.x + blockIdx.x * blockDim.x;
  if (index < n) {
    c[index] = a[index] *  b[index];
  }
}

float CPU_big_dot(float *A,float *B,int n) {
  float *c;
  float result = 0;

  c = (float *) malloc(n * sizeof(float));
  for (int i = 0; i < n; i++) {
    c[i] = A[i] * B[i];
    result = result + c[i];
  }  
  return result;
}

float GPU_big_dot(float *A,float *B,int n) {
  
  int size = n * sizeof(float);
  float *C;
  float *d_a, *d_b, *d_c;
  float result = 0;
  long long t_kernel;
  long long t_g2c;
  long long t_c2g;

  char kernel_ex[30] = "Kernel execution time";
  char g2c[50] = "Data transfer from GPU to CPU time";
  char c2g[70] = "Memory allocation and data transfer from CPU to GPU time";
  
  // allocate memory for device copies
  t_c2g = start_timer();
  C = (float *) malloc(size);
  cudaMalloc((void **) &d_a, size);
  cudaMalloc((void **) &d_b, size);
  cudaMalloc((void **) &d_c, size);
  cudaMemcpy(d_a, A, size, cudaMemcpyHostToDevice);
  cudaMemcpy(d_b, B, size, cudaMemcpyHostToDevice);
  stop_timer(t_c2g,c2g);

  // execute the kernel
  t_kernel = start_timer();
  multiply<<<(n + THREADS_PER_BLOCK - 1)/THREADS_PER_BLOCK, THREADS_PER_BLOCK>>>(d_a, d_b, d_c, n);
  stop_timer(t_kernel,kernel_ex);

  // copy the result to CPU
  t_g2c=start_timer();
  cudaMemcpy(C, d_c, size, cudaMemcpyDeviceToHost);
  stop_timer(t_g2c,g2c);

  for (int i = 0; i < n; i++) { 
    result = result + C[i];
   }
  
  // free memory
  cudaFree(d_a); 
  cudaFree(d_b);
  cudaFree(d_c);
  cudaFree(C);
  return result;
}

void random_vec(float *x, int size) {
  srand((unsigned int)time(NULL));
  float a = rand() % 100 + 1;
  for (int i = 0; i < size; i++) {
     x[i] =(((float) rand() /(float)(RAND_MAX)) * a);      
  }
}
