__kernel void matmul(__global __read_only float* A, __global __read_only float* B,
		     __global __write_only float* C){
  size_t r = get_global_id(0), c = get_global_id(1);
  size_t size = get_global_size(0);
  float result = 0.0;

  for(size_t i = 0; i < size; i++)
    result += A[r * size + i] * B[i * size + c];
  C[r * size + c] = result;
}
