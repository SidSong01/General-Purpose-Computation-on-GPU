# Implemwntation with Atomics and Parallel Reduction

---

Two kernel functions.

* kernel1: use shared memory and parallel reduction to calculate partial sum on each thread block.

* kernel2: use shared memory, parallel reduction, and atomic function or atomic lock to perform the entire computation on GPU.

### The cuda code is

`assignment2_fnal.cu`

### The execuable file after Nvidia c compiler：

`hw2`

---

## Hardware information

Nvidia Tesla K20m
