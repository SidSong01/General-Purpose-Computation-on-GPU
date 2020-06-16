[//]: # (Image References)

[image1]: ./1.png

# Overview

The dot product of two vectors $a=(a_1, a_2, ... a_n-1)$ and $𝑏 = (𝑏_1, 𝑏_2, …,𝑏_n-1)$, written 𝑎 ∙ 𝑏, is simply the sum of the component-by-component products: $$a \cdot b = \sum_{i=0}^n-1 a_i \times b_i $$

Dot products are used extensively in computing and have a wide range of applications. For instance, in 3D graphics (n = 3), we often make use of the fact that $𝑎 \cdot 𝑏 = |𝑎||𝑏| \cos \theta $, where $| |$ denotes vector length and $\theta$ is the angle between the two vectors.

---

## In this program

It is a CUDA code for computing in parallel the dot product of two (N = 1024*1024) random single precision floating point vectors.

The output is the time comparison, results comparision, and Speed up =CPU time/GPU time.

Detailed GPU time are presented as follows:

* Memory allocation and data transfer from CPU to GPU time
* Kernel execution time
* Data transfer from GPU to CPU time

## How to run

* Make it executabel
`nvcc -o name assignment1_final.cu`

* Run

`./name`

## Specification of the GPU used for this assignment

![alt text][image1]
