# Overview

The dot product of two vectors $a=(a_1, a_2, ... a_n-1)$ and $𝑏 = (𝑏_1, 𝑏_2, …,𝑏_n-1)$, written 𝑎 ∙ 𝑏, is simply the sum of the component-by-component products: $$a \cdot b = \sum_{i=0}^n-1 a_i \times b_i $$

Dot products are used extensively in computing and have a wide range of applications. For instance, in 3D graphics (n = 3), we often make use of the fact that 𝑎 ∙ 𝑏 = |𝑎||𝑏|𝑐𝑜𝑠𝜃, where | | denotes vector length and 𝜃 is the angle between the two vectors.

CUDA code to compute in parallel the dot product of two (N = 1024*1024) random single precision floating point vectors.
