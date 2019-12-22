#include <stdio.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <bits/stdc++.h>

__global__ void myScatterKernel(const uint32_t* in, uint32_t* scan, uint32_t* out, int n, int num0);

void MyScatter(const uint32_t* in, uint32_t *scan, uint32_t *out, int n, int blockSizeX);