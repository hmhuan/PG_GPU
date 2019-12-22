#include <stdio.h>
#include <stdint.h>
#include <thrust/device_vector.h>
#include <thrust/copy.h>
#include <thrust/sort.h>
#include <bits/stdc++.h>


__global__ void reductionPhase(int *d_in, int *d_scan, int n);

__global__ void postReductionPhase(int *d_in, int *d_out, int n);


__global__ void addPhase(int *d_in, int *d_out, int n);

void MyScan(const uint32_t *in, uint32_t* res,int n, int sizeBlock);
