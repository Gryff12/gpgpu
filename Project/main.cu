#include <stdio.h>
#include <iostream>
#include "test.h"

__global__ void print_kernel() {
	printf(
			"Hello from block %d, thread %d\n",
			blockIdx.x, threadIdx.x);
}

int main() {

	hello();

	print_kernel<<<3, 3>>>();
	cudaDeviceSynchronize();

	return 0;
}
