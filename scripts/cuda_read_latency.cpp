#include <cuda_runtime.h>
#include <device_launch_parameters.h>
#include <chrono>
#include <iostream>

__global__ void latencyTestKernel(int* globalMem, int* result, int numTests) {
    int index = blockIdx.x * blockDim.x + threadIdx.x;
    if (index < numTests) {
        auto start = std::chrono::high_resolution_clock::now();
        int temp = globalMem[index];
        *result = temp;
        auto end = std::chrono::high_resolution_clock::now();
        result[1] = std::chrono::duration_cast<std::chrono::nanoseconds>(end - start).count();
    }
}

int main() {
    const int numTests = 100000;
    int* globalMem;
    int* result;
    int* h_result;

    cudaMalloc((void**)&globalMem, sizeof(int) * numTests);
    cudaMalloc((void**)&result, sizeof(int) * 2);
    cudaMallocHost((void**)&h_result, sizeof(int) * 2);

    int* h_globalMem = (int*)malloc(sizeof(int) * numTests);
    for (int i = 0; i < numTests; ++i) {
        h_globalMem[i] = i;
    }
    cudaMemcpy(globalMem, h_globalMem, sizeof(int) * numTests, cudaMemcpyHostToDevice);

    latencyTestKernel<<<1, numTests>>>(globalMem, result, numTests);
    cudaDeviceSynchronize();

    cudaMemcpy(h_result, result, sizeof(int) * 2, cudaMemcpyDeviceToHost);

    std::cout << "Average read latency: " << h_result[1] / numTests << " ns" << std::endl;

    cudaFree(globalMem);
    cudaFree(result);
    cudaFreeHost(h_result);
    free(h_globalMem);

    return 0;
}