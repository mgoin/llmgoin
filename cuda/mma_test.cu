// nvcc -arch=sm_90 mma_test.cu -o mma_test
// ./mma_test
// Test passed: all computed values are correct.

#include <cuda.h>
#include <iostream>
#include <cstdio>
#include <cstdlib>
#include <cstdint>

// Macro for CUDA error checking.
#define cudaCheckError(ans) { gpuAssert((ans), __FILE__, __LINE__); }
inline void gpuAssert(cudaError_t code, const char *file, int line, bool abort = true) {
    if (code != cudaSuccess) {
        std::cerr << "CUDA Error: " << cudaGetErrorString(code)
                  << " " << file << " " << line << std::endl;
        if (abort) exit(code);
    }
}

constexpr int M = 16;
constexpr int K = 256;
constexpr int N = 8;

struct FragmentA {
    uint32_t data[M * K / 32];
};

struct FragmentB {
    uint32_t data[N * K / 32];
};

struct FragmentC {
    int data[M * N];
};

__device__ void mma_operator(FragmentC &d, const FragmentA &a, const FragmentB &b, const FragmentC &c) {
    uint32_t const *A = reinterpret_cast<uint32_t const *>(&a);
    uint32_t const *B = reinterpret_cast<uint32_t const *>(&b);
    int const *C = reinterpret_cast<int const *>(&c);
    int *D = reinterpret_cast<int *>(&d);
    asm volatile(
        "mma.sync.aligned.m16n8k256.row.col.s32.b1.b1.s32.and.popc {%0,%1,%2,%3}, "
        "{%4,%5,%6,%7}, "
        "{%8,%9}, {%10,%11,%12,%13};\n"
        : "=r"(D[0]), "=r"(D[1]), "=r"(D[2]), "=r"(D[3])
        : "r"(A[0]), "r"(A[1]), "r"(A[2]), "r"(A[3]),
          "r"(B[0]), "r"(B[1]),
          "r"(C[0]), "r"(C[1]), "r"(C[2]), "r"(C[3])
    );
}

__global__ void mma_kernel(FragmentA *a, FragmentB *b, FragmentC *c, FragmentC *d_out) {
    FragmentC d;
    mma_operator(d, *a, *b, *c);
    *d_out = d;
}

int main() {
    // Allocate and initialize host data.
    FragmentA h_a;
    FragmentB h_b;
    FragmentC h_c = {0}; // bias matrix initialized to zero
    FragmentC h_d_out;

    // Initialize input fragments with all ones.
    for (int i = 0; i < M * K / 32; i++) {
        h_a.data[i] = 0xFFFFFFFF;
    }
    for (int i = 0; i < N * K / 32; i++) {
        h_b.data[i] = 0xFFFFFFFF;
    }
    for (int i = 0; i < M * N; i++) {
        h_c.data[i] = 0;
    }

    // Allocate device memory.
    FragmentA *d_a;
    FragmentB *d_b;
    FragmentC *d_c, *d_out;
    cudaCheckError(cudaMalloc(&d_a, sizeof(FragmentA)));
    cudaCheckError(cudaMalloc(&d_b, sizeof(FragmentB)));
    cudaCheckError(cudaMalloc(&d_c, sizeof(FragmentC)));
    cudaCheckError(cudaMalloc(&d_out, sizeof(FragmentC)));

    // Copy data from host to device.
    cudaCheckError(cudaMemcpy(d_a, &h_a, sizeof(FragmentA), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_b, &h_b, sizeof(FragmentB), cudaMemcpyHostToDevice));
    cudaCheckError(cudaMemcpy(d_c, &h_c, sizeof(FragmentC), cudaMemcpyHostToDevice));

    // Launch kernel. (Using 1 block of 32 threads.)
    mma_kernel<<<1, 32>>>(d_a, d_b, d_c, d_out);
    cudaCheckError(cudaGetLastError());
    cudaCheckError(cudaDeviceSynchronize());

    // Copy the result back to host.
    cudaCheckError(cudaMemcpy(&h_d_out, d_out, sizeof(FragmentC), cudaMemcpyDeviceToHost));

    // Test correctness:
    // With inputs of all ones and a zero bias, each computed value (from the 256 bit-products)
    // should be the popcount over 256 bits (i.e. 8 * 32 = 256).
    bool test_passed = true;
    const int expected = 256;
    for (int i = 0; i < 4; i++) {
        if (h_d_out.data[i] != expected) {
            std::cerr << "Error: At index " << i << ", expected " 
                      << expected << " but got " << h_d_out.data[i] << std::endl;
            test_passed = false;
        }
    }
    if (test_passed) {
        std::cout << "Test passed: all computed values are correct." << std::endl;
    } else {
        std::cout << "Test failed." << std::endl;
    }

    // Free device memory.
    cudaFree(d_a);
    cudaFree(d_b);
    cudaFree(d_c);
    cudaFree(d_out);

    return 0;
}
