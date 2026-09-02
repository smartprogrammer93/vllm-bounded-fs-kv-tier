// Pluggable-allocator shim: hand torch cudaMallocManaged memory instead of
// cudaMalloc, so a pool allocated through it is host-addressable and can be
// written straight to disk with pwritev. Placement hints are deliberately
// omitted: the benchmark's warmup iterations touch every page from the GPU,
// which migrates them, and the CUDA 13 advise/prefetch signatures differ.
#include <cuda_runtime.h>
#include <stdio.h>

extern "C" void *managed_malloc(ssize_t size, int device, cudaStream_t stream) {
  (void)device; (void)stream;
  void *p = NULL;
  cudaError_t rc = cudaMallocManaged(&p, (size_t)size, cudaMemAttachGlobal);
  if (rc != cudaSuccess) {
    fprintf(stderr, "managed_malloc(%zd): %s\n", size, cudaGetErrorString(rc));
    return NULL;
  }
  return p;
}

extern "C" void managed_free(void *ptr, ssize_t size, int device, cudaStream_t stream) {
  (void)size; (void)device; (void)stream;
  cudaFree(ptr);
}
