// Pluggable-allocator shim with an env-selected backend, so the SAME kernels can
// be run over pools from each allocation type:
//   W28_BACKEND=managed   cudaMallocManaged        (host-addressable, W28b)
//   W28_BACKEND=pinned    cudaHostAlloc            (host memory, GPU-visible)
//   W28_BACKEND=pageable  posix_memalign           (plain malloc; needs ATS)
// The point is bandwidth under a scattered/paged access pattern, which is what
// attention actually does -- a contiguous stream flatters every backend.
#include <cuda_runtime.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>

static int backend = -1;   // 0 managed, 1 pinned, 2 pageable

static void pick(void) {
  const char *b = getenv("W28_BACKEND");
  if (!b) b = "managed";
  if (!strcmp(b, "pinned")) backend = 1;
  else if (!strcmp(b, "pageable")) backend = 2;
  else backend = 0;
  fprintf(stderr, "[shim] backend=%s\n", b);
}

extern "C" void *managed_malloc(ssize_t size, int device, cudaStream_t stream) {
  (void)device; (void)stream;
  if (backend < 0) pick();
  void *p = NULL;
  cudaError_t rc = cudaSuccess;
  if (backend == 0) {
    rc = cudaMallocManaged(&p, (size_t)size, cudaMemAttachGlobal);
  } else if (backend == 1) {
    rc = cudaHostAlloc(&p, (size_t)size, cudaHostAllocMapped);
  } else {
    if (posix_memalign(&p, 4096, (size_t)size) != 0) p = NULL;
  }
  if (!p || rc != cudaSuccess) {
    fprintf(stderr, "[shim] alloc(%zd) failed: %s\n", size,
            rc == cudaSuccess ? "posix_memalign" : cudaGetErrorString(rc));
    return NULL;
  }
  return p;
}

extern "C" void managed_free(void *ptr, ssize_t size, int device, cudaStream_t stream) {
  (void)size; (void)device; (void)stream;
  if (backend == 0) cudaFree(ptr);
  else if (backend == 1) cudaFreeHost(ptr);
  else free(ptr);
}
