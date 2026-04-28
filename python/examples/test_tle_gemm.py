"""
TLE GEMM Integration Test — spine-triton native version

Adapted from FlagTree/python/test/tle/integration/test_tle_gemm.py.
Uses spine-triton native TLE APIs (tle.alloc, tle.local_ptr, tle.copy)
instead of FlagTree tle.gpu.* namespace.

Workflow:
  1. tle.alloc  → allocate local buffer (scope="l2")
  2. tle.local_ptr → generate pointer tensor from buffer + indices
  3. tle.copy   → synchronous global→local copy (tl.load + tl.store)
  4. tl.load    → load from local pointer tensor
  5. tl.dot     → accumulate GEMM tile
"""

import torch
import triton
import triton.language as tl
from triton.language.extra import tle
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def tle_gemm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    K,
    stride_am,
    stride_ak,
    stride_bk,
    stride_bn,
    stride_cm,
    stride_cn,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    BLOCK_K: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    offs_m = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_n = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)

    accumulator = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # --- allocate local buffers via tle.alloc (reuses smt.alloc → xsmt.alloc) ---
    a_local = tle.alloc([BLOCK_M, BLOCK_K], dtype=tl.float32, scope="l2")
    b_local = tle.alloc([BLOCK_K, BLOCK_N], dtype=tl.float32, scope="l2")

    # --- build local pointer tensors via tle.local_ptr ---
    a_row_ids = tl.broadcast_to(tl.arange(0, BLOCK_M)[:, None], (BLOCK_M, BLOCK_K))
    a_col_ids = tl.broadcast_to(tl.arange(0, BLOCK_K)[None, :], (BLOCK_M, BLOCK_K))
    a_local_ptrs = tle.local_ptr(a_local, (a_row_ids, a_col_ids))

    b_row_ids = tl.broadcast_to(tl.arange(0, BLOCK_K)[:, None], (BLOCK_K, BLOCK_N))
    b_col_ids = tl.broadcast_to(tl.arange(0, BLOCK_N)[None, :], (BLOCK_K, BLOCK_N))
    b_local_ptrs = tle.local_ptr(b_local, (b_row_ids, b_col_ids))

    for k_start in range(0, K, BLOCK_K):
        k_offs = k_start + tl.arange(0, BLOCK_K)

        # global pointers for this tile
        a_ptrs = a_ptr + offs_m[:, None] * stride_am + k_offs[None, :] * stride_ak
        b_ptrs = b_ptr + k_offs[:, None] * stride_bk + offs_n[None, :] * stride_bn

        # synchronous copy: global → local
        # tle.copy(src, dst) expands to tl.load(src) + tl.store(dst, value)
        tle.copy(a_ptrs, a_local_ptrs, [BLOCK_M, BLOCK_N])
        tle.copy(b_ptrs, b_local_ptrs, [BLOCK_M, BLOCK_N])

        # load from local and accumulate
        a_tile = tl.load(a_local_ptrs)
        b_tile = tl.load(b_local_ptrs)
        accumulator += tl.dot(a_tile, b_tile)

    # store result
    c_ptrs = c_ptr + offs_m[:, None] * stride_cm + offs_n[None, :] * stride_cn
    mask = (offs_m[:, None] < M) & (offs_n[None, :] < N)
    tl.store(c_ptrs, accumulator, mask=mask)


def run_tle_gemm(a, b, c, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64):
    assert a.shape[1] == b.shape[0]
    M, K = a.shape
    _, N = b.shape

    grid = (triton.cdiv(M, BLOCK_M), triton.cdiv(N, BLOCK_N))
    tle_gemm_kernel[grid](
        a,
        b,
        c,
        M,
        N,
        K,
        a.stride(0),
        a.stride(1),
        b.stride(0),
        b.stride(1),
        c.stride(0),
        c.stride(1),
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
    )


def test_tle_gemm_square():
    """Basic square GEMM: 128x128 x 128x128"""
    torch.manual_seed(42)
    M, N, K = 128, 128, 128
    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)
    c = torch.empty(M, N, dtype=torch.float32)

    run_tle_gemm(a, b, c, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64)

    expected = torch.matmul(a, b)
    torch.testing.assert_close(c, expected, atol=1e-2, rtol=1e-2)
    print("[PASS] test_tle_gemm_square")


def test_tle_gemm_rectangular():
    """Rectangular GEMM: 128x256 x 256x64"""
    torch.manual_seed(123)
    M, N, K = 128, 64, 256
    a = torch.randn(M, K, dtype=torch.float32)
    b = torch.randn(K, N, dtype=torch.float32)
    c = torch.empty(M, N, dtype=torch.float32)

    run_tle_gemm(a, b, c, BLOCK_M=64, BLOCK_N=64, BLOCK_K=64)

    expected = torch.matmul(a, b)
    torch.testing.assert_close(c, expected, atol=1e-2, rtol=1e-2)
    print("[PASS] test_tle_gemm_rectangular")


if __name__ == "__main__":
    test_tle_gemm_square()
    test_tle_gemm_rectangular()
    print("[ALL PASS] test_tle_gemm")
