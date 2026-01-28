import time
import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())


# ==================== MM Kernel ====================
@triton.jit
def mm_kernel(
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
):
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        offsets=[0, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        order=[1, 0],
    )

    a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
    a = smt.view(
        a_descriptor_load, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K)
    )
    b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
    b = smt.view(
        b_descriptor_load, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N)
    )

    accumulator = smt.dot(a, b)
    accumulator = smt.view(accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))
    c = accumulator.to(c_ptr.dtype.element_ty)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# ==================== ADDMM Kernel ====================
@triton.jit
def addmm_kernel(
    a_ptr,
    b_ptr,
    bias_ptr,
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
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
):
    """addmm: bias + A @ B"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        offsets=[0, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        order=[1, 0],
    )

    a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
    a = smt.view(
        a_descriptor_load, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K)
    )
    b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
    b = smt.view(
        b_descriptor_load, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N)
    )

    accumulator = smt.dot(a, b)
    accumulator = smt.view(accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))

    # Load bias and broadcast
    bias_block_ptr = tl.make_block_ptr(
        base=bias_ptr,
        shape=[N],
        strides=[1],
        offsets=[pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_N],
        order=[0],
    )
    bias = tl.load(bias_block_ptr, boundary_check=(0,))
    # Broadcast bias to [BLOCK_SIZE_M, BLOCK_SIZE_N]
    bias_broadcast = bias[None, :]

    c = accumulator.to(c_ptr.dtype.element_ty) + bias_broadcast

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# ==================== BMM Kernel ====================
@triton.jit
def bmm_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    B,
    M,
    N,
    K,
    stride_ab,
    stride_am,
    stride_ak,
    stride_bb,
    stride_bk,
    stride_bn,
    stride_cb,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
    BLOCK_SIZE_K: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
):
    """bmm: batched matrix multiplication"""
    pid_b = tl.program_id(0)
    pid_m = tl.program_id(1)
    pid_n = tl.program_id(2)

    # Offset to the batch
    a_batch_offset = pid_b * stride_ab
    b_batch_offset = pid_b * stride_bb
    c_batch_offset = pid_b * stride_cb

    a_block_ptr = tl.make_block_ptr(
        base=a_ptr + a_batch_offset,
        shape=[M, K],
        strides=[stride_am, stride_ak],
        offsets=[pid_m * BLOCK_SIZE_M, 0],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
        order=[1, 0],
    )

    b_block_ptr = tl.make_block_ptr(
        base=b_ptr + b_batch_offset,
        shape=[K, N],
        strides=[stride_bk, stride_bn],
        offsets=[0, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_K, BLOCK_SIZE_N],
        order=[1, 0],
    )

    a_descriptor_load = smt.descriptor_load(a_block_ptr, (0, 0))
    a = smt.view(
        a_descriptor_load, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K)
    )
    b_descriptor_load = smt.descriptor_load(b_block_ptr, (0, 0))
    b = smt.view(
        b_descriptor_load, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N)
    )

    accumulator = smt.dot(a, b)
    accumulator = smt.view(accumulator, (0, 0), (BLOCK_SIZE_M, BLOCK_SIZE_N), (1, 1))
    c = accumulator.to(c_ptr.dtype.element_ty)

    c_block_ptr = tl.make_block_ptr(
        base=c_ptr + c_batch_offset,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# ==================== MV Kernel ====================
@triton.jit
def mv_kernel(
    A,
    B,
    C,
    N,
    M,
    stride_an,
    stride_am,
    stride_bm,
    stride_cn,
    BLOCK_N: tl.constexpr,
    BLOCK_M: tl.constexpr,
):
    """mv: matrix-vector multiplication, C = A @ B
    A: [N, M] matrix
    B: [M] vector
    C: [N] output vector
    """
    pid = tl.program_id(0)

    # Initialize accumulator [BLOCK_N, BLOCK_M]
    acc = tl.zeros((BLOCK_N, BLOCK_M), dtype=tl.float32)

    # Loop over M dimension
    for m in range(0, M, BLOCK_M):
        # Load matrix block [BLOCK_N, BLOCK_M]
        A_block_ptr = tl.make_block_ptr(
            base=A,
            shape=[N, M],
            strides=[stride_an, stride_am],
            offsets=[pid * BLOCK_N, m],
            block_shape=[BLOCK_N, BLOCK_M],
            order=[1, 0],
        )
        a = tl.load(A_block_ptr, boundary_check=(0, 1)).to(tl.float32)

        # Load vector block [BLOCK_M]
        B_block_ptr = tl.make_block_ptr(
            base=B,
            shape=[M],
            strides=[stride_bm],
            offsets=[m],
            block_shape=[BLOCK_M],
            order=[0],
        )
        b = tl.load(B_block_ptr, boundary_check=(0,)).to(tl.float32)

        # Accumulate: a * b (broadcast b to [BLOCK_N, BLOCK_M])
        acc += a * b[None, :]

    # Sum over M dimension to get [BLOCK_N]
    acc = tl.sum(acc, axis=1)

    # Store output [BLOCK_N, 1]
    C_block_ptr = tl.make_block_ptr(
        base=C,
        shape=[N, 1],
        strides=[stride_cn, 1],
        offsets=[pid * BLOCK_N, 0],
        block_shape=[BLOCK_N, 1],
        order=[1, 0],
    )
    tl.store(C_block_ptr, acc[:, None].to(C.dtype.element_ty), boundary_check=(0, 1))


# ==================== Outer Kernel ====================
@triton.jit
def outer_kernel(
    a_ptr,
    b_ptr,
    c_ptr,
    M,
    N,
    stride_cm,
    stride_cn,
    BLOCK_SIZE_M: tl.constexpr,
    BLOCK_SIZE_N: tl.constexpr,
):
    """outer: outer product, c[i,j] = a[i] * b[j]"""
    pid_m = tl.program_id(0)
    pid_n = tl.program_id(1)

    # Load a block [BLOCK_SIZE_M]
    a_block_ptr = tl.make_block_ptr(
        base=a_ptr,
        shape=[M],
        strides=[1],
        offsets=[pid_m * BLOCK_SIZE_M],
        block_shape=[BLOCK_SIZE_M],
        order=[0],
    )
    a = tl.load(a_block_ptr, boundary_check=(0,))

    # Load b block [BLOCK_SIZE_N]
    b_block_ptr = tl.make_block_ptr(
        base=b_ptr,
        shape=[N],
        strides=[1],
        offsets=[pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_N],
        order=[0],
    )
    b = tl.load(b_block_ptr, boundary_check=(0,))

    # Compute outer product: a[:, None] * b[None, :] -> [BLOCK_SIZE_M, BLOCK_SIZE_N]
    c = a[:, None] * b[None, :]

    # Store output
    c_block_ptr = tl.make_block_ptr(
        base=c_ptr,
        shape=[M, N],
        strides=[stride_cm, stride_cn],
        offsets=[pid_m * BLOCK_SIZE_M, pid_n * BLOCK_SIZE_N],
        block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_N],
        order=[1, 0],
    )
    tl.store(c_block_ptr, c, boundary_check=(0, 1))


# ==================== Wrapper Functions ====================

def triton_mm(a, b, block_size_m=256, block_size_n=256, micro_m=None, micro_n=None, micro_k=None):
    """Matrix multiplication using Triton kernel: C = A @ B"""
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    # Set MICRO parameters based on dtype
    if a.dtype == torch.float32:
        micro_m = micro_m if micro_m is not None else 8
        micro_n = micro_n if micro_n is not None else 32
        micro_k = micro_k if micro_k is not None else 32
    else:  # float16
        micro_m = micro_m if micro_m is not None else 16
        micro_n = micro_n if micro_n is not None else 32
        micro_k = micro_k if micro_k is not None else 8

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    BLOCK_SIZE_K = triton.next_power_of_2(K)

    mm_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        MICRO_M=micro_m,
        MICRO_N=micro_n,
        MICRO_K=micro_k,
    )
    return c


def triton_addmm(bias, a, b, block_size_m=256, block_size_n=256, micro_m=None, micro_n=None, micro_k=None):
    """Addmm using Triton kernel: C = bias + A @ B"""
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()

    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape

    # Set MICRO parameters based on dtype
    if a.dtype == torch.float32:
        micro_m = micro_m if micro_m is not None else 8
        micro_n = micro_n if micro_n is not None else 32
        micro_k = micro_k if micro_k is not None else 32
    else:  # float16
        micro_m = micro_m if micro_m is not None else 16
        micro_n = micro_n if micro_n is not None else 32
        micro_k = micro_k if micro_k is not None else 8

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    BLOCK_SIZE_K = triton.next_power_of_2(K)

    addmm_kernel[grid](
        a, b, bias, c, M, N, K,
        a.stride(0), a.stride(1),
        b.stride(0), b.stride(1),
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        MICRO_M=micro_m,
        MICRO_N=micro_n,
        MICRO_K=micro_k,
    )
    return c


def triton_bmm(a, b, block_size_m=256, block_size_n=256, micro_m=None, micro_n=None, micro_k=None):
    """Batched matrix multiplication using Triton kernel: C = A @ B (batched)"""
    if a.stride(0) > 1 and a.stride(1) > 1 and a.stride(2) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1 and b.stride(2) > 1:
        b = b.contiguous()

    assert a.shape[2] == b.shape[1], "incompatible dimensions"
    assert a.shape[0] == b.shape[0], "batch size mismatch"
    B, M, K = a.shape
    _, _, N = b.shape

    # Set MICRO parameters based on dtype
    if a.dtype == torch.float32:
        micro_m = micro_m if micro_m is not None else 8
        micro_n = micro_n if micro_n is not None else 32
        micro_k = micro_k if micro_k is not None else 32
    else:  # float16
        micro_m = micro_m if micro_m is not None else 16
        micro_n = micro_n if micro_n is not None else 32
        micro_k = micro_k if micro_k is not None else 8

    c = torch.empty((B, M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (
            B,
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    BLOCK_SIZE_K = triton.next_power_of_2(K)

    bmm_kernel[grid](
        a, b, c, B, M, N, K,
        a.stride(0), a.stride(1), a.stride(2),
        b.stride(0), b.stride(1), b.stride(2),
        c.stride(0), c.stride(1), c.stride(2),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        MICRO_M=micro_m,
        MICRO_N=micro_n,
        MICRO_K=micro_k,
    )
    return c


def triton_mv(mat, vec, block_n=128, block_m=64):
    """Matrix-vector multiplication using Triton kernel: out = mat @ vec
    mat: [N, M] matrix
    vec: [M] vector
    out: [N] output vector
    """
    if mat.stride(0) > 1 and mat.stride(1) > 1:
        mat = mat.contiguous()

    N, M = mat.shape
    assert vec.shape[0] == M, "incompatible dimensions"

    out = torch.empty((N,), device=mat.device, dtype=mat.dtype)

    grid = (triton.cdiv(N, block_n),)

    mv_kernel[grid](
        mat, vec, out,
        N, M,
        mat.stride(0), mat.stride(1),
        vec.stride(0),
        out.stride(0),
        BLOCK_N=block_n,
        BLOCK_M=block_m,
    )
    return out


def triton_outer(a, b, block_size_m=128, block_size_n=128):
    """Outer product using Triton kernel: C[i,j] = a[i] * b[j]"""
    M = a.shape[0]
    N = b.shape[0]

    c = torch.empty((M, N), device=a.device, dtype=a.dtype)

    def grid(META):
        return (
            triton.cdiv(M, META["BLOCK_SIZE_M"]),
            triton.cdiv(N, META["BLOCK_SIZE_N"]),
        )

    outer_kernel[grid](
        a, b, c, M, N,
        c.stride(0), c.stride(1),
        BLOCK_SIZE_M=block_size_m,
        BLOCK_SIZE_N=block_size_n,
    )
    return c


# ==================== Validation Functions ====================

def validate_mm(test_name, M, N, K, dtype=torch.float32, atol=1e-2):
    """Validate mm kernel against PyTorch reference"""
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=dtype, device="cpu", requires_grad=False)
    B = torch.randn((K, N), dtype=dtype, device="cpu", requires_grad=False)

    output_triton = triton_mm(A, B)
    output_torch = torch.mm(A, B)

    max_diff = torch.max(torch.abs(output_triton - output_torch)).item()
    is_close = torch.allclose(output_triton, output_torch, atol=atol, rtol=0)

    status = "✅ PASS" if is_close else "❌ FAIL"
    print(f"  {status} | {test_name:20} | Shape: ({M}, {N}, {K}) | Max diff: {max_diff:.6f}")
    return is_close


def validate_addmm(test_name, M, N, K, dtype=torch.float32, atol=1e-2):
    """Validate addmm kernel against PyTorch reference"""
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=dtype, device="cpu", requires_grad=False)
    B = torch.randn((K, N), dtype=dtype, device="cpu", requires_grad=False)
    bias = torch.randn((N,), dtype=dtype, device="cpu", requires_grad=False)

    output_triton = triton_addmm(bias, A, B)
    output_torch = torch.addmm(bias, A, B)

    max_diff = torch.max(torch.abs(output_triton - output_torch)).item()
    is_close = torch.allclose(output_triton, output_torch, atol=atol, rtol=0)

    status = "✅ PASS" if is_close else "❌ FAIL"
    print(f"  {status} | {test_name:20} | Shape: ({M}, {N}, {K}) | Max diff: {max_diff:.6f}")
    return is_close


def validate_bmm(test_name, B, M, N, K, dtype=torch.float32, atol=1e-2):
    """Validate bmm kernel against PyTorch reference"""
    torch.manual_seed(0)
    A = torch.randn((B, M, K), dtype=dtype, device="cpu", requires_grad=False)
    Bmat = torch.randn((B, K, N), dtype=dtype, device="cpu", requires_grad=False)

    output_triton = triton_bmm(A, Bmat)
    output_torch = torch.bmm(A, Bmat)

    max_diff = torch.max(torch.abs(output_triton - output_torch)).item()
    is_close = torch.allclose(output_triton, output_torch, atol=atol, rtol=0)

    status = "✅ PASS" if is_close else "❌ FAIL"
    print(f"  {status} | {test_name:20} | Shape: ({B}, {M}, {N}, {K}) | Max diff: {max_diff:.6f}")
    return is_close


def validate_mv(test_name, M, N, dtype=torch.float32, atol=1e-4):
    """Validate mv kernel against PyTorch reference"""
    torch.manual_seed(0)
    mat = torch.randn((M, N), dtype=dtype, device="cpu", requires_grad=False)
    vec = torch.randn((N,), dtype=dtype, device="cpu", requires_grad=False)

    output_triton = triton_mv(mat, vec)
    output_torch = torch.mv(mat, vec)

    max_diff = torch.max(torch.abs(output_triton - output_torch)).item()
    is_close = torch.allclose(output_triton, output_torch, atol=atol, rtol=0)

    status = "✅ PASS" if is_close else "❌ FAIL"
    print(f"  {status} | {test_name:20} | Shape: ({M}, {N}) | Max diff: {max_diff:.6f}")
    return is_close


def validate_outer(test_name, M, N, dtype=torch.float32, atol=1e-4):
    """Validate outer kernel against PyTorch reference"""
    torch.manual_seed(0)
    a = torch.randn((M,), dtype=dtype, device="cpu", requires_grad=False)
    b = torch.randn((N,), dtype=dtype, device="cpu", requires_grad=False)

    output_triton = triton_outer(a, b)
    output_torch = torch.outer(a, b)

    max_diff = torch.max(torch.abs(output_triton - output_torch)).item()
    is_close = torch.allclose(output_triton, output_torch, atol=atol, rtol=0)

    status = "✅ PASS" if is_close else "❌ FAIL"
    print(f"  {status} | {test_name:20} | Shape: ({M}, {N}) | Max diff: {max_diff:.6f}")
    return is_close


# ==================== Benchmark Functions ====================

def benchmark_mm(M, N, K, dtype=torch.float32, num_warmup=5, num_iterations=100):
    """Benchmark mm: Triton vs PyTorch"""
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=dtype, device="cpu", requires_grad=False)
    B = torch.randn((K, N), dtype=dtype, device="cpu", requires_grad=False)

    for _ in range(num_warmup):
        _ = triton_mm(A, B)
    start = time.time()
    for _ in range(num_iterations):
        _ = triton_mm(A, B)
    triton_time = 1000 * (time.time() - start) / num_iterations


    start = time.time()
    for _ in range(10):
        _ = torch.mm(A, B)
    torch_time = 1000 * (time.time() - start) / 10

    ops = 2 * M * N * K
    triton_gflops = ops / (triton_time / 1000) / 1e9
    torch_gflops = ops / (torch_time / 1000) / 1e9
    speedup = torch_time / triton_time if triton_time > 0 else 0

    return triton_time, torch_time, triton_gflops, torch_gflops, speedup


def benchmark_addmm(M, N, K, dtype=torch.float32, num_warmup=5, num_iterations=100):
    """Benchmark addmm: Triton vs PyTorch"""
    torch.manual_seed(0)
    A = torch.randn((M, K), dtype=dtype, device="cpu", requires_grad=False)
    B = torch.randn((K, N), dtype=dtype, device="cpu", requires_grad=False)
    bias = torch.randn((N,), dtype=dtype, device="cpu", requires_grad=False)

    for _ in range(num_warmup):
        _ = triton_addmm(bias, A, B)
    start = time.time()
    for _ in range(num_iterations):
        _ = triton_addmm(bias, A, B)
    triton_time = 1000 * (time.time() - start) / num_iterations

    start = time.time()
    for _ in range(10):
        _ = torch.addmm(bias, A, B)
    torch_time = 1000 * (time.time() - start) / 10

    ops = 2 * M * N * K + M * N
    triton_gflops = ops / (triton_time / 1000) / 1e9
    torch_gflops = ops / (torch_time / 1000) / 1e9
    speedup = torch_time / triton_time if triton_time > 0 else 0

    return triton_time, torch_time, triton_gflops, torch_gflops, speedup


def benchmark_bmm(Batch, M, N, K, dtype=torch.float32, num_warmup=5, num_iterations=100):
    """Benchmark bmm: Triton vs PyTorch"""
    torch.manual_seed(0)
    A = torch.randn((Batch, M, K), dtype=dtype, device="cpu", requires_grad=False)
    B = torch.randn((Batch, K, N), dtype=dtype, device="cpu", requires_grad=False)

    for _ in range(num_warmup):
        _ = triton_bmm(A, B)
    start = time.time()
    for _ in range(num_iterations):
        _ = triton_bmm(A, B)
    triton_time = 1000 * (time.time() - start) / num_iterations

    start = time.time()
    for _ in range(10):
        _ = torch.bmm(A, B)
    torch_time = 1000 * (time.time() - start) / 10

    ops = 2 * Batch * M * N * K
    triton_gflops = ops / (triton_time / 1000) / 1e9
    torch_gflops = ops / (torch_time / 1000) / 1e9
    speedup = torch_time / triton_time if triton_time > 0 else 0

    return triton_time, torch_time, triton_gflops, torch_gflops, speedup


def benchmark_mv(M, N, dtype=torch.float32, num_warmup=5, num_iterations=100):
    """Benchmark mv: Triton vs PyTorch"""
    torch.manual_seed(0)
    mat = torch.randn((M, N), dtype=dtype, device="cpu", requires_grad=False)
    vec = torch.randn((N,), dtype=dtype, device="cpu", requires_grad=False)

    for _ in range(num_warmup):
        _ = triton_mv(mat, vec)
    start = time.time()
    for _ in range(num_iterations):
        _ = triton_mv(mat, vec)
    triton_time = 1000 * (time.time() - start) / num_iterations

    start = time.time()
    for _ in range(10):
        _ = torch.mv(mat, vec)
    torch_time = 1000 * (time.time() - start) / 10

    ops = 2 * M * N
    triton_gflops = ops / (triton_time / 1000) / 1e9
    torch_gflops = ops / (torch_time / 1000) / 1e9
    speedup = torch_time / triton_time if triton_time > 0 else 0

    return triton_time, torch_time, triton_gflops, torch_gflops, speedup


def benchmark_outer(M, N, dtype=torch.float32, num_warmup=5, num_iterations=100):
    """Benchmark outer: Triton vs PyTorch"""
    torch.manual_seed(0)
    a = torch.randn((M,), dtype=dtype, device="cpu", requires_grad=False)
    b = torch.randn((N,), dtype=dtype, device="cpu", requires_grad=False)

    for _ in range(num_warmup):
        _ = triton_outer(a, b)
    start = time.time()
    for _ in range(num_iterations):
        _ = triton_outer(a, b)
    triton_time = 1000 * (time.time() - start) / num_iterations

    start = time.time()
    for _ in range(10):
        _ = torch.outer(a, b)
    torch_time = 1000 * (time.time() - start) / 10

    ops = M * N
    triton_gflops = ops / (triton_time / 1000) / 1e9
    torch_gflops = ops / (torch_time / 1000) / 1e9
    speedup = torch_time / triton_time if triton_time > 0 else 0

    return triton_time, torch_time, triton_gflops, torch_gflops, speedup


if __name__ == "__main__":
    test_warm_up = 5
    test_iterations = 100

    # (M, N, K) shapes for matrix multiplication
    test_shape_list = [
        (1024, 1024, 1024),
        (512, 512, 512),
        (256, 256, 256),
        (128, 128, 128),
    ]

    # (B, M, N, K) shapes for batched matrix multiplication
    test_bmm_shape_list = [
        (4, 256, 256, 256),
        (8, 128, 128, 128),
        (16, 64, 64, 64),
    ]

    # (M, N) shapes for mv and outer
    test_mv_shape_list = [
        (1024, 1024),
        (512, 512),
        (256, 256),
    ]

    test_dtype_list = [torch.float32, torch.float16]

    print("=" * 120)
    print("Custom Matrix Operations - Triton vs PyTorch Benchmark")
    print("=" * 120)

    # ==================== MM ====================
    print(f"\n{'#' * 120}")
    print(f"# {'MATRIX MULTIPLICATION (mm)':^116} #")
    print(f"{'#' * 120}\n")

    for test_dtype in test_dtype_list:
        print(f"  dtype: {test_dtype}")
        print(f"  {'-' * 116}")
        print(
            f"  {'Shape (M,N,K)':25} | {'Triton (ms)':15} | {'PyTorch (ms)':15} | "
            f"{'Triton GFLOPS':15} | {'PyTorch GFLOPS':15} | {'Speedup':10}"
        )
        print(f"  {'-' * 116}")

        for test_shape in test_shape_list:
            M, N, K = test_shape
            try:
                triton_time, torch_time, triton_gflops, torch_gflops, speedup = benchmark_mm(
                    M, N, K, dtype=test_dtype, num_warmup=test_warm_up, num_iterations=test_iterations,
                )
                print(
                    f"  {str(test_shape):25} | {triton_time:15.4f} | {torch_time:15.4f} | "
                    f"{triton_gflops:15.2f} | {torch_gflops:15.2f} | {speedup:10.2f}x"
                )
            except Exception as e:
                print(f"  {str(test_shape):25} | Failed: {str(e)}")
        print()

    # ==================== ADDMM ====================
    print(f"\n{'#' * 120}")
    print(f"# {'ADDMM (bias + A @ B)':^116} #")
    print(f"{'#' * 120}\n")

    for test_dtype in test_dtype_list:
        print(f"  dtype: {test_dtype}")
        print(f"  {'-' * 116}")
        print(
            f"  {'Shape (M,N,K)':25} | {'Triton (ms)':15} | {'PyTorch (ms)':15} | "
            f"{'Triton GFLOPS':15} | {'PyTorch GFLOPS':15} | {'Speedup':10}"
        )
        print(f"  {'-' * 116}")

        for test_shape in test_shape_list:
            M, N, K = test_shape
            try:
                triton_time, torch_time, triton_gflops, torch_gflops, speedup = benchmark_addmm(
                    M, N, K, dtype=test_dtype, num_warmup=test_warm_up, num_iterations=test_iterations,
                )
                print(
                    f"  {str(test_shape):25} | {triton_time:15.4f} | {torch_time:15.4f} | "
                    f"{triton_gflops:15.2f} | {torch_gflops:15.2f} | {speedup:10.2f}x"
                )
            except Exception as e:
                print(f"  {str(test_shape):25} | Failed: {str(e)}")
        print()

    # ==================== BMM ====================
    print(f"\n{'#' * 120}")
    print(f"# {'BATCHED MATRIX MULTIPLICATION (bmm)':^116} #")
    print(f"{'#' * 120}\n")

    for test_dtype in test_dtype_list:
        print(f"  dtype: {test_dtype}")
        print(f"  {'-' * 116}")
        print(
            f"  {'Shape (B,M,N,K)':25} | {'Triton (ms)':15} | {'PyTorch (ms)':15} | "
            f"{'Triton GFLOPS':15} | {'PyTorch GFLOPS':15} | {'Speedup':10}"
        )
        print(f"  {'-' * 116}")

        for test_shape in test_bmm_shape_list:
            B, M, N, K = test_shape
            try:
                triton_time, torch_time, triton_gflops, torch_gflops, speedup = benchmark_bmm(
                    B, M, N, K, dtype=test_dtype, num_warmup=test_warm_up, num_iterations=test_iterations,
                )
                print(
                    f"  {str(test_shape):25} | {triton_time:15.4f} | {torch_time:15.4f} | "
                    f"{triton_gflops:15.2f} | {torch_gflops:15.2f} | {speedup:10.2f}x"
                )
            except Exception as e:
                print(f"  {str(test_shape):25} | Failed: {str(e)}")
        print()

    # ==================== MV ====================
    print(f"\n{'#' * 120}")
    print(f"# {'MATRIX-VECTOR MULTIPLICATION (mv)':^116} #")
    print(f"{'#' * 120}\n")

    for test_dtype in test_dtype_list:
        print(f"  dtype: {test_dtype}")
        print(f"  {'-' * 116}")
        print(
            f"  {'Shape (M,N)':25} | {'Triton (ms)':15} | {'PyTorch (ms)':15} | "
            f"{'Triton GFLOPS':15} | {'PyTorch GFLOPS':15} | {'Speedup':10}"
        )
        print(f"  {'-' * 116}")

        for test_shape in test_mv_shape_list:
            M, N = test_shape
            try:
                triton_time, torch_time, triton_gflops, torch_gflops, speedup = benchmark_mv(
                    M, N, dtype=test_dtype, num_warmup=test_warm_up, num_iterations=test_iterations,
                )
                print(
                    f"  {str(test_shape):25} | {triton_time:15.4f} | {torch_time:15.4f} | "
                    f"{triton_gflops:15.2f} | {torch_gflops:15.2f} | {speedup:10.2f}x"
                )
            except Exception as e:
                print(f"  {str(test_shape):25} | Failed: {str(e)}")
        print()

    # ==================== OUTER ====================
    print(f"\n{'#' * 120}")
    print(f"# {'OUTER PRODUCT (outer)':^116} #")
    print(f"{'#' * 120}\n")

    for test_dtype in test_dtype_list:
        print(f"  dtype: {test_dtype}")
        print(f"  {'-' * 116}")
        print(
            f"  {'Shape (M,N)':25} | {'Triton (ms)':15} | {'PyTorch (ms)':15} | "
            f"{'Triton GFLOPS':15} | {'PyTorch GFLOPS':15} | {'Speedup':10}"
        )
        print(f"  {'-' * 116}")

        for test_shape in test_mv_shape_list:
            M, N = test_shape
            try:
                triton_time, torch_time, triton_gflops, torch_gflops, speedup = benchmark_outer(
                    M, N, dtype=test_dtype, num_warmup=test_warm_up, num_iterations=test_iterations,
                )
                print(
                    f"  {str(test_shape):25} | {triton_time:15.4f} | {torch_time:15.4f} | "
                    f"{triton_gflops:15.2f} | {torch_gflops:15.2f} | {speedup:10.2f}x"
                )
            except Exception as e:
                print(f"  {str(test_shape):25} | Failed: {str(e)}")
        print()

    # ==================== Validation ====================
    print(f"\n{'#' * 120}")
    print(f"# {'VALIDATION':^116} #")
    print(f"{'#' * 120}\n")

    print("  Validating Triton kernels against PyTorch reference...")
    print(f"  {'-' * 116}")

    all_passed = True

    # Validate mm
    print("\n  --- mm ---")
    for test_dtype in test_dtype_list:
        atol = 1e-2 if test_dtype == torch.float16 else 1e-4
        for test_shape in test_shape_list:
            M, N, K = test_shape
            test_name = f"mm_{test_dtype}".replace("torch.", "")
            try:
                passed = validate_mm(test_name, M, N, K, dtype=test_dtype, atol=atol)
                all_passed = all_passed and passed
            except Exception as e:
                print(f"  ❌ FAIL | {test_name:20} | Shape: ({M}, {N}, {K}) | Error: {str(e)}")
                all_passed = False

    # Validate addmm
    print("\n  --- addmm ---")
    for test_dtype in test_dtype_list:
        atol = 1e-2 if test_dtype == torch.float16 else 1e-4
        for test_shape in test_shape_list:
            M, N, K = test_shape
            test_name = f"addmm_{test_dtype}".replace("torch.", "")
            try:
                passed = validate_addmm(test_name, M, N, K, dtype=test_dtype, atol=atol)
                all_passed = all_passed and passed
            except Exception as e:
                print(f"  ❌ FAIL | {test_name:20} | Shape: ({M}, {N}, {K}) | Error: {str(e)}")
                all_passed = False

    # Validate bmm
    print("\n  --- bmm ---")
    for test_dtype in test_dtype_list:
        atol = 1e-2 if test_dtype == torch.float16 else 1e-4
        for test_shape in test_bmm_shape_list:
            B, M, N, K = test_shape
            test_name = f"bmm_{test_dtype}".replace("torch.", "")
            try:
                passed = validate_bmm(test_name, B, M, N, K, dtype=test_dtype, atol=atol)
                all_passed = all_passed and passed
            except Exception as e:
                print(f"  ❌ FAIL | {test_name:20} | Shape: ({B}, {M}, {N}, {K}) | Error: {str(e)}")
                all_passed = False

    # Validate mv
    print("\n  --- mv ---")
    for test_dtype in test_dtype_list:
        atol = 1e-2 if test_dtype == torch.float16 else 1e-4
        for test_shape in test_mv_shape_list:
            M, N = test_shape
            test_name = f"mv_{test_dtype}".replace("torch.", "")
            try:
                passed = validate_mv(test_name, M, N, dtype=test_dtype, atol=atol)
                all_passed = all_passed and passed
            except Exception as e:
                print(f"  ❌ FAIL | {test_name:20} | Shape: ({M}, {N}) | Error: {str(e)}")
                all_passed = False

    # Validate outer
    print("\n  --- outer ---")
    for test_dtype in test_dtype_list:
        atol = 1e-2 if test_dtype == torch.float16 else 1e-4
        for test_shape in test_mv_shape_list:
            M, N = test_shape
            test_name = f"outer_{test_dtype}".replace("torch.", "")
            try:
                passed = validate_outer(test_name, M, N, dtype=test_dtype, atol=atol)
                all_passed = all_passed and passed
            except Exception as e:
                print(f"  ❌ FAIL | {test_name:20} | Shape: ({M}, {N}) | Error: {str(e)}")
                all_passed = False

    print(f"\n  {'-' * 116}")
    if all_passed:
        print("  ✅ All validations passed!")
    else:
        print("  ❌ Some validations failed!")

    print("\n" + "=" * 120)
