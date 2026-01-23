import time
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
from triton.language.extra.cpu import libdevice as tl_extra_shim


# ============================================================================
# Elementwise Math Function Kernels
# ============================================================================


@triton.jit
def exp_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Exponential function: exp(x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)
    y = tl_extra_shim.exp(x)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def cos_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Cosine function: cos(x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)
    y = tl.cos(x)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def sin_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Sine function: sin(x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)
    y = tl.sin(x)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def tanh_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Hyperbolic tangent function: tanh(x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)
    y = tl_extra_shim.tanh(x)
    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def erf_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """Error function: erf(x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)
    y = tl_extra_shim.erf(x)
    tl.store(out_ptr + offsets, y, mask=mask)


# ============================================================================
# Performance Benchmark Functions
# ============================================================================


def benchmark_triton_kernel(kernel, x, num_iterations=100):
    """Benchmark a Triton kernel"""
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Warmup
    for _ in range(10):
        kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    elapsed = (time.time() - start) / num_iterations * 1000  # Convert to ms

    return elapsed


def benchmark_torch_op(op_func, x, num_iterations=100):
    """Benchmark a PyTorch operation"""
    # Warmup
    for _ in range(10):
        _ = op_func(x)

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        _ = op_func(x)
    elapsed = (time.time() - start) / num_iterations * 1000  # Convert to ms

    return elapsed


if __name__ == "__main__":
    test_warm_up = 5
    test_iterations = 100

    test_shape_list = [(1024, 1024), (512, 512), (256, 256), (128, 128), (32, 32)]
    test_dtype_list = [torch.float32]

    test_ops = {
        "exp": (exp_kernel, torch.exp),
        "cos": (cos_kernel, torch.cos),
        "sin": (sin_kernel, torch.sin),
        "tanh": (tanh_kernel, torch.tanh),
        "erf": (erf_kernel, torch.erf),
    }

    print("=" * 100)
    print("Elementwise Math Functions - Triton vs PyTorch Benchmark")
    print("=" * 100)

    for op_name, (triton_kernel, torch_func) in test_ops.items():
        print(f"\n{'#' * 100}")
        print(f"{'#':^100}")
        print(f"# {op_name.upper():^96} #")
        print(f"{'#':^100}")
        print(f"{'#' * 100}\n")

        for test_dtype in test_dtype_list:
            print(f"  dtype: {test_dtype}")
            print(f"  {'-' * 96}")
            print(f"  {'Shape':20} | {'Triton (ms)':20} | {'PyTorch (ms)':20} | {'Speedup':15}")
            print(f"  {'-' * 96}")

            for test_shape in test_shape_list:
                x = torch.randn(test_shape, dtype=test_dtype, device="cpu")

                triton_time = benchmark_triton_kernel(triton_kernel, x, test_iterations)
                torch_time = benchmark_torch_op(torch_func, x, test_iterations)
                speedup = torch_time / triton_time if triton_time > 0 else 0

                print(
                    f"  {str(test_shape):20} | {triton_time:20.4f} | {torch_time:20.4f} | {speedup:15.2f}x"
                )

            print()

    print("=" * 100)
