import time
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
from triton.language.extra.cpu import libdevice as tl_extra_shim


@triton.jit
def silu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """SiLU: x * sigmoid(x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)

    # SiLU = x / (1 + exp(-x))
    # Optimized to match FlagGems:
    # x_fp32 = x.to(tl.float32)
    # y = tl.fdiv(x_fp32, (1.0 + tl.exp(-x_fp32)))
    x_fp32 = x.to(tl.float32)
    y = x_fp32 / (1.0 + tl.exp(-x_fp32))

    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def relu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """ReLU: max(0, x)"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)

    # ReLU = max(0, x) - optimizing using tl.where as seen in FlagGems
    y = tl.where(x > 0, x, 0)

    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def gelu_none_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GeLU using erf: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)

    # GeLU formula: 0.5 * x * (1 + erf(x / sqrt(2)))
    scale: tl.constexpr = 0.7071067811  # 1 / sqrt(2)
    y = 0.5 * x * (1.0 + tl_extra_shim.erf(x * scale))

    tl.store(out_ptr + offsets, y, mask=mask)


@triton.jit
def gelu_tanh_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
):
    """GeLU using tanh approximation"""
    pid = tl.program_id(0)
    block_start = pid * BLOCK_SIZE
    offsets = block_start + tl.arange(0, BLOCK_SIZE)
    mask = offsets < n_elements

    x = tl.load(in_ptr + offsets, mask=mask)

    # FlagGems optimized implementation:
    # 0.5 * x * (1 + tanh(x * 0.79788456 * (1 + 0.044715 * x^2)))

    sqrt_2_over_pi: tl.constexpr = 0.79788456
    coeff: tl.constexpr = 0.044715

    x_fp32 = x.to(tl.float32)
    x_sq = x_fp32 * x_fp32

    # x * 0.79788456 * (1 + 0.044715 * x^2)
    inner = x_fp32 * sqrt_2_over_pi * (1.0 + coeff * x_sq)

    y = 0.5 * x * (1.0 + tl_extra_shim.tanh(inner))

    tl.store(out_ptr + offsets, y, mask=mask)


def validate_op(kernel, op_func, test_name, atol=1e-4, dtype=torch.float32, size=64*64):
    """Generic function to validate a kernel against PyTorch reference"""
    torch.manual_seed(0)
    x = torch.randn(size, dtype=dtype)
    output_triton = torch.empty_like(x)
    output_torch = op_func(x)

    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    max_diff = torch.max(torch.abs(output_triton - output_torch)).item()
    is_close = torch.allclose(output_triton, output_torch, atol=atol)

    status = "✅ PASS" if is_close else "❌ FAIL"
    print(f"{status} | {test_name:20} | Max diff: {max_diff:.6f}")

    return is_close


def benchmark_op(kernel, op_func, test_name, num_iterations=100, size=1024, dtype=torch.float32):
    """Generic function to benchmark a kernel"""
    torch.manual_seed(0)
    x = torch.randn(size, dtype=dtype)
    output = torch.empty_like(x)

    n_elements = x.numel()
    BLOCK_SIZE = 128
    grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

    # Warmup
    for _ in range(10):
        kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)

    # Benchmark Triton
    start = time.time()
    for _ in range(num_iterations):
        kernel[grid](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE)
    triton_time = (time.time() - start) / num_iterations * 1000  # Convert to ms

    # Benchmark PyTorch
    start = time.time()
    for _ in range(num_iterations):
        _ = op_func(x)
    torch_time = (time.time() - start) / num_iterations * 1000  # Convert to ms

    print(f"triton: dtype {dtype} size {size}, cost {triton_time:.3f} ms")
    print(f"torch:  dtype {dtype} size {size}, cost {torch_time:.3f} ms")


if __name__ == "__main__":
    test_warm_up = 5
    test_iterations = 100

    test_size_list = [1024 * 1024, 512 * 512, 256 * 256, 128 * 128, 32 * 32]
    test_dtype_list = [torch.float32, torch.float16]

    test_op_list = {
        "silu": (silu_kernel, torch.nn.functional.silu, 1e-4),
        "relu": (relu_kernel, torch.nn.functional.relu, 1e-4),
        "gelu_none": (gelu_none_kernel, lambda x: torch.nn.functional.gelu(x, approximate="none"), 1e-4),
        "gelu_tanh": (gelu_tanh_kernel, lambda x: torch.nn.functional.gelu(x, approximate="tanh"), 1e-3),
    }

    for op_name, (kernel, op_func, atol) in test_op_list.items():
        print(f"############ {op_name} ############")

        for test_dtype in test_dtype_list:
            for test_size in test_size_list:
                x = torch.randn(test_size, dtype=test_dtype, device="cpu")

                # Triton benchmark
                output_triton = torch.empty_like(x)
                n_elements = x.numel()
                # FlagGems uses dynamic BLOCK_SIZE (MAX_TILE_SIZE=1024 for Spacemit)
                # Increasing BLOCK_SIZE from 128 to 1024 to match optimization
                BLOCK_SIZE = 1024
                grid = (triton.cdiv(n_elements, BLOCK_SIZE),)

                for _ in range(test_warm_up):
                    kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=BLOCK_SIZE)

                start = time.time()
                for _ in range(test_iterations):
                    kernel[grid](x, output_triton, n_elements, BLOCK_SIZE=BLOCK_SIZE)
                end = time.time()

                print(
                    f"triton: dtype {test_dtype} size {test_size}, cost {1000 * (end - start) / test_iterations:.3f} ms"
                )

                # PyTorch benchmark
                for _ in range(test_warm_up):
                    _ = op_func(x)

                start = time.time()
                for _ in range(test_iterations):
                    _ = op_func(x)
                end = time.time()

                print(
                    f"torch:  dtype {test_dtype} size {test_size}, cost {1000 * (end - start) / test_iterations:.3f} ms"
                )