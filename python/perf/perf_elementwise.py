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
    num_ctas: tl.constexpr,
):
    """Exponential function: exp(x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        x = tl.load(in_block_ptr, boundary_check=(0,))
        y = tl_extra_shim.exp(x)
        tl.store(out_block_ptr, y, boundary_check=(0,))


@triton.jit
def cos_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """Cosine function: cos(x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        x = tl.load(in_block_ptr, boundary_check=(0,))
        y = tl.cos(x)
        tl.store(out_block_ptr, y, boundary_check=(0,))


@triton.jit
def sin_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """Sine function: sin(x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        x = tl.load(in_block_ptr, boundary_check=(0,))
        y = tl.sin(x)
        tl.store(out_block_ptr, y, boundary_check=(0,))


@triton.jit
def tanh_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """Hyperbolic tangent function: tanh(x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        x = tl.load(in_block_ptr, boundary_check=(0,))
        y = tl_extra_shim.tanh(x)
        tl.store(out_block_ptr, y, boundary_check=(0,))


@triton.jit
def erf_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """Error function: erf(x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        x = tl.load(in_block_ptr, boundary_check=(0,))
        y = tl_extra_shim.erf(x)
        tl.store(out_block_ptr, y, boundary_check=(0,))


# ============================================================================
# Activation Function Kernels
# ============================================================================


@triton.jit
def silu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """SiLU: x * sigmoid(x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        x = tl.load(in_block_ptr, boundary_check=(0,))

        # SiLU = x / (1 + exp(-x))
        x_fp32 = x.to(tl.float32)
        y = x_fp32 / (1.0 + tl.exp(-x_fp32))
        y = y.to(x.dtype)

        tl.store(out_block_ptr, y, boundary_check=(0,))


@triton.jit
def relu_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """ReLU: max(0, x)"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        x = tl.load(in_block_ptr, boundary_check=(0,))

        # ReLU = max(0, x)
        y = tl.where(x > 0, x, 0)

        tl.store(out_block_ptr, y, boundary_check=(0,))


@triton.jit
def gelu_none_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """GeLU using erf: 0.5 * x * (1 + erf(x / sqrt(2)))"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        x = tl.load(in_block_ptr, boundary_check=(0,))

        # GeLU formula: 0.5 * x * (1 + erf(x / sqrt(2)))
        scale: tl.constexpr = 0.7071067811  # 1 / sqrt(2)
        y = 0.5 * x * (1.0 + tl_extra_shim.erf(x * scale))

        tl.store(out_block_ptr, y, boundary_check=(0,))


@triton.jit
def gelu_tanh_kernel(
    in_ptr,
    out_ptr,
    n_elements,
    BLOCK_SIZE: tl.constexpr,
    num_ctas: tl.constexpr,
):
    """GeLU using tanh approximation"""
    pid = tl.program_id(0)
    num_blocks = tl.cdiv(n_elements, BLOCK_SIZE)
    sub_num = tl.cdiv(max(num_blocks - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_idx = pid + num_ctas * block_idx
        block_start = task_idx * BLOCK_SIZE

        in_block_ptr = tl.make_block_ptr(
            base=in_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )
        out_block_ptr = tl.make_block_ptr(
            base=out_ptr,
            shape=(n_elements,),
            strides=(1,),
            offsets=(block_start,),
            block_shape=(BLOCK_SIZE,),
            order=(0,),
        )

        x = tl.load(in_block_ptr, boundary_check=(0,))

        # GeLU tanh approximation:
        # 0.5 * x * (1 + tanh(x * 0.79788456 * (1 + 0.044715 * x^2)))
        sqrt_2_over_pi: tl.constexpr = 0.79788456
        coeff: tl.constexpr = 0.044715

        x_fp32 = x.to(tl.float32)
        x_sq = x_fp32 * x_fp32
        inner = x_fp32 * sqrt_2_over_pi * (1.0 + coeff * x_sq)

        y = 0.5 * x_fp32 * (1.0 + tl_extra_shim.tanh(inner))
        y = y.to(x.dtype)

        tl.store(out_block_ptr, y, boundary_check=(0,))


# ============================================================================
# Performance Benchmark Functions
# ============================================================================


def benchmark_triton_kernel(kernel, x, num_iterations=100, num_ctas=16, num_warmup=10):
    """Benchmark a Triton kernel"""
    output = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 32

    # Warmup
    for _ in range(num_warmup):
        kernel[(num_ctas,)](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_ctas=num_ctas)

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        kernel[(num_ctas,)](x, output, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_ctas=num_ctas)
    elapsed = (time.time() - start) / num_iterations * 1000  # Convert to ms

    return elapsed


def benchmark_torch_op(op_func, x, num_iterations=100, num_warmup=10):
    """Benchmark a PyTorch operation"""
    # Warmup
    for _ in range(num_warmup):
        _ = op_func(x)

    # Benchmark
    start = time.time()
    for _ in range(num_iterations):
        _ = op_func(x)
    elapsed = (time.time() - start) / num_iterations * 1000  # Convert to ms

    return elapsed


def validate_kernel(kernel, torch_func, x, num_ctas=16, atol=1e-4, rtol=1e-4):
    """Validate Triton kernel output against PyTorch reference"""
    output_triton = torch.empty_like(x)
    n_elements = x.numel()
    BLOCK_SIZE = 32

    kernel[(num_ctas,)](x, output_triton, n_elements, BLOCK_SIZE=BLOCK_SIZE, num_ctas=num_ctas)
    output_torch = torch_func(x)

    is_close = torch.allclose(output_triton, output_torch, atol=atol, rtol=rtol)
    max_diff = torch.max(torch.abs(output_triton - output_torch)).item()

    return is_close, max_diff


if __name__ == "__main__":
    test_warm_up = 5
    test_iterations = 100
    num_ctas = 16

    test_shape_list = [(1024, 1024), (512, 512), (256, 256), (128, 128), (32, 32)]
    test_dtype_list = [torch.float32]

    # (kernel, torch_func, atol, rtol)
    test_ops = {
        # Math functions
        "exp": (exp_kernel, torch.exp, 1e-4, 1e-4),
        "cos": (cos_kernel, torch.cos, 1e-4, 1e-4),
        "sin": (sin_kernel, torch.sin, 1e-4, 1e-4),
        "tanh": (tanh_kernel, torch.tanh, 1e-4, 1e-4),
        "erf": (erf_kernel, torch.erf, 1e-4, 1e-4),
        # Activation functions
        "silu": (silu_kernel, torch.nn.functional.silu, 1e-4, 1e-4),
        "relu": (relu_kernel, torch.nn.functional.relu, 1e-4, 1e-4),
        "gelu_none": (gelu_none_kernel, lambda x: torch.nn.functional.gelu(x, approximate="none"), 1e-4, 1e-4),
        "gelu_tanh": (gelu_tanh_kernel, lambda x: torch.nn.functional.gelu(x, approximate="tanh"), 1e-3, 1e-3),
    }

    print("=" * 100)
    print("Elementwise Operations - Triton vs PyTorch Benchmark")
    print("=" * 100)

    for op_name, (triton_kernel, torch_func, atol, rtol) in test_ops.items():
        print(f"\n{'#' * 100}")
        print(f"{'#':^100}")
        print(f"# {op_name.upper():^96} #")
        print(f"{'#':^100}")
        print(f"{'#' * 100}\n")

        for test_dtype in test_dtype_list:
            print(f"  dtype: {test_dtype}")
            print(f"  {'-' * 96}")
            print(f"  {'Shape':20} | {'Triton (ms)':15} | {'PyTorch (ms)':15} | {'Speedup':10} | {'Status':15}")
            print(f"  {'-' * 96}")

            for test_shape in test_shape_list:
                x = torch.randn(test_shape, dtype=test_dtype, device="cpu")

                # Correctness validation
                is_correct, max_diff = validate_kernel(triton_kernel, torch_func, x, num_ctas, atol, rtol)
                status = "✅ PASS" if is_correct else f"❌ FAIL (diff={max_diff:.2e})"

                # Performance benchmark
                triton_time = benchmark_triton_kernel(triton_kernel, x, test_iterations, num_ctas, test_warm_up)
                torch_time = benchmark_torch_op(torch_func, x, test_iterations, test_warm_up)
                speedup = torch_time / triton_time if triton_time > 0 else 0

                print(
                    f"  {str(test_shape):20} | {triton_time:15.4f} | {torch_time:15.4f} | {speedup:10.2f}x | {status}"
                )

            print()

    print("=" * 100)
