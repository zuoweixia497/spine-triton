import time
import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def softmax_kernel(
    output_ptr, input_ptr,
    input_row_stride, output_row_stride,
    n_rows, n_cols,
    ROW_SIZE: tl.constexpr,
    COL_SIZE: tl.constexpr
):
    row_start = tl.program_id(0) * ROW_SIZE

    for row_idx in range(row_start, row_start + ROW_SIZE):
        denominator = 0.0
        row_max = tl.full((COL_SIZE,), -float('inf'), dtype=tl.float32)

        for col_idx in range(0, n_cols, COL_SIZE):
            input_block_ptr = tl.make_block_ptr(
                base=input_ptr + row_idx * input_row_stride,
                shape=(n_cols),
                strides=(1,),
                offsets=(col_idx,),
                block_shape=(COL_SIZE,),
                order=(0,)
            )
            row = tl.load(input_block_ptr, boundary_check=(0,), padding_option="neg_inf")
            # row = tl.where(col_idx < n_cols, row, -float('inf'))
            row_max = tl.maximum(row, row_max)

        row_max_total = tl.max(row_max, axis=0)

        for col_idx in range(0, n_cols, COL_SIZE):
            input_block_ptr = tl.make_block_ptr(
                base=input_ptr + row_idx * input_row_stride,
                shape=(n_cols),
                strides=(1,),
                offsets=(col_idx,),
                block_shape=(COL_SIZE,),
                order=(0,)
            )
            output_block_ptr = tl.make_block_ptr(
                base=output_ptr + row_idx * output_row_stride,
                shape=(n_cols,),
                strides=(1,),
                offsets=(col_idx,),
                block_shape=(COL_SIZE,),
                order=(0,)
            )
            row = tl.load(input_block_ptr, boundary_check=(0,), padding_option="neg_inf")
            # row = tl.where(col_idx < n_cols, row, -float('inf'))
            row_minus_max = row - row_max_total
            numerator = tl.exp(row_minus_max)
            denominator += tl.sum(numerator, axis=0)
            tl.store(output_block_ptr, numerator.to(input_ptr.type.element_ty), boundary_check=(0,))

        for col_idx in range(0, n_cols, COL_SIZE):
            output_block_ptr = tl.make_block_ptr(
                base=output_ptr + row_idx * output_row_stride,
                shape=(n_cols,),
                strides=(1,),
                offsets=(col_idx,),
                block_shape=(COL_SIZE,),
                order=(0,)
            )
            exp_out = tl.load(output_block_ptr, boundary_check=(0,))
            softmax_output = exp_out.to(tl.float32) / denominator
            tl.store(output_block_ptr, softmax_output.to(input_ptr.type.element_ty), boundary_check=(0,))


def softmax(x):

    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    COL_SIZE = 64
    ROW_SIZE = min(64, n_rows)

    def grid(META): return (
        triton.cdiv(n_rows, META["ROW_SIZE"]),
    )

    softmax_kernel[grid](
        y, x,
        x.stride(0), y.stride(0),
        n_rows, n_cols,
        ROW_SIZE=ROW_SIZE,
        COL_SIZE=COL_SIZE,
    )

    return y

if __name__ == "__main__":
    torch.manual_seed(0)

    test_shape_list = [(1024, 1024), (512, 512), (256, 256), (128, 128), (32, 32)]
    test_dtype_list = [torch.float32, torch.float16]
    test_warm_up = 5
    test_iterations = 100

    # ====================================================================
    # Phase 1: Correctness Validation
    # ====================================================================
    print("=" * 80)
    print("Phase 1: Correctness Validation")
    print("=" * 80)

    for test_dtype in test_dtype_list:
        for test_shape in test_shape_list:
            M, N = test_shape
            x = torch.randn(M, N, device='cpu', dtype=test_dtype)
            y_triton = softmax(x)
            y_torch = torch.softmax(x, dim=1)
            is_correct = torch.allclose(y_triton, y_torch, atol=1e-2, rtol=1e-2)
            max_diff = torch.max(torch.abs(y_triton - y_torch)).item()
            status = "✅ PASS" if is_correct else f"❌ FAIL (max_diff={max_diff:.2e})"
            print(f"  softmax      | {str(test_dtype):15} | {str(test_shape):15} | {status}")

    # ====================================================================
    # Phase 2: Triton Performance
    # ====================================================================
    print()
    print("=" * 80)
    print("Phase 2: Triton Kernel Performance")
    print("=" * 80)

    for test_dtype in test_dtype_list:
        print(f"\n  dtype: {test_dtype}")
        header = f"  {'Op':12}"
        for shape in test_shape_list:
            header += f" | {str(shape):>22}"
        print(f"  {'-' * (16 + 25 * len(test_shape_list))}")
        print(header)
        print(f"  {'-' * (16 + 25 * len(test_shape_list))}")

        row = f"  {'softmax':12}"
        for test_shape in test_shape_list:
            M, N = test_shape
            x = torch.randn(M, N, device='cpu', dtype=test_dtype)

            # Warmup
            for _ in range(test_warm_up):
                _ = softmax(x)

            # Benchmark
            start = time.time()
            for _ in range(test_iterations):
                _ = softmax(x)
            triton_time = (time.time() - start) / test_iterations * 1000  # ms

            n_elements = x.numel()
            throughput = n_elements / (triton_time / 1000)  # elements/sec
            if throughput >= 1e9:
                tp_str = f"{triton_time:.2f}ms {throughput/1e9:.2f}G/s"
            elif throughput >= 1e6:
                tp_str = f"{triton_time:.2f}ms {throughput/1e6:.2f}M/s"
            else:
                tp_str = f"{triton_time:.2f}ms {throughput:.0f}/s"
            row += f" | {tp_str:>22}"
        print(row)

        print()

    print("=" * 80)