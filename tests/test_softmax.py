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
            tl.store(output_block_ptr, numerator, boundary_check=(0,))

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
            softmax_output = exp_out / denominator
            tl.store(output_block_ptr, softmax_output, boundary_check=(0,))


def softmax(x):

    n_rows, n_cols = x.shape
    y = torch.empty_like(x)

    COL_SIZE = 64
    ROW_SIZE = 64

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

    M, N = 1024, 1000
    dtype = torch.float32

    x = torch.randn(M, N, device='cpu', dtype=dtype)

    y_torch = torch.softmax(x, dim=1)

    y_triton = softmax(x)

    if torch.allclose(y_triton, y_torch, atol=1e-5, rtol=1e-5):
        print("✅ Success! Triton result matches PyTorch result.")
    else:
        print("❌ Failure! Results do not match.")

    start = time.time()
    for _ in range(1000):
        C = softmax(x)
    end = time.time()
    print(f"Time: {end - start}")

    start = time.time()
    for _ in range(1000):
        C = torch.softmax(x, dim=1)
    end = time.time()
    print(f"Time: {end - start}")