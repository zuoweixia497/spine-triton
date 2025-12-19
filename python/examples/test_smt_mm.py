import torch
import triton
import triton.language as tl
import triton.language.extra.smt as smt
from triton.backends.spine_triton.driver import CPUDriver

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
    EVEN_K: tl.constexpr,
    SUB_BLK_M: tl.constexpr,
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

    accumulator = tl.zeros((BLOCK_SIZE_M, BLOCK_SIZE_N), dtype=a_ptr.type.element_ty)

    if EVEN_K:
        b = smt.descriptor_load(b_block_ptr, (0, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
        sub_num = tl.cdiv(min(BLOCK_SIZE_M, M - BLOCK_SIZE_M * pid_m), SUB_BLK_M)
        for s in smt.parallel(0, sub_num):
            a = smt.descriptor_load(a_block_ptr,  (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
            accumulator_view = smt.view(accumulator, (s * SUB_BLK_M, 0), (SUB_BLK_M, BLOCK_SIZE_N), (MICRO_M, MICRO_N))
            accumulator_view = smt.dot(a, b, accumulator_view)
    else:
        for k in range(0, tl.cdiv(K, BLOCK_SIZE_K)):
            b = smt.descriptor_load(b_block_ptr, (k, 0), (BLOCK_SIZE_K, BLOCK_SIZE_N), (MICRO_K, MICRO_N))
            sub_num = tl.cdiv(min(BLOCK_SIZE_M, M - BLOCK_SIZE_M * pid_m), SUB_BLK_M)
            for s in smt.parallel(0, sub_num):
                a = smt.descriptor_load(a_block_ptr,  (s * SUB_BLK_M, k), (SUB_BLK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))
                accumulator_view = smt.view(accumulator, (s * SUB_BLK_M, k), (SUB_BLK_M, BLOCK_SIZE_N), (MICRO_M, MICRO_N))
                accumulator_view = smt.dot(a, b, accumulator_view)

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

def triton_mm(a, b):
    # handle non-contiguous inputs if necessary
    if a.stride(0) > 1 and a.stride(1) > 1:
        a = a.contiguous()
    if b.stride(0) > 1 and b.stride(1) > 1:
        b = b.contiguous()
    # checks constraints
    assert a.shape[1] == b.shape[0], "incompatible dimensions"
    M, K = a.shape
    _, N = b.shape
    # allocates output
    c = torch.empty((M, N), device=a.device, dtype=a.dtype)
    # launch kernel
    grid = lambda META: (
        triton.cdiv(M, META["BLOCK_SIZE_M"]),
        triton.cdiv(N, META["BLOCK_SIZE_N"]),
    )
    BLOCK_SIZE_K = triton.next_power_of_2(K)

    mm_kernel[grid](
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
        BLOCK_SIZE_M=128,
        BLOCK_SIZE_N=128,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        SUB_BLK_M=32,
        MICRO_M=8,
        MICRO_N=16,
        MICRO_K=8,
        EVEN_K=True
    )
    return c

if __name__ == "__main__":

    M, N, K = 256, 256, 512
    A = torch.randn((M, K), dtype=torch.float32, device="cpu", requires_grad=False)
    B = torch.randn((K, N), dtype=torch.float32, device="cpu", requires_grad=False)

    C = triton_mm(A, B)

    C_ref = torch.mm(A, B)

    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=0)

    print("PASS")
