# python3 -m pytest tests/test_smt.py -v -s
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
driver = CPUDriver()
driver.set_current_arch_id("0xA03C")
triton.runtime.driver.set_active(driver)
import triton.language as tl
import pytest
import triton.language.extra.smt as smt


@pytest.mark.parametrize(
    "SUB_BLK_M, BLOCK_SIZE_M, BLOCK_SIZE_K, MICRO_M, MICRO_K",
    [
        (64, 128, 512, 16, 16),
    ]
)
def test_descriptor_load(SUB_BLK_M, BLOCK_SIZE_M, BLOCK_SIZE_K, MICRO_M, MICRO_K):

    @triton.jit
    def descriptor_load(
        a_ptr,
        output_ptr,
        M,
        K,
        stride_am,
        stride_ak,
        stride_om0,
        stride_om1,
        stride_om2,
        stride_om3,
        num_blocks_m: tl.constexpr,
        num_blocks_k: tl.constexpr,
        SUB_BLK_M: tl.constexpr,
        BLOCK_SIZE_M: tl.constexpr,
        BLOCK_SIZE_K: tl.constexpr,
        MICRO_M: tl.constexpr,
        MICRO_K: tl.constexpr,
    ):
        pid = tl.program_id(0)

        a_block_ptr = tl.make_block_ptr(
            base=a_ptr,
            shape=[M, K],
            strides=[stride_am, stride_ak],
            offsets=[pid * BLOCK_SIZE_M, 0],
            block_shape=[BLOCK_SIZE_M, BLOCK_SIZE_K],
            order=[1, 0],
        )

        a = smt.descriptor_load(a_block_ptr, (0, 0), (SUB_BLK_M, BLOCK_SIZE_K), (MICRO_M, MICRO_K))

        output_block_ptr = tl.make_block_ptr(
            base=output_ptr,
            shape=[num_blocks_m, num_blocks_k, MICRO_M, MICRO_K],
            strides=[stride_om0, stride_om1, stride_om2, stride_om3],
            offsets=[0, 0, 0, 0],
            block_shape=[num_blocks_m, num_blocks_k, MICRO_M, MICRO_K],
            order=[3, 2, 1, 0],
        )

        tl.store(output_block_ptr, a)

    torch.manual_seed(42)
    M, K = 256, BLOCK_SIZE_K
    device = "cpu"
    a = torch.randn((M, K), dtype=torch.float32, device=device)

    num_blocks_m = (SUB_BLK_M + MICRO_M - 1) // MICRO_M
    num_blocks_k = (BLOCK_SIZE_K + MICRO_K - 1) // MICRO_K
    output_shape = (num_blocks_m, num_blocks_k, MICRO_M, MICRO_K)
    output = torch.zeros(output_shape, dtype=torch.float32, device=device)

    grid = (triton.cdiv(M, BLOCK_SIZE_M),)

    descriptor_load[grid](
        a, output,
        M, K,
        a.stride(0), a.stride(1),
        output.stride(0), output.stride(1), output.stride(2), output.stride(3),
        num_blocks_m=num_blocks_m,
        num_blocks_k=num_blocks_k,
        SUB_BLK_M=SUB_BLK_M,
        BLOCK_SIZE_M=BLOCK_SIZE_M,
        BLOCK_SIZE_K=BLOCK_SIZE_K,
        MICRO_M=MICRO_M,
        MICRO_K=MICRO_K
    )

    print("output:", output)
    print("output.shape:", output.shape)
