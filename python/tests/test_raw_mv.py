import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import pytest
from triton.language.extra.spine_raw import spine_raw, In, InOut, call as _sr_call
import triton.language.extra.spine_raw as sr_mod


@spine_raw(name="linalg")
def mv_macc_block(
    B:   In["memref<*xf16, #ptr.generic_space>"],
    A:   In["memref<*xf16, #ptr.generic_space>"],
    col: In["index"],
    M:   In["index"],
    nk:  In["index"],
    C:   InOut["memref<*xf32, #ptr.generic_space>"],
):
    NSUB = 4
    # 单一 TCM staging buffer:编译期 unroll 后 pack→load→macc 每个 sub-tile
    # 顺序执行,load_2d 已把数据读进独立 vector 寄存器(r),故 buf 可安全复用。
    # 4 个独立累加链保留在 accs[] 里。相比每 sub-tile 一个 buf(4 个并发 TCM
    # block × num_threads=4 = 16 会超订 8-block pool 导致 spineTcmAcquire 失败),
    # 单 buf 峰值只占 1 个 TCM block(×4 thread = 4 ≤ 8),恰好放得下。
    # 不能用 free_tcm/memref.dealloc:K3 lowering 拒绝 "memory free side-effect"
    # (mm 同样无显式 dealloc),TCM block 在 kernel 作用域退出时由 runtime 自动归还。
    buf = sr_mod.alloc_tcm_2d(32, 64, "f16")
    accs = [sr_mod.splat_2d(0.0, 1, 64, "f32") for _ in range(NSUB)]
    for kb in sr_mod.range(nk):
        koff = kb * 32
        lhs = sr_mod.view_2d(B, 1, 32, "f16", koff)
        for i in range(NSUB):                        # compile-time unroll → 4 份
            sr_mod.pack_2d_t_into(buf, A, col + i * 64, 32, 64, M, "f16", koff)
            r = sr_mod.load_2d(buf, 32, 64, "f16")
            accs[i] = sr_mod.batch_macc(lhs, r, accs[i])
    for i in range(NSUB):
        sr_mod.store_2d_at(C, col + i * 64, 1, 64, accs[i])


@triton.jit(do_not_specialize=["M", "NK"])
def _mv_macc_host(B, A, C, M, NK, NB: tl.constexpr):
    pid = tl.program_id(0)
    _sr_call(mv_macc_block, outputs=[], inputs=[B, A, pid * NB, M, NK, C])


def raw_mv(inp, vec):
    """Standalone spine_raw mv — no flag_gems dependency."""
    _BLOCK_M = 32
    _PACK_N = 64
    _NB = 4 * _PACK_N

    if inp.stride(0) > 1 and inp.stride(1) > 1:
        inp = inp.contiguous()
    assert inp.shape[1] == vec.shape[0], "incompatible dimensions"
    N, M = inp.shape
    assert inp.dtype == torch.float16, f"requires f16, got {inp.dtype}"
    assert M % _BLOCK_M == 0, f"M={M} not multiple of {_BLOCK_M}"
    assert N % _NB == 0, f"N={N} not multiple of {_NB}"
    assert inp.stride(1) == 1 and inp.stride(0) == M, "must be row-major contiguous"

    a = inp.contiguous()
    b = vec.contiguous()
    c = torch.empty(N, device=inp.device, dtype=torch.float32)
    grid = (N // _NB,)
    _mv_macc_host[grid](b, a, c, M, M // _BLOCK_M, NB=_NB)
    return c.to(torch.float16)


@pytest.mark.parametrize("N, M", [
    (256, 32),
    (256, 64),
    (512, 32),
    (1024, 64),
    (4096, 32),
    (4096, 64),
    (4096, 128),
    (4096, 256),
])
def test_raw_mv_correctness(N, M):
    A = torch.randn(N, M, dtype=torch.float16)
    B = torch.randn(M, dtype=torch.float16)
    out = raw_mv(A, B)
    ref = torch.mv(A.float(), B.float())
    ok = torch.allclose(out.float(), ref, rtol=1e-2, atol=1e-2)
    max_diff = (out.float() - ref).abs().max().item()
    assert ok, f"N={N} M={M} max_diff={max_diff:.4e}"
