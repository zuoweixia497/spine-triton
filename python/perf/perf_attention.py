import time
import torch
import triton
import triton.language as tl
from triton.backends.spine_triton.driver import CPUDriver
import triton.language.extra.smt as smt


triton.runtime.driver.set_active(CPUDriver())


@triton.jit
def _attn_fwd_inner(
    acc,
    l_i_2d,
    m_i_2d,
    Q_block_ptr,
    K_block_ptr,
    V_block_ptr,
    start_m,
    qk_scale,
    BLOCK_M: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    offs_m: tl.constexpr,
    offs_n: tl.constexpr,
    N_CTX: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
    num_m_tiles: tl.constexpr,
    num_n_tiles: tl.constexpr,
):

    if STAGE == 1:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = 0, start_m * BLOCK_M
    elif STAGE == 2:
        tl.static_assert(BLOCK_M >= BLOCK_N)
        lo, hi = start_m * BLOCK_M, (start_m + 1) * BLOCK_M
        lo = tl.multiple_of(lo, BLOCK_M)
    else:
        lo, hi = 0, N_CTX

    K_block_ptr = tl.advance(K_block_ptr, (lo, 0))
    V_block_ptr = tl.advance(V_block_ptr, (lo, 0))

    q_descriptor_load = smt.descriptor_load(Q_block_ptr, (0, 0))
    q = smt.view(q_descriptor_load, (0, 0), (BLOCK_M, HEAD_DIM), (MICRO_M, MICRO_K))

    offs_m_4d = tl.reshape(offs_m, (num_m_tiles, 1, MICRO_M, 1))
    offs_n_4d = tl.reshape(offs_n, (1, num_n_tiles, 1, MICRO_N))

    for start_n in tl.range(lo, hi, BLOCK_N):
        start_n = tl.multiple_of(start_n, BLOCK_N)
        k_descriptor_load = smt.descriptor_load(K_block_ptr, (0, 0))
        k = smt.view(k_descriptor_load, (0, 0), (BLOCK_N, HEAD_DIM), (MICRO_N, MICRO_K))
        trans_k = tl.permute(k, (1, 0, 3, 2))

        # Dot product usually accumulates in float32 for tensor cores
        qk = smt.dot(q, trans_k)

        qk = qk * qk_scale

        mask_n = (start_n + offs_n) < N_CTX
        mask_n_4d = tl.reshape(mask_n, (1, num_n_tiles, 1, MICRO_N))
        qk = tl.where(mask_n_4d, qk, -1.0e6)

        if STAGE == 2:
            mask_causal = offs_m_4d >= (start_n + offs_n_4d)
            mask_n = (start_n + offs_n_4d) < N_CTX
            mask = mask_causal & mask_n
            qk = tl.where(mask, qk, -1.0e6)

        # qk_max calc
        qk_max_3 = tl.max(qk, axis=3)
        m_ij_2d = tl.max(qk_max_3, axis=1)

        # Update max stats
        m_ij_2d = tl.maximum(m_i_2d, m_ij_2d)

        m_ij_bc = tl.reshape(m_ij_2d, (num_m_tiles, 1, MICRO_M, 1))
        qk = qk - m_ij_bc

        p = tl.math.exp(qk)

        p_sum_3 = tl.sum(p, axis=3)
        l_ij_2d = tl.sum(p_sum_3, axis=1)

        # Scale accumulator with new max
        alpha_2d = tl.math.exp(m_i_2d - m_ij_2d)
        l_i_2d = l_i_2d * alpha_2d + l_ij_2d

        alpha_bc = tl.reshape(alpha_2d, (num_m_tiles, 1, MICRO_M, 1))
        acc = acc * alpha_bc

        v_descriptor_load = smt.descriptor_load(V_block_ptr, (0, 0))
        v = smt.view(v_descriptor_load, (0, 0), (BLOCK_N, HEAD_DIM), (MICRO_K, MICRO_N))

        # Cast P to V's dtype for dot product if necessary, or keep float32 if precision needed
        p_cast = p.to(v.dtype)
        p_cast = smt.view(p_cast, (0, 0), (BLOCK_M, BLOCK_N), (MICRO_M, MICRO_K))

        acc += smt.dot(p_cast, v)
        m_i_2d = m_ij_2d

        V_block_ptr = tl.advance(V_block_ptr, (BLOCK_N, 0))
        K_block_ptr = tl.advance(K_block_ptr, (BLOCK_N, 0))

    return acc, l_i_2d, m_i_2d


@triton.jit
def _attn_fwd(
    Q, K, V, M, Out, acc_buffer, # Added acc_buffer arg if needed, or create inside
    sm_scale,
    stride_qz: tl.constexpr, stride_qh: tl.constexpr, stride_qm: tl.constexpr, stride_qk: tl.constexpr,
    stride_kz: tl.constexpr, stride_kh: tl.constexpr, stride_kn: tl.constexpr, stride_kk: tl.constexpr,
    stride_vz: tl.constexpr, stride_vh: tl.constexpr, stride_vn: tl.constexpr, stride_vk: tl.constexpr,
    stride_oz: tl.constexpr, stride_oh: tl.constexpr, stride_om: tl.constexpr, stride_on: tl.constexpr,
    Z: tl.constexpr,
    H: tl.constexpr,
    N_CTX: tl.constexpr,
    HEAD_DIM: tl.constexpr,
    BLOCK_M: tl.constexpr,
    BLOCK_N: tl.constexpr,
    STAGE: tl.constexpr,
    MICRO_M: tl.constexpr,
    MICRO_K: tl.constexpr,
    MICRO_N: tl.constexpr,
    num_m_tiles: tl.constexpr,
    num_n_tiles: tl.constexpr,
    num_k_tiles: tl.constexpr,
    num_ctas: tl.constexpr,
):

    NUM_BLOCKS_M = tl.cdiv(N_CTX, BLOCK_M)
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    pid = tl.program_id(0)
    sub_num = tl.cdiv(max(NUM_BLOCKS - pid, 0), num_ctas)

    for block_idx in tl.range(0, sub_num):
        task_hz_idx = (pid + num_ctas * block_idx) // NUM_BLOCKS_M
        task_m_idx = (pid + num_ctas * block_idx) % NUM_BLOCKS_M
        off_z = task_hz_idx // H
        off_h = task_hz_idx % H
        qvk_offset = off_z.to(tl.int64) * stride_qz + off_h.to(tl.int64) * stride_qh

        Q_block_ptr = tl.make_block_ptr(
            base=Q + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_qm, stride_qk),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )
        V_block_ptr = tl.make_block_ptr(
            base=V + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_vn, stride_vk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        K_block_ptr = tl.make_block_ptr(
            base=K + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_kn, stride_kk),
            offsets=(0, 0),
            block_shape=(BLOCK_N, HEAD_DIM),
            order=(1, 0),
        )
        O_block_ptr = tl.make_block_ptr(
            base=Out + qvk_offset,
            shape=(N_CTX, HEAD_DIM),
            strides=(stride_om, stride_on),
            offsets=(task_m_idx * BLOCK_M, 0),
            block_shape=(BLOCK_M, HEAD_DIM),
            order=(1, 0),
        )

        offs_m = task_m_idx * BLOCK_M + tl.arange(0, BLOCK_M)
        offs_n = tl.arange(0, BLOCK_N)
        math_dtype = Q.dtype.element_ty

        m_i_2d = tl.zeros([num_m_tiles, MICRO_M], dtype=math_dtype) - float("inf")
        l_i_2d = tl.zeros([num_m_tiles, MICRO_M], dtype=math_dtype) + 1.0

        acc_4d = tl.zeros([num_m_tiles, num_k_tiles, MICRO_M, MICRO_N], dtype=math_dtype)

        if STAGE & 1:
            acc_4d, l_i_2d, m_i_2d = _attn_fwd_inner(
                acc_4d,
                l_i_2d,
                m_i_2d,
                Q_block_ptr,
                K_block_ptr,
                V_block_ptr,
                task_m_idx,
                sm_scale,
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,
                4 - STAGE,
                offs_m,
                offs_n,
                N_CTX,
                MICRO_M,
                MICRO_K,
                MICRO_N,
                num_m_tiles,
                num_n_tiles,
            )

        if STAGE & 2:
            acc_4d, l_i_2d, m_i_2d = _attn_fwd_inner(
                acc_4d,
                l_i_2d,
                m_i_2d,
                Q_block_ptr,
                K_block_ptr,
                V_block_ptr,
                task_m_idx,
                sm_scale,
                BLOCK_M,
                HEAD_DIM,
                BLOCK_N,
                2,
                offs_m,
                offs_n,
                N_CTX,
                MICRO_M,
                MICRO_K,
                MICRO_N,
                num_m_tiles,
                num_n_tiles,
            )

        # Reshape and normalize
        acc_2d = smt.view(acc_4d, (0, 0), (BLOCK_M, HEAD_DIM), (1, 1))
        m_i = tl.reshape(m_i_2d, (BLOCK_M,))
        l_i = tl.reshape(l_i_2d, (BLOCK_M,))

        # Normalize with LogSumExp logic
        m_i = m_i + tl.math.log(l_i)
        accumulator = acc_2d / l_i[:, None]

        mask_m = offs_m < N_CTX
        m_ptrs = M + task_hz_idx * N_CTX + offs_m
        tl.store(m_ptrs, m_i.to(M.type.element_ty), mask=mask_m)

        # Cast back to output type (e.g. float16) before storing
        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty), boundary_check=(0, 1))


def flash_attention_forward(q, k, v, sm_scale, is_causal=False, num_ctas=16):
    """
    Flash Attention forward pass with dynamic MICRO block size support.
    """
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    o = torch.empty_like(q)
    # Placeholder for acc, though kernel initializes its own
    acc = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K),
        dtype=torch.float32, # Debugging usually safer with float32
        device=q.device,
    )
    M = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32
    )

    if is_causal:
        STAGE = 3
    else:
        STAGE = 1

    if q.dtype == torch.float32:
        MICRO_M = 8
        MICRO_N = 32
        MICRO_K = 32
    else :  # float16
        MICRO_M = 16
        MICRO_N = 32
        MICRO_K = 8

    BLOCK_M = 32
    BLOCK_N = 32

    # Calculate tiles
    num_m_tiles = BLOCK_M // MICRO_M
    num_n_tiles = BLOCK_N // MICRO_N
    num_k_tiles = HEAD_DIM_K // MICRO_N

    _attn_fwd[(num_ctas,)](
        q,
        k,
        v,
        M,
        o,
        acc,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0],
        q.shape[1],
        N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=STAGE,
        MICRO_M=MICRO_M,
        MICRO_K=MICRO_K,
        MICRO_N=MICRO_N,
        num_m_tiles=num_m_tiles,
        num_n_tiles=num_n_tiles,
        num_k_tiles=num_k_tiles,
        num_ctas=num_ctas,
    )

    return o


def flash_attention_forward_with_config(
    q,
    k,
    v,
    sm_scale,
    is_causal=False,
    block_m=64,
    block_n=64,
    num_ctas=16,
):
    """Flash Attention forward pass with explicit Triton config parameters."""
    HEAD_DIM_Q, HEAD_DIM_K = q.shape[-1], k.shape[-1]
    HEAD_DIM_V = v.shape[-1]
    assert HEAD_DIM_Q == HEAD_DIM_K and HEAD_DIM_K == HEAD_DIM_V
    assert HEAD_DIM_K in {16, 32, 64, 128, 256}

    o = torch.empty_like(q)
    acc = torch.empty(
        (q.shape[0], q.shape[1], q.shape[2], HEAD_DIM_K),
        dtype=torch.float32,
        device=q.device,
    )
    M = torch.empty((q.shape[0], q.shape[1], q.shape[2]), device=q.device, dtype=torch.float32)

    STAGE = 3 if is_causal else 1

    if q.dtype == torch.float32:
        MICRO_M = 8
        MICRO_N = 32
        MICRO_K = 32
    else:
        MICRO_M = 16
        MICRO_N = 32
        MICRO_K = 8

    BLOCK_M = block_m
    BLOCK_N = block_n

    # Keep the descriptor tiling valid for current dtype/micro config.
    assert BLOCK_M % MICRO_M == 0, "BLOCK_M must be divisible by MICRO_M"
    assert BLOCK_N % MICRO_N == 0, "BLOCK_N must be divisible by MICRO_N"

    num_m_tiles = BLOCK_M // MICRO_M
    num_n_tiles = BLOCK_N // MICRO_N
    num_k_tiles = HEAD_DIM_K // MICRO_N

    _attn_fwd[(num_ctas,)](
        q,
        k,
        v,
        M,
        o,
        acc,
        sm_scale,
        q.stride(0), q.stride(1), q.stride(2), q.stride(3),
        k.stride(0), k.stride(1), k.stride(2), k.stride(3),
        v.stride(0), v.stride(1), v.stride(2), v.stride(3),
        o.stride(0), o.stride(1), o.stride(2), o.stride(3),
        q.shape[0],
        q.shape[1],
        N_CTX=q.shape[2],
        HEAD_DIM=HEAD_DIM_K,
        BLOCK_M=BLOCK_M,
        BLOCK_N=BLOCK_N,
        STAGE=STAGE,
        MICRO_M=MICRO_M,
        MICRO_K=MICRO_K,
        MICRO_N=MICRO_N,
        num_m_tiles=num_m_tiles,
        num_n_tiles=num_n_tiles,
        num_k_tiles=num_k_tiles,
        num_ctas=num_ctas,
    )

    return o


def _best_of_repeats(run_once, num_warmup=5, num_iterations=20, num_repeats=3):
    for _ in range(num_warmup):
        run_once()

    best_ms = float("inf")
    for _ in range(num_repeats):
        start = time.time()
        for _ in range(num_iterations):
            run_once()
        elapsed_ms = 1000 * (time.time() - start) / num_iterations
        best_ms = min(best_ms, elapsed_ms)

    return best_ms


def _tune_best_config(configs, run_with_config, num_warmup=3, num_iterations=20, num_repeats=2):
    best_config = None
    best_time_ms = float("inf")

    for config in configs:
        try:
            config_time_ms = _best_of_repeats(
                lambda: run_with_config(config.kwargs),
                num_warmup=num_warmup,
                num_iterations=num_iterations,
                num_repeats=num_repeats,
            )
        except Exception:
            continue

        if config_time_ms < best_time_ms:
            best_time_ms = config_time_ms
            best_config = config

    if best_config is None:
        raise RuntimeError("No valid Triton config found for current attention input.")

    return best_config, best_time_ms


ATTN_TUNING_CONFIGS = [
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "num_ctas": 8}, num_warps=1),
    triton.Config({"BLOCK_M": 64, "BLOCK_N": 64, "num_ctas": 16}, num_warps=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "num_ctas": 8}, num_warps=1),
    triton.Config({"BLOCK_M": 128, "BLOCK_N": 64, "num_ctas": 16}, num_warps=1),
]


def pytorch_attention(q, k, v, sm_scale, is_causal=False):
    """
    PyTorch native attention implementation for reference.
    """
    # q, k, v are potentially float16, cast to float32 for reference accuracy if needed
    # But usually we compare in the same dtype
    attn = torch.matmul(q, k.transpose(-2, -1)) * sm_scale

    if is_causal:
        seq_len = q.shape[2]
        mask = torch.triu(torch.ones(seq_len, seq_len, device=q.device), diagonal=1).bool()
        attn = attn.masked_fill(mask, float('-inf'))

    attn = torch.softmax(attn, dim=-1)
    output = torch.matmul(attn, v)
    return output


if __name__ == "__main__":
    test_warm_up = 5
    test_iterations = 20
    test_repeats = 2
    num_ctas = 16

    # Test configurations
    test_shape_list = [
        (1, 8, 128, 64),
        (1, 8, 256, 64),
        (1, 8, 512, 64),
    ]

    test_dtype_list = [torch.float16, torch.float32]
    test_causal_list = [False, True]

    # ====================================================================
    # Phase 1: Correctness Validation
    # ====================================================================
    print("=" * 80)
    print("Phase 1: Correctness Validation")
    print("=" * 80)

    for is_causal in test_causal_list:
        causal_str = "Causal" if is_causal else "Non-Causal"
        for test_dtype in test_dtype_list:
            for test_shape in test_shape_list:
                try:
                    batch, heads, seq_len, head_dim = test_shape
                    q = torch.randn(test_shape, dtype=test_dtype, device="cpu")
                    k = torch.randn(test_shape, dtype=test_dtype, device="cpu")
                    v = torch.randn(test_shape, dtype=test_dtype, device="cpu")
                    sm_scale = 1.0 / (head_dim ** 0.5)

                    out = flash_attention_forward(q, k, v, sm_scale, is_causal, num_ctas)
                    ref = pytorch_attention(q, k, v, sm_scale, is_causal)

                    atol = 1e-2 if test_dtype == torch.float16 else 1e-4
                    is_correct = torch.allclose(out, ref, atol=atol, rtol=atol)
                    max_diff = (out - ref).abs().max().item()
                    status = "✅ PASS" if is_correct else f"❌ FAIL (max_diff={max_diff:.2e})"
                    print(f"  {causal_str:10} | {str(test_dtype):15} | {str(test_shape):25} | {status}")
                except Exception as e:
                    print(f"  {causal_str:10} | {str(test_dtype):15} | {str(test_shape):25} | ❌ ERROR: {e}")

    # ====================================================================
    # Phase 2: Triton Performance
    # ====================================================================
    print()
    print("=" * 80)
    print("Phase 2: Triton Kernel Performance")
    print("=" * 80)

    for is_causal in test_causal_list:
        causal_str = "Causal" if is_causal else "Non-Causal"

        for test_dtype in test_dtype_list:
            print(f"\n  {causal_str} | dtype: {test_dtype}")
            header = f"  {'Shape':25}"
            header += f" | {'Time (ms)':>12} | {'GFLOPS':>12} | {'Config':>28}"
            print(f"  {'-' * 92}")
            print(header)
            print(f"  {'-' * 92}")

            for test_shape in test_shape_list:
                try:
                    batch, heads, seq_len, head_dim = test_shape
                    q = torch.randn(test_shape, dtype=test_dtype, device="cpu")
                    k = torch.randn(test_shape, dtype=test_dtype, device="cpu")
                    v = torch.randn(test_shape, dtype=test_dtype, device="cpu")
                    sm_scale = 1.0 / (head_dim ** 0.5)

                    best_config, _ = _tune_best_config(
                        ATTN_TUNING_CONFIGS,
                        lambda meta: flash_attention_forward_with_config(
                            q,
                            k,
                            v,
                            sm_scale,
                            is_causal=is_causal,
                            block_m=meta["BLOCK_M"],
                            block_n=meta["BLOCK_N"],
                            num_ctas=meta["num_ctas"],
                        ),
                        num_warmup=max(1, test_warm_up // 2),
                        num_iterations=test_iterations,
                        num_repeats=test_repeats,
                    )

                    triton_time = _best_of_repeats(
                        lambda: flash_attention_forward_with_config(
                            q,
                            k,
                            v,
                            sm_scale,
                            is_causal=is_causal,
                            block_m=best_config.kwargs["BLOCK_M"],
                            block_n=best_config.kwargs["BLOCK_N"],
                            num_ctas=best_config.kwargs["num_ctas"],
                        ),
                        num_warmup=test_warm_up,
                        num_iterations=test_iterations,
                        num_repeats=test_repeats,
                    )

                    flops = 4 * batch * heads * seq_len * seq_len * head_dim
                    gflops = flops / 1e9 / (triton_time / 1000)

                    config_str = (
                        f"BM={best_config.kwargs['BLOCK_M']},"
                        f"BN={best_config.kwargs['BLOCK_N']},"
                        f"CTA={best_config.kwargs['num_ctas']}"
                    )
                    print(f"  {str(test_shape):25} | {triton_time:12.2f} | {gflops:12.2f} | {config_str:28}")
                except Exception as e:
                    print(f"  {str(test_shape):25} | ERROR: {e}")

            print()

    print("=" * 80)