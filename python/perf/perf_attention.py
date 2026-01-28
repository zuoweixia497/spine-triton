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

        if STAGE == 2:
            mask = offs_m_4d >= (start_n + offs_n_4d)
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
    num_cores: tl.constexpr,
):

    NUM_BLOCKS_M = N_CTX // BLOCK_M
    NUM_BLOCKS = NUM_BLOCKS_M * Z * H

    pid = tl.program_id(0)
    sub_num = tl.cdiv(max(NUM_BLOCKS - pid, 0), num_cores)

    for block_idx in smt.parallel(0, sub_num):
        task_hz_idx = (pid + num_cores * block_idx) // NUM_BLOCKS_M
        task_m_idx = (pid + num_cores * block_idx) % NUM_BLOCKS_M
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

        m_i_2d = tl.zeros([num_m_tiles, MICRO_M], dtype=tl.float32) - float("inf")
        l_i_2d = tl.zeros([num_m_tiles, MICRO_M], dtype=tl.float32) + 1.0

        acc_4d = tl.zeros([num_m_tiles, num_k_tiles, MICRO_M, MICRO_N], dtype=tl.float32)

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

        m_ptrs = M + task_hz_idx * N_CTX + offs_m
        tl.store(m_ptrs, m_i.to(M.type.element_ty))

        # Cast back to output type (e.g. float16) before storing
        tl.store(O_block_ptr, accumulator.to(Out.type.element_ty))


def flash_attention_forward(q, k, v, sm_scale, is_causal=False, num_cores=20):
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

    _attn_fwd[(num_cores,)](
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
        num_cores=num_cores,
    )

    return o


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
    num_cores = 20

    # Test configurations
    test_shape_list = [
        (1, 8, 128, 64),
        (1, 8, 256, 64),
        (1, 8, 512, 64),
    ]

    test_dtype_list = [torch.float16, torch.float32]
    test_causal_list = [False, True]

    print("Flash Attention Performance Test")
    print("=================================")

    for is_causal in test_causal_list:
        causal_str = "Causal" if is_causal else "Non-Causal"
        print(f"\n--- {causal_str} Attention ---")

        for test_dtype in test_dtype_list:
            print(f"\n[Testing Dtype: {test_dtype}]")
            for test_shape in test_shape_list:
                try:
                    batch, heads, seq_len, head_dim = test_shape

                    q = torch.randn(test_shape, dtype=test_dtype, device="cpu")
                    k = torch.randn(test_shape, dtype=test_dtype, device="cpu")
                    v = torch.randn(test_shape, dtype=test_dtype, device="cpu")
                    sm_scale = 1.0 / (head_dim ** 0.5)

                    # Warm up
                    for _ in range(test_warm_up):
                        out = flash_attention_forward(q, k, v, sm_scale, is_causal, num_cores)

                    # Benchmark triton kernel
                    start = time.time()
                    for _ in range(test_iterations):
                        out = flash_attention_forward(q, k, v, sm_scale, is_causal, num_cores)
                    end = time.time()
                    triton_time = 1000 * (end - start) / test_iterations

                    # Benchmark pytorch reference
                    for _ in range(test_warm_up):
                        ref = pytorch_attention(q, k, v, sm_scale, is_causal)

                    start = time.time()
                    for _ in range(test_iterations):
                        ref = pytorch_attention(q, k, v, sm_scale, is_causal)
                    end = time.time()
                    pytorch_time = 1000 * (end - start) / test_iterations

                    flops = 4 * batch * heads * seq_len * seq_len * head_dim
                    gflops_triton = test_iterations * flops / 1e9 / (triton_time * test_iterations / 1000)
                    gflops_pytorch = test_iterations * flops / 1e9 / (pytorch_time * test_iterations / 1000)
                    speedup = pytorch_time / triton_time

                    print(
                        f"shape {test_shape}: "
                        f"Triton {triton_time:.3f} ms ({gflops_triton:.2f} GFLOPS), "
                        f"PyTorch {pytorch_time:.3f} ms ({gflops_pytorch:.2f} GFLOPS), "
                        f"Speedup {speedup:.2f}x"
                    )

                    # Verify correctness
                    # Lower tolerance slightly for float16 accumulation differences
                    atol = 1e-2 if test_dtype == torch.float16 else 1e-4
                    if not torch.allclose(out, ref, atol=atol, rtol=atol):
                        max_diff = (out - ref).abs().max().item()
                        print(f"  WARNING: Results differ! Max diff: {max_diff:.6f}")

                except Exception as e:
                    import traceback
                    traceback.print_exc()
                    print(f"dtype {test_dtype} shape {test_shape}, Failed: {str(e)}")

    print("\n=================================")
    print("Performance test completed.")