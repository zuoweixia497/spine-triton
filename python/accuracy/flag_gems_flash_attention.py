import numpy as np
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    Z, H, N_CTX, HEAD_DIM, dtype = 1, 1, 128, 128, torch.float32
    DEVICE = "cpu"
    # if N_CTX % BM != 0 or N_CTX % BN != 0 or HEAD_DIM % 16 != 0:
    #     pytest.skip("Skipping non-divisible case")

    torch.manual_seed(20)
    q = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    k = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    v = (
        torch.empty((Z, H, N_CTX, HEAD_DIM), dtype=dtype, device=DEVICE)
        .normal_(mean=0.0, std=0.5)
        .requires_grad_()
    )
    sm_scale = 0.5


    torch_result = torch.nn.functional.scaled_dot_product_attention(
        q,
        k,
        v,
        scale=sm_scale,
        is_causal=False,
    )


    import flag_gems.runtime.backend._spacemit.ops as sp
    flaggem_result = sp.flash_attention(q, k, v, sm_scale)

    print("flaggem_result", flaggem_result)
    print("torch_result", torch_result)
    torch.testing.assert_close(flaggem_result, torch_result, atol=1e-2, rtol=0)
    print("PASS")