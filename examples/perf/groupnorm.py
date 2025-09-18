import time
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    torch.set_num_threads(2)
    N, C, H, W, num_groups = 1, 32, 32, 32, 8
    wb_none = False
    dtype = torch.float32
    res_inp = torch.randn(size=(N, C, H, W), dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
        res_weight = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(size=(C,), dtype=dtype, device=flag_gems.device)
    eps = 1e-5
    with flag_gems.use_gems():
        for i in range(0, 50):
            C = torch.nn.functional.group_norm(
                res_inp, num_groups, weight=res_weight, bias=res_bias, eps=eps
            )
        start1 = time.time()
        for i in range(0, 1000):
            C = torch.nn.functional.group_norm(
                res_inp, num_groups, weight=res_weight, bias=res_bias, eps=eps
            )
        end1 = time.time()
    print("triton time ms:", (end1 - start1))

    start2 = time.time()
    for i in range(0, 10):
        C_ref = torch.nn.functional.group_norm(
            res_inp, num_groups, weight=res_weight, bias=res_bias, eps=eps
        )
    end2 = time.time()
    print("torch time ms:", ((end2 - start2) / 10) * 1000)
    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=0)

    print("PASS")
