import time
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    torch.set_num_threads(2)
    shape = (512, 512)
    wb_none = False
    dtype = torch.float32
    res_inp = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    if wb_none:
        res_weight = None
        res_bias = None
    else:
        res_weight = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
        res_bias = torch.randn(shape[1:], dtype=dtype, device=flag_gems.device)
    eps = 1e-5

    with flag_gems.use_gems():
        for i in range(0, 50):
            ref_out = torch.layer_norm(
                res_inp,
                shape[1:],
                weight=res_weight,
                bias=res_bias,
                eps=eps,
            )
        start1 = time.time()
        for i in range(0, 1000):
            ref_out = torch.layer_norm(
                res_inp,
                shape[1:],
                weight=res_weight,
                bias=res_bias,
                eps=eps,
            )
        end1 = time.time()
    print("triton time ms:", (end1 - start1))

    start2 = time.time()
    for i in range(0, 10):
        C_ref = torch.layer_norm(
            res_inp,
            shape[1:],
            weight=res_weight,
            bias=res_bias,
            eps=eps,
        )
    end2 = time.time()
    print("torch time ms:", ((end2 - start2) / 10) * 1000)
    torch.testing.assert_close(ref_out, C_ref, atol=1e-2, rtol=0)

    print("PASS")
