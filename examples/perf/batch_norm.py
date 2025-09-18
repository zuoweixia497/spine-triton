import time
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    torch.set_num_threads(2)
    shape = (512, 512)
    affine = False
    C = shape[1]
    dtype = torch.float32
    inp = torch.randn(size=shape, dtype=dtype, device=flag_gems.device)
    weight = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )
    bias = (
        torch.randn(size=(C,), dtype=dtype, device=flag_gems.device) if affine else None
    )

    running_mean = torch.zeros(size=(C,), dtype=dtype, device=flag_gems.device)
    running_var = torch.ones(size=(C,), dtype=dtype, device=flag_gems.device)

    eps = 1e-5

    with flag_gems.use_gems():
        for i in range(0, 50):
            C = torch.nn.functional.batch_norm(
                inp,
                running_mean,
                running_var,
                weight=weight,
                bias=bias,
                eps=eps,
            )
        start1 = time.time()
        for i in range(0, 1000):
            C = torch.nn.functional.batch_norm(
                inp,
                running_mean,
                running_var,
                weight=weight,
                bias=bias,
                eps=eps,
            )
        end1 = time.time()
    print("triton time s:", (end1 - start1) / 1000)

    start2 = time.time()
    for i in range(0, 10):
        C_ref = torch.nn.functional.batch_norm(
            inp,
            running_mean,
            running_var,
            weight=weight,
            bias=bias,
            eps=eps,
        )
    end2 = time.time()
    print("torch time s:", (end2 - start2) / 10)
    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=0)

    print("PASS")
