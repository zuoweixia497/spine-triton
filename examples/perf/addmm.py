import time
import numpy as np
from functools import wraps
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    M, N, K = 256, 256, 512
    A = torch.randn(
        (M, K), dtype=torch.float32, device=flag_gems.device, requires_grad=False
    )
    B = torch.randn(
        (K, N), dtype=torch.float32, device=flag_gems.device, requires_grad=False
    )
    bias = torch.randn(
        (N,), dtype=torch.float32, device=flag_gems.device, requires_grad=False
    )
    # bias = torch.zeros((N,), dtype=torch.float32, device=flag_gems.device, requires_grad=False)

    with flag_gems.use_gems():
        with torch.no_grad():
            for i in range(0, 50):
                C = torch.addmm(bias, A, B, beta=2, alpha=3)
            start1 = time.time()
            for i in range(0, 1000):
                C = torch.addmm(bias, A, B, beta=2, alpha=3)
            end1 = time.time()
    print("triton time s:", (end1 - start1) / 1000)
    print("triton gtops:", M * N * K * 1000 / (end1 - start1) * 2 / 1024 / 1024 / 1024)

    start2 = time.time()
    for i in range(0, 10):
        C_ref = torch.addmm(bias, A, B, beta=2, alpha=3)
    end2 = time.time()

    print("triton time s:", (end2 - start2) / 10)
    print("triton gtops:", M * N * K * 10 / (end2 - start2) * 2 / 1024 / 1024 / 1024)

    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=0)

    print("PASS")
