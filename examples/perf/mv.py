import time
import numpy as np
from functools import wraps
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    torch.set_num_threads(2)
    M, N = 256, 256
    dtype = torch.float32
    matrix = torch.randn((N, M), dtype=dtype, device=flag_gems.device)
    vector = torch.randn((M,), dtype=dtype, device=flag_gems.device)

    with flag_gems.use_gems():
        with torch.no_grad():
            for i in range(0, 10):
                res_out = torch.mv(matrix, vector)
            start1 = time.time()
            for i in range(0, 1000):
                res_out = torch.mv(matrix, vector)
            end1 = time.time()
    print("triton time s:", (end1 - start1)/1000)

    start2 = time.time()
    for i in range(0, 10):
        ref_out = torch.mv(matrix, vector)
    end2 = time.time()
    print("torch time s:", (end2 - start2)/10)

    torch.testing.assert_close(res_out, ref_out, atol=1e-2, rtol=0)

    print("PASS")