import time
import numpy as np
from functools import wraps
import torch
import triton
import sys
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    torch.set_num_threads(2)
    K = 512
    inp = torch.randn(
        [K], dtype=torch.float32, device=flag_gems.device, requires_grad=False
    )
    for i in range(0, 10):
        with flag_gems.use_gems():
            with torch.no_grad():
                C = torch.softmax(inp, -1)
    start1 = time.time()
    for i in range(0, 100):
        with flag_gems.use_gems():
            with torch.no_grad():
                C = torch.softmax(inp, -1)
    end1 = time.time()
    print("triton time ms:", ((end1 - start1) / 100) * 1000)

    start2 = time.time()
    for i in range(0, 10):
        C_ref = torch.softmax(inp, -1)
    end2 = time.time()
    print("torch time ms:", ((end2 - start2) / 10) * 1000)

    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=0)

    print("PASS")
