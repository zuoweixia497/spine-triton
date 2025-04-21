import time
import numpy as np
from functools import wraps
import torch
import triton
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    M, K = 256, 512
    inp = torch.randn([M, K], dtype=torch.float32, device=flag_gems.device, requires_grad=False)
    with flag_gems.use_gems():
        with torch.no_grad():
            C = torch.tanh(inp)

    C_ref = torch.tanh(inp)

    torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=0)

    print("PASS")