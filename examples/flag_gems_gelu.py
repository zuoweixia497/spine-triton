import time
import numpy as np
from functools import wraps
import torch
import triton
from triton.backends.triton_shared.driver import CPUDriver
triton.runtime.driver.set_active(CPUDriver())
import flag_gems

def generate_grid(start, end, step):
    num_points = int((end - start) / step) + 1
    grid = torch.linspace(start, end, num_points, dtype=torch.float32, device=flag_gems.device, requires_grad=False)
    return grid

if __name__ == "__main__":
    inp = generate_grid(-2, 20, 0.0001)
    for approximate in ("none", "tanh"):
        with flag_gems.use_gems():
            with torch.no_grad():
                C = torch.nn.functional.gelu(inp, approximate=approximate)

        C_ref = torch.nn.functional.gelu(inp, approximate=approximate)

        print(C)
        print(C_ref)
        torch.testing.assert_close(C, C_ref, atol=1e-2, rtol=1e-2)

        print("PASS")