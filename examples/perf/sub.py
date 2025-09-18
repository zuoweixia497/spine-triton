import time
import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

if __name__ == "__main__":
    torch.set_num_threads(2)
    shape = (512, 512)
    dtype = torch.float32
    approximate = "none"
    inp1 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    inp2 = torch.randn(shape, dtype=dtype, device=flag_gems.device)
    with flag_gems.use_gems():
        res_out = torch.sub(inp1, inp2)
        for i in range(0, 50):
            res_out = torch.sub(inp1, inp2)
        start1 = time.time()
        for i in range(0, 1000):
            res_out = torch.sub(inp1, inp2)
        end1 = time.time()
    print("triton time ms:", (end1 - start1))

    start2 = time.time()
    for i in range(0, 10):
        ref_out = torch.sub(inp1, inp2)
    end2 = time.time()
    print("torch time ms:", ((end2 - start2) / 10) * 1000)
    torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)

    print("PASS")
