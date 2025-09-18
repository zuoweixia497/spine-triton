import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems

shape = (32, 6, 6)
kernel = (64, 6, 2)
groups = 1
stride = 1
padding = 0
dtype = torch.float32
bias = False
dilation = 1


inp = torch.randn(shape, dtype=dtype, device=flag_gems.device, requires_grad=True)

torch.backends.cudnn.allow_tf32 = False
weight = torch.randn(kernel, dtype=dtype, device=flag_gems.device, requires_grad=True)
if bias is True:
    bias = torch.randn(
        [weight.shape[0]], dtype=dtype, device=flag_gems.device, requires_grad=True
    )

else:
    bias = None


import time

start2 = time.time()
for i in range(0, 10):
    ref_out = torch.nn.functional.conv1d(
        inp,
        weight,
        bias=bias,
        groups=groups,
        stride=stride,
        padding=padding,
        dilation=dilation,
    ).to(dtype)
end2 = time.time()
print("torch time ms:", ((end2 - start2) / 10) * 1000)

import flag_gems.runtime.backend._spacemit.ops as sp

for i in range(0, 10):
    res_out = sp.conv1d(
        inp,
        weight,
        bias=bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )

start1 = time.time()
for i in range(0, 100):
    res_out = sp.conv1d(
        inp,
        weight,
        bias=bias,
        padding=padding,
        stride=stride,
        dilation=dilation,
        groups=groups,
    )
end1 = time.time()
print("triton time ms:", ((end1 - start1) / 100) * 1000)
torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)

print("PASS")
