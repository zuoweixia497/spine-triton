import torch
import triton
from triton.backends.spine_triton.driver import CPUDriver

triton.runtime.driver.set_active(CPUDriver())
import flag_gems


# shape = (1, 2, 5, 5)
# kernel =  (1, 2, 3, 3)
# groups = 1
# stride = (1,1)
# padding = (0,0)
# dtype = torch.float32
# bias = False
# dilation = (1,1)


# shape = (1, 64, 56, 56)
# kernel =  (128, 64, 1, 1)
# groups = 1
# stride = (2,2)
# padding = (0,0)
# dtype = torch.float32
# bias = False
# dilation = (1,1)

# shape = (1, 64, 56, 56)
# kernel =  (64, 64, 3, 3)
# groups = 1
# stride = (1,1)
# padding = (1,1)
# dtype = torch.float32
# bias = False
# dilation = (1,1)

# shape = (1, 64, 56, 56)
# kernel =  (128, 64, 3, 3)
# groups = 1
# stride = (2,2)
# padding = (1,1)
# dtype = torch.float32
# bias = False
# dilation = (1,1)

# shape = (1, 3, 224, 224)
# kernel =  (64, 3, 7, 7)
# groups = 1
# stride = (2,2)
# padding = (3,3)
# dtype = torch.float32
# bias = False
# dilation = (1,1)

shape = (1, 128, 28, 28)
kernel = (128, 128, 3, 3)
groups = 1
stride = (1, 1)
padding = (1, 1)
dtype = torch.float32
bias = False
dilation = (1, 1)

# shape = (1, 64, 56, 56)
# kernel =  (128, 64, 1, 1)
# groups = 1
# stride = (2,2)
# padding = (0,0)
# dtype = torch.float32
# bias = False
# dilation = (1,1)

# shape = (1, 128, 28, 28)
# kernel =  (256, 128, 1, 1)
# groups = 1
# stride = (2,2)
# padding = (0,0)
# dtype = torch.float32
# bias = False
# dilation = (1,1)


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
    ref_out = torch.nn.functional.conv2d(
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

for i in range(0, 10):
    with flag_gems.use_gems():
        res_out = torch.nn.functional.conv2d(
            inp,
            weight,
            bias=bias,
            groups=groups,
            stride=stride,
            padding=padding,
            dilation=dilation,
        ).to(dtype)

start1 = time.time()
for i in range(0, 100):
    with flag_gems.use_gems():
        res_out = torch.nn.functional.conv2d(
            inp,
            weight,
            bias=bias,
            groups=groups,
            stride=stride,
            padding=padding,
            dilation=dilation,
        ).to(dtype)
end1 = time.time()
print("triton time ms:", ((end1 - start1) / 100) * 1000)
torch.testing.assert_close(ref_out, res_out, atol=1e-2, rtol=0)

print("PASS")
