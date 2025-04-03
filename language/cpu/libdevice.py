from triton.language import core

@core.extern
def erf(arg0, _builder=None):
    return core.tensor(_builder.create_erf(arg0.handle), arg0.type)

@core.extern
def pow(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("__rvv_powf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("__rvv_powd", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def exp(arg0, _builder=None):
    return core.tensor(_builder.create_exp(arg0.handle), arg0.type)

def exp2(arg0):
    ...

@core.extern
def tanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("__rvv_tanhf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("__rvv_tanhd", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

def trunc(arg0):
    ...

def rsqrt(arg0):
    ...

def div_rn(arg0, arg1):
    ...

def div_rz(arg0, arg1):
    ...

def div_rd(arg0, arg1):
    ...

def fmod(arg0, arg1):
    ...

def div_ru(arg0, arg1):
    ...

def rint(arg0):
    ...

def finitef(arg0):
    ...

def isinf(arg0):
    ...

def isnan(arg0):
    ...

def isfinited(arg0):
    ...