from triton.language import core

@core.extern
def abs(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("int32"), ): ("linalg.abs", core.dtype("int32")),
            (core.dtype("int64"), ): ("linalg.abs", core.dtype("int64")),
            (core.dtype("fp32"), ): ("linalg.abs", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.abs", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def ceil(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp64"), ): ("linalg.ceil", core.dtype("fp64")),
            (core.dtype("fp32"), ): ("linalg.ceil", core.dtype("fp32")),
        }, is_pure=True, _builder=_builder)

@core.extern
def exp(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.exp", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.exp", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def floor(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.floor", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.floor", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def log(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.log", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.log", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def round(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.round", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.round", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def rsqrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.rsqrt", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.rsqrt", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def sqrt(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.sqrt", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.sqrt", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def tanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.tanh", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.tanh", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def erf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.erf", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.erf", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def pow(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("linalg.powf", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("linalg.powf", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def gelu_tanh(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.gelu_tanh", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.gelu_tanh", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def gelu_none(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.gelu_none", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.gelu_none", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def silu(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.silu", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.silu", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

# TODO: the following lower implementation
@core.extern
def acos(arg0, _builder=None):
    return core.tensor(_builder.create_acos(arg0.handle), arg0.type)

@core.extern
def acosh(arg0, _builder=None):
    return core.tensor(_builder.create_acosh(arg0.handle), arg0.type)

@core.extern
def asin(arg0, _builder=None):
    return core.tensor(_builder.create_asin(arg0.handle), arg0.type)

@core.extern
def asinh(arg0, _builder=None):
    return core.tensor(_builder.create_asinh(arg0.handle), arg0.type)

@core.extern
def atan(arg0, _builder=None):
    return core.tensor(_builder.create_atan(arg0.handle), arg0.type)

@core.extern
def atanh(arg0, _builder=None):
    return core.tensor(_builder.create_atanh(arg0.handle), arg0.type)

@core.extern
def atan2(arg0, _builder=None):
    return core.tensor(_builder.create_atan2(arg0.handle), arg0.type)

@core.extern
def cbrt(arg0, _builder=None):
    return core.tensor(_builder.create_cbrt(arg0.handle), arg0.type)

@core.extern
def cos(arg0, _builder=None):
    return core.tensor(_builder.create_cos(arg0.handle), arg0.type)

@core.extern
def cosh(arg0, _builder=None):
    return core.tensor(_builder.create_cosh(arg0.handle), arg0.type)

@core.extern
def exp2(arg0, _builder=None):
    return core.tensor(_builder.create_exp2(arg0.handle), arg0.type)

@core.extern
def expm1(arg0, _builder=None):
    return core.tensor(_builder.create_expm1(arg0.handle), arg0.type)

@core.extern
def log2(arg0, _builder=None):
    return core.tensor(_builder.create_log2(arg0.handle), arg0.type)


@core.extern
def log10(arg0, _builder=None):
    return core.tensor(_builder.create_log10(arg0.handle), arg0.type)


@core.extern
def log1p(arg0, _builder=None):
    return core.tensor(_builder.create_log1p(arg0.handle), arg0.type)


@core.extern
def sin(arg0, _builder=None):
    return core.tensor(_builder.create_sin(arg0.handle), arg0.type)


@core.extern
def sinh(arg0, _builder=None):
    return core.tensor(_builder.create_sinh(arg0.handle), arg0.type)


@core.extern
def tan(arg0, _builder=None):
    return core.tensor(_builder.create_tan(arg0.handle), arg0.type)


@core.extern
def trunc(arg0, _builder=None):
    return core.tensor(_builder.create_trunc(arg0.handle), arg0.type)

@core.extern
def div_rn(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("linalg.div_rn", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("linalg.div_rn", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def div_rz(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("linalg.div_rz", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("linalg.div_rz", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def div_rd(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("linalg.div_rd", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("linalg.div_rd", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def fmod(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("linalg.fmod", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("linalg.fmod", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def div_ru(arg0, arg1, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0, arg1], {
            (core.dtype("fp32"), core.dtype("fp32")): ("linalg.div_ru", core.dtype("fp32")),
            (core.dtype("fp64"), core.dtype("fp64")): ("linalg.div_ru", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)

@core.extern
def rint(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("linalg.rint", core.dtype("fp32")),
            (core.dtype("fp64"), ): ("linalg.rint", core.dtype("fp64")),
        }, is_pure=True, _builder=_builder)


@core.extern
def finitef(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp32"), ): ("linalg.finitef", core.dtype("int32")),
    }, is_pure=True, _builder=_builder).to(core.int1, _builder=_builder)


@core.extern
def isinf(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [arg0], {
            (core.dtype("fp32"), ): ("linalg.isinf", core.dtype("int32")),
            (core.dtype("fp64"), ): ("linalg.isinf", core.dtype("int32")),
        }, is_pure=True, _builder=_builder).to(core.int1, _builder=_builder)

@core.extern
def isnan(arg0, _builder=None):
    return core.extern_elementwise(
        "", "", [
            arg0,
        ], {
            (core.dtype("fp32"), ): ("linalg.isnan", core.dtype("int32")),
            (core.dtype("fp64"), ): ("linalg.isnan", core.dtype("int32")),
        }, is_pure=True, _builder=_builder).to(core.int1, _builder=_builder)

@core.extern
def isfinited(arg0, _builder=None):
    return core.extern_elementwise("", "", [arg0], {
        (core.dtype("fp64"), ): ("linalg.isfinited", core.dtype("int32")),
    }, is_pure=True, _builder=_builder).to(core.int1, _builder=_builder)