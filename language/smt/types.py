import triton.language.core as tl
from typing import List, Tuple
from triton._C.libtriton import ir
from triton.language.semantic import TritonSemantic


class buffered_tensor(tl.tensor):
    def __init__(self, handle, element_ty: tl.dtype, shape: List, copies: int, storage: str, semantic: TritonSemantic = None):
        buf_type = buffered_tensor_type(element_ty, shape, copies, storage, semantic)
        super().__init__(handle, buf_type)

        self.type = buf_type
        self.shape = shape
        self.element_ty = element_ty
        self.copies = copies
        self.storage = storage
        self.semantic = semantic

    def __getitem__(self, buffer_idx):
       return buffer_view(self, buffer_idx, _semantic=self.semantic)

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)

    def __str__(self) -> str:
        return f"buffered_tensor<{self.element_ty}, {self.shape}, copies={self.copies}>"

    def make_permute(self, handle, dims):
        return buffered_tensor(
            handle,
            self.element_ty,
            [self.shape[d] for d in dims],
            self.type.copies,
            self.type.storage,
        )


class buffered_tensor_type(tl.pointer_type):

    def __init__(self, element_ty: tl.dtype, shape: List, copies: int, storage: str, semantic: TritonSemantic = None):
        super().__init__(element_ty, shape)
        self.storage = storage
        self.copies = copies
        self.semantic = semantic
        self.element_ty = element_ty
        self.shape = shape

        assert semantic or copies == 0, "buffered_tensor array must be created with a builder"

    def _unflatten_ir(self, handles: List[ir.value], cursor: int) -> Tuple[buffered_tensor, int]:
        value = buffered_tensor(handles[cursor], self.scalar, self.shape, self.copies, self.storage, self.semantic)
        return value, cursor + 1

    def mangle(self) -> str:
        elt = self.scalar.mangle()
        shape = '_'.join(map(str, self.shape))
        if self.copies > 0:
            shape += f'_{self.copies}'
        return f'buffered_{elt}S{shape}'

    def __str__(self) -> str:
        return f"buffered_tensor_ptr_<{self.element_ty}, {self.shape}, {self.copies}>"

    def __eq__(self, other) -> bool:
        return (type(self) is type(other) and self.shape == other.shape
                and self.copies == other.copies)

    def _flatten_ir_types(self, builder: ir.builder, out: List[ir.type]) -> None:
        out.append(self.to_ir(builder))

    def to_ir(self, builder: ir.builder) -> None:
        shape = self.shape
        builder = self.semantic.builder
        if self.copies >= 1:
            shape = [self.copies] + list(shape)

        return builder.create_smt_buffer_type(
            shape,
            self.element_ty.to_ir(builder),
            self.copies,
            self.storage.value,
        )

    @property
    def scalar(self):
        return self.element_ty.scalar if hasattr(self.element_ty, 'scalar') else self.element_ty

    def _flatten_ir(self, handles) -> None:
        handles.append(self.handle)


@tl.builtin
def buffer_view(
    local_allocated_buffers: buffered_tensor,
    buffer_idx: int,
    _semantic=None,
) -> tl.tensor:
    """
    Returns a subview of the buffer.
    """
    buffer_idx = _semantic._convert_elem_to_ir_value(buffer_idx, require_i64=False)
    view_handle = _semantic.builder.create_buffer_tensor_subview(local_allocated_buffers.handle, buffer_idx)

    original_shape = local_allocated_buffers.shape
    if local_allocated_buffers.type.copies == 0:
        if len(original_shape) == 1:
            new_shape = [1]
        else:
            new_shape = original_shape[1:]
    else:
        new_shape = original_shape[1:]

    element_ty = local_allocated_buffers.element_ty

    return tl.tensor(view_handle, tl.pointer_type(tl.block_type(element_ty, new_shape)))