from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence, Tuple, Union

from ..frisk_ffi import ir

DTypeLike = Union[str, ir.type]
ShapeLike = Sequence[int]

__all__ = [
    "create_context",
    "create_session",
    "FriskHelper",
]


def create_context(*, load_dialects: bool = True, disable_multithreading: bool = True) -> ir.context:
    """Return a context with all Frisk dialects registered."""
    ctx = ir.context()
    if disable_multithreading:
        ctx.disable_multithreading()
    if load_dialects:
        ir.frisk.load_dialects(ctx)
    return ctx


def create_session(
    *, load_dialects: bool = True, disable_multithreading: bool = True
) -> "FriskHelper":
    """Convenience wrapper that returns a ready-to-use FriskHelper."""
    return FriskHelper.create(
        load_dialects=load_dialects,
        disable_multithreading=disable_multithreading,
    )


@dataclass
class FriskHelper:
    """Helper utility that wires together context, builder, and module creation."""

    context: ir.context
    builder: ir.builder
    module: ir.module

    @classmethod
    def create(
        cls, *, load_dialects: bool = True, disable_multithreading: bool = True
    ) -> "FriskHelper":
        ctx = create_context(
            load_dialects=load_dialects, disable_multithreading=disable_multithreading
        )
        builder = ir.builder(ctx)
        module = builder.create_module()
        # Keep the context alive on the module object (matches existing tests)
        module.context = ctx
        return cls(ctx, builder, module)

    # ------------------------------------------------------------------ #
    # Type helpers
    # ------------------------------------------------------------------ #
    def memref_type(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = "fp16",
        *,
        space: ir.MemorySpace = ir.MemorySpace.GLOBAL,
        layout: ir.AffineMap | None = None,
    ) -> ir.memref_buffer:
        """Create a memref type with the requested shape/memory space."""
        element_type = self._as_type(dtype)
        dims = [int(dim) for dim in shape]
        affine_map = layout if layout is not None else ir.AffineMap()
        return self.builder.get_memref_ty(element_type, dims, affine_map, space)

    # ------------------------------------------------------------------ #
    # Op helpers
    # ------------------------------------------------------------------ #
    def new_kernel(
        self, name: str, arg_types: Sequence[ir.type]
    ) -> Tuple[ir.KernelOp, ir.block]:
        """Insert a KernelOp into the module and return the op plus entry block."""
        kernel_type = self.builder.get_kernel_ty(list(arg_types))
        kernel = self.builder.create_kernel_op(self.module, name, kernel_type)
        self.module.push_back(kernel)
        entry = kernel.add_entry_block()
        self.builder.set_insertion_point_to_start(entry)
        return kernel, entry

    def alloc_buffer(
        self,
        shape: ShapeLike,
        dtype: DTypeLike = "fp16",
        *,
        space: ir.MemorySpace = ir.MemorySpace.LOCAL,
        alignment: int = 0,
    ) -> ir.value:
        """Allocate a buffer in the requested memory space and return its SSA value."""
        dtype_str = self._as_dtype_str(dtype)
        op = self.builder.create_alloc_buffer_op(list(shape), dtype_str, space, alignment)
        return ir.OpState.get_result(op, 0)

    def create_gemm(
        self,
        A: ir.value,
        B: ir.value,
        C: ir.value,
        *,
        trans_a: bool = False,
        trans_b: bool = False,
        clear_accum: bool = False,
    ) -> ir.GemmOp:
        """Light-weight wrapper around builder.create_gemm_op with kwargs."""
        return self.builder.create_gemm_op(A, B, C, trans_a, trans_b, clear_accum)

    # ------------------------------------------------------------------ #
    # Convenience utilities
    # ------------------------------------------------------------------ #
    def ensure_insertion_block(self) -> ir.block:
        """Return the current insertion block, creating one when required."""
        block = self.builder.get_insertion_block()
        if block is not None:
            return block
        new_block = self.builder.new_block()
        self.module.get_region(0).push_back(new_block)
        self.builder.set_insertion_point_to_start(new_block)
        return new_block

    def dump(self, msg: str | None = None) -> None:
        """Print the module for debugging."""
        if msg:
            print(msg)
        print(self.module)

    # ------------------------------------------------------------------ #
    # Internal helpers
    # ------------------------------------------------------------------ #
    def _as_type(self, dtype: DTypeLike) -> ir.type:
        if isinstance(dtype, ir.type):
            return dtype
        key = dtype.lower()
        if key == "fp16":
            return self.builder.get_f16_ty()
        if key == "bf16":
            return self.builder.get_bf16_ty()
        if key == "fp32":
            return self.builder.get_f32_ty()
        if key == "fp64":
            return self.builder.get_f64_ty()
        if key == "int64":
            return self.builder.get_int64_ty()
        raise ValueError(f"Unsupported dtype '{dtype}'.")

    def _as_dtype_str(self, dtype: DTypeLike) -> str:
        if isinstance(dtype, str):
            return dtype
        if dtype.is_f16():
            return "fp16"
        if dtype.is_f32():
            return "fp32"
        if dtype.is_f64():
            return "fp64"
        raise ValueError("Unsupported MLIR type for allocation helper.")
