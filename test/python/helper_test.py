import frisk as F


def build_sample_module():
    helper = F.create_session()

    f16 = helper.builder.get_f16_ty()
    a_ty = helper.memref_type((128, 64), f16, space=F.MemorySpace.SHARED)
    b_ty = helper.memref_type((64, 128), f16, space=F.MemorySpace.SHARED)
    c_ty = helper.memref_type((128, 128), f16, space=F.MemorySpace.LOCAL)

    kernel, entry = helper.new_kernel("helper_demo", [a_ty, b_ty, c_ty])
    args = [entry.get_argument(i) for i in range(3)]

    helper.create_gemm(args[0], args[1], args[2])

    scratch = helper.alloc_buffer(
        (32, 32), "fp16", space=F.MemorySpace.SHARED, alignment=64
    )
    assert "memref" in str(scratch.get_type())

    assert kernel.verify()
    assert helper.module.verify()
    return helper.module


if __name__ == "__main__":
    mod = build_sample_module()
    mod.dump("Helper utilities emitted the following module:")
