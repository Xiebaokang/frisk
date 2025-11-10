import frisk as F

def create_gemmop():
  context = F.context()
  F.frisk.load_dialects(context)
  builder = F.builder(context)
  module = builder.create_module()
  module.context = context

  shape = [4096, 4096]
  shape2 = [4096]
  dtype = builder.get_f16_ty()
  # create funcOp
  # mem_type = builder.get_memref_ty(dtype, shape, F.AffineMap(), F.MemorySpace.GLOBAL)
  # func_type = builder.get_function_ty([mem_type, mem_type, mem_type], [])
  # fn = builder.get_or_insert_function(module, "test_gemm", func_type)
  # module.push_back(fn)
  # entry = fn.add_entry_block()

  # create kernelOp
  mem_type = builder.get_memref_ty(dtype, shape, F.AffineMap(), F.MemorySpace.GLOBAL)
  mem2_type = builder.get_memref_ty(dtype, shape2, F.AffineMap(), F.MemorySpace.GLOBAL)
  kernel_type = builder.get_kernel_ty([mem_type, mem_type, mem_type, mem2_type])
  kernel = builder.create_kernel_op(module, "test_gemm", kernel_type)
  module.push_back(kernel)
  entry1 = kernel.add_entry_block()
  builder.set_insertion_point_to_start(entry1)
  #create parallelOp
  prarllel = builder.create_parallel_op([8, 8], 128)
  entry2 = prarllel.add_entry_block()
  builder.set_insertion_point_to_start(entry2)
  
  input_val = [entry1.get_argument(0), entry1.get_argument(1), 
               entry1.get_argument(2), entry1.get_argument(3)]
  
  shared_A = builder.create_alloc_buffer_op([128, 32], "fp16", F.MemorySpace.SHARED, 1024)
  shared_B = builder.create_alloc_buffer_op([32, 128], "fp16", F.MemorySpace.SHARED, 1024)
  shared_C = builder.create_alloc_buffer_op([128, 128], "fp16", F.MemorySpace.LOCAL)


  # body 
  def for_body_test(b, iv):
    # fill
    fill = builder.create_fill_op(input_val[0], 0.0, "fp16")
    # reduce op
    reduce = builder.create_reduce_op(input_val[0], input_val[3], "max", 1, True)
    # copy
    cp = builder.create_copy_op(input_val[0], input_val[1], [iv, iv], [iv, iv])
    # print(cp.get_src_extents())
    # print(cp.get_dst_indices())
    def block_body_test(b, ivs):
      # create gemmOp
      out = b.create_gemm_op(input_val[0], input_val[1], input_val[2])
    # create blockOp
    builder.create_block_op([128, 128], block_body_test)
  # create forOp
  builder.create_for_op(0, 128, 1, for_body_test)

  # vr = F.value_range([out.get_matrix_A(), out.get_matrix_B(), out.get_matrix_B()])
  # vr = F.value_range()
  # print(vr)
  return module

if __name__ == "__main__":
  module = create_gemmop()
  module.dump()