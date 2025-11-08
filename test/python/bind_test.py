import frisk as F

def create_gemmop():
  context = F.context()
  F.frisk.load_dialects(context)
  builder = F.builder(context)
  module = builder.create_module()
  module.context = context

  shape = [4096, 4096]
  dtype = builder.get_f16_ty()
  # create funcOp
  # mem_type = builder.get_memref_ty(dtype, shape, F.AffineMap(), F.MemorySpace.GLOBAL)
  # func_type = builder.get_function_ty([mem_type, mem_type, mem_type], [])
  # fn = builder.get_or_insert_function(module, "test_gemm", func_type)
  # module.push_back(fn)
  # entry = fn.add_entry_block()

  # create kernelOp
  mem_type = builder.get_memref_ty(dtype, shape, F.AffineMap(), F.MemorySpace.GLOBAL)
  kernel_type = builder.get_kernel_ty([mem_type, mem_type, mem_type])
  kernel = builder.create_kernel_op(module, "test_gemm", kernel_type)
  module.push_back(kernel)
  entry1 = kernel.add_entry_block()
  builder.set_insertion_point_to_start(entry1)
  #create parallelOp
  prarllel = builder.create_parallel_op([8, 8], 128)
  entry2 = prarllel.add_entry_block()
  builder.set_insertion_point_to_start(entry2)
  # body 
  def sub_for(b, iv):
    def sub_gemm(b, ivs):
      # create gemmOp
      input_val = [entry1.get_argument(0), entry1.get_argument(1), entry1.get_argument(2)]
      out = b.create_gemm_op(input_val[0], input_val[1], input_val[2])
    # create blockOp
    builder.create_block_op([128, 128], sub_gemm)
  # create forOp
  builder.create_for_op(0, 128, 1, sub_for)

  # vr = F.value_range([out.get_matrix_A(), out.get_matrix_B(), out.get_matrix_B()])
  # vr = F.value_range()
  # print(vr)
  return module

if __name__ == "__main__":
  module = create_gemmop()
  module.dump()