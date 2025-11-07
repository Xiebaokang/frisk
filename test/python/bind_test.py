import frisk as F

def create_gemmop():
  context = F.context()
  F.frisk.load_dialects(context)
  builder = F.builder(context)
  module = builder.create_module()
  module.context = context

  shape = [4096, 4096]
  dtype = builder.get_f16_ty()
  func_type = builder.get_memref_ty(dtype, shape, F.AffineMap(), F.MemorySpace.GLOBAL)
  func_type = builder.get_function_ty([func_type, func_type, func_type], [])

  fn = builder.get_or_insert_function(module, "test_gemm", func_type)
  module.push_back(fn)
  entry = fn.add_entry_block()
  builder.set_insertion_point_to_start(entry)

  input_val = [entry.get_argument(0), entry.get_argument(1), entry.get_argument(2)]
  out = builder.create_gemm(input_val[0], input_val[1], input_val[2])
  builder.ret([])
  return module

if __name__ == "__main__":
  module = create_gemmop()
  module.dump()