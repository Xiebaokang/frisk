import frisk as F

def test1():
  context = F.context()
  builder = F.builder(context)
  # create expr
  dim0 = builder.get_affine_dim_expr(0)
  symbol = builder.get_affine_symbol_expr(0)
  cst0 = builder.get_affine_constant_expr(0)
  cst5 = builder.get_affine_constant_expr(5)
  expr = dim0 * cst5 + symbol + cst0
  print(expr)
  print(expr.get_kind())
  expr = F.get_affine_binary_op_expr(expr.get_kind(), expr, expr)
  print(expr)
  binary_expr = expr.as_binary()
  print(binary_expr.get_lhs(), binary_expr.get_rhs())
  dim_expr = expr.as_dim()
  print(dim_expr)
  # create map
  map = F.AffineMap.get(1, 1, [expr], context)
  print(map)
  expr0 = map.get_result(0)
  exprs = map.get_results()
  print(expr0, exprs)

"""
other test
"""

if __name__ == "__main__":
  test1()
