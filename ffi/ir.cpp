#include <pybind11/functional.h>
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>

#include "mlir/IR/Builders.h"
#include "mlir/IR/BuiltinOps.h"
#include "mlir/IR/BuiltinTypes.h"
#include "mlir/IR/MLIRContext.h"
#include "mlir/IR/Verifier.h"

// Dialect
#include "Dialect/Frisk/IR/FriskDialect.h"

namespace py = pybind11;

namespace mlir::frisk {

  // *** C++ 绑定 python  IR 创建部分 ***
using ret = py::return_value_policy;

  // frisk自己的dialect相关的函数，与Python绑定部分
void init_ffi_ir_frisk(py::module_ &&m) {
  m.def("load_dialects", [](MLIRContext &context) {
    mlir::DialectRegistry registry;
    // clang-format off
    registry.insert<mlir::frisk::FriskDialect,
                    mlir::affine::AffineDialect,
                    mlir::func::FuncDialect,
                    mlir::arith::ArithDialect,
                    mlir::math::MathDialect,
                    mlir::scf::SCFDialect,
                    mlir::cf::ControlFlowDialect>();
    // clang-format on
    context.appendDialectRegistry(registry);
    context.loadAllAvailableDialects();
  });

  py::class_<GemmOp, OpState>(m, "GemmOp", py::module_local())
      .def("from_operation", [](Operation &operation) -> py::object {
        if (auto op = dyn_cast<GemmOp>(operation))
          return py::cast(op);
        else
          return py::none();
      })
      .def("get_matrix_A", &GemmOp::getMatrixA)
      .def("get_matrix_B", &GemmOp::getMatrixB)
      .def("get_matrix_C", &GemmOp::getMatrixC)
      .def("get_all_matrix", &GemmOp::getAllMatrix);
  // kernel op
  py::class_<KernelOp, OpState>(m, "KernelOp", py::module_local())
      .def("arg", [](KernelOp &self, unsigned idx) -> BlockArgument {
          if (idx >= self.getNumArguments())
            throw pybind11::index_error("Kernel argument index out of range");
          return self.getArgument(idx);
        })
      .def("get_num_arguments", &KernelOp::getNumArguments)
      .def("add_entry_block", [](KernelOp &self) -> Block * { return self.addEntryBlock(); }, ret::reference)
      .def_property_readonly("type", &KernelOp::getFunctionType);
  // parallel op
  py::class_<ParallelOp, OpState>(m, "ParallelOp", py::module_local())
      .def("get_ivs", [](ParallelOp &self) -> std::vector<BlockArgument> {
        std::vector<BlockArgument> ivs;
        for (auto &&arg : self.getIVs()) { ivs.push_back(arg); }
        return ivs;
        })
      .def("get_grid", &ParallelOp::getGrid)
      .def("get_thread_num", &ParallelOp::getThreadNum)
      .def("get_grid_dim", &ParallelOp::getGridDims)
      .def("add_entry_block", [](ParallelOp &self) -> Block * { return self.addEntryBlock(); }, ret::reference);
  // block Op
  py::class_<BlockOp, OpState>(m, "BlockOp", py::module_local())
      .def("get_ivs", [](BlockOp &self) -> std::vector<BlockArgument> {
        std::vector<BlockArgument> ivs;
        for (auto &&arg : self.getIVs()) { ivs.push_back(arg); }
        return ivs;
        })
      .def("get_block_ranges", &BlockOp::getBlockRanges)
      .def("get_iteration_num", &BlockOp::getIterationNum)
      .def("get_block_dim", &BlockOp::getBlockDim);
  // for Op
  py::class_<ForOp, OpState>(m, "ForOp", py::module_local())
      .def("get_induction_var", &ForOp::getInductionVar)
      .def("get_lower_bound", &ForOp::getLower)
      .def("get_upper_bound", &ForOp::getUpper)
      .def("get_step", &ForOp::getStep);
  // alloc buffer op
  py::class_<AllocBufferOp, OpState>(m, "AllocBufferOp", py::module_local())
    .def("get_alignment", &AllocBufferOp::getAlignment)
    .def("get_memroy_space", [](AllocBufferOp &self) {
      auto memSpace = self.getMemorySpace();
      return static_cast<MemorySpace>(memSpace);
    })
    .def("get_result", &AllocBufferOp::getResult)
    .def("get_memref_ty", &AllocBufferOp::getMemRefType);
  // copy op
  py::class_<CopyOp, OpState>(m, "CopyOp", py::module_local())
    .def("get_src_affine_map", &CopyOp::getSrcMap)
    .def("get_dst_affine_map", &CopyOp::getDstMap)
    .def("get_src_memref", &CopyOp::getSrcMemRef)
    .def("get_dst_memref", &CopyOp::getDstMemRef)
    .def("get_src_memref_ty", &CopyOp::getSrcMemRefType)
    .def("get_dst_memref_ty", &CopyOp::getDstMemRefType)
    .def("get_src_indices", [](CopyOp &self) -> ValueRange { return self.getSrcIndices(); })
    .def("get_dst_indices", [](CopyOp &self) -> ValueRange { return self.getDstIndices(); })
    .def("get_src_extents", [](CopyOp &self) -> std::vector<int64_t> { 
      return std::vector<int64_t>(self.getSrcExtents()); 
    })
    .def("get_dst_extents", [](CopyOp &self) -> std::vector<int64_t> { 
      return std::vector<int64_t>(self.getDstExtents()); 
    });
  // fill op
  py::class_<FillOp, OpState>(m, "FillOp", py::module_local())
    .def("get_memref", &FillOp::getMemref);
  // reduce op
  py::class_<ReduceOp, OpState>(m, "ReduceOp", py::module_local())
    .def("get_src_memref", &ReduceOp::getSrc)
    .def("get_dst_memref", &ReduceOp::getDst)
    .def("get_reduce_dim", &ReduceOp::getDim);
}

  // OpBuilder 构建器的与python绑定部分
void init_ffi_ir_builder(py::module_ &m) {
  py::class_<OpBuilder::InsertPoint>(m, "InsertPoint", py::module_local());

  // memroy space
  py::enum_<MemorySpace>(m, "MemorySpace", py::arithmetic())
      .value("GLOBAL", MemorySpace::global)
      .value("SHARED", MemorySpace::shared)
      .value("LOCAL", MemorySpace::local)
      .export_values();

  // builder
  py::class_<OpBuilderWithLoc>(m, "builder")
      .def(py::init<MLIRContext *>())
      .def("create_module", [](OpBuilderWithLoc &self) -> ModuleOp { return self.create<ModuleOp>(); })
      // insertion block/point
      .def("set_insertion_point_to_start",
           [](OpBuilderWithLoc &self, Block &block) -> void { self.setInsertionPointToStart(block); })
      .def("set_insertion_point_to_end",
           [](OpBuilderWithLoc &self, Block &block) { self.setInsertionPointToEnd(block); })
      .def("set_insertion_point_after", [](OpBuilderWithLoc &self, Operation &op) { self.setInsertionPointAfter(op); })
      .def(
          "get_insertion_block",
          [](OpBuilderWithLoc &self) -> Block * { return self.getBuilder().getInsertionBlock(); }, ret::reference)
      .def("get_insertion_point", [](OpBuilderWithLoc &self) { return self.getBuilder().saveInsertionPoint(); })
      .def("restore_insertion_point",
           [](OpBuilderWithLoc &self, OpBuilder::InsertPoint pt) { self.restoreInsertionPoint(pt); })
      // Attr
      .def("get_bool_attr", [](OpBuilderWithLoc &self, bool value) { return self.getBuilder().getBoolAttr(value); })
      .def("get_int32_attr",
           [](OpBuilderWithLoc &self, int32_t value) { return self.getBuilder().getI32IntegerAttr(value); })
      .def("get_float_attr", [](OpBuilderWithLoc &self, float value, const std::string &dtype) { 
            if (dtype == "e4m3fn") {
              return FloatAttr::get(Float8E4M3FNType(), value);
            } else if (dtype == "e5m2") {
              return FloatAttr::get(Float8E5M2Type(), value);
            } else if (dtype == "fp16") {
              return FloatAttr::get(Float16Type(), value);
            } else if (dtype == "fp32") {
              return FloatAttr::get(Float32Type(), value);
            } else if (dtype == "fp64") {
              return FloatAttr::get(Float64Type(), value);
            }
            throw std::invalid_argument("invalid float type");
          })
      // constant
      .def("get_int64",
           [](OpBuilderWithLoc &self, int64_t v) -> Value {
             return Value(self.create<arith::ConstantIntOp>(v, self.getBuilder().getI64Type()));
           })
      .def("get_f16",
           [](OpBuilderWithLoc &self, float v) -> Value {
             return self.create<arith::ConstantOp>(self.getBuilder().getF16FloatAttr(v));
           })
      .def("get_f32",
           [](OpBuilderWithLoc &self, float v) -> Value {
             return self.create<arith::ConstantOp>(self.getBuilder().getF32FloatAttr(v));
           })
      .def("get_f64",
           [](OpBuilderWithLoc &self, double v) -> Value {
             return self.create<arith::ConstantOp>(self.getBuilder().getF64FloatAttr(v));
           })
      .def("get_f16_tensor",
           [](OpBuilderWithLoc &self, std::vector<float> &v, std::vector<int64_t> &shape) -> Value {
             auto elem_type = self.getBuilder().getF16Type();
             auto tensor_type = RankedTensorType::get(shape, elem_type);
             llvm::SmallVector<Attribute> vals;
             for (auto e : v) {
               vals.push_back(self.getBuilder().getF16FloatAttr(e));
             }
             auto data = DenseElementsAttr::get(tensor_type, vals);
             return self.create<arith::ConstantOp>(tensor_type, data);
           })
      .def("get_f32_tensor",
           [](OpBuilderWithLoc &self, std::vector<float> &v, std::vector<int64_t> &shape) -> Value {
             auto elem_type = self.getBuilder().getF32Type();
             auto tensor_type = RankedTensorType::get(shape, elem_type);
             llvm::SmallVector<Attribute> vals;
             for (auto e : v) {
               vals.push_back(self.getBuilder().getF32FloatAttr(e));
             }
             auto data = DenseElementsAttr::get(tensor_type, vals);
             return self.create<arith::ConstantOp>(tensor_type, data);
           })
      .def("get_f64_tensor",
           [](OpBuilderWithLoc &self, std::vector<double> &v, std::vector<int64_t> &shape) -> Value {
             auto elem_type = self.getBuilder().getF64Type();
             auto tensor_type = RankedTensorType::get(shape, elem_type);
             llvm::SmallVector<Attribute> vals;
             for (auto e : v) {
               vals.push_back(self.getBuilder().getF64FloatAttr(e));
             }
             auto data = DenseElementsAttr::get(tensor_type, vals);
             return self.create<arith::ConstantOp>(tensor_type, data);
           })
      .def("get_int64_tensor",
           [](OpBuilderWithLoc &self, std::vector<int64_t> &v, std::vector<int64_t> &shape) -> Value {
             auto elem_type = self.getBuilder().getI64Type();
             auto tensor_type = RankedTensorType::get(shape, elem_type);
             llvm::SmallVector<Attribute> vals;
             for (auto e : v) {
               vals.push_back(self.getBuilder().getI64IntegerAttr(e));
             }
             auto data = DenseElementsAttr::get(tensor_type, vals);
             return self.create<arith::ConstantOp>(tensor_type, data);
           })

      // builder get type
      .def("get_bool_ty", [](OpBuilderWithLoc &self) -> Type { return self.getBuilder().getI8Type(); })
      .def("get_int64_ty", [](OpBuilderWithLoc &self) -> Type { return self.getBuilder().getI64Type(); })
      .def("get_f64_ty", [](OpBuilderWithLoc &self) -> Type { return self.getBuilder().getF64Type(); })
      .def("get_f16_ty", [](OpBuilderWithLoc &self) -> Type { return self.getBuilder().getF16Type(); })
      .def("get_bf16_ty", [](OpBuilderWithLoc &self) -> Type { return self.getBuilder().getBF16Type(); })
      .def("get_f32_ty", [](OpBuilderWithLoc &self) -> Type { return self.getBuilder().getF32Type(); })
      .def("get_ranked_tensor_ty",
           [](OpBuilderWithLoc &self, Type &elementType, std::vector<int64_t> &shape) -> Type {
             return RankedTensorType::get(shape, elementType);
           })
      .def("get_memref_ty", 
           [](OpBuilderWithLoc &self, Type &elementType, std::vector<int64_t> &shape, AffineMap map, MemorySpace space) -> Type {
             if (!map) {
              return MemRefType::get(shape, elementType, {}, static_cast<unsigned>(space));
             }
             return MemRefType::get(shape, elementType, map, static_cast<unsigned>(space));
           }, py::arg("elementType"), py::arg("shape"), py::arg("map") = AffineMap(), py::arg("space") = MemorySpace::global)
      .def("get_function_ty",
           [](OpBuilderWithLoc &self, std::vector<Type> inTypes, std::vector<Type> outTypes) -> Type {
             return self.getBuilder().getFunctionType(inTypes, outTypes);
           })
      .def("get_kernel_ty",
           [](OpBuilderWithLoc &self, std::vector<Type> inTypes) -> Type {
             std::vector<Type> outTypes{};
             return self.getBuilder().getFunctionType(inTypes, outTypes);
           })

      // builder locs
      .def("set_loc", [](OpBuilderWithLoc &self, Location loc) { self.setLastLoc(loc); })
      .def("set_loc", 
        [](OpBuilderWithLoc &self, const std::string &fileName, int line, int column) { self.setLastLoc(fileName, line, column); })
      .def("get_loc", [](OpBuilderWithLoc &self) -> Location { return self.getLastLoc(); })

      // builder op
      .def("clone", [](OpBuilderWithLoc &self, Operation &op) -> Operation * { return self.clone(op); }, ret::reference)
      .def("get_or_insert_function",
           [](OpBuilderWithLoc &self, ModuleOp &module, std::string &funcName, Type &funcType) -> func::FuncOp {
             if (Operation *funcOperation = module.lookupSymbol(funcName))
               return llvm::dyn_cast<func::FuncOp>(funcOperation);
             if (auto funcTy = dyn_cast<FunctionType>(funcType)) {
               return self.create<func::FuncOp>(funcName, funcTy);
             }
             throw std::invalid_argument("invalid function type");
           })
      .def("create_block",
          [](OpBuilderWithLoc &self) -> Block * {
            Region *parent = self.getBuilder().getBlock()->getParent();
            return self.getBuilder().createBlock(parent);
          },
          ret::reference)
      .def("create_block_with_parent",
          [](OpBuilderWithLoc &self, Region &parent, std::vector<Type> &argTypes) -> Block * {
            // TODO: update arg loc
            auto loc = self.getBuilder().getUnknownLoc();
            llvm::SmallVector<Location, 8> argLocs(argTypes.size(), loc);
            return self.getBuilder().createBlock(&parent, {}, argTypes, argLocs);
          },
          ret::reference)
      .def("new_block", [](OpBuilderWithLoc &self) -> Block * { return new Block(); }, ret::reference)
      .def("ret",
          [](OpBuilderWithLoc &self, std::vector<Value> &vals) -> OpState { return self.create<func::ReturnOp>(vals); })
      // affine expr
      .def("get_affine_dim_expr", 
        [](OpBuilderWithLoc &self, unsigned position) -> AffineExpr { return self.getBuilder().getAffineDimExpr(position); })
      .def("get_affine_symbol_expr", 
        [](OpBuilderWithLoc &self, unsigned position) -> AffineExpr { return self.getBuilder().getAffineSymbolExpr(position); })
      .def("get_affine_constant_expr", 
        [](OpBuilderWithLoc &self, int64_t constant) -> AffineExpr { return self.getBuilder().getAffineConstantExpr(constant); })
      // affine map
        .def("get_empty_affine_map", 
        [](OpBuilderWithLoc &self) -> AffineMap { return self.getBuilder().getEmptyAffineMap(); })
      .def("get_dim_identity_map", 
        [](OpBuilderWithLoc &self) -> AffineMap { return self.getBuilder().getDimIdentityMap(); })
      .def("get_multi_dim_identity_map", 
        [](OpBuilderWithLoc &self, unsigned rank) -> AffineMap { return self.getBuilder().getMultiDimIdentityMap(rank); })
      .def("get_symbol_identity_map", 
        [](OpBuilderWithLoc &self) -> AffineMap { return self.getBuilder().getSymbolIdentityMap(); })
      .def("get_single_dim_shift_affine_map", 
        [](OpBuilderWithLoc &self, int64_t shift) -> AffineMap { return self.getBuilder().getSingleDimShiftAffineMap(shift); })
      .def("get_shifted_affine_map", 
        [](OpBuilderWithLoc &self, AffineMap map, int64_t shift) -> AffineMap { return self.getBuilder().getShiftedAffineMap(map, shift); })
      // frisk
      .def("create_gemm_op",
        [](OpBuilderWithLoc &self, Value &A, Value &B, Value &C) { return self.create<GemmOp>(A, B, C); })
      .def("create_kernel_op", [](OpBuilderWithLoc &self, ModuleOp &module, std::string &KernelName, Type &KernelType) {
        if (Operation *kernelOperation = module.lookupSymbol(KernelName))
          return llvm::dyn_cast<KernelOp>(kernelOperation);
        if (auto kernelTy = dyn_cast<FunctionType>(KernelType)) {
          return self.create<KernelOp>(KernelName, kernelTy);
        }
        throw std::invalid_argument("invalid kernel function type");
      })
      .def("create_parallel_op", [](OpBuilderWithLoc &self, std::vector<int64_t> &gridSize, int64_t threadNum) {
        auto builder = self.getBuilder();
        Block *block = builder.getInsertionBlock();
        if (block->getOperations().size() == 1) {
          return self.create<ParallelOp>(gridSize, threadNum);
        }
        throw std::invalid_argument("invalid parallel create");
      })
      .def("create_block_op", 
        [](OpBuilderWithLoc &self, std::vector<int64_t> &blockRange, py::function pyFunc) {
          return self.create<BlockOp>(blockRange, [&](ValueRange args) -> void {
            py::gil_scoped_acquire acquire;
            try {
              pyFunc(self, args);
            } catch (const py::error_already_set &e) {
              llvm::errs() << "Python exception in frisk block op: " << e.what() << "\n";
              throw;
            }
          });
        })
      .def("create_for_op", 
        [](OpBuilderWithLoc &self, int64_t lower, int64_t upper, int64_t step, py::function pyFunc) {
          return self.create<ForOp>(lower, upper, step, [&](Value arg) -> void {
            py::gil_scoped_acquire acquire;
            try {
              pyFunc(self, arg);
            } catch (const py::error_already_set &e) {
              llvm::errs() << "Python exception in frisk for op: " << e.what() << "\n";
              throw;
            }
          });
        })
      .def("create_alloc_buffer_op", 
        [](OpBuilderWithLoc &self, std::vector<int64_t> shape, const std::string &dtype, MemorySpace space, int64_t alignment) {
          MLIRContext *context = self.getBuilder().getContext();
          if (dtype == "e4m3fn") {
            return self.create<AllocBufferOp>(ArrayRef<int64_t>(shape), 
                    Float8E4M3FNType::get(context), alignment, static_cast<int64_t>(space));
          } else if (dtype == "e5m2") {
            return self.create<AllocBufferOp>(ArrayRef<int64_t>(shape), 
                    Float8E5M2Type::get(context), alignment, static_cast<int64_t>(space));
          } else if (dtype == "fp16") {
            return self.create<AllocBufferOp>(ArrayRef<int64_t>(shape), 
                    Float16Type::get(context), alignment, static_cast<int64_t>(space));
          } else if (dtype == "fp32") {
            return self.create<AllocBufferOp>(ArrayRef<int64_t>(shape), 
                    Float32Type::get(context), alignment, static_cast<int64_t>(space));
          } else if (dtype == "fp64") {
            return self.create<AllocBufferOp>(ArrayRef<int64_t>(shape), 
                    Float64Type::get(context), alignment, static_cast<int64_t>(space));
          } else {
            throw std::invalid_argument("invalid float type");
          }
      }, py::arg("shape"), py::arg("dtype"), py::arg("alignment") = 0, py::arg("space") = MemorySpace::global)
      .def("create_copy_op", 
        [](OpBuilderWithLoc &self, 
              Value src, Value dst, 
              std::vector<Value> srcIndices, std::vector<Value> dstIndices, 
              AffineMap srcMap, AffineMap dstMap, 
              std::vector<int64_t> srcExtents, std::vector<int64_t> dstExtents) {
          if (!srcMap && !dstMap && srcExtents.empty() && dstExtents.empty()) {
            return self.create<CopyOp>(src, dst, ValueRange(srcIndices), ValueRange(dstIndices));
          } else if (srcMap && dstMap && srcExtents.empty() && dstExtents.empty()) {
            return self.create<CopyOp>(src, dst, srcMap, dstMap, ValueRange(srcIndices), ValueRange(dstIndices));
          } else if (srcMap && dstMap && !srcExtents.empty() && !dstExtents.empty()) {
            return self.create<CopyOp>(src, dst, srcMap, dstMap, 
              ValueRange(srcIndices), ValueRange(dstIndices), ArrayRef(srcExtents), ArrayRef(dstExtents));
          } else {
            llvm::errs() << "copy op map error."<< "\n";
            throw;
          }
      }, py::arg("src"), py::arg("dst"), 
      py::arg("srcIndices"), py::arg("dstIndices"), 
      py::arg("srcMap") = AffineMap(), py::arg("dstMap") = AffineMap(),
      py::arg("srcExtents") = py::list(), py::arg("dstExtents") = py::list())
      .def("create_fill_op", [](OpBuilderWithLoc &self, Value memref, float value, const std::string &dtype) {
        FloatAttr attr;
        MLIRContext *context = self.getBuilder().getContext();
        if (dtype == "e4m3fn") {
          attr = FloatAttr::get(Float8E4M3FNType::get(context), value);
        } else if (dtype == "e5m2") {
          attr =  FloatAttr::get(Float8E5M2Type::get(context), value);
        } else if (dtype == "fp16") {
          attr =  FloatAttr::get(Float16Type::get(context), value);
        } else if (dtype == "fp32") {
          attr =  FloatAttr::get(Float32Type::get(context), value);
        } else if (dtype == "fp64") {
          attr =  FloatAttr::get(Float64Type::get(context), value);
        } else {
          throw std::invalid_argument("invalid float type");
        }
        return self.create<FillOp>(memref, attr);
      })
      .def("create_reduce_op", [](OpBuilderWithLoc &self, Value src, Value dst, const std::string &kind, int64_t dim, bool clear) {
        return self.create<ReduceOp>(src, dst, kind, dim, clear);
      }, py::arg("src"), py::arg("dst"), py::arg("kind"), py::arg("dim"), py::arg("clear")=true);

}

  // 其他dialect和IR构建所需函数与python绑定部分
void init_ffi_ir_common_op(py::module_ &m) {
  // dynamic_attr is used to transfer ownership of the MLIR context to the module
  py::class_<ModuleOp, OpState>(m, "module", py::module_local(), py::dynamic_attr())
      .def("dump", 
           [](ModuleOp &self, const std::string &log) -> void {
            if (!log.empty()) {
              llvm::outs() << log << "\n";
            }
            llvm::outs() << self << "\n";
            // self.dump();
            }, py::arg("log") = "")
      .def("__str__",
           [](ModuleOp &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = OpPrintingFlags();
             printingFlags.enableDebugInfo();
             self.print(os, printingFlags);
             return str;
           })
      .def("push_back", [](ModuleOp &self, func::FuncOp &funcOp) -> void { self.push_back(funcOp); })
      .def("push_back", [](ModuleOp &self, KernelOp &kernelOp) -> void { self.push_back(kernelOp); })
      .def("has_function",
           [](ModuleOp &self, std::string &funcName) -> bool {
             if (self.lookupSymbol<func::FuncOp>(funcName))
               return true;
             return false;
           })
      .def("get_function",
           [](ModuleOp &self, std::string &funcName) -> func::FuncOp {
             return self.lookupSymbol<func::FuncOp>(funcName);
           })
      .def("get_kernel",
           [](ModuleOp &self, std::string &kernelName) -> KernelOp { return self.lookupSymbol<KernelOp>(kernelName); })
      .def("erase_kernel",
           [](ModuleOp &self, std::string &kernelName) -> void {
            auto kernel = self.lookupSymbol<KernelOp>(kernelName);
            assert(kernel->use_empty() && "KernelOp still has uses!");
            kernel->erase();
          })
      .def("get_int_attr",
           [](ModuleOp &self, std::string name) -> py::object {
             auto ret = self->getAttrOfType<IntegerAttr>(name);
             if (!ret)
               return py::none();
             return py::int_(ret.getInt());
           })
      .def("walk", [](ModuleOp &self, const std::function<void(Operation *)> &fn) { self.walk(fn); });

  py::class_<arith::ConstantOp, OpState>(m, "constant", py::module_local())
      .def("get_splat_float_value", [](arith::ConstantOp &self) -> py::object {
        if (auto dense_attr = dyn_cast<DenseElementsAttr>(self.getValueAttr())) {
          if (dense_attr.isSplat()) {
            double val = dense_attr.getSplatValue<FloatAttr>().getValueAsDouble();
            return py::float_(val);
          }
        }
        return py::none();
      });

  py::class_<func::FuncOp, OpState>(m, "function", py::module_local())
      .def("arg",
           [](func::FuncOp &self, unsigned idx) -> BlockArgument {
             if (idx >= self.getNumArguments())
               throw pybind11::index_error("Function argument index out of range");
             return self.getArgument(idx);
           })
      .def("get_num_arguments", &func::FuncOp::getNumArguments)
      .def(
          "get_callable_region", [](func::FuncOp &self) -> Region * { return self.getCallableRegion(); },
          ret::reference)
      .def(
          "get_terminator",
          [](func::FuncOp &self) -> Operation * { return self.getCallableRegion()->front().getTerminator(); },
          ret::reference)
      .def(
          "add_entry_block", [](func::FuncOp &self) -> Block * { return self.addEntryBlock(); }, ret::reference)
      .def(
          "set_arg_attr",
          [](func::FuncOp &self, int arg_no, const std::string &name, int val) {
            if (arg_no >= self.getNumArguments())
              throw pybind11::index_error("Function argument index out of range");
            // set arg attributes "name" to value "val"
            auto attrTy = IntegerType::get(self.getContext(), 32);
            self.setArgAttr(arg_no, name, IntegerAttr::get(attrTy, val));
          },
          ret::reference)
      .def_property_readonly("type", &func::FuncOp::getFunctionType)
      .def("reset_type", &func::FuncOp::setType);
}

void init_ffi_ir_operation(py::module_ &m) {
  py::class_<OpState>(m, "OpState", py::module_local())
      .def("set_attr", [](OpState &self, std::string &name, Attribute &attr) -> void { self->setAttr(name, attr); })
      .def("get_num_results", [](OpState &self) -> unsigned { return self->getNumResults(); })
      .def("get_result",
           [](OpState &self, unsigned idx) -> Value {
             if (idx >= self->getNumResults())
               throw pybind11::index_error("Op result index out of range");
             return self->getResult(idx);
           })
      .def("get_region",
          [](OpState &self, unsigned idx) -> Region & {
            if (idx >= self->getNumRegions())
              throw pybind11::index_error("Op region index out of range");
            return self->getRegion(idx);
          },
          ret::reference)
      .def("dump", [](OpState &self) { self->dump(); })
      .def("__str__",
           [](OpState &self) -> std::string {
             std::string str;
             llvm::raw_string_ostream os(str);
             auto printingFlags = OpPrintingFlags();
             printingFlags.enableDebugInfo();
             self->print(os, printingFlags);
             return str;
           })
      .def("append_operand", [](OpState &self, Value &val) { self->insertOperands(self->getNumOperands(), val); })
      .def("verify", [](OpState &self) -> bool { return succeeded(verify(self.getOperation())); });

  py::class_<Operation, std::unique_ptr<Operation, py::nodelete>>(m, "operation", py::module_local())
      .def("__str__",
           [](Operation &self) {
             llvm::StringRef opName = self.getName().getStringRef();
             return opName.str();
           })
      .def("__repr__",
           [](Operation &self) {
             llvm::StringRef opName = self.getName().getStringRef();
             return opName.str();
           })
      .def("get_name",
           [](Operation &self) {
             llvm::StringRef opName = self.getName().getStringRef();
             return opName.str();
           })
      .def("get_num_operands", &Operation::getNumOperands)
      .def("get_operand", &Operation::getOperand)
      .def("set_operand", &Operation::setOperand)
      .def("get_num_results", &Operation::getNumResults)
      .def("get_result", &Operation::getResult)
      .def("get_num_regions", &Operation::getNumRegions)
      .def("get_region", &Operation::getRegion, ret::reference)
      .def("get_block", &Operation::getBlock, ret::reference)
      .def("get_str_attr",
           [](Operation &self, const std::string &name) -> py::object {
             auto ret = self.getAttrOfType<StringAttr>(name);
             if (!ret)
               return py::none();
             return py::str(ret.getValue().str());
           })
      .def("get_flat_symbol_ref_attr",
           [](Operation &self, const std::string &name) -> py::object {
             auto ret = self.getAttrOfType<FlatSymbolRefAttr>(name);
             if (!ret)
               return py::none();
             return py::str(ret.getValue().str());
           })
      .def("is_constant_op",
           [](Operation &self) -> bool { return isa<arith::ConstantOp>(self) || isa<arith::ConstantIntOp>(self); })
      .def("to_constant_op",
           [](Operation &self) -> py::object {
             if (auto op = dyn_cast<arith::ConstantOp>(self)) {
               return py::cast(op);
             }
             return py::none();
           });
}

void init_ffi_ir_affine_expr_map(py::module_ &m) {
  py::class_<AffineMap>(m, "AffineMap", py::module_local())
    // 构造函数
    .def(py::init<>())
    // 静态工厂方法
    .def_static("get", [](MLIRContext* context) -> AffineMap { return AffineMap::get(context); })
    .def_static("get", 
      [](unsigned dimCount, unsigned symbolCount, MLIRContext *context) -> AffineMap { 
        return AffineMap::get(dimCount, symbolCount, context); 
      })
    .def_static("get", 
      [](unsigned dimCount, unsigned symbolCount, AffineExpr result) -> AffineMap { 
        return AffineMap::get(dimCount, symbolCount, result); 
      })
    .def_static("get", 
      [](unsigned dimCount, unsigned symbolCount, std::vector<AffineExpr> results, MLIRContext *context) -> AffineMap {
        return AffineMap::get(dimCount, symbolCount, ArrayRef<AffineExpr>(results), context); 
      })
    .def_static("get_constant_map", 
      [](int64_t val, MLIRContext *context) -> AffineMap { return AffineMap::getConstantMap(val, context); })
    .def_static("get_multi_dim_identity_map", 
      [](unsigned numDims, MLIRContext *context) -> AffineMap { return AffineMap::getMultiDimIdentityMap(numDims, context); })
    .def_static("get_minor_identity_map", 
      [](unsigned dims, unsigned results, MLIRContext *context) -> AffineMap { 
        return AffineMap::getMinorIdentityMap(dims, results, context); 
      })
    .def_static("get_permutation_map", 
      [](std::vector<unsigned> permutation, MLIRContext *context) -> AffineMap { 
        return AffineMap::getPermutationMap(ArrayRef<unsigned>(permutation), context); 
      })
    .def_static("get_permutation_map", 
      [](std::vector<int64_t> permutation, MLIRContext *context) -> AffineMap { 
        return AffineMap::getPermutationMap(ArrayRef<int64_t>(permutation), context); 
      })
    .def_static("get_multi_dim_map_with_targets", 
      [](unsigned numDims, MLIRContext *context) -> AffineMap { return AffineMap::getMultiDimIdentityMap(numDims, context); })
    // 比较运算符
    .def("__eq__", [](const AffineMap &self, const AffineMap &other) { return self == other; })
    .def("__ne__", [](const AffineMap &self, const AffineMap &other) { return self != other; })
    .def("__bool__", [](const AffineMap &self) { return static_cast<bool>(self); })
    .def("__nonzero__", [](const AffineMap &self) { return static_cast<bool>(self); })
    // 字符串表示
    .def("__str__", [](AffineMap &self) {
      std::string str;
      llvm::raw_string_ostream os(str);
      self.print(os);
      return str;
    })
    .def("__repr__", [](AffineMap &self) {
      std::string str;
      llvm::raw_string_ostream os(str);
      os << "AffineMap(";
      self.print(os);
      os << ")";
      return str;
    })
    // 属性查询
    .def("get_num_dims", &AffineMap::getNumDims)
    .def("get_num_symbols", &AffineMap::getNumSymbols)
    .def("get_num_results", &AffineMap::getNumResults)
    .def("get_num_inputs", &AffineMap::getNumInputs)
    // 结果访问
    .def("get_results", [](AffineMap &self) -> std::vector<AffineExpr> {
      return std::vector<AffineExpr>(self.getResults().begin(), self.getResults().end());
    })
    .def("get_result", &AffineMap::getResult)
    // 类型检查方法
    .def("is_identity", &AffineMap::isIdentity)
    .def("is_symbol_identity", &AffineMap::isSymbolIdentity)
    .def("is_minor_identity", &AffineMap::isMinorIdentity)
    .def("is_empty", &AffineMap::isEmpty)
    .def("is_single_constant", &AffineMap::isSingleConstant)
    .def("is_constant", &AffineMap::isConstant)
    .def("is_permutation", &AffineMap::isPermutation)
    .def("is_projected_permutation", &AffineMap::isProjectedPermutation, py::arg("allow_zero_in_results") = false)
    // 常量结果获取
    .def("get_single_constant_result", &AffineMap::getSingleConstantResult)
    .def("get_constant_results", [](AffineMap &self) {
      auto results = self.getConstantResults();
      return std::vector<int64_t>(results.begin(), results.end());
    })
    // 维度位置查询
    .def("get_dim_position", &AffineMap::getDimPosition, py::arg("idx"))
    .def("get_result_position", [](AffineMap &self, AffineExpr input) -> py::object {
      auto result = self.getResultPosition(input);
      if (result.has_value())
        return py::cast(result.value());
      return py::none();
    }, py::arg("input"));


  // AffineExprKind
  py::enum_<AffineExprKind>(m, "AffineExprKind", py::module_local())
    .value("Add", AffineExprKind::Add)
    .value("Mul", AffineExprKind::Mul) 
    .value("Mod", AffineExprKind::Mod)
    .value("FloorDiv", AffineExprKind::FloorDiv)
    .value("CeilDiv", AffineExprKind::CeilDiv)
    .value("LAST_AFFINE_BINARY_OP", AffineExprKind::LAST_AFFINE_BINARY_OP)
    .value("Constant", AffineExprKind::Constant)
    .value("DimId", AffineExprKind::DimId)
    .value("SymbolId", AffineExprKind::SymbolId)
    .export_values();

  py::class_<AffineExpr>(m, "AffineExpr", py::module_local())
    // 构造函数
    .def(py::init<>())
    // 比较运算符
    .def("__eq__", [](const AffineExpr &self, const AffineExpr &other) { return self == other; })
    .def("__eq__", [](const AffineExpr &self, int64_t v) { return self == v; })
    .def("__ne__", [](const AffineExpr &self, const AffineExpr &other) { return self != other; })
    .def("__ne__", [](const AffineExpr &self, int64_t v) { return self != v; })
    .def("__bool__", [](const AffineExpr &self) { return static_cast<bool>(self); })
    .def("__nonzero__", [](const AffineExpr &self) { return static_cast<bool>(self); })
    // 字符串表示
    .def("__str__", [](AffineExpr &self) {
      std::string str;
      llvm::raw_string_ostream os(str);
      self.print(os);
      return str;
    })
    .def("__repr__", [](AffineExpr &self) {
      std::string str;
      llvm::raw_string_ostream os(str);
      os << "AffineExpr(";
      self.print(os);
      os << ")";
      return str;
    })
    // 属性查询方法
    .def("get_context", &AffineExpr::getContext)
    .def("get_kind", &AffineExpr::getKind)
    .def("is_symbolic_or_constant", &AffineExpr::isSymbolicOrConstant)
    .def("is_pure_affine", &AffineExpr::isPureAffine)
    .def("get_largest_known_divisor", &AffineExpr::getLargestKnownDivisor)
    .def("is_multiple_of", &AffineExpr::isMultipleOf)
    .def("is_function_of_dim", &AffineExpr::isFunctionOfDim)
    .def("is_function_of_symbol", &AffineExpr::isFunctionOfSymbol)
    // 替换方法
    .def("replace_dims_and_symbols", &AffineExpr::replaceDimsAndSymbols)
    .def("replace_dims", &AffineExpr::replaceDims)
    .def("replace_symbols", &AffineExpr::replaceSymbols)
    .def("replace", 
      [](const AffineExpr &self, AffineExpr target, AffineExpr replacement) { return self.replace(target, replacement); })
    .def("replace_with_map", 
      [](const AffineExpr &self, const DenseMap<AffineExpr, AffineExpr> &map) { return self.replace(map); })
    // 移位方法
    .def("shift_dims", &AffineExpr::shiftDims, py::arg("num_dims"), py::arg("shift"), py::arg("offset") = 0)
    .def("shift_symbols", &AffineExpr::shiftSymbols, py::arg("num_symbols"), py::arg("shift"), py::arg("offset") = 0)
    // 算术运算符
    .def("__add__", [](const AffineExpr &self, int64_t v) { return self + v; })
    .def("__add__", [](const AffineExpr &self, const AffineExpr &other) { return self + other; })
    .def("__radd__", [](const AffineExpr &self, int64_t v) { return self + v; })
    .def("__neg__", [](const AffineExpr &self) { return -self; })
    .def("__sub__", [](const AffineExpr &self, int64_t v) { return self - v; })
    .def("__sub__", [](const AffineExpr &self, const AffineExpr &other) { return self - other; })
    .def("__rsub__", [](const AffineExpr &self, int64_t v) { return AffineExpr() + v - self; })
    .def("__mul__", [](const AffineExpr &self, int64_t v) { return self * v; })
    .def("__mul__", [](const AffineExpr &self, const AffineExpr &other) { return self * other; })
    .def("__rmul__", [](const AffineExpr &self, int64_t v) { return self * v; })
    // 特殊的算术操作
    .def("floor_div", [](const AffineExpr &self, uint64_t v) { return self.floorDiv(v); })
    .def("floor_div", [](const AffineExpr &self, AffineExpr other) { return self.floorDiv(other); })
    .def("ceil_div", [](const AffineExpr &self, uint64_t v) { return self.ceilDiv(v); })
    .def("ceil_div", [](const AffineExpr &self, AffineExpr other) { return self.ceilDiv(other); })
    .def("__mod__", [](const AffineExpr &self, uint64_t v) { return self % v; })
    .def("__mod__", [](const AffineExpr &self, AffineExpr other) { return self % other; })
    // 组合方法
    .def("compose", &AffineExpr::compose)
    // dyn_cast
    .def("as_binary", [](AffineExpr &self) -> py::object {
      if (auto binary = dyn_cast<AffineBinaryOpExpr>(self)) {
        return py::cast(binary);
      }
      return py::none();
    })
    .def("as_dim", [](AffineExpr &self) -> py::object {
      if (auto dim = dyn_cast<AffineDimExpr>(self)) {
        return py::cast(dim);
      }
      return py::none();
    })
    .def("as_symbol", [](AffineExpr &self) -> py::object {
      if (auto symbol = dyn_cast<AffineSymbolExpr>(self)) {
        return py::cast(symbol);
      }
      return py::none();
    })
    .def("as_constant", [](AffineExpr &self) -> py::object {
      if (auto constant = dyn_cast<AffineConstantExpr>(self)) {
        return py::cast(constant);
      }
      return py::none();
    });
    
  // AffineBinaryOpExpr 绑定
  py::class_<AffineBinaryOpExpr, AffineExpr>(m, "AffineBinaryOpExpr", py::module_local())
    .def("get_lhs", &AffineBinaryOpExpr::getLHS)
    .def("get_rhs", &AffineBinaryOpExpr::getRHS);

  // AffineDimExpr 绑定
  py::class_<AffineDimExpr, AffineExpr>(m, "AffineDimExpr", py::module_local())
    .def("get_position", &AffineDimExpr::getPosition);

  // AffineSymbolExpr 绑定  
  py::class_<AffineSymbolExpr, AffineExpr>(m, "AffineSymbolExpr", py::module_local())
    .def("get_position", &AffineSymbolExpr::getPosition);

  // AffineConstantExpr 绑定
  py::class_<AffineConstantExpr, AffineExpr>(m, "AffineConstantExpr", py::module_local())
    .def("get_value", &AffineConstantExpr::getValue);

  //基础表达式创建函数
  m.def("get_affine_dim_expr", &getAffineDimExpr);
  m.def("get_affine_symbol_expr", &getAffineSymbolExpr);
  m.def("get_affine_constant_expr", &getAffineConstantExpr);
  m.def("get_affine_constant_exprs", &getAffineConstantExprs);
  m.def("get_affine_binary_op_expr", &getAffineBinaryOpExpr);
  // 复杂表达式构造
  m.def("get_affine_expr_from_flat_form", &getAffineExprFromFlatForm);
  // 表达式简化
  m.def("simplify_affine_expr", &simplifyAffineExpr);
  // 边界分析
  m.def("get_bound_for_affine_expr", &getBoundForAffineExpr);
}

void init_ffi_ir_common(py::module_ &m) {
  py::class_<MLIRContext>(m, "context", py::module_local())
      .def(py::init<>())
      .def("disable_multithreading", [](MLIRContext &self) { self.disableMultithreading(); });

  py::class_<Type>(m, "type", py::module_local())
      .def("__str__",
           [](Type &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return os.str();
           })
      .def("to_ranked_tensor_ty",
           [](Type &self) -> py::object {
             if (auto ty = dyn_cast<RankedTensorType>(self)) {
               return py::cast(ty);
             }
             return py::none();
           })
      .def("is_f16", &Type::isF16)
      .def("is_f32", &Type::isF32)
      .def("is_f64", &Type::isF64)
      .def("is_float", [](Type &self) -> bool { return isa<FloatType>(self); })
      .def("is_index", &Type::isIndex)
      .def("is_int", [](Type &self) -> bool { return isa<IntegerType>(self); })
      .def("is_signed_int", py::overload_cast<>(&Type::isSignedInteger, py::const_))
      .def("is_unsigned_int", py::overload_cast<>(&Type::isUnsignedInteger, py::const_))
      .def("is_signless_int", py::overload_cast<>(&Type::isSignlessInteger, py::const_));

  py::class_<RankedTensorType, Type>(m, "ranked_tensor", py::module_local())
      .def("get_rank", &RankedTensorType::getRank)
      .def("get_shape", [](RankedTensorType &self) -> std::vector<int64_t> { return self.getShape(); })
      .def("get_element_ty", &RankedTensorType::getElementType);
  
  py::class_<MemRefType, Type>(m, "memref_buffer", py::module_local())
      .def("get_rank", &MemRefType::getRank)
      .def("get_dim_size", &MemRefType::getDimSize)
      .def("get_elem_width", &MemRefType::getElementTypeBitWidth)
      .def("get_shape", [](MemRefType &self) -> std::vector<int64_t> { return self.getShape(); })
      .def("get_element_ty", &MemRefType::getElementType)
      .def("get_memory_space", [](MemRefType &self) -> std::optional<MemorySpace> {
        if (auto spaceAttr = dyn_cast<IntegerAttr>(self.getMemorySpace())) {
          int64_t spaceValue = spaceAttr.getInt();
          return std::optional<MemorySpace>{static_cast<MemorySpace>(spaceValue)};
        }
        return std::optional<MemorySpace>{};
      })
      .def("get_layout_map", [](MemRefType &self) -> AffineMap {
        auto layout = self.getLayout();
        return layout.getAffineMap();
      });

  py::class_<FunctionType>(m, "function_type", py::module_local()).def("param_types", [](FunctionType &self) {
    return std::vector<Type>(self.getInputs().begin(), self.getInputs().end());
  });

  py::class_<Location>(m, "location", py::module_local())
      .def("__str__", [](Location &self) {
        std::string str;
        llvm::raw_string_ostream os(str);
        self.print(os);
        return os.str();
      });

  py::class_<Value>(m, "value", py::module_local())
      .def("get_context", &Value::getContext)
      .def(
          "get_defining_op", [](Value &self) -> Operation * { return self.getDefiningOp(); }, ret::reference)
      .def("get_users",
           [](Value &self) -> py::list {
             py::list ret;
             for (auto &use : self.getUses()) {
               ret.append(use.getOwner());
             }
             return ret;
           })
      .def("replace_all_uses_with", [](Value &self, Value &newValue) { self.replaceAllUsesWith(newValue); })
      .def("get_type", &Value::getType)
      .def("id", [](Value &self) { return (uint64_t)self.getImpl(); })
      .def("__eq__", &Value::operator==)
      .def("__hash__", [](Value &self) -> unsigned { return hash_value(self); })
      .def("__repr__", [](mlir::Value self) {
          return "Value(" + std::to_string(reinterpret_cast<uintptr_t>(self.getAsOpaquePointer())) + ")";
        })
      .def("__str__", [](mlir::Value self) {
          std::string str;
          llvm::raw_string_ostream os(str);
          self.printAsOperand(os, mlir::OpPrintingFlags());
          return os.str();
        });
  
  py::class_<ValueRange>(m, "value_range", py::module_local())
      .def(py::init<>())
      .def(py::init<mlir::Value>())
      .def(py::init<const std::vector<mlir::Value> &>())
      .def("empty", &mlir::ValueRange::empty)
      .def("size", &mlir::ValueRange::size)
      .def("__len__", &mlir::ValueRange::size)
      .def("__getitem__", [](const mlir::ValueRange &self, size_t index) {
          if (index >= self.size()) { throw py::index_error("ValueRange index out of range"); }
          return self[index];
        }, py::return_value_policy::reference_internal)
      .def("__iter__", 
        [](const mlir::ValueRange &self) { return py::make_iterator(self.begin(), self.end()); }, py::keep_alive<0, 1>())
      .def("to_vector", 
        [](const mlir::ValueRange &self) { return std::vector<mlir::Value>(self.begin(), self.end()); })
      .def("__eq__", [](const mlir::ValueRange &self, const mlir::ValueRange &other) { return self == other; })
      .def("__ne__", [](const mlir::ValueRange &self, const mlir::ValueRange &other) { return self != other; })
      .def("__repr__", [](const mlir::ValueRange &self) {
          std::string repr = "ValueRange([";
          for (size_t i = 0; i < self.size(); ++i) {
            if (i > 0) repr += ", ";
            repr += "Value(" + std::to_string(reinterpret_cast<uintptr_t>(self[i].getAsOpaquePointer())) + ")";
          }
          repr += "])";
          return repr;
        })

      .def("__str__", [](const mlir::ValueRange &self) {
          return "ValueRange(size=" + std::to_string(self.size()) + ")";
        });
      

  py::class_<OpResult, Value>(m, "op_result", py::module_local());

  py::class_<BlockArgument, Value>(m, "block_argument", py::module_local());

  py::class_<Region>(m, "region", py::module_local())
      .def("get_parent_region", &Region::getParentRegion, ret::reference)
      .def("size", [](Region &self) { return self.getBlocks().size(); })
      .def(
          "front", [](Region &self) -> Block * { return &self.front(); }, ret::reference)
      .def("empty", &Region::empty)
      .def("walk", [](Block &self, const std::function<void(Operation *)> &fn) { self.walk(fn); })
      .def("id", [](Region &self) { return (uint64_t)&self; });

  py::class_<Block>(m, "block", py::module_local())
      .def("arg",
           [](Block &self, int index) -> BlockArgument {
             if (index >= self.getNumArguments())
               throw pybind11::index_error("Block argument index out of range");
             return self.getArgument(index);
           })
      .def("add_argument",
           [](Block &self, Type ty) {
             auto loc = UnknownLoc::get(ty.getContext());
             self.addArgument(ty, loc);
           })
      .def("get_num_arguments", &Block::getNumArguments)
      .def("get_argument", &Block::getArgument)
      .def("dump", &Block::dump)
      .def("move_before", [](Block &self, Block &dst) { self.moveBefore(&dst); })
      .def("insert_before", &Block::insertBefore)
      .def("get_parent", &Block::getParent, ret::reference)
      .def("merge_block_before",
           [](Block &self, Block &dst) {
             // ref: RewriterBase::mergeBlocks()
             if (self.getNumArguments() != 0)
               throw std::runtime_error("This block has arguments, don't merge");
             dst.getOperations().splice(dst.begin(), self.getOperations());
             self.dropAllUses();
             self.erase();
           })
      .def("replace_use_in_block_with",
           [](Block &self, Value &v, Value &newVal) {
             v.replaceUsesWithIf(newVal, [&](OpOperand &operand) {
               Operation *user = operand.getOwner();
               Block *currentBlock = user->getBlock();
               while (currentBlock) {
                 if (currentBlock == &self)
                   return true;
                 // Move up one level
                 currentBlock = currentBlock->getParent()->getParentOp()->getBlock();
               }
               return false;
             });
           })
      .def("__str__",
           [](Block &self) {
             std::string str;
             llvm::raw_string_ostream os(str);
             self.print(os);
             return str;
           })
      .def("has_terminator", [](Block &self) { return !self.empty() && self.back().hasTrait<OpTrait::IsTerminator>(); })
      .def("has_return", [](Block &self) { return !self.empty() && self.back().hasTrait<OpTrait::ReturnLike>(); })
      .def("erase", [](Block &self) { self.erase(); })
      .def("walk", [](Block &self, const std::function<void(Operation *)> &fn) { self.walk(fn); })
      .def("id", [](Block &self) { return (uint64_t)&self; });

  py::class_<Attribute>(m, "attribute", py::module_local());
  py::class_<IntegerAttr, Attribute>(m, "integer_attr", py::module_local());
  py::class_<BoolAttr, Attribute>(m, "bool_attr", py::module_local());
}

void init_ffi_ir(py::module_ &&m) {
  init_ffi_ir_common(m);
  init_ffi_ir_operation(m);
  init_ffi_ir_common_op(m);
  init_ffi_ir_affine_expr_map(m);
  init_ffi_ir_builder(m);
  init_ffi_ir_frisk(m.def_submodule("frisk"));
}

} // namespace mlir::frisk