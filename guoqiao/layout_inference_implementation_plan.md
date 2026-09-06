# Frisk Layout Inference System Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** 在 Frisk 中实现一个面向 NVIDIA SM90/SM90a 的 MLIR-native 布局系统闭环，覆盖组合布局代数、Distributed/Storage IR、约束传播、有限候选求解、显式 conversion、正确性验证和性能守门。

**Architecture:** 采用“基础能力 + 纵向切片 + 逐类扩展”。Local/Register 使用带 `DistributedEncodingAttr` 的 RankedTensor SSA，Shared/Global 使用 MemRef 与 `StorageLayoutAttr` binding；Op 只收集约束，solver 只选择 assignment，materializer 在求解后统一改写 IR。现有 `LayoutAttr + DenseMap<Value, Attribute>` 仅作为迁移期语义基线，通过 adapter 接入新系统。

**Tech Stack:** C++17、LLVM/MLIR ODS/TableGen、MLIR Pass/Dialect Conversion、Affine/Presburger、SCF、MemRef、Tensor、GPU/NVGPU/NVVM、LLVM ADT/APInt、CMake/Ninja、llvm-lit/FileCheck、CTest、CUDA/Nsight 性能工具。

> **执行状态（2026-09-06）：M0、M1、M2 已完成并通过 Gate；下一未完成里程碑为 M3（Task 14）。**

## Global Constraints

- 首个目标仅为 NVIDIA SM90/SM90a；不得在通用 solver 中散布 target 字符串判断。
- 完整 WGMMA/TMA/mbarrier 指令 lowering 不在本计划范围；本计划实现布局契约、静态成本和 conversion 微基准所需的最小 lowering adapter。
- TileLang 语义参考固定为 `6623b12d232b343648a5ba99992e3e6f0d6376d2`，不得链接或 vendor TVM/TIRX 类型。
- Local/Register 的最终公共表示必须是 RankedTensor SSA；不得继续扩展 local-memory MemRef 作为长期寄存器模型。
- Shared/Global 保持 MemRef；XOR storage mapping 通过 `frisk.layout_view`/binding 表达，不强塞进 MemRef semi-affine layout。
- `frisk.convert_layout` 和 rematerialization 是普通候选，不以“零转换方案无解”为启用条件。
- Hard constraint 不得被代价模型违反；Affine/Presburger 的 `unknown` 不得当作证明成功。
- 所有 worklist、constraint、candidate 和 tie-break 必须使用稳定 ID，结果不得依赖 DenseMap/指针遍历顺序。
- 用户 IR 错误、unsupported shape 和求解冲突必须返回带位置的诊断；不得用 assertion 处理。
- 每项公共语义或架构边界发生变化时，必须在同一任务中同步维护 `guoqiao/layout_inference_design.md`。
- 现有用户修改属于用户；不得覆盖 `build.sh` 或其他与当前任务无关的工作区变更。
- 正确性零容忍：race、OOB、invalid ownership、未解析 LayoutVar 和 nondeterministic IR 必须为 0。
- 性能守门：受支持 kernel 固定环境中位运行时间回退不超过 3%；常规 component candidate 上限 256，首版 beam width 32，布局推断不超过完整编译时间的 10%。

---

## 0. 执行规则

### 0.1 每个任务的固定循环

```text
确认前置任务和接口
  -> 写一个会失败的最小测试
  -> 运行并确认失败原因正确
  -> 写满足该测试的最小实现
  -> 运行局部测试
  -> 运行本里程碑回归
  -> 检查设计文档是否漂移
  -> 提交一个可独立评审的 commit
```

不得把多个未验证任务合并后一次测试。一个任务的退出条件未满足时，不进入依赖它的任务。

### 0.2 基础构建命令

首次配置或 CMake 结构变化后：

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
```

日常构建：

```bash
cmake --build build --target FriskIR FriskTransforms frisk-opt check-frisk --parallel 32
```

现有基线：

```bash
./build/test_pass/frisk_attr_test
./build/test_pass/frisk_reduce_layout_test
```

### 0.3 文档维护检查

每个任务提交前运行：

```bash
git diff --check
rg -n "LayoutAttr|DistributedEncodingAttr|StorageLayoutAttr|convert_layout|CostVector" \
  guoqiao/layout_inference_design.md \
  guoqiao/layout_inference_strategy_comparison.md
```

如果实现改变了属性字段、映射方向、hard/soft 边界、候选优先级、conversion 策略、里程碑范围或性能阈值，必须先更新设计文档再提交。

## 1. 里程碑与依赖

| 里程碑              | 任务   | 可独立验收的结果                                                                  |
| ------------------- | ------ | --------------------------------------------------------------------------------- |
| M0 基础设施         | 1–4   | 原生测试可由 CTest/lit 运行；`frisk-opt` 可加载 dialect/pass；MemoryEffect 正确 |
| M1 布局代数         | 5–9   | Affine/BitLinear/Product 与 Distributed/Storage Attr 可解析、组合和验证           |
| M2 Storage 切片     | 10–13 | alloc/view/copy 完成 constraint → propagation → binding → verifier             |
| M3 Distributed 切片 | 14–17 | Tensor 多 consumer 可选择共同布局或物化`frisk.convert_layout`                   |
| M4 完整传播迁移     | 18–22 | alias/region/Copy/Fill/Gemm/Reduce 进入新 solver；旧生产路径退役                  |
| M5 候选与 SM90      | 23–26 | 全局候选选择、SM90 契约、conversion 优化和 controlled relaxation 完整             |
| M6 硬化验收         | 27–29 | property/differential/performance/compile-time 守门通过，文档与实现一致           |

## 2. 目标代码结构

```text
include/Dialect/Frisk/IR/
  FriskLayoutAttrs.td
  FriskLayoutAttrInterfaces.td
  FriskLayoutOpInterfaces.td
  FriskLayoutInterfaces.h
  FriskLayoutOps.td

include/Dialect/Frisk/Analysis/
  LayoutCommon.h
  LayoutAlgebra.h
  LayoutConstraint.h
  LayoutSolver.h
  LayoutVerifier.h
  LayoutTarget.h
  LegacyLayoutAdapter.h
  LayoutStatistics.h

include/Dialect/Frisk/Target/SM90/
  SM90LayoutTarget.h

include/Dialect/Frisk/Transforms/
  LayoutTypeConverter.h
  PassPipelines.h
  Passes.h
  Passes.td

lib/Dialect/Frisk/IR/
  FriskLayoutAttrs.cpp
  FriskLayoutInterfaces.cpp
  FriskLayoutOps.cpp

lib/Dialect/Frisk/Analysis/
  LayoutAlgebra.cpp
  LayoutConstraint.cpp
  LayoutPropagation.cpp
  LayoutCandidateSolver.cpp
  LayoutCostModel.cpp
  LayoutVerifier.cpp
  LegacyLayoutAdapter.cpp
  LayoutStatistics.cpp

lib/Dialect/Frisk/Target/SM90/
  SM90LayoutTarget.cpp
  SM90CopyConstraints.cpp
  SM90GemmConstraints.cpp
  SM90ReduceConstraints.cpp
  SM90CostModel.cpp

lib/Dialect/Frisk/Transforms/
  NormalizeLayoutIR.cpp
  LayoutInfer.cpp
  LayoutTypeConverter.cpp
  MaterializeLayouts.cpp
  OptimizeLayoutConversions.cpp
  PassPipelines.cpp

lib/Conversion/FriskLayoutToGPU/
  CMakeLists.txt
  TestLayoutConversionLowering.cpp

tools/frisk-opt/
  CMakeLists.txt
  frisk-opt.cpp

test/Dialect/Frisk/layout/
test/Transforms/
unittests/Dialect/Frisk/Layout/
benchmark/layout/conversion/
benchmark/layout/solver/
```

文件按职责拆分。`FriskOps.cpp` 中现有 GEMM/shared layout 构造只在迁移任务中移动，不提前做无关重构。

---

## M0：基础设施与现状冻结

### Task 1: 将现有原生测试纳入 CTest 并冻结基线

**Files:**

- Modify: `CMakeLists.txt`
- Modify: `test_pass/CMakeLists.txt`
- Create: `guoqiao/layout_baseline.md`
- Verify: `test_pass/attr_test.cpp`
- Verify: `test_pass/reduce_layout_test.cpp`

**Interfaces:**

- Consumes: 当前 `FriskIR`、`frisk_attr_test`、`frisk_reduce_layout_test`。
- Produces: `ctest --test-dir build` 可发现并运行两个 legacy baseline；后续所有任务使用同一入口做回归。

- [X] **Step 1: 记录当前测试发现状态**

Run:

```bash
ctest --test-dir build -N
```

Expected: 输出 `Total Tests: 0`，证明当前 CMake 尚未注册测试。

- [X] **Step 2: 运行两个现有可执行测试并保存基线结论**

Run:

```bash
./build/test_pass/frisk_attr_test
./build/test_pass/frisk_reduce_layout_test
```

Expected: 两个进程退出码均为 0；将覆盖的 target/shape/reduce case 和当前已知限制写入 `guoqiao/layout_baseline.md`，不得复制大段日志。

- [X] **Step 3: 注册 CTest**

在顶层 `CMakeLists.txt` 的 `project` 命令后加入：

```cmake
include(CTest)
enable_testing()
```

在 `test_pass/CMakeLists.txt` 两个 executable 定义后加入：

```cmake
add_test(NAME FriskAttrTest COMMAND frisk_attr_test)
add_test(NAME FriskReduceLayoutTest COMMAND frisk_reduce_layout_test)
```

- [X] **Step 4: 重新配置并验证测试发现**

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
cmake --build build --target frisk_attr_test frisk_reduce_layout_test --parallel 32
ctest --test-dir build --output-on-failure
```

Expected: `100% tests passed, 0 tests failed out of 2`。

- [X] **Step 5: 提交基线**

```bash
git add CMakeLists.txt test_pass/CMakeLists.txt guoqiao/layout_baseline.md
git commit -m "test: register legacy layout baselines"
```

### Task 2: 编译并注册真正的布局推断 pass

**Files:**

- Modify: `include/Dialect/Frisk/Transforms/Passes.h:1-18`
- Modify: `include/Dialect/Frisk/Transforms/Passes.td:1-17`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt:1-13`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp:1-25`
- Create: `test_pass/layout_pass_test.cpp`
- Modify: `test_pass/CMakeLists.txt`

**Interfaces:**

- Consumes: MLIR `OperationPass<FunctionOpInterface>`、生成的 `impl::FriskInferLayoutsBase`。
- Produces: `std::unique_ptr<Pass> createFriskInferLayoutsPass()` 和注册参数 `frisk-infer-layouts`。

- [X] **Step 1: 写会失败的 pass 构造测试**

创建 `test_pass/layout_pass_test.cpp`，核心断言为：

```cpp
#include "Dialect/Frisk/Transforms/Passes.h"

int main() {
  std::unique_ptr<mlir::Pass> pass = mlir::frisk::createFriskInferLayoutsPass();
  if (!pass || pass->getArgument() != "frisk-infer-layouts")
    return 1;
  return 0;
}
```

Run:

```bash
cmake --build build --target frisk_layout_pass_test --parallel 32
```

Expected: FAIL，原因是 target/`createFriskInferLayoutsPass` 尚不存在。

- [X] **Step 2: 统一 pass 名称和构造器**

将 `Passes.td` 定义改为：

```tablegen
def FriskInferLayouts : InterfacePass<"frisk-infer-layouts", "FunctionOpInterface"> {
  let summary = "Infer and materialize Frisk layouts";
  // MLIR 21 pass TableGen expects a call expression, including `()`.
  let constructor = "mlir::frisk::createFriskInferLayoutsPass()";
  let dependentDialects = ["::mlir::frisk::FriskDialect"];
}
```

删除 `Passes.h` 中旧的 `createLayoutInferPass()` 手写声明，改为：

```cpp
std::unique_ptr<Pass> createFriskInferLayoutsPass();
```

- [X] **Step 3: 实现最小可运行 pass**

`LayoutInfer.cpp` 使用生成基类：

```cpp
namespace {
class FriskInferLayoutsPass final
    : public impl::FriskInferLayoutsBase<FriskInferLayoutsPass> {
public:
  void runOnOperation() override {
    getOperation()->setAttr("frisk.layout_inference_ran",
                            UnitAttr::get(&getContext()));
  }
};
} // namespace

std::unique_ptr<Pass> createFriskInferLayoutsPass() {
  return std::make_unique<FriskInferLayoutsPass>();
}
```

将 `LayoutInfer.cpp` 加入 `FriskTransforms` source list，并链接 `MLIRFuncDialect`、`MLIRIR`、`MLIRPass`、`FriskIR`。

- [X] **Step 4: 注册并运行测试**

在 `test_pass/CMakeLists.txt` 新增 `frisk_layout_pass_test`，链接 `FriskTransforms`，并注册 CTest。

Run:

```bash
cmake --build build --target frisk_layout_pass_test --parallel 32
ctest --test-dir build -R FriskLayoutPassTest --output-on-failure
```

Expected: `FriskLayoutPassTest` PASS。

- [X] **Step 5: 提交 pass 驱动**

```bash
git add include/Dialect/Frisk/Transforms lib/Dialect/Frisk/Transforms test_pass
git commit -m "feat: add executable layout inference pass"
```

### Task 3: 增加 `frisk-opt` 与 MLIR lit 测试入口

**Files:**

- Modify: `CMakeLists.txt`
- Create: `tools/CMakeLists.txt`
- Create: `tools/frisk-opt/CMakeLists.txt`
- Create: `tools/frisk-opt/frisk-opt.cpp`
- Create: `test/CMakeLists.txt`
- Create: `test/lit.cfg.py`
- Create: `test/lit.site.cfg.py.in`
- Create: `test/Transforms/infer-layouts-smoke.mlir`

**Interfaces:**

- Consumes: `FriskDialect`、`registerFriskPasses()`、`MlirOptMain`。
- Produces: `build/bin/frisk-opt` 和 `check-frisk` target。

- [X] **Step 1: 写命令行 smoke test**

创建：

```mlir
// RUN: frisk-opt %s -pass-pipeline='builtin.module(func.func(frisk-infer-layouts))' | FileCheck %s

module {
  func.func @smoke() {
    return
  }
}

// CHECK: frisk.layout_inference_ran
```

Run:

```bash
cmake --build build --target check-frisk --parallel 32
```

Expected: FAIL，原因是 `check-frisk`/`frisk-opt` 尚不存在。

- [X] **Step 2: 实现 opt driver**

`tools/frisk-opt/frisk-opt.cpp`：

```cpp
#include "Dialect/Frisk/IR/FriskDialect.h"
#include "Dialect/Frisk/Transforms/Passes.h"
#include "mlir/InitAllDialects.h"
#include "mlir/InitAllPasses.h"
#include "mlir/Tools/mlir-opt/MlirOptMain.h"

int main(int argc, char **argv) {
  mlir::DialectRegistry registry;
  mlir::registerAllDialects(registry);
  registry.insert<mlir::frisk::FriskDialect>();
  mlir::registerAllPasses();
  mlir::frisk::registerFriskPasses();
  return mlir::failed(
      mlir::MlirOptMain(argc, argv, "Frisk optimizer\n", registry));
}
```

对应 CMake 必须同时链接 `registerAllDialects()`/`registerAllPasses()` 引用的
MLIR 静态库集合；仅链接 `MLIROptLib` 不会传递所有注册实现：

```cmake
add_llvm_executable(frisk-opt frisk-opt.cpp)
set_target_properties(frisk-opt PROPERTIES
  RUNTIME_OUTPUT_DIRECTORY ${PROJECT_BINARY_DIR}/bin
)
llvm_update_compile_flags(frisk-opt)
get_property(dialect_libs GLOBAL PROPERTY MLIR_DIALECT_LIBS)
get_property(conversion_libs GLOBAL PROPERTY MLIR_CONVERSION_LIBS)
get_property(extension_libs GLOBAL PROPERTY MLIR_EXTENSION_LIBS)
target_link_libraries(frisk-opt PRIVATE
  ${dialect_libs}
  ${conversion_libs}
  ${extension_libs}
  MLIROptLib
  MLIRTransforms
  MLIRTransformUtils
  MLIRSupport
  MLIRIR
  FriskIR
  FriskTransforms
)
```

- [X] **Step 3: 配置 lit**

`test/lit.cfg.py` 必须配置：

```python
config.name = "FRISK"
config.test_format = lit.formats.ShTest(not llvm_config.use_lit_shell)
config.suffixes = [".mlir"]
config.test_source_root = os.path.dirname(__file__)
config.test_exec_root = os.path.join(config.frisk_obj_root, "test")
llvm_config.use_default_substitutions()
llvm_config.add_tool_substitutions(
    [ToolSubst("frisk-opt", unresolved="fatal")],
    [config.frisk_tools_dir, config.llvm_tools_dir],
)
```

`test/CMakeLists.txt` 使用 `configure_lit_site_cfg`，并按以下方式创建目标：

```cmake
# 顶层必须在 add_subdirectory(test) 前提供 Python3_EXECUTABLE，避免依赖
# LLVM 构建树中 llvm-lit 脚本自身可能不可移植的 shebang。
find_package(Python3 REQUIRED COMPONENTS Interpreter)

add_lit_testsuite(check-frisk "Running Frisk regression tests"
  ${CMAKE_CURRENT_BINARY_DIR}
  DEPENDS frisk-opt
)
```

顶层增加 `add_subdirectory(tools)` 与 `add_subdirectory(test)`。

- [X] **Step 4: 验证 driver 和 lit**

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
cmake --build build --target frisk-opt check-frisk --parallel 32
```

Expected: lit 报告 `1 passed`。

- [X] **Step 5: 提交测试入口**

```bash
git add CMakeLists.txt tools test
git commit -m "test: add frisk-opt and lit harness"
```

### Task 4: 修正 MemoryEffect 与容器 Op 副作用传播

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskOps.td`
- Create: `test_pass/memory_effect_test.cpp`
- Modify: `test_pass/CMakeLists.txt`
- Modify: `guoqiao/layout_inference_design.md` only if the implemented effect model differs from Section 8.2

**Interfaces:**

- Consumes: MLIR `MemoryEffectOpInterface`、`RecursiveMemoryEffects`。
- Produces: Alloc/Copy/Fill/Gemm/Reduce 的准确 Allocate/Read/Write effect；Kernel/Parallel/Block/For 递归暴露 region effects。

- [X] **Step 1: 写失败的 effect 测试**

测试构造 Copy、Fill、Gemm、Reduce、Alloc，并断言：

```cpp
if (mlir::isMemoryEffectFree(copy) || mlir::isMemoryEffectFree(gemm) ||
    mlir::isMemoryEffectFree(reduce))
  return 1;
if (!mlir::hasEffect<mlir::MemoryEffects::Write>(fill) ||
    !mlir::hasEffect<mlir::MemoryEffects::Allocate>(alloc))
  return 1;
```

Run:

```bash
cmake --build build --target frisk_memory_effect_test --parallel 32
```

Expected: FAIL，因为现有 ODS 把多个内存 Op 标记为 `Pure`。

- [X] **Step 2: 修正 ODS effects**

采用以下边界：

```tablegen
// 容器
Frisk_Op<"kernel", [RecursiveMemoryEffects,
                     ImplicitFriskTerminator]>
Frisk_Op<"parallel", [RecursiveMemoryEffects,
                       HasParent<"KernelOp">,
                       ImplicitFriskTerminator]>
Frisk_Op<"block", [RecursiveMemoryEffects, ImplicitFriskTerminator]>
Frisk_Op<"for", [RecursiveMemoryEffects, ImplicitFriskTerminator]>

// 空终止符必须显式无副作用，否则容器递归 effect 查询会返回 unknown。
Frisk_Op<"end", [Pure, Terminator, ReturnLike]>

// 数据操作：删除 Pure，按 operand/result 声明
Arg<AnyMemRef, "", [MemRead]>:$src
Arg<AnyMemRef, "", [MemWrite]>:$dst
Res<AnyMemRef, "", [MemAlloc]>:$result
```

Gemm 的 A/B 为 `MemRead`，C 为 `MemRead, MemWrite`；Reduce 的 src 为 `MemRead`、dst 为 `MemWrite`；Copy/FIll 使用现有 operand effect 并删除 `Pure`。

- [X] **Step 3: 生成、编译并运行 effect 测试**

Run:

```bash
cmake --build build --target FriskTableGen FriskIR frisk_memory_effect_test --parallel 32
ctest --test-dir build -R FriskMemoryEffectTest --output-on-failure
```

Expected: PASS。

- [X] **Step 4: 运行 M0 全量回归**

Run:

```bash
ctest --test-dir build --output-on-failure
cmake --build build --target check-frisk --parallel 32
```

Expected: 所有 CTest 和 lit tests PASS。

- [X] **Step 5: 提交 MemoryEffect 修复**

```bash
git add include/Dialect/Frisk/IR/FriskOps.td test_pass guoqiao/layout_inference_design.md
git commit -m "fix: model Frisk memory effects"
```

### M0 Gate

必须同时满足：

```bash
ctest --test-dir build --output-on-failure
cmake --build build --target check-frisk --parallel 32
build/bin/frisk-opt --help-list | rg "frisk-infer-layouts"
```

Expected: CTest/lit 全部通过，pass 出现在完整帮助列表中。MLIR 21 的普通
`--help` 不展开 pass 清单，因此这里必须使用 `--help-list`。未满足时不得进入 M1。

### M0 执行记录（2026-08-30）

状态：**完成**。

独立提交：

```text
29fdf5e test: register legacy layout baselines
487e4e8 feat: add executable layout inference pass
4c57a18 test: add frisk-opt and lit harness
7af9e04 fix: model Frisk memory effects
```

最终 Gate 结果：

```text
ctest --test-dir build --output-on-failure
  4/4 passed

cmake --build build --target check-frisk --parallel 32
  Total Discovered Tests: 1
  Passed: 1

build/bin/frisk-opt --help-list | rg "frisk-infer-layouts"
  --frisk-infer-layouts - Infer and materialize Frisk layouts
```

实施中确认并回写计划的工具链约束：

1. MLIR 21 pass TableGen 的 `constructor` 必须是带 `()` 的调用表达式。
2. 调用 `registerAllDialects()`/`registerAllPasses()` 的静态链接 driver 必须链接 MLIR dialect、conversion、extension 及核心 transform 库集合。
3. lit 配置前必须提供 `Python3_EXECUTABLE`；`frisk-opt` 必须显式输出到 `build/bin`。
4. RecursiveMemoryEffects 容器中的 `frisk.end` 必须显式 `Pure`，否则递归 effect 查询会返回 unknown。
5. MLIR 21 的 pass 注册验收使用 `frisk-opt --help-list`；普通 `--help` 不展示 pass 清单。

---

## M1：组合布局代数与新 Attr

### Task 5: 建立布局公共类型和生成式 Interface

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutCommon.h`
- Create: `include/Dialect/Frisk/IR/FriskLayoutAttrInterfaces.td`
- Create: `include/Dialect/Frisk/IR/FriskLayoutOpInterfaces.td`
- Create: `include/Dialect/Frisk/IR/FriskLayoutInterfaces.h`
- Create: `lib/Dialect/Frisk/IR/FriskLayoutInterfaces.cpp`
- Modify: `include/Dialect/Frisk/IR/CMakeLists.txt`
- Modify: `lib/Dialect/Frisk/IR/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/LayoutInterfaceTest.cpp`
- Create: `unittests/Dialect/Frisk/Layout/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/CMakeLists.txt`
- Create: `unittests/CMakeLists.txt`
- Modify: `CMakeLists.txt`

**Interfaces:**

- Produces:

```cpp
enum class LayoutKind { Distributed, Storage, Instruction };
enum class ProofStatus { Proven, Disproven, Unknown };

struct LayoutProof {
  ProofStatus status;
  SmallVector<int64_t> counterexample;
  std::string reason;
};

struct CostVector {
  uint64_t instructionPathAndWork;
  uint64_t memoryTransactions;
  uint64_t bankConflictDegree;
  uint64_t conversionBytesAndSync;
  uint64_t spillRiskAndRegisters;
  uint64_t sharedBytesAndOccupancy;
  uint64_t replication;
  uint64_t codeSize;
  uint64_t deterministicTieBreak;
};

class LayoutMapAttrInterface;
class LayoutEncodingAttrInterface;
class LayoutConstraintOpInterface;
class LayoutConstraintBuilder;
```

- [X] **Step 1: 写 Interface 生成失败测试**

测试包含新头文件并做静态检查：

```cpp
static_assert(std::is_enum_v<mlir::frisk::LayoutKind>);
static_assert(std::is_enum_v<mlir::frisk::ProofStatus>);
```

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
```

Expected: FAIL，原因是头文件和 unit test target 尚不存在。

- [X] **Step 2: 定义公共结果类型**

`LayoutCommon.h` 定义上述枚举，并增加：

```cpp
struct LayoutDimension {
  StringAttr name;
  int64_t extent;
};

using LayoutMapAttr = LayoutMapAttrInterface;
```

`extent == ShapedType::kDynamic` 只允许出现在 affine outer；BitLinear 的 bit width 必须静态。

- [X] **Step 3: 定义 Attr/Encoding 接口**

`FriskLayoutAttrInterfaces.td` 至少声明：

```tablegen
def LayoutMapAttrInterface : AttrInterface<"LayoutMapAttrInterface"> {
  let cppNamespace = "::mlir::frisk";
  let methods = [
    InterfaceMethod<"Return canonical map", "::mlir::FailureOr<::mlir::Attribute>",
                    "canonicalizeMap", (ins)>,
    InterfaceMethod<"Verify named dimensions", "::mlir::LogicalResult",
                    "verifyMap", (ins "::mlir::Location":$loc)>
  ];
}

def LayoutEncodingAttrInterface : AttrInterface<"LayoutEncodingAttrInterface"> {
  let cppNamespace = "::mlir::frisk";
  let methods = [
    InterfaceMethod<"Expand encoding", "::mlir::FailureOr<::mlir::Attribute>",
                    "getCanonicalMap", (ins "::mlir::ShapedType":$type)>,
    InterfaceMethod<"Verify encoding", "::mlir::LogicalResult",
                    "verifyForType", (ins "::mlir::ShapedType":$type,
                                           "::mlir::Location":$loc)>,
    InterfaceMethod<"Return encoding kind", "::mlir::frisk::LayoutKind",
                    "getKind", (ins)>
  ];
}
```

`FriskLayoutOpInterfaces.td` 声明 `collectLayoutConstraints(LayoutConstraintBuilder&)`；此时只生成声明，Task 11 再提供 builder 实现。

- [X] **Step 4: 接入 TableGen 和 unit test**

为 Attr interface 和 Op interface 分别生成 decl/def，避免两个 generator 写同一文件；`FriskIR` 增加 `FriskLayoutInterfaces.cpp`。Unit test 使用：

```cmake
add_unittest(FriskUnitTests FriskLayoutUnitTests
  LayoutInterfaceTest.cpp
)
target_link_libraries(FriskLayoutUnitTests PRIVATE FriskIR MLIRIR)
```

Frisk 是外部 MLIR 工程；若 LLVM 配置未导出 `llvm_gtest` target，`unittests/CMakeLists.txt`
必须先从 `${LLVM_MAIN_SRC_DIR}/../third-party/unittest` 引入与当前 LLVM 源码匹配的
gtest/gmock，再调用 `add_unittest`。不能依赖系统 gtest 或裸 `-lllvm_gtest`。

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
cmake --build build --target FriskLayoutUnitTests --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
```

Expected: unit test PASS。

- [X] **Step 5: 提交接口骨架**

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk unittests CMakeLists.txt
git commit -m "feat: define layout interfaces and proof types"
```

### Task 6: 实现 GF(2) Matrix 核心

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutAlgebra.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutAlgebra.cpp`
- Create: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Modify: `lib/Dialect/Frisk/CMakeLists.txt`
- Modify: `unittests/Dialect/Frisk/Layout/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/GF2MatrixTest.cpp`

**Interfaces:**

- Produces:

```cpp
class GF2Matrix {
public:
  static FailureOr<GF2Matrix> get(unsigned rows, unsigned cols,
                                  ArrayRef<APInt> rowBits);
  unsigned getNumRows() const;
  unsigned getNumColumns() const;
  APInt apply(const APInt &input) const;
  GF2Matrix transpose() const;
  FailureOr<GF2Matrix> compose(const GF2Matrix &rhs) const;
  unsigned rank() const;
  SmallVector<APInt> kernelBasis() const;
  FailureOr<GF2Matrix> inverse() const;
  FailureOr<GF2Matrix> rightInverse() const;
  bool operator==(const GF2Matrix &rhs) const;
};
```

- [X] **Step 1: 写矩阵红灯测试**

至少覆盖 identity、XOR、rank-deficient、inverse 和 compose 顺序：

```cpp
TEST(GF2MatrixTest, ComposeUsesRhsThenLhs) {
  GF2Matrix distributed = makeMatrix({0b01, 0b11}, 2);
  GF2Matrix storage = makeMatrix({0b10, 0b11}, 2);
  auto composed = storage.compose(distributed);
  ASSERT_TRUE(succeeded(composed));
  EXPECT_EQ(composed->apply(APInt(2, 0b11)),
            storage.apply(distributed.apply(APInt(2, 0b11))));
}
```

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
```

Expected: FAIL，`GF2Matrix` 未定义。

- [X] **Step 2: 实现构造、apply、transpose 和 compose**

内部使用 `SmallVector<APInt> rows`。乘加规则固定为：

```cpp
bool bit = (rows[row] & input).popcount() & 1;
```

`lhs.compose(rhs)` 表示 `lhs(rhs(x))`；维度不匹配返回 failure，不截断 APInt。

- [X] **Step 3: 实现消元、kernel、inverse/right-inverse**

使用确定性 Gauss-Jordan：pivot 按 column 从小到大、row 从小到大选择。`inverse()` 仅方阵满秩成功；`rightInverse()` 仅输出空间被覆盖时成功。

- [X] **Step 4: 增加穷举 oracle 并验证**

对输入 bit 数不超过 8 的矩阵枚举所有输入，比较 `compose/apply`；对可逆矩阵验证 `A.inverse()(A(x)) == x`。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='GF2MatrixTest.*'
```

Expected: 所有 GF2Matrix tests PASS。

- [X] **Step 5: 提交 GF(2) 核心**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis unittests
git commit -m "feat: add deterministic GF2 matrix algebra"
```

### Task 7: 实现 `BitLinearLayoutMapAttr`

**Files:**

- Create: `include/Dialect/Frisk/IR/FriskLayoutAttrs.td`
- Modify: `include/Dialect/Frisk/IR/FriskAttributes.td`
- Create: `lib/Dialect/Frisk/IR/FriskLayoutAttrs.cpp`
- Modify: `lib/Dialect/Frisk/IR/CMakeLists.txt`
- Create: `test/Dialect/Frisk/layout/bit-linear-attr.mlir`
- Create: `unittests/Dialect/Frisk/Layout/BitLinearLayoutTest.cpp`
- Modify: `unittests/Dialect/Frisk/Layout/CMakeLists.txt`

**Interfaces:**

- Produces `BitLinearLayoutMapAttr`，字段固定为：

```text
input_names       : ArrayAttr<StringAttr>
input_bit_widths  : DenseI64ArrayAttr
output_names      : ArrayAttr<StringAttr>
output_bit_widths : DenseI64ArrayAttr
matrix            : DenseIntElementsAttr<tensor<rows x cols x i1>>
```

- [X] **Step 1: 写 parse/verify 红灯测试**

测试包含一个合法 XOR map 和三个非法 map：matrix shape 错、重复 dimension name、bit width 为 0。

```mlir
// RUN: frisk-opt %s --split-input-file --verify-diagnostics | FileCheck %s
module attributes {
  frisk.map = #frisk.bit_linear<
    inputs = ["lane", "register"], input_bits = [2, 1],
    outputs = ["m", "n"], output_bits = [1, 2],
    matrix = dense<[[1, 0, 0], [0, 1, 1], [0, 0, 1]]> : tensor<3x3xi1>>
} {}
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，attribute 尚未定义。

- [X] **Step 2: 定义 ODS Attr 和 verifier**

Attr 实现 `LayoutMapAttrInterface`。Verifier 检查：名称唯一、name/width 数量一致、所有 width > 0、matrix 行数等于输出总 bit、列数等于输入总 bit。

- [X] **Step 3: 接入 GF2 algebra**

实现：

```cpp
FailureOr<GF2Matrix> BitLinearLayoutMapAttr::getMatrixValue() const;
FailureOr<Attribute> BitLinearLayoutMapAttr::canonicalizeMap() const;
LayoutProof checkInjective(BitLinearLayoutMapAttr map);
LayoutProof checkSurjective(BitLinearLayoutMapAttr map);
FailureOr<BitLinearLayoutMapAttr>
composeBitLinear(BitLinearLayoutMapAttr lhs, BitLinearLayoutMapAttr rhs);
```

Canonical form 保持命名维度顺序，但删除 width 为 0 的禁止状态、规范 DenseElements 存储并拒绝重复名称；不得按 hash 顺序重排。

- [X] **Step 4: 验证 parse/print、compose 和 replication**

Unit test 检查全零 input column 被识别为 kernel/replication，FileCheck 检查 parse-print 稳定。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='BitLinearLayoutTest.*'
```

Expected: unit/lit tests PASS。

- [X] **Step 5: 提交 BitLinear Attr**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk/IR test unittests
git commit -m "feat: add bit-linear layout attribute"
```

### Task 8: 实现 Affine outer 与 Product layout

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutAttrs.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutAttrs.cpp`
- Modify: `include/Dialect/Frisk/Analysis/LayoutAlgebra.h`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutAlgebra.cpp`
- Create: `unittests/Dialect/Frisk/Layout/ProductLayoutTest.cpp`
- Create: `test/Dialect/Frisk/layout/product-layout.mlir`

**Interfaces:**

- Produces:

```text
AffineLayoutMapAttr(input_names, input_extents,
                    output_names, output_extents, affine_map)
ProductLayoutMapAttr(outer, inner, split_extents)
```

```cpp
FailureOr<Attribute> composeLayoutMaps(Attribute lhs, Attribute rhs);
FailureOr<Attribute> projectLayoutMap(Attribute map,
                                      ArrayRef<StringRef> outputs);
FailureOr<Attribute> permuteLayoutMap(Attribute map,
                                      ArrayRef<StringRef> outputs);
LayoutProof checkCoverage(Attribute map, ArrayRef<int64_t> logicalShape);
LayoutProof checkInjectivity(Attribute map, ArrayRef<int64_t> domainShape);
```

- [X] **Step 1: 写 non-power-of-two 与 carry 红灯测试**

测试概念语义 `logical = outer * innerExtent + inner`；Attr 中将 Affine outer
规范存为已缩放的 `outer_base = outer * innerExtent`，实际求值为
`logical = outer_base + inner`。覆盖 shape 6 的 ragged predicate，以及 outer base
未对齐时禁止直接拼接 GF(2) 内层。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，Affine/Product API 尚不存在。

- [X] **Step 2: 实现 `AffineLayoutMapAttr` verifier**

检查 AffineMap dim 数、symbol 数、name/extent 数量、正静态 extent 或 `ShapedType::kDynamic`。动态 extent 只允许作为 symbol/bounds，不得进入 BitLinear matrix。

- [X] **Step 3: 实现 Product compose 和 canonicalization**

固定组合流程：

```text
按名字对齐维度
-> 验证 inner extent 为 2 的幂
-> 验证 outer base 不向 inner bit 产生 carry
-> 分别 compose outer/inner
-> 合并 predicate
-> canonicalize factor order
```

Product verifier 无法证明无 carry时直接拒绝构造；独立 proof API 对动态/超限情形返回
`ProofStatus::Unknown`，不得生成不安全 Product，也不得把 `Unknown` 当作成功。

- [X] **Step 4: 运行枚举 oracle**

对每维 extent 不超过 8 的 product map 枚举 `(outer, inner)`，比较 canonical map 与逐层求值；覆盖 transpose、projection、padding、ragged。

实现额外设置 65,536 个 domain point 的硬上限：完整 BitLinear domain 优先使用
GF(2) rank 精确证明；静态 Affine/Product 仅作为 verifier/测试的有界参考证明，动态、
含 symbol 或超限时返回 `Unknown`，不得进入 solver 的逐候选热循环。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='ProductLayoutTest.*'
```

Expected: PASS。

- [X] **Step 5: 提交组合代数**

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk test unittests
git commit -m "feat: compose affine and bit-linear layouts"
```

### Task 9: 实现 Distributed/Storage Encoding 与 legacy adapter

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutAttrs.td`
- Modify: `include/Dialect/Frisk/IR/FriskAttributes.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutAttrs.cpp`
- Create: `include/Dialect/Frisk/Analysis/LegacyLayoutAdapter.h`
- Create: `lib/Dialect/Frisk/Analysis/LegacyLayoutAdapter.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/LayoutEncodingTest.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LegacyLayoutAdapterTest.cpp`
- Create: `test/Dialect/Frisk/layout/layout-encoding.mlir`

**Interfaces:**

- Produces:

```cpp
FailureOr<DistributedEncodingAttr>
convertLegacyDistributed(LayoutAttr legacy, ShapedType type,
                         Location loc);

FailureOr<StorageLayoutAttr>
convertLegacyStorage(LayoutAttr legacy, MemRefType type,
                     Location loc);
```

Encoding 字段：

```text
DistributedEncodingAttr(map, topology, replication)
StorageLayoutAttr(map, memory_space, alignment, vector_granularity)
```

ODS assembly format 固定为命名字段；后续测试中的 `#smem_layout` 和 `#enc` 必须在文件顶部完整定义，例如：

```mlir
#smem_layout = #frisk.storage<
  map = #smem_map,
  memory_space = #frisk<memory_space Shared>,
  alignment = 128,
  vector_granularity = 16>
#enc = #frisk.distributed<
  map = #blocked_map,
  topology = [8, 32, 4, 1, 1],
  replication = 1>
```

`topology` 的字段顺序固定为 `(register, lane, warp, warp_group, cta)`；parser、printer 和 verifier 共同使用这一顺序，禁止依靠调用点注释猜测。

- [X] **Step 1: 写 type verifier 和 baseline conversion 红灯测试**

测试：encoding logical shape 不匹配、Storage map 非单射、shared alignment 非正、现有 `sm90_ss` Gemm A/B/C legacy layout 转换。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，新 Encoding/adapter 不存在。

- [X] **Step 2: 实现 Encoding verifier**

Distributed 检查 carrier names 只来自 `register/lane/warp/warp_group/cta`、coverage、replication 与 topology；对 BitLinear map 精确验证
`replication = product(topology) / 2^rank(B_D)`。Storage 检查 memory space、live
domain injectivity、`(byte_offset, bit_offset)`、byte 单位的 alignment/vector
granularity。

- [X] **Step 3: 实现显式支持集的 legacy adapter**

Adapter 仅支持当前 SM90 baseline 中可证明的静态 2 次幂 fragment、linear/padded
shared 和能从完整真值表恢复 GF(2) basis 的 swizzle。SM80 旧路径继续回归，但不作为
当前 SM90-only adapter 的支持承诺。无法识别时：

```cpp
return emitError(loc)
       << "legacy layout cannot be represented by the canonical layout algebra";
```

不得猜测、不得退化为 linear layout。对 swizzle 使用现有语义逐点枚举构造 GF(2) basis，并验证全域后才返回。

实现细节：adapter 单独编译为 `FriskLegacyLayoutAdapter`，依赖 `FriskIR` 和
`FriskAnalysis`，从而避免 `FriskIR <-> FriskAnalysis` 循环。多结果 legacy storage
tuple 采用由各结果最大 extent 确定的 row-major flatten；只有单结果 map 可进入
Affine fallback。枚举上限为 65,536，超限返回 failure。

- [X] **Step 4: 对现有 baseline 做等价枚举**

对每个 `(thread, register)` 和 logical shared point，比较 legacy 与新 canonical map；测试失败必须打印首个反例。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutEncodingTest.*:LegacyLayoutAdapterTest.*'
```

Expected: `sm90_ss` 和 `sm90_rs` 所有已转换 operand PASS；其中 `sm90_rs` A 的
legacy `replicate=2` 必须成为一个零 carrier bit，并由 GF(2) rank 验证为 2。

- [X] **Step 5: 维护设计文档并提交**

如果最终 Attr 参数或 canonicalization 与设计文档 Section 5/6 不同，先同步修改。

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk test unittests \
  guoqiao/layout_inference_design.md
git commit -m "feat: add distributed and storage encodings"
```

### M1 Gate

Run:

```bash
cmake --build build --target FriskIR FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
ctest --test-dir build --output-on-failure
```

Expected: 全部通过；现有 SM90 layout baseline 的新旧枚举无差异；无 `unknown` 被当作 proven 的测试路径。

**执行结果（2026-08-30）：PASS。** `FriskLayoutUnitTests` 运行 6 个 suite、
24 个测试全部通过；lit 4/4 通过；CTest 4/4 通过。`sm90_ss` 与 `sm90_rs`
adapter 对每个 logical point 完成新旧差分，动态 Affine proof 的 `Unknown` 路径有独立
测试且未被接受为 proven。

---

## M2：Storage 纵向切片

### Task 10: 新增 `frisk.layout_view` Storage anchor

**Files:**

- Create: `include/Dialect/Frisk/IR/FriskLayoutOps.td`
- Modify: `include/Dialect/Frisk/IR/FriskOps.td`
- Create: `lib/Dialect/Frisk/IR/FriskLayoutOps.cpp`
- Modify: `lib/Dialect/Frisk/IR/CMakeLists.txt`
- Create: `test/Dialect/Frisk/layout/layout-view.mlir`

**Interfaces:**

- Produces:

```mlir
%view = frisk.layout_view %source
  {layout = #smem_layout}
  : memref<64x64xf16, #frisk<memory_space Shared>>
    -> memref<64x64xf16, #frisk<memory_space Shared>>
```

`layout` 在推断前可省略，在 materialization 后必须存在。

- [X] **Step 1: 写 parse/verify 红灯测试**

覆盖合法 view、source/result type 不同、global view 绑定 shared layout、非单射 storage map。

```mlir
// expected-error@+1 {{source and result must have identical memref types}}
%bad = "frisk.layout_view"(%arg0) :
  (memref<64x64xf16>) -> memref<32x128xf16>
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，Op 尚不存在。

- [X] **Step 2: 定义 Op 和 ViewLike 语义**

ODS 约束：一个 `AnyMemRef` operand、同类型 `AnyMemRef` result、可选 `StorageLayoutAttr`，实现 `ViewLikeOpInterface` 和无副作用 view 语义。C++ verifier 调用 `layout.verifyForType(getResult().getType(), getLoc())`。

```cpp
Value LayoutViewOp::getViewSource() { return getSource(); }
```

- [X] **Step 3: 实现 canonicalization**

只允许以下折叠：

```text
layout_view(layout_view(x, L), L) -> layout_view(x, L)
layout_view(x, identity storage)  -> x，仅当无下游 layout anchor 依赖该 Op
```

两个不同 storage binding 不得折叠或覆盖。

- [X] **Step 4: 运行 IR tests**

Run:

```bash
cmake --build build --target FriskIR check-frisk --parallel 32
```

Expected: layout-view tests PASS，非法 case 输出指定诊断。

- [X] **Step 5: 提交 layout view**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk/IR test
git commit -m "feat: add storage layout view anchor"
```

### Task 11: 建立 LayoutVar、Constraint 和 Provenance graph

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutConstraint.h`
- Create: `include/Dialect/Frisk/Analysis/LayoutTarget.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutConstraint.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/LayoutConstraintTest.cpp`

**Interfaces:**

- Produces:

```cpp
using LayoutVarID = uint32_t;
using LayoutConstraintID = uint32_t;
using ProvenanceID = uint32_t;

enum class LayoutState { Uninitialized, CandidateSet, Resolved, Conflict };
enum class ConstraintStrength { Hard, Soft };
enum class ConstraintKind {
  SameLayout,
  TransformLayout,
  RequireEncoding,
  InstructionContract,
  StorageAccess,
  AliasLayout,
  Ownership,
  ResourceLimit,
  Preference
};

struct LayoutCandidate {
  Attribute value;
  ProvenanceID provenance;
  uint64_t stableOrdinal;
};

struct LayoutVar {
  LayoutVarID id;
  LayoutKind kind;
  Type shapedType;
  SmallVector<LayoutCandidate> candidates;
  LayoutState state;
};

enum class EdgeResolutionKind { KeepCommonLayout, Convert, Rematerialize };

struct LayoutConversionEdge {
  OpOperand *use;
  Attribute sourceEncoding;
  Attribute targetEncoding;
  EdgeResolutionKind resolution;
  uint64_t bytes;
  uint64_t synchronizationCost;
};

class LayoutConstraintGraph {
public:
  LayoutVarID addVariable(LayoutKind kind, Type type, StringRef stableName);
  LayoutConstraintID addConstraint(ConstraintKind kind,
                                   ConstraintStrength strength,
                                   ArrayRef<LayoutVarID> vars,
                                   Operation *source, StringRef rule,
                                   StringRef reason);
  LogicalResult verifyInvariants(Location loc) const;
};

struct CandidateAssignment;

class LayoutTarget {
public:
  virtual ~LayoutTarget() = default;
  virtual void enumerateCandidates(
      const LayoutVar &var,
      SmallVectorImpl<LayoutCandidate> &out) const = 0;
  virtual LogicalResult verifyCandidate(const LayoutVar &var,
                                        Attribute candidate,
                                        Location loc) const = 0;
  virtual FailureOr<CostVector>
  evaluate(const CandidateAssignment &assignment) const = 0;
};
```

- [X] **Step 1: 写 stable-ID 和 invariant 红灯测试**

测试相同逻辑输入在不同插入顺序下，按 `stableName` finalize 后 ID/constraint 顺序一致；重复变量名、无效 var 引用和空 hard constraint 必须失败。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，graph 尚不存在。

- [X] **Step 2: 实现 graph ownership**

Graph 拥有 vars、constraints、provenance；`Operation*` 只用于诊断，不参与排序或 hash。稳定名称来自：

```text
function symbol / block ordinal / operation ordinal / result-or-operand ordinal / layout kind
```

- [X] **Step 3: 实现 provenance chain**

```cpp
struct LayoutProvenance {
  ProvenanceID id;
  std::optional<ProvenanceID> parent;
  Operation *source;
  std::string rule;
  std::string reason;
};
```

提供 `printProvenanceChain(ProvenanceID, raw_ostream&)`，检测 parent cycle。

- [X] **Step 4: 验证确定性和诊断**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutConstraintTest.*'
```

Expected: PASS；两种插入顺序 dump 完全相同。

- [X] **Step 5: 提交 constraint graph**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis unittests
git commit -m "feat: add deterministic layout constraint graph"
```

### Task 12: 实现 Storage constraint collector 和 strict/common propagation

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutSolver.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutPropagation.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/CMakeLists.txt`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`
- Modify: `lib/Dialect/Frisk/CMakeLists.txt`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutPropagationTest.cpp`
- Create: `test/Transforms/storage-propagation.mlir`

**Interfaces:**

- Consumes the `LayoutConversionEdge` and `EdgeResolutionKind` definitions from Task 11; produces edge enumeration and cost evaluation:

```cpp
class LayoutConstraintBuilder {
public:
  LayoutVarID getOrCreateStorageVar(Value anchor);
  LayoutVarID getOrCreateDistributedVar(Value value);
  LogicalResult require(LayoutVarID var, Attribute encoding,
                        Operation *source, StringRef rule);
  LogicalResult same(LayoutVarID lhs, LayoutVarID rhs,
                     Operation *source, StringRef rule);
};

FailureOr<LayoutConstraintGraph> collectLayoutConstraints(Operation *root,
                                                          LayoutTarget &target);
LogicalResult propagateStrict(LayoutConstraintGraph &graph);
LogicalResult propagateCommonToFixedPoint(LayoutConstraintGraph &graph);
```

- [X] **Step 1: 写 storage propagation 红灯测试**

输入包含两个 `layout_view` 和一个 Copy；src 有 storage binding、dst 未绑定。预期 pass 后 dst 获得数学等价 binding。另一个 split case 给 src/dst 不兼容 hard seed，预期诊断包含两条 provenance。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，pass 仍只写 smoke attr。

- [X] **Step 2: 收集 Storage variables 和 constraints**

Collector 规则：

```text
layout_view with layout -> RequireEncoding(hard)
layout_view without layout -> unresolved Storage var
same-source compatible views -> AliasLayout(hard)
copy whole-tile src/dst -> StorageAccess(hard relation) + coalescing Preference
```

Task 12 只接受 whole-tile、静态 shape copy；其他 case 返回明确 unsupported diagnostic。

- [X] **Step 3: 实现 bootstrap SM90 storage candidates**

`SM90LayoutTarget` 首版只生成：linear、transpose、必要 padding、32B/64B/128B XOR swizzle。每个候选必须先通过 Storage verifier；候选按固定枚举序号排序。

这里的 candidate generation 仅用于初始化尚无 hard seed 的变量 domain；它发生在 fixed-point propagation 之前，不执行全局优选。完整候选生成、剪枝和 beam search 由 Task 23 接管。

- [X] **Step 4: 实现单调传播**

Strict 只处理 singleton seed、SameLayout 和 exact AliasLayout。Common 只执行交集/关系投影：

```cpp
while (!worklist.empty()) {
  LayoutConstraintID id = worklist.pop_front();
  ChangeResult changed = applyConstraint(graph, id);
  if (changed == ChangeResult::Change)
    enqueueAdjacentConstraintsInStableOrder(graph, id, worklist);
}
```

空集合进入 Conflict 并输出两条 seed provenance；传播循环内部不得新增 candidate。

- [X] **Step 5: 验证收敛和顺序独立**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutPropagationTest.*'
```

Expected: PASS；随机打乱初始 constraint vector 后 canonical graph dump 不变。

- [X] **Step 6: 提交 storage propagation**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk test unittests
git commit -m "feat: propagate storage layout constraints"
```

### Task 13: 物化 Storage binding 并验证第一条端到端链路

**Files:**

- Create: `include/Dialect/Frisk/Analysis/LayoutVerifier.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutVerifier.cpp`
- Create: `lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt`
- Create: `test/Transforms/storage-materialization.mlir`
- Create: `test/Transforms/storage-conflict.mlir`
- Create: `unittests/Dialect/Frisk/Layout/LayoutVerifierTest.cpp`

**Interfaces:**

- Produces:

```cpp
struct LayoutSolution {
  DenseMap<LayoutVarID, Attribute> assignments;
  SmallVector<LayoutConversionEdge> conversions;
};

struct BootstrapSolverLimits {
  unsigned maxVariables = 8;
  unsigned maxDomainSize = 4;
};

FailureOr<LayoutSolution> solveBootstrapLayoutGraph(
    LayoutConstraintGraph &graph, LayoutTarget &target,
    BootstrapSolverLimits limits = {});

LogicalResult verifySolvedLayoutGraph(const LayoutConstraintGraph &graph,
                                      const LayoutSolution &solution,
                                      LayoutTarget &target,
                                      Location loc);

LogicalResult materializeLayouts(Operation *root,
                                 const LayoutConstraintGraph &graph,
                                 const LayoutSolution &solution);

LogicalResult verifyMaterializedLayouts(Operation *root,
                                        LayoutTarget &target);
```

- [X] **Step 1: 写未物化和错误物化红灯测试**

覆盖 unresolved layout_view、错误 memory space、非单射 map、alias views 不一致。诊断检查：

```mlir
// expected-error@+1 {{unresolved storage layout for layout variable}}
%v = "frisk.layout_view"(%arg0) : (memref<64x64xf16>) -> memref<64x64xf16>
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，verifier/materializer 尚不存在。

- [X] **Step 2: 实现严格受限的 bootstrap resolver**

为使 M2/M3 在完整求解器之前可执行，实现只面向纵向切片的 exhaustive resolver：component 最多 8 个变量、每个 domain 最多 4 个候选；只接受满足全部 hard constraint 的 assignment，并按 candidate stable ordinal 序列决定唯一结果。超过边界直接诊断 `bootstrap layout solver limit exceeded`，不得截断或猜测。

M2 不枚举 conversion/rematerialization；Task 15 扩展该 resolver 处理一个多 consumer conversion edge，Task 23 必须以正式 CostVector/beam solver 替换并删除它。

- [X] **Step 3: 实现 solved graph verifier**

逐 assignment 检查 kind、type、coverage/injectivity、alias 和 Copy relation。失败信息包含稳定 var 名、candidate、constraint rule 和 provenance chain。

- [X] **Step 4: 实现 Storage materialization**

Materializer 只给现有 `LayoutViewOp` 设置求解出的 `layout` attr；若 solution 中的 Attr 与已有 hard binding 不等价则立即失败，不覆盖。

- [X] **Step 5: 将 pass 变成真正 orchestrator**

`FriskInferLayoutsPass::runOnOperation()` 顺序固定为：

```cpp
auto graph = collectLayoutConstraints(getOperation(), target);
propagateStrict(*graph);
propagateCommonToFixedPoint(*graph);
auto solution = solveBootstrapLayoutGraph(*graph, target);
verifySolvedLayoutGraph(*graph, solution, target, getOperation().getLoc());
materializeLayouts(getOperation(), *graph, solution);
verifyMaterializedLayouts(getOperation(), target);
```

任一步失败调用 `signalPassFailure()`；删除 M0 的 smoke attr。

- [X] **Step 6: 运行端到端测试**

Run:

```bash
cmake --build build --target FriskTransforms FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutVerifierTest.*'
```

Expected: Storage 输入推断并打印 canonical binding；冲突 case 输出两条 provenance；所有测试 PASS。

- [X] **Step 7: 同步设计并提交**

如果实际 pass 阶段边界与设计 Section 8 不同，先更新设计文档。

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk test unittests \
  guoqiao/layout_inference_design.md
git commit -m "feat: materialize and verify storage layouts"
```

### M2 Gate

Run:

```bash
build/bin/frisk-opt test/Transforms/storage-materialization.mlir \
  -frisk-infer-layouts -verify-each
cmake --build build --target check-frisk FriskLayoutUnitTests --parallel 32
```

Expected: alloc/view/copy Storage slice 完整通过；`LayoutInfer.cpp` 不再是 pass 壳；无 conversion 或 Tensor 逻辑被提前混入。

---

## M3：Distributed Tensor 与 conversion 纵向切片

### Task 14: 新增 Tensor carrier 和 `frisk.convert_layout`

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutOps.cpp`
- Create: `test/Dialect/Frisk/layout/tile-carriers.mlir`
- Create: `test/Dialect/Frisk/layout/convert-layout.mlir`

**Interfaces:**

- Produces:

```mlir
%tile = frisk.tile_load %view
  : memref<64x64xf16, #frisk<memory_space Shared>>
    -> tensor<64x64xf16>

%converted = frisk.convert_layout %tile
  : tensor<64x64xf16, #src> -> tensor<64x64xf16, #dst>

frisk.tile_store %tile, %view
  : tensor<64x64xf16, #enc>,
    memref<64x64xf16, #frisk<memory_space Shared>>
```

首版 load/store 只表示 whole-tile、静态 shape；切片和动态 offset 在 M4 后扩展。

- [ ] **Step 1: 写 Op verifier 红灯测试**

覆盖：load shape/dtype 不匹配、store shape 不匹配、convert source/target shape 或 element type 不同、identity convert。

```mlir
// expected-error@+1 {{source and target must have identical shape and element type}}
%bad = "frisk.convert_layout"(%x) :
  (tensor<64x64xf16, #src>) -> tensor<32x128xf16, #dst>
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，新 Ops 尚不存在。

- [ ] **Step 2: 定义 carrier Ops**

`TileLoadOp` 具有 MemRead，`TileStoreOp` 具有 MemWrite，`ConvertLayoutOp` 为 pure。Convert verifier 要求 source/target 均有合法 `DistributedEncodingAttr` 且不相等；identity 输入交给 canonicalizer 删除，parser verifier 测试使用 canonicalization 前合法表示时不得崩溃。

- [ ] **Step 3: 实现基础 canonicalization**

```text
convert_layout(x, src == dst) -> x
convert_layout(convert_layout(x, A -> B), B -> A) -> x
```

第二条仅在没有 side effect 且 canonical map 精确等价时应用。

- [ ] **Step 4: 运行 IR tests**

Run:

```bash
cmake --build build --target FriskIR check-frisk --parallel 32
```

Expected: carrier/convert tests PASS。

- [ ] **Step 5: 提交 Tensor carrier**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk/IR test
git commit -m "feat: add distributed tensor carriers"
```

### Task 15: 收集 Distributed constraints 并支持多 consumer

**Files:**

- Modify: `lib/Dialect/Frisk/Analysis/LayoutConstraint.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutPropagation.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`
- Create: `unittests/Dialect/Frisk/Layout/DistributedPropagationTest.cpp`
- Create: `test/Transforms/distributed-propagation.mlir`
- Create: `test/Transforms/multi-consumer-layout.mlir`

**Interfaces:**

- Extends:

```cpp
LayoutVarID LayoutConstraintBuilder::getOrCreateDistributedVar(Value value);
LogicalResult LayoutConstraintBuilder::transform(
    LayoutVarID src, LayoutVarID dst, Attribute coordinateTransform,
    Operation *source, StringRef rule);
LogicalResult LayoutConstraintBuilder::storageAccess(
    LayoutVarID distributed, LayoutVarID storage, AccessKind access,
    Operation *source, StringRef rule);

FailureOr<LayoutSolution> solveBootstrapLayoutGraph(
    LayoutConstraintGraph &graph, LayoutTarget &target,
    BootstrapSolverLimits limits);
```

- [ ] **Step 1: 写双 consumer 红灯测试**

构造一个 `tile_load` 结果被两个 `tile_store` 使用，两个 destination view 分别绑定 linear 和 transpose storage。预期 analysis 保留两个 distributed candidate，而不是第一个 consumer 锁定结果。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，collector 尚未创建 Distributed var。

- [ ] **Step 2: 收集基础 Distributed rules**

规则固定为：

```text
tile_load/store -> StorageAccess
arith/math elementwise tensor op -> SameLayout(all tensor operands/results)
tensor.transpose -> TransformLayout(permutation)
existing tensor encoding -> RequireEncoding(hard)
existing convert_layout -> source/target RequireEncoding(hard) + Convertible edge
```

对于未知 layout-bearing Op，pass 必须报 `operation has no layout constraint model`。

- [ ] **Step 3: 生成基础 distributed candidates**

SM90 bootstrap candidates 只包含 blocked、lane-striped、warp-striped、fully-replicated；每个候选展开为 canonical map 并通过 type verifier。

- [ ] **Step 4: 扩展 propagation 为双向关系投影**

StorageAccess 和 transpose 必须同时支持 producer→consumer、consumer→producer。候选集合只缩小；多 consumer 的 union-of-requirements 在 solve 前保留为 candidate alternatives，不得原地覆盖。

- [ ] **Step 5: 扩展 bootstrap resolver，闭合第一条多 consumer 路径**

在既有 8-variable/4-candidate 上限内，允许每个可转换 consumer edge 枚举 `KeepCommonLayout` 或 `Convert`。目标顺序固定为：先满足 hard constraint，再最小化 conversion 数，最后比较 assignment/edge stable key；本阶段不宣称性能最优，也不枚举 rematerialization。求解结果必须把选中的 conversion 写入 `LayoutSolution::conversions`，供 Task 16 直接消费。

- [ ] **Step 6: 验证 fixed-point 与多 consumer**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='DistributedPropagationTest.*'
```

Expected: PASS；两个 consumer 候选均存在；bootstrap solution 明确选择共同布局或单个 consumer-edge conversion；顺序打乱结果不变。

- [ ] **Step 7: 提交 Distributed propagation**

```bash
git add lib/Dialect/Frisk test unittests
git commit -m "feat: propagate distributed layout constraints"
```

### Task 16: 物化 Tensor encoding、SCF 类型和 conversion edge

**Files:**

- Modify: `lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp`
- Create: `include/Dialect/Frisk/Transforms/LayoutTypeConverter.h`
- Create: `lib/Dialect/Frisk/Transforms/LayoutTypeConverter.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt`
- Create: `test/Transforms/materialize-distributed.mlir`
- Create: `test/Transforms/materialize-scf.mlir`
- Create: `test/Transforms/materialize-conversion.mlir`

**Interfaces:**

- Produces:

```cpp
class LayoutTypeConverter : public TypeConverter {
public:
  LayoutTypeConverter(const LayoutConstraintGraph &graph,
                      const LayoutSolution &solution);
  FailureOr<RankedTensorType> convertLayoutBearingTensor(Value value) const;
};

LogicalResult materializeDistributedLayouts(
    Operation *root, const LayoutConstraintGraph &graph,
    const LayoutSolution &solution,
    ArrayRef<LayoutConversionEdge> conversions);
```

- [ ] **Step 1: 写 SCF 和 conversion 红灯测试**

覆盖 `scf.if` 两个 yield、`scf.for` init/block argument/yield/result，以及两个 consumer 需要不同 encoding 时在 consumer edge 前出现一次 conversion。

```mlir
// CHECK: %[[CVT:.*]] = frisk.convert_layout %[[PRODUCER]]
// CHECK: frisk.tile_store %[[CVT]], %[[DST]]
```

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，Tensor type 尚未被改写。

- [ ] **Step 2: 实现非 region Op 的 type rewrite**

按 dominance order 克隆需要更改 result type 的 Op，使用 `IRMapping` 替换 operands/results；不得直接让 result type 与 Op verifier 暂时不一致。

- [ ] **Step 3: 实现 SCF 一致重写**

固定规则：

```text
scf.if: each yield operand encoding == corresponding result encoding
scf.for: init == iter_arg == yield == result
scf.while: before args == condition args == after args == results
```

若分支内部需要转换，在 yield 前插入；不得修改 join 后类型逃避 hard constraint。

- [ ] **Step 4: 物化 solver 选中的 conversion edge**

只有 `LayoutSolution` 显式列出的 edge 才插入 `frisk.convert_layout`。source/target 相等时视为 solver/materializer bug 并失败；materializer 不重新比较成本。

- [ ] **Step 5: 运行 materialization tests**

Run:

```bash
cmake --build build --target FriskTransforms check-frisk --parallel 32
```

Expected: Tensor/SCF/conversion tests PASS，`-verify-each` 无错误。

- [ ] **Step 6: 提交 Distributed materialization**

```bash
git add include/Dialect/Frisk/Transforms lib/Dialect/Frisk/Transforms test
git commit -m "feat: materialize distributed layouts and conversions"
```

### Task 17: 实现基础 conversion cleanup 和最小测试 lowering adapter

**Files:**

- Create: `lib/Dialect/Frisk/Transforms/OptimizeLayoutConversions.cpp`
- Modify: `include/Dialect/Frisk/Transforms/Passes.td`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt`
- Create: `lib/Conversion/FriskLayoutToGPU/CMakeLists.txt`
- Create: `lib/Conversion/FriskLayoutToGPU/TestLayoutConversionLowering.cpp`
- Modify: `lib/Conversion/CMakeLists.txt`
- Create: `test/Transforms/optimize-layout-conversions.mlir`
- Create: `test/Conversion/FriskLayoutToGPU/warp-shuffle.mlir`
- Create: `test/Conversion/FriskLayoutToGPU/shared-exchange.mlir`

**Interfaces:**

- Produces:

```cpp
std::unique_ptr<Pass> createOptimizeLayoutConversionsPass();
std::unique_ptr<Pass> createTestLowerLayoutConversionsPass();
```

测试 lowering 仅支持静态、单 CTA、BitLinear→BitLinear redistribution：warp 内优先用 shuffle，跨 warp 使用 workgroup scratch + barrier exchange。动态 carrier、跨 CTA 和需要 cluster 通信的 conversion 明确报 unsupported。

- [ ] **Step 1: 写 cleanup/lowering 红灯测试**

覆盖 identity、A→B→A、相邻 A→B→C 合并，一个 lane permutation conversion 产生 `gpu.shuffle`，以及一个跨 warp permutation 产生 workgroup scratch store/barrier/load。动态 carrier 预期诊断 `test lowering requires a static single-CTA redistribution`。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，passes 尚不存在。

- [ ] **Step 2: 实现 type-safe cleanup patterns**

只实现数学上可证明的：identity elimination、相邻 conversion compose、inverse pair elimination。Pattern 必须调用 canonical map equality，不比较 Attr 指针。

- [ ] **Step 3: 实现单 warp redistribution 计划**

计算：

```text
R = rightInverse(D_src) compose D_dst
```

对每个目标 carrier 得到 source lane/register；若 source 不覆盖目标 logical element，lowering 失败。生成 `gpu.shuffle` 前验证所有 lane index 在 `[0, 31]`。

- [ ] **Step 4: 实现单 CTA shared exchange**

先用 canonical maps 为每个 live logical element证明唯一 source owner 和所有 destination owners，再以 logical linear index 分配 scratch slot：source owner 写入，执行一次 `gpu.barrier`，destination owner 读取。scratch 字节数按 element bit width、tile volume 和 alignment 精确计算；sub-byte 暂不支持。若存在缺失 source、重复 writer、越界 slot 或 scratch 超过 target limit，lowering 必须失败并打印首个 logical coordinate 反例。

- [ ] **Step 5: 注册 passes 并运行测试**

Run:

```bash
cmake --build build --target FriskTransforms check-frisk --parallel 32
```

Expected: cleanup、single-warp 和 shared-exchange FileCheck PASS；unsupported case 使用预期诊断失败；生成 IR 通过 `-verify-each`。

- [ ] **Step 6: 提交 conversion MVP**

```bash
git add include/Dialect/Frisk/Transforms lib/Conversion \
  lib/Dialect/Frisk/Transforms test
git commit -m "feat: optimize and test-lower layout conversions"
```

### M3 Gate

Run:

```bash
cmake --build build --target check-frisk FriskLayoutUnitTests --parallel 32
build/bin/frisk-opt test/Transforms/multi-consumer-layout.mlir \
  -frisk-infer-layouts -frisk-optimize-layout-conversions -verify-each
```

Expected: 多 consumer IR 合法；共同布局或 conversion 由 solution 明确决定；SCF 类型一致；单 warp 与单 CTA shared-exchange conversion 可以 lower，非支持路径明确失败。

---

## M4：完整约束传播与旧 IR 迁移

### Task 18: 完成 alias/view 与 region graph

**Files:**

- Modify: `lib/Dialect/Frisk/Analysis/LayoutConstraint.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutPropagation.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutVerifier.cpp`
- Create: `unittests/Dialect/Frisk/Layout/AliasRegionTest.cpp`
- Create: `test/Transforms/layout-alias.mlir`
- Create: `test/Transforms/layout-scf-fixed-point.mlir`

**Interfaces:**

- Extends graph with：

```cpp
struct RegionLayoutEdge {
  Value incoming;
  BlockArgument blockArgument;
  Value yielded;
  Value result;
};

LogicalResult collectAliasGroups(Operation *root,
                                 LayoutConstraintBuilder &builder);
LogicalResult collectRegionEdges(Operation *root,
                                 LayoutConstraintBuilder &builder);
```

- [ ] **Step 1: 写 alias/region 红灯测试**

覆盖 nested layout_view、memref.cast/subview、`scf.if` join、`scf.for` loop-carried tensor、循环回边需要多轮才能收敛，以及两个 alias storage seed 冲突。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: 至少一个 case FAIL 或 unresolved。

- [ ] **Step 2: 建立 storage alias groups**

使用 `ViewLikeOpInterface` 和 MLIR alias/view flow；同一底层 allocation 的 views 通过 `AliasLayout` 关联。关系保存 logical coordinate transform，不简单要求 Attr 文本相等。

- [ ] **Step 3: 建立 structured region edges**

为 `scf.if/for/while` 收集 block argument、yield、condition/result 映射。Region edge 是 hard constraint；转换只能插在 branch/yield 边界，不能跨越 side-effecting op。

- [ ] **Step 4: 验证循环 fixed-point**

Worklist 必须在 loop backedge 上收敛。记录初始候选总数、成功删除候选数和 queue pop 数：成功删除数不得超过初始候选总数；queue pop 数不得超过 `初始约束数 + Σ(每次 domain change 的相邻约束数)`。测试按这一由单调 lattice 和实际 degree 推导的上界断言，不使用固定“最多循环 N 次”掩盖振荡。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='AliasRegionTest.*'
```

Expected: PASS。

- [ ] **Step 5: 提交 alias/region 支持**

```bash
git add lib/Dialect/Frisk test unittests
git commit -m "feat: propagate layouts through aliases and regions"
```

### Task 19: 迁移 Copy、Fill 和 Parallel 约束

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskOps.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90CopyConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/CMakeLists.txt`
- Create: `test/Transforms/infer-copy-fill-parallel.mlir`
- Create: `unittests/Dialect/Frisk/Layout/CopyFillConstraintTest.cpp`

**Interfaces:**

- Produces：

```cpp
LogicalResult collectCopyConstraints(CopyOp op,
                                     LayoutConstraintBuilder &builder,
                                     LayoutTarget &target);
LogicalResult collectFillConstraints(FillOp op,
                                     LayoutConstraintBuilder &builder);
LogicalResult collectParallelConstraints(ParallelOp op,
                                         LayoutConstraintBuilder &builder);
```

- [ ] **Step 1: 写正反向推断红灯测试**

Copy 分别测试已知 src 推 dst、已知 dst 反推 src；Fill 测试 unique writer/replication；Parallel 测试 threads=128 seed warp-group topology，threads 与 encoding 不一致时报错。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，collector 尚未覆盖三类 Op。

- [ ] **Step 2: 实现 Copy 约束**

Copy 根据 operand role 创建 Storage 或 Distributed var，建立 `StorageAccess`/`SameLayout`/`Preference`。首版 vector width 只接受 `{1, 2, 4, 8, 16}` bytes，alignment 不满足时删除候选，不修改 shape。

- [ ] **Step 3: 实现 Fill/Parallel 约束**

Fill 对每个 live logical point要求唯一 writer，允许明确 replicated read 但不允许 replicated store。Parallel 将 thread count/topology 写入 ResourceLimit/Ownership，不直接生成最终 layout。

- [ ] **Step 4: 删除 Parallel 的递归 `DenseMap` 调度使用**

保留旧方法定义供 adapter tests 使用，但生产 `FriskInferLayoutsPass` 不再调用 `ParallelOp::inferLayout`。增加测试统计 legacy 方法调用次数为 0。

- [ ] **Step 5: 运行迁移测试**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
ctest --test-dir build --output-on-failure
```

Expected: Copy/Fill/Parallel 新测试和 legacy tests 均 PASS。

- [ ] **Step 6: 提交基础 Op 迁移**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk test unittests
git commit -m "feat: migrate copy fill and parallel constraints"
```

### Task 20: 增加内部 Tensor MMA Op 并迁移 Gemm 约束

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutOps.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90GemmConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/GemmConstraintTest.cpp`
- Create: `test/Transforms/infer-gemm-layout.mlir`

**Interfaces:**

- Produces internal operand type constraint and op：

```tablegen
def Frisk_MemRefOrRankedTensor :
    AnyTypeOf<[AnyMemRef, AnyRankedTensor]>;
```

```mlir
%result = frisk.mma %a, %b, %init
  {m = 128, n = 128, k = 64, trans_a = false, trans_b = false,
   policy = #frisk<gemm_warp_policy Square>}
  : (memref<128x64xbf16, #frisk<memory_space Shared>>,
     memref<64x128xbf16, #frisk<memory_space Shared>>,
     tensor<128x128xf32>) -> tensor<128x128xf32>
```

```cpp
LogicalResult collectGemmConstraints(MmaOp op,
                                     LayoutConstraintBuilder &builder,
                                     LayoutTarget &target);
```

- [ ] **Step 1: 写 MMA verifier 和 constraint 红灯测试**

覆盖 BF16/FP16 `m64nNxk16` SM90 候选、A/B major/transpose、accumulator shape、128-thread warp-group、非法 dtype/shape，以及不同 warp policy。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，内部 MMA Op/constraints 尚不存在。

- [ ] **Step 2: 定义 target-neutral `MmaOp`**

Op 只保存数学 tile 语义和 policy，不保存 PTX mnemonic。A/B 可为 Shared MemRef view 或 RankedTensor；init/result 必须是 RankedTensor。Verifier 只检查 shape/dtype 的数学一致性，SM90 合法性放 target rules。

- [ ] **Step 3: 从现有 Gemm 逻辑提取 canonical candidates**

将 `FriskOps.cpp:1285-1406` 的 target/layout 构造移动到 `SM90GemmConstraints.cpp`，输出 `InstructionContract`、`RequireEncoding` 候选和 provenance；不得直接写 layout map。

```text
A/B shared -> Storage candidates + WGMMA descriptor contract
A/B tensor -> DotOperand distributed candidates
result/init -> Mma/Distributed accumulator candidates
```

- [ ] **Step 4: 与 legacy Gemm 枚举差分**

对现有 `sm80_ss`、`sm90_ss` 和 local/shared 组合，比较新候选中至少存在一个与 legacy map 数学等价的 assignment；新系统可另外保留更优候选。

- [ ] **Step 5: 运行 tests**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='GemmConstraintTest.*'
```

Expected: PASS；非法 SM90 contract 输出目标规则名称。

- [ ] **Step 6: 提交 Gemm 迁移**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk test unittests
git commit -m "feat: model gemm as layout constraints"
```

### Task 21: 增加 Tensor Reduce Op 并迁移 Reduce ownership

**Files:**

- Modify: `include/Dialect/Frisk/IR/FriskLayoutOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskLayoutOps.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90ReduceConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/ReduceConstraintTest.cpp`
- Create: `test/Transforms/infer-reduce-layout.mlir`

**Interfaces:**

- Produces：

```mlir
%result = frisk.reduce_tensor %src
  {kind = "sum", dim = 1}
  : tensor<64x64xf32> -> tensor<64xf32>
```

```cpp
LogicalResult collectReduceConstraints(ReduceTensorOp op,
                                       LayoutConstraintBuilder &builder,
                                       LayoutTarget &target);
```

- [ ] **Step 1: 写 ownership 红灯测试**

覆盖 reduce dim projection、lane shuffle、replicated result、elected owner、thread count 与 replicate 不整除、非 2 次幂 reduce extent 的 guarded/shared fallback。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，新 Reduce Op/constraints 不存在。

- [ ] **Step 2: 定义 `ReduceTensorOp`**

Verifier 检查 result shape 等于删除 reduce dim 后的 shape、dtype/kind 支持、dim 合法。Op 不直接选择 shuffle/shared tree。

- [ ] **Step 3: 将现有 Reduce map 逻辑转成 Transform/Ownership constraints**

`FriskOps_Reduce.cpp:235-369` 的 dimension projection、replicate condense 语义迁到 target-neutral transform；SM90 rule 决定 shuffle/shared tree candidate。无法证明 unique owner 时删除 store candidate并给反例。

- [ ] **Step 4: 与 legacy Reduce cases 差分**

复用 `reduce_layout_test.cpp` case table，比较 output coordinate、replication 和 owner，而不是 Attr 文本。

- [ ] **Step 5: 运行 tests**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='ReduceConstraintTest.*'
```

Expected: PASS；legacy executable 仍 PASS。

- [ ] **Step 6: 提交 Reduce 迁移**

```bash
git add include/Dialect/Frisk/IR lib/Dialect/Frisk test unittests
git commit -m "feat: model reduce ownership constraints"
```

### Task 22: 实现 legacy normalization 并退役旧生产路径

**Files:**

- Create: `lib/Dialect/Frisk/Transforms/NormalizeLayoutIR.cpp`
- Create: `include/Dialect/Frisk/Transforms/PassPipelines.h`
- Create: `lib/Dialect/Frisk/Transforms/PassPipelines.cpp`
- Modify: `include/Dialect/Frisk/Transforms/Passes.td`
- Modify: `lib/Dialect/Frisk/Transforms/CMakeLists.txt`
- Modify: `include/Dialect/Frisk/IR/FriskOps.td`
- Modify: `lib/Dialect/Frisk/IR/FriskOps.cpp`
- Modify: `lib/Dialect/Frisk/IR/FriskOps_Reduce.cpp`
- Modify: `include/Dialect/Frisk/IR/FriskInterfaces.td`
- Create: `test/Transforms/normalize-legacy-layout-ir.mlir`
- Create: `test/Transforms/no-legacy-layout-map.mlir`

**Interfaces:**

- Produces: `std::unique_ptr<Pass> createNormalizeLayoutIRPass()`，参数 `frisk-normalize-layout-ir`。
- Produces: `void buildFriskLayoutPipeline(OpPassManager &pm)`，注册参数 `frisk-layout-pipeline`。
- Pipeline contract:

```text
legacy Buffer IR
  -> frisk-normalize-layout-ir
  -> Tensor Mma/Reduce + layout_view/tile load/store
  -> frisk-infer-layouts
```

- [ ] **Step 1: 写 normalization 红灯测试**

输入使用现有 local MemRef Gemm C、Reduce src/dst；输出必须含 `frisk.mma`、`frisk.reduce_tensor`、Tensor SSA 和必要 bridge，不得含 local fragment 作为新 solver LayoutVar。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，normalization pass 尚不存在。

- [ ] **Step 2: 实现 local MemRef 到 Tensor SSA normalization**

按 def-use slice 重写完整 fragment 生命周期。仅支持可证明无 alias escape 的 local alloc；地址被未知 Op 使用时诊断：

```text
legacy local fragment escapes supported normalization boundary
```

不得部分转换同一 alias group。

- [ ] **Step 3: 注册默认布局 pipeline**

在 `PassPipelines.h/.cpp` 提供：

```cpp
void buildFriskLayoutPipeline(OpPassManager &pm) {
  pm.addPass(createNormalizeLayoutIRPass());
  pm.addPass(createFriskInferLayoutsPass());
  pm.addPass(createOptimizeLayoutConversionsPass());
}
```

在 pass 注册实现中增加：

```cpp
PassPipelineRegistration<> layoutPipeline(
    "frisk-layout-pipeline",
    "Normalize, infer, materialize, and optimize Frisk layouts",
    [](OpPassManager &pm) { buildFriskLayoutPipeline(pm); });
```

`frisk-opt` 启动时调用 `registerFriskPasses()` 后必须同时注册上述 pipeline；`PassPipelines.cpp` 加入 `FriskTransforms` source list，不能依赖静态初始化碰巧被链接保留。

- [ ] **Step 4: 删除旧生产接口**

从 ODS/C++ 删除 `ParallelOp::inferLayout`、`GemmOp::inferLayout`、`ReduceOp::inferLayout` 及旧 `LayoutInterface::inferLayout(DenseMap&)`。Legacy adapter 和 baseline oracle 保留在 Analysis/test，不得被生产 pass 调用。

- [ ] **Step 5: 验证仓库无生产 DenseMap 路径**

Run:

```bash
rg -n "inferLayout.*DenseMap|DenseMap<Value, Attribute>.*layout" \
  include/Dialect lib/Dialect
```

Expected: 无匹配；允许匹配仅存在于 `LegacyLayoutAdapter` 和 test 文件时，应使用更窄路径再次确认生产目录为 0。

- [ ] **Step 6: 运行 M4 全量回归**

Run:

```bash
cmake --build build --target FriskIR FriskTransforms FriskLayoutUnitTests check-frisk --parallel 32
ctest --test-dir build --output-on-failure
```

Expected: 所有 tests PASS；legacy input 经 normalization 后通过 `-verify-each`。

- [ ] **Step 7: 同步设计文档并提交**

确认设计 Section 5、7、12 不再把旧 interface 写成生产路径。

```bash
git add include/Dialect/Frisk lib/Dialect/Frisk test \
  guoqiao/layout_inference_design.md
git commit -m "refactor: normalize legacy fragments to tensor layout IR"
```

### M4 Gate

Run:

```bash
build/bin/frisk-opt test/Transforms/no-legacy-layout-map.mlir \
  -frisk-layout-pipeline -verify-each
rg -n "inferLayout.*DenseMap|DenseMap<Value, Attribute>.*layout" \
  include/Dialect lib/Dialect
```

Expected: pipeline 成功；生产目录旧接口匹配数为 0；Copy/Fill/Gemm/Reduce/Parallel/SCF 都由新 graph 处理。

---

## M5：有限候选、SM90 契约与 conversion 优化

### Task 23: 实现连通分量候选求解和 CostVector

**Files:**

- Create: `lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp`
- Create: `lib/Dialect/Frisk/Analysis/LayoutCostModel.cpp`
- Modify: `include/Dialect/Frisk/Analysis/LayoutSolver.h`
- Modify: `include/Dialect/Frisk/Analysis/LayoutTarget.h`
- Modify: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Create: `unittests/Dialect/Frisk/Layout/LayoutCandidateSolverTest.cpp`
- Create: `test/Transforms/layout-candidate-selection.mlir`

**Interfaces:**

- Consumes the `CostVector` definition from Task 5; produces assignment and solver APIs:

```cpp
bool operator<(const CostVector &lhs, const CostVector &rhs);

struct CandidateAssignment {
  DenseMap<LayoutVarID, unsigned> candidateIndex;
  SmallVector<LayoutConversionEdge> conversions;
  CostVector cost;
};

struct SolverOptions {
  unsigned maxCandidatesPerComponent = 256;
  unsigned beamWidth = 32;
};

FailureOr<LayoutSolution> solveLayoutGraph(LayoutConstraintGraph &graph,
                                           LayoutTarget &target,
                                           SolverOptions options);
```

- [ ] **Step 1: 写全局选择红灯测试**

构造三个变量的 component，使逐点最小选择违反中间 SameLayout，而全局 assignment 有唯一合法最小值；另测相同 cost 依赖 stable tie-break，不依赖 candidate 插入顺序。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，candidate solver 不存在。

- [ ] **Step 2: 构建连通分量和 hard pruning**

只使用 hard constraint 建 component。候选先 canonical hash 去重，再逐个验证；pruning 必须记录：

```cpp
struct CandidateRejection {
  LayoutVarID var;
  unsigned candidateIndex;
  LayoutConstraintID constraint;
  std::string reason;
};
```

正式 solver 必须先覆盖 bootstrap 已支持的 `KeepCommonLayout/Convert` edge resolution：从 edge 中已有的 bytes/synchronization 上界构造保守 conversion cost，并保持 M3 的 conversion materialization 接口。完成生产接线后，删除 `solveBootstrapLayoutGraph`、`BootstrapSolverLimits` 及生产 pass 对它们的调用；用相同的 M2/M3 case 验证正式 solver 保持语义结果，并允许因完整 CostVector 选择更优的合法 assignment。Rematerialization 和更精确的 critical-path conversion cost 留到 Task 24。

- [ ] **Step 3: 实现有上限 beam search**

变量选择顺序固定为：candidate 数最少、hard degree 最大、stable ID 最小。每扩展一个变量立即做 partial hard check 和 lower-bound cost；beam 使用 `CostVector + stable assignment key` 排序。

超过 256 个原始组合时不静默截断：打印 component ID、vars、每个 domain size、pruning 数和 beamWidth。

- [ ] **Step 4: 实现 CostVector 比较**

非法候选在进入 cost 前已经删除。比较顺序按照字段声明；`deterministicTieBreak` 只能解决其他字段完全相等，不能编码性能偏好。

- [ ] **Step 5: 验证最优性和确定性**

对总组合数不超过 4096 的随机小图，用 exhaustive solver 作为 oracle，比较 beam/exact assignment；首版测试图设置 beamWidth 足够覆盖 exact solution。

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutCandidateSolverTest.*'
```

Expected: PASS。

- [ ] **Step 6: 提交候选 solver**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis test unittests
git commit -m "feat: select global layout candidate assignments"
```

### Task 24: 将 conversion placement 和 rematerialization 纳入普通候选

**Files:**

- Modify: `include/Dialect/Frisk/Analysis/LayoutConstraint.h`
- Modify: `include/Dialect/Frisk/Analysis/LayoutSolver.h`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/LayoutCostModel.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutConversionChoiceTest.cpp`
- Create: `test/Transforms/layout-conversion-choice.mlir`

**Interfaces:**

- Produces:

```cpp
enum class EdgeResolutionKind { KeepCommonLayout, Convert, Rematerialize };

struct LayoutConversionEdge {
  OpOperand *use;
  Attribute sourceEncoding;
  Attribute targetEncoding;
  EdgeResolutionKind resolution;
  uint64_t bytes;
  uint64_t synchronizationCost;
};

bool isRematerializable(Operation *op);
FailureOr<CostVector> evaluateConversionEdge(const LayoutConversionEdge &edge,
                                             LayoutTarget &target);
```

- [ ] **Step 1: 写“零转换不是绝对优先”的红灯测试**

构造：

```text
方案 A：零 conversion，instructionPathAndWork = 100
方案 B：一次 conversion，instructionPathAndWork = 10，conversion cost = 8
```

预期选择 B。再构造主路径相同而 conversion 更重的 case，预期选择零 conversion。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，solver 尚未枚举 edge resolution。

- [ ] **Step 2: 枚举 edge resolution**

对于 `Convertible` soft conflict 同时生成：共同布局、consumer-edge conversion、pure producer rematerialization。Hard seed 冲突不得通过 conversion 掩盖。

- [ ] **Step 3: 计算 conversion 成本**

至少包含元素 bit 数、tile volume、是否跨 warp、是否需要 shared staging、barrier 数和所在 loop depth。未知 lowering 使用保守上界并标记 provenance，不能记为 0。

- [ ] **Step 4: 限制 rematerialization**

只允许 `MemoryEffectOpInterface` 证明无副作用、无 region、无随机/时钟语义、成本低于 conversion 的 backward slice。最大 slice op 数首版设为 8；超过时不枚举。

- [ ] **Step 5: 验证选择与 IR 物化**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutConversionChoiceTest.*'
```

Expected: 两个成本方向测试均 PASS；FileCheck 中 conversion 数和位置与 solution 一致。

- [ ] **Step 6: 同步 conversion 设计并提交**

确认 `layout_inference_design.md` Section 8.7–8.9 与实际普通候选策略一致。

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis \
  test unittests guoqiao/layout_inference_design.md
git commit -m "feat: solve layout conversion placement globally"
```

### Task 25: 完成数据驱动的 SM90 WGMMA/TMA 布局契约

**Files:**

- Modify: `lib/Dialect/Frisk/Target/SM90/SM90LayoutTarget.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90CopyConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90GemmConstraints.cpp`
- Modify: `lib/Dialect/Frisk/Target/SM90/SM90ReduceConstraints.cpp`
- Create: `lib/Dialect/Frisk/Target/SM90/SM90CostModel.cpp`
- Create: `include/Dialect/Frisk/Target/SM90/SM90LayoutTarget.h`
- Create: `unittests/Dialect/Frisk/Layout/SM90ContractTest.cpp`
- Create: `test/Transforms/sm90-layout-contracts.mlir`

**Interfaces:**

- Produces数据表：

```cpp
enum class SM90ElementType { F16, BF16, F32, TF32, I8 };

struct WGMMAContract {
  int64_t m;
  int64_t n;
  int64_t k;
  SM90ElementType aType;
  SM90ElementType bType;
  SM90ElementType accType;
  bool transA;
  bool transB;
  unsigned threads = 128;
};

struct TMAContract {
  unsigned rank;
  unsigned swizzleBytes;
  unsigned baseAlignmentBytes;
  unsigned innermostGranularityBytes;
};
```

- [ ] **Step 1: 写已知合法/非法契约红灯测试**

至少覆盖 BF16/FP16 WGMMA、A/B major mode、128-thread warp-group、32B/64B/128B swizzle、128B base alignment、TMA box/granularity、ragged fallback。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，规则表/成本尚不完整。

- [ ] **Step 2: 固化 target 查询**

`SM90LayoutTarget` 从 `#nvvm.target`/DLTI 查询 chip/features；迁移期允许读取 `frisk.target = "sm_90"`，但立即规范化为 target object。通用 Analysis 不允许比较字符串。

- [ ] **Step 3: 实现 WGMMA candidate expansion**

每条 contract 展开为 accumulator、dot operand、shared storage canonical maps；展开后调用通用 verifier。MMA Attr 只是 contract view，必须能返回 canonical `DistributedEncodingAttr`。

- [ ] **Step 4: 实现 TMA/Copy candidates**

TMA 不合法时保留 cp.async/vector copy candidate；不能把 fallback 当 hard error。Swizzle/padding 必须同时计入 allocation size、bank 和 descriptor legality。

- [ ] **Step 5: 实现 shape-aware target cost**

成本来自实际 tile volume、transaction、bank、barrier 和资源估计，不使用一个对所有 shape 固定的“WGMMA 永远优先”常数。Static contract 未命中时只删除对应 candidate，不影响仍正确的 fallback。

- [ ] **Step 6: 与现有 layout tool/legacy baseline 差分**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests frisk_layout_tool check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='SM90ContractTest.*'
./build/exp/layout/frisk_layout_tool \
  --target=sm_90 --block-m=128 --block-n=128 --block-k=64 \
  --threads=128 --a-space=shared --b-space=shared --dtype=fp16 --ldmatrix
```

Expected: tests PASS；核心 baseline canonical map 等价；非法 TMA case 仍有合法 fallback。

- [ ] **Step 7: 提交 SM90 规则库**

```bash
git add include/Dialect/Frisk/Target lib/Dialect/Frisk/Target test unittests
git commit -m "feat: add SM90 layout contract library"
```

### Task 26: 实现 controlled relaxation、conversion hoist 和受控重计算

**Files:**

- Modify: `lib/Dialect/Frisk/Analysis/LayoutCandidateSolver.cpp`
- Modify: `lib/Dialect/Frisk/Transforms/OptimizeLayoutConversions.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutRelaxationTest.cpp`
- Create: `test/Transforms/layout-relaxation.mlir`
- Create: `test/Transforms/hoist-layout-conversion.mlir`

**Interfaces:**

- Produces:

```cpp
enum class RelaxationKind {
  Replication,
  StoragePaddingOrNarrowerVector,
  GuardedRagged,
  TMAFallback,
  WGMMAFallback
};

FailureOr<LayoutSolution> attemptControlledRelaxation(
    LayoutConstraintGraph &graph, LayoutTarget &target,
    SolverOptions options, SmallVectorImpl<RelaxationRecord> &records);
```

- [ ] **Step 1: 写 relaxation 顺序红灯测试**

覆盖：普通 conversion 候选存在时不进入 relaxation；TMA alignment 不满足选择 cp.async；ragged shape 选择 guarded candidate；WGMMA contract 合法时不得为减少 conversion 回退 SIMT。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，relaxation records 不存在。

- [ ] **Step 2: 实现严格有序 relaxation**

仅当普通 candidate domain 为空时依次尝试：replication、padding/窄 vector、guarded ragged、TMA fallback、最后 WGMMA illegal fallback。每一步从原始 graph 快照重建 domain，不能累积未选松弛。

- [ ] **Step 3: 记录原因和增量成本**

```cpp
struct RelaxationRecord {
  RelaxationKind kind;
  LayoutConstraintID trigger;
  CostVector incrementalCost;
  std::string reason;
};
```

诊断/debug IR 可打印 records；release IR 不保留非必要 debug attr。

- [ ] **Step 4: 实现 conversion hoist**

允许越过 pure elementwise、broadcast/extend 和 loop-invariant producer；不得越过 memory effects、未知 region、barrier 或 ownership boundary。Hoist 前后都运行 dominance 和 verifier。

- [ ] **Step 5: 实现受控 rematerialization rewrite**

只重写 solver 已选择 `Rematerialize` 的 slice；使用 `IRMapping`，共享已有 rematerialized value，避免指数复制；完成后删除死 conversion 并运行 canonicalizer。

- [ ] **Step 6: 验证 relaxation/hoist**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutRelaxationTest.*'
```

Expected: PASS；每个 case 的 relaxation kind、conversion count 和位置稳定。

- [ ] **Step 7: 同步设计并提交**

```bash
git add lib/Dialect/Frisk test unittests guoqiao/layout_inference_design.md
git commit -m "feat: relax and optimize layout assignments safely"
```

### M5 Gate

Run:

```bash
cmake --build build --target FriskLayoutUnitTests check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
build/bin/frisk-opt test/Transforms/layout-conversion-choice.mlir \
  -frisk-layout-pipeline -verify-each
```

Expected: exhaustive 小图 oracle、全局成本选择、SM90 contracts、conversion/rematerialization 和 relaxation 全部通过；不存在“零转换绝对优先”行为。

---

## M6：正确性、差分测试与性能验收

### Task 27: 建立 property、随机图和确定性测试

**Files:**

- Create: `unittests/Dialect/Frisk/Layout/LayoutPropertyTest.cpp`
- Create: `unittests/Dialect/Frisk/Layout/LayoutSolverPropertyTest.cpp`
- Create: `test/Transforms/layout-diagnostics.mlir`
- Create: `test/Transforms/layout-determinism.mlir`
- Create: `tools/layout-reduce/CMakeLists.txt`
- Create: `tools/layout-reduce/layout-reduce.cpp`
- Modify: `tools/CMakeLists.txt`

**Interfaces:**

- Produces固定 seed property runner 和失败图最小化工具：

```cpp
struct LayoutPropertyOptions {
  uint64_t seed;
  unsigned cases;
  unsigned maxBits;
  unsigned maxVars;
};

FailureOr<LayoutConstraintGraph>
reduceFailingLayoutGraph(const LayoutConstraintGraph &graph,
                         function_ref<bool(const LayoutConstraintGraph &)> fails);
```

- [ ] **Step 1: 写会暴露顺序问题的随机测试**

固定 seeds `{1, 7, 42, 20260816}`，生成最多 8 input bits、8 vars、16 constraints 的可枚举小图；对每张图执行 20 次稳定 shuffle，并比较 canonical solution dump。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: 在最小化工具和完整确定性保障实现前，测试应至少缺少目标或失败。

- [ ] **Step 2: 增加代数 properties**

必须验证：

```text
identity compose A == A
(A compose B)(x) == A(B(x))
inverse(A)(A(x)) == x when invertible
rightInverse covers every required logical point
canonicalize(canonicalize(A)) == canonicalize(A)
Product evaluation == outer/inner staged evaluation
```

- [ ] **Step 3: 增加 solver properties**

小图与 exhaustive solver 比较合法性和最小 CostVector；所有 hard constraint 在 solution 上重新执行；空 domain 必须有至少一个 rejection/provenance record。

- [ ] **Step 4: 实现失败图最小化**

按稳定顺序尝试删除 constraint、var、candidate，保留仍触发同一 failure signature 的最小子图；输出可被 unit test parser 重新加载的文本，不输出地址。

- [ ] **Step 5: 验证 diagnostics 和 determinism**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests layout-reduce check-frisk --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='LayoutPropertyTest.*:LayoutSolverPropertyTest.*'
```

Expected: 所有固定 seed PASS；重复执行 canonical output 字节相同。

- [ ] **Step 6: 提交 property suite**

```bash
git add unittests test tools
git commit -m "test: add layout property and determinism coverage"
```

### Task 28: 建立固定 TileLang 语义差分 corpus

**Files:**

- Create: `tools/layout-reference/export_tilelang_layouts.py`
- Create: `test/Dialect/Frisk/layout/Inputs/tilelang_layout_corpus.json`
- Create: `unittests/Dialect/Frisk/Layout/TileLangDifferentialTest.cpp`
- Create: `guoqiao/tilelang_layout_reference.md`

**Interfaces:**

- Corpus schema：

```json
{
  "schema_version": 1,
  "tilelang_commit": "6623b12d232b343648a5ba99992e3e6f0d6376d2",
  "cases": [
    {
      "name": "sm90_bf16_gemm_128x128x64",
      "shape": [128, 128, 64],
      "dtype": "bf16",
      "carrier_extents": {"register": 8, "lane": 32, "warp": 4},
      "distributed_points": [],
      "storage_points": [],
      "replication": 1,
      "writer_owners": []
    }
  ]
}
```

测试只读取提交到仓库的 JSON，不要求 CI 安装 TileLang。

- [ ] **Step 1: 写 corpus reader 红灯测试**

测试拒绝 schema 版本错误、commit 不匹配、重复 logical point owner 和越界 storage offset。

Run: `cmake --build build --target FriskLayoutUnitTests --parallel 32`。

Expected: FAIL，corpus/reader 尚不存在。

- [ ] **Step 2: 实现固定版本 exporter**

脚本启动时执行：

```python
expected = "6623b12d232b343648a5ba99992e3e6f0d6376d2"
if args.revision != expected:
    raise SystemExit(f"unsupported TileLang revision: {args.revision}")
actual = subprocess.check_output(
    ["git", "-C", args.tilelang, "rev-parse", "HEAD"], text=True
).strip()
if actual != expected:
    raise SystemExit(f"TileLang revision mismatch: {actual} != {expected}")
```

Exporter 输出枚举语义，不输出 TileLang Attr 文本或 Python 对象 repr；JSON key 排序、数字格式稳定。

- [ ] **Step 3: 生成首批 corpus**

至少包含：

```text
SM90 BF16/FP16 GEMM
linear/transpose/32B/64B/128B shared swizzle
copy vector widths
reduce dim 0/1
non-power-of-two broadcast
ragged padding guard
dtype-changing view
unused/floating fragment
owner compatibility
```

不得直接从 `/home/baopeihua/tilelang` 当前 checkout 导入模块。先以 pinned commit 创建 detached worktree，再让 exporter 检查该 worktree 的 `HEAD`：

```bash
git -C /home/baopeihua/tilelang worktree add --detach \
  /tmp/tilelang-layout-reference-6623b12d \
  6623b12d232b343648a5ba99992e3e6f0d6376d2
python tools/layout-reference/export_tilelang_layouts.py \
  --tilelang /tmp/tilelang-layout-reference-6623b12d \
  --revision 6623b12d232b343648a5ba99992e3e6f0d6376d2 \
  --output test/Dialect/Frisk/layout/Inputs/tilelang_layout_corpus.json
```

Expected: exporter 成功，第二次执行产生完全相同文件。生成和审计结束后可执行：

```bash
git -C /home/baopeihua/tilelang worktree remove \
  /tmp/tilelang-layout-reference-6623b12d
```

Exporter 还必须把 pinned worktree 插入独立 Python 进程的首个 import path，并拒绝从系统 site-packages 导入 TileLang；启动日志打印实际导入模块路径和 commit，文档审计时一并核对。

- [ ] **Step 4: 实现语义比较**

Frisk 对每个 case 枚举 canonical map，比较 coverage、logical ownership、replication、writer owner 和 storage bit offset。允许不同构造文本，只比较语义；失败打印首个 carrier/logical coordinate 反例。

- [ ] **Step 5: 运行差分测试**

Run:

```bash
cmake --build build --target FriskLayoutUnitTests --parallel 32
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests \
  --gtest_filter='TileLangDifferentialTest.*'
```

Expected: corpus 全部 PASS。

- [ ] **Step 6: 记录参考策略并提交**

`guoqiao/tilelang_layout_reference.md` 记录 commit、schema、重新生成命令、采用/拒绝的语义和更新审计流程。

```bash
git add tools/layout-reference test/Dialect/Frisk/layout/Inputs \
  unittests/Dialect/Frisk/Layout/TileLangDifferentialTest.cpp \
  guoqiao/tilelang_layout_reference.md
git commit -m "test: add pinned TileLang layout corpus"
```

### Task 29: 建立性能、编译开销守门并完成系统验收

**Files:**

- Modify: `CMakeLists.txt`
- Create: `include/Dialect/Frisk/Analysis/LayoutStatistics.h`
- Create: `lib/Dialect/Frisk/Analysis/LayoutStatistics.cpp`
- Modify: `lib/Dialect/Frisk/Analysis/CMakeLists.txt`
- Modify: `lib/Dialect/Frisk/Transforms/LayoutInfer.cpp`
- Create: `benchmark/CMakeLists.txt`
- Create: `benchmark/layout/CMakeLists.txt`
- Create: `benchmark/layout/solver/CMakeLists.txt`
- Create: `benchmark/layout/solver/layout_solver_bench.cpp`
- Create: `benchmark/layout/conversion/CMakeLists.txt`
- Create: `benchmark/layout/conversion/layout_conversion_bench.cu`
- Create: `benchmark/layout/run_layout_benchmarks.py`
- Create: `benchmark/layout/baseline/sm90_layout_baseline.json`
- Create: `guoqiao/layout_inference_acceptance.md`
- Modify: `guoqiao/layout_inference_design.md` if measured limits require an approved change

**Interfaces:**

- Produces：

```cpp
struct LayoutStatistics {
  uint64_t variables;
  uint64_t constraints;
  uint64_t components;
  uint64_t generatedCandidates;
  uint64_t prunedCandidates;
  uint64_t peakComponentCandidates;
  uint64_t conversions;
  uint64_t conversionBytes;
  uint64_t estimatedBankConflicts;
  uint64_t estimatedMemoryTransactions;
  std::chrono::nanoseconds solveTime;
};
```

Pass option `-frisk-layout-stats-file=<path>` 输出稳定 JSON；默认不写文件。

- [ ] **Step 1: 写 statistics 红灯测试**

用固定小图检查 vars/constraints/candidates/conversions 数量和稳定 JSON key；连续两次运行除时间字段外完全一致。

Run: `cmake --build build --target check-frisk --parallel 32`。

Expected: FAIL，statistics option 尚不存在。

- [ ] **Step 2: 实现统计和 solver microbenchmark**

`layout_solver_bench` 使用固定 corpus，分别测 10、50、100、250 Op graph；预热 5 次、测量 30 次，报告 median/p95、candidate peak 和解 hash。不得在计时区间打印 IR。

- [ ] **Step 3: 实现可选 CUDA conversion microbenchmark**

CMake option：

```cmake
option(FRISK_ENABLE_CUDA_BENCHMARKS
       "Build SM90 layout conversion benchmarks" OFF)
```

启用时要求 `nvcc` 和 SM90；至少比较 identity、单 warp shuffle、shared exchange 三类 conversion。每类 case 同时提供人工编写的 `reference_*` kernel 和通过 `TestLayoutConversionLowering.cpp` 生成/调用的 `inferred_*` 路径；两者使用同一输入和校验器。每个 case 校验输出，再预热 100 次、测量 1000 次；记录 GPU、driver、clock policy 和编译 flags。

顶层仅在 `FRISK_ENABLE_CUDA_BENCHMARKS=ON` 时执行 `enable_language(CUDA)` 并加入 conversion 子目录；默认 CPU/CI 构建不得依赖 CUDA toolkit。

- [ ] **Step 4: 实现 benchmark runner 和 baseline 比较**

Runner 读取/写入：

```json
{
  "environment": {},
  "solver": {},
  "conversions": {},
  "supported_kernels": {}
}
```

baseline JSON 中 conversion/runtime 数值必须来自同一固定环境下的 `reference_*` kernel，不能把新 solver 首次运行结果自封为基线。比较规则：正确性失败立即退出；已有 lowering 的 `inferred_*` 对相应 `reference_*` median 回退超过 3% 失败；布局推断占完整编译时间超过 10% 失败；peak component candidates 超过 256 失败。时间噪声在 3% 内视为持平。

本计划的 runtime 守门范围只覆盖 conversion adapter 和仓库中已经存在、能走新布局 pipeline 的 kernel；尚无 WGMMA/TMA lowering 的路径只做静态 contract/cost 验证，不伪造 runtime 数据。

- [ ] **Step 5: 运行 CPU/solver 验收**

Run:

```bash
cmake --build build --target frisk_layout_solver_bench check-frisk FriskLayoutUnitTests --parallel 32
python benchmark/layout/run_layout_benchmarks.py \
  --solver build/benchmark/layout/solver/frisk_layout_solver_bench \
  --baseline benchmark/layout/baseline/sm90_layout_baseline.json \
  --check
```

Expected: correctness/compile-time/candidate gates PASS。

- [ ] **Step 6: 在 SM90 环境运行 conversion/runtime 验收**

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Release \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir \
  -DFRISK_ENABLE_CUDA_BENCHMARKS=ON
cmake --build build --target frisk_layout_conversion_bench --parallel 32
python benchmark/layout/run_layout_benchmarks.py \
  --conversion build/benchmark/layout/conversion/frisk_layout_conversion_bench \
  --record-reference \
  --output-baseline benchmark/layout/baseline/sm90_layout_baseline.json
python benchmark/layout/run_layout_benchmarks.py \
  --conversion build/benchmark/layout/conversion/frisk_layout_conversion_bench \
  --baseline benchmark/layout/baseline/sm90_layout_baseline.json \
  --check
```

Expected: 第一条 runner 命令在记录 GPU/driver/clock/flags 后生成 reference baseline；第二条 GPU correctness PASS，且受支持 case 不超过已批准回退阈值。baseline 的新增或更新必须由性能评审确认；没有 SM90 机器时不得生成 baseline，也不得把本步骤标记完成。

- [ ] **Step 7: 执行完整最终验证**

Run:

```bash
cmake -S . -B build -G Ninja \
  -DCMAKE_BUILD_TYPE=Debug \
  -DCMAKE_LINKER=lld \
  -DLLVM_ENABLE_ASSERTIONS=ON \
  -DMLIR_DIR=/data0/xiebaokang/rocm-llvm-project/build/lib/cmake/mlir
cmake --build build --target FriskIR FriskTransforms frisk-opt \
  FriskLayoutUnitTests check-frisk --parallel 32
ctest --test-dir build --output-on-failure
./build/unittests/Dialect/Frisk/FriskLayoutUnitTests
git diff --check
```

Expected: 所有构建、CTest、lit、unit tests 和 diff check PASS。

- [ ] **Step 8: 编写验收记录并同步设计**

`layout_inference_acceptance.md` 必须列出每个 Global Constraint 的证据命令、结果、未覆盖项和后续 WGMMA/TMA lowering 边界。若实测要求改变阈值，必须先获得架构评审再修改设计文档，不能为让 benchmark 通过而直接放宽。

- [ ] **Step 9: 提交性能守门和验收记录**

```bash
git add include/Dialect/Frisk/Analysis lib/Dialect/Frisk/Analysis \
  lib/Dialect/Frisk/Transforms benchmark guoqiao/layout_inference_acceptance.md \
  guoqiao/layout_inference_design.md
git commit -m "test: gate layout correctness and performance"
```

### M6 Gate

必须具备以下证据：

```text
CTest/lit/unit/property 全部通过
TileLang 固定 corpus 全部通过
相同输入重复编译产生相同布局 IR
race/OOB/invalid ownership/unresolved var 为 0
常规 component candidates <= 256
beam width = 32
布局推断编译时间占比 <= 10%
受支持 SM90 runtime case 中位回退 <= 3%
尚无 lowering 的 WGMMA/TMA 路径仅声明静态契约通过
```

任一项缺少证据时，布局系统不能标记完成。

---

## 3. 最终 Definition of Done

- [ ] M0–M6 的全部 Gate 均有最新命令输出。
- [ ] `frisk-layout-pipeline` 对所有受支持 Frisk kernel 生成合法、已解析、可验证的布局 IR。
- [ ] 生产源码中不存在 `inferLayout(builder, DenseMap<Value, Attribute>&)`。
- [ ] Local/Register 的生产 IR 不再以 local MemRef 作为求解载体。
- [ ] Distributed/Storage layout 映射方向、Attr 字段和 verifier 与设计文档一致。
- [ ] Hard conflict 输出 provenance chain 和坐标反例。
- [ ] 多 consumer 可以选择共同布局、conversion 或 rematerialization。
- [ ] conversion 是普通成本候选，且不会被零转换绝对优先规则压制。
- [ ] SM90 WGMMA/TMA/shared swizzle contracts 有正反例测试。
- [ ] TileLang 固定语义 corpus、property tests 和顺序确定性测试通过。
- [ ] CPU/solver 与 SM90 conversion/runtime 性能守门通过。
- [ ] `guoqiao/layout_inference_design.md`、对比文档、实施计划和验收记录不存在语义冲突。
- [ ] 完整 WGMMA/TMA lowering 的未交付边界在验收记录中明确保留。

## 4. 计划执行方式

严格按 Task 1 → Task 29 顺序执行。每完成一个 Task：

1. 运行该任务局部测试；
2. 运行所在里程碑已有回归；
3. 检查设计文档漂移；
4. 提交独立 commit；
5. 进行一次 spec compliance review 和一次 code quality review；
6. 只有 review 通过才进入下一个 Task。

建议在每个 M0–M6 Gate 后由用户进行一次阶段验收。M2、M3、M4、M5 的 Gate 是架构关口，不建议并行跨越。
