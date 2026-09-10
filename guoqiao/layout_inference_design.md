# Frisk SM90 MLIR-native 布局推断系统设计

> 状态：MLIR-native 双域 IR 架构已确认（原架构候选“方案 C”）
> 日期：2026-08-16
> 范围：NVIDIA SM90/SM90a；布局推断、布局验证与布局物化
> 核心选择：Local/Register Tile 使用 `RankedTensorType + EncodingAttr`，Shared/Global 保持 MemRef，由统一约束系统连接分布式布局与存储布局
> 实施状态（2026-09-10）：M0–M2 已完成；M3 Task 14–17 已实现、通过逐任务审查和功能 Gate，整分支最终审查进行中。Storage 与 Distributed Tensor 均形成 collect → solve → materialize → verify 闭环，conversion 支持静态单 CTA 测试降低；可执行 GPU 运行验收仍属 M6。

## 1. 结论先行

Frisk 应采用已经确认的 MLIR-native 双域 IR 架构，并进行以下重构：

1. Local/Register Tile 改为不可变 SSA Tensor，布局写入 `RankedTensorType` 的 encoding；不再用 MemRef 模拟寄存器片段。
2. Shared/Global 数据继续使用 MemRef，以保留 MLIR 在别名、内存效应、Buffer、GPU/NVGPU lowering 方面的成熟能力。
3. 不再用当前单一 `LayoutAttr` 同时表达线程分布和内存排布，而是明确分为：
   - `DistributedEncodingAttr`：描述寄存器值如何分布到 `register/lane/warp/warp-group/CTA`；
   - `StorageLayoutAttr`：描述逻辑坐标到 global/shared 物理地址的映射；
   - `MmaEncodingAttr`、`DotOperandEncodingAttr`：描述 WGMMA 的指令契约。
4. 推断器采用“硬约束传播 + 候选生成 + 全局代价选择 + 显式转换物化”，而不是每个 Op 直接修改一个 `DenseMap<Value, Attribute>`。
5. 布局代数采用“仿射外层 × GF(2) 位线性内层”的组合表示：外层处理切块、动态边界、padding 和非 2 次幂；内层处理 lane/register 分布和 XOR swizzle。这是本方案区别于机械迁移 TileLang、也区别于只复制 Triton encoding 类层次的主要创新点。
6. TileLang 应使用最新代码作为语义参考，但必须固定到经过审计的 commit，不能依赖浮动的 `main`，也不能继续以当前旧版本作为实现依据。建议：
   - 设计与差分测试参考快照：`6623b12d232b343648a5ba99992e3e6f0d6376d2`；
   - 稳定回归基线：`v0.1.13`（`8001cc4ccf6149382d2019654a19f59c1d4d0482`）；
   - Frisk 不链接、不 vendor TileLang，仅移植语义、测试思想和硬件规则。

最终目标不是“在 MLIR 中重写 TileLang”，而是建设 Frisk 自己的布局 IR、约束系统和 SM90 规则库。MLIR 带来的优势主要是长期架构、可组合性、验证能力和 lowering 基础；性能不会因为使用 MLIR 自动变好，必须由 SM90 规则、代价模型以及最终代码质量共同保证。

## 2. 目标、边界与非目标

### 2.1 当前目标

- 建立长期独立、可扩展的布局系统，核心抽象不绑定 TVM/TIR/TIRX。
- 第一阶段只支持 NVIDIA SM90/SM90a，优先覆盖 H100 上的 WGMMA、TMA、mbarrier、shared-memory swizzle 和普通 SIMT/warp-level 路径。
- 复用 MLIR 的 SSA、ODS、Attr/Op/Type Interface、MemoryEffect、Alias/View、DataFlow、Dialect Conversion、GPU/NVGPU/NVVM 等基础设施。
- 对 GEMM、Copy、Fill、Reduce、Elementwise、Broadcast、Transpose、Reshape/View、控制流和别名建立完整推断规则。
- 推断结果必须可验证、可解释、可复现，并真正进入 lowering；不能停留在外部 map 中。
- 以“无正确性回退、无系统性性能回退”为创新边界。

### 2.2 第一阶段明确不做

- 不支持 SM100/TMEM/tcgen05。
- 不支持 AMD MFMA/LDS、Metal、CPU 等后端。
- 不以 CTA cluster/multicast 为首个里程碑；IR 中预留 `cluster/cta` 层级，SM90a 后续可启用。
- 不追求一次性支持任意动态 layout；SM90 指令内层 tile 必须静态，动态性主要存在于外层 shape、边界和 predicate。
- 不把通用求解器设计成任意 ILP/SMT 优化器；热路径使用有限候选、精确代数和确定性搜索。
- 不保留 TileLang/TVM 数据结构作为 Frisk 的公共 ABI。

### 2.3 当前实施计划的交付边界

当前实施计划采用“基础能力 + 纵向切片 + 逐类扩展”的方案 A，交付布局系统闭环：

- 组合布局代数、Distributed/Storage IR 和 verifier；
- constraint graph、strict/common propagation、候选生成与全局代价选择；
- layout encoding/storage binding/`frisk.convert_layout` 物化与转换优化；
- TileLang 语义差分、正确性、确定性、编译开销和性能守门。

本计划实现 WGMMA/TMA 的布局契约、候选、合法性和静态成本，并为 conversion 提供足够完成真实微基准的最小 lowering adapter；完整 WGMMA/TMA/mbarrier 指令 lowering 作为后续独立计划。对尚无 lowering 的路径只能宣称布局契约和静态质量通过，不能提前宣称运行时性能达标。

## 3. 当前代码与参考系统审计

### 3.1 Frisk 当前状态

当前实现已经完成 Storage 纵向切片，但 Distributed Tensor、显式 conversion 和完整 SM90 指令契约仍未实现：

- `LayoutAttr` 同时包含 `forwardIndex`、`forwardThread` 和 `replicateSize`，混合了存储布局、线程分布和复制语义。
- `forwardIndex/forwardThread` 只用 `AffineMapAttr` 表示，无法自然表达 XOR swizzle。
- `LayoutInterface::inferLayout` 通过 `DenseMap<Value, Attribute>` 原地更新布局，接口没有表达约束、冲突原因、候选和代价。
- `frisk.layout_view` 已提供 Shared/Global Storage binding；module 级 `frisk-infer-layouts` 已依次执行约束收集、strict/common 传播、受限求解、物化和二次验证。
- constraint graph 使用逐组件长度前缀编码的 fully-qualified ancestor symbol path，加上 block/op/result 组成无歧义稳定名称；hard conflict 可输出两条 seed provenance。
- M2 bootstrap resolver 只处理每连通分量最多 8 个变量、每个 domain 最多 4 个候选，不枚举 conversion/rematerialization；二维 Storage domain 为 linear、transpose、padding 和按行字节选择的一个 32B/64B/128B XOR 候选。
- `frisk-infer-layouts` 缺少 target 属性时按 `sm_90` bootstrap 默认值处理；若显式给出 target，则只接受 `sm90/sm_90/sm90a/sm_90a`，其他 target 在构造候选前拒绝。
- materialized verifier 会枚举静态 logical domain，验证每个元素的 bit range 不重叠，并证明最大地址不超过底层 MemRef strided/affine type 表达的静态容量；无法证明时保守拒绝。
- `gemm/copy/fill/reduce` 的 MemoryEffect 已修正；M2 collector 目前只接收两个 operand 都由 `layout_view` 锚定的静态 whole-tile Copy，其他 Copy 明确诊断 unsupported。
- M2 alias group 覆盖任意深度、零变换的 `layout_view` chain；`memref.cast`、`subview` 等 transform-aware alias 归入 M4 的统一 AliasAnalysis/region 传播。
- Parallel 内的递归推断只覆盖部分操作，没有形成统一的 Op Interface 调度。
- Storage binding 已进入 Frisk IR，但推断结果还没有完整进入 Frisk → NVGPU/NVVM 的 conversion pipeline，因此当前布局正确并不等于最终代码质量正确。

因此，第一步必须先修正 IR 语义和接口，而不是继续向现有 `LayoutAttr` 追加特殊字段。

### 3.2 TileLang 版本结论

本次已执行 `git fetch origin --prune`。本机 TileLang 状态如下：

| 项目                | 版本                                                             |
| ------------------- | ---------------------------------------------------------------- |
| 本地 HEAD           | `f8083581f13370455f9b4618fb1ae99fb1fb5480`，2026-01-19         |
| 远端`origin/main` | `6623b12d232b343648a5ba99992e3e6f0d6376d2`，2026-08-15         |
| 距离                | 本地落后 707 commits，无本地分叉                                 |
| 最新稳定标签        | `v0.1.13`，commit `8001cc4ccf6149382d2019654a19f59c1d4d0482` |

布局相关目录从旧版本到远端最新版本有明显大规模演进：审计范围内 24 个文件约 `+6969/-955`，新增约 2300 行的 `cute_layout.cc`、约 630 行的 `cute_layout.h`，并加入了 repeat/expand、线性化索引、高维 GEMM、swizzle 合并、任意 swizzled SMEM 的 TMA/GMMA lowering、dtype-changing view、ragged 边界、owner 验证和 Z3 上下文修复等能力。

TileLang 的五阶段推断骨架仍有参考价值：严格传播、普通 fixed-point、free-mode 松弛和 alias finalize；但其内部已经深度绑定 TVM Analyzer、PrimExpr、Buffer、TIRX 和 Z3。`tirx` 是 TileLang 所依赖 TVM 分支中的新一代/扩展 TIR 命名空间与 IR API，不应迁入 Frisk。

版本策略必须是：

- 不再使用旧 `f8083581` 作为正确性依据；它只能帮助理解算法演进。
- 以最新审计 commit `6623b12d` 建立一次语义快照，吸收标签之后的 correctness 修复。
- 以 `v0.1.13` 建立可重复运行的稳定差分测试环境。
- 每季度或每次 TileLang 布局重大版本后做一次“语义审计”，只同步规则与测试，不追踪其内部实现。
- 在 Frisk 仓库中维护 `TileLangReference.md` 或测试清单，记录参考 commit、采用/拒绝的行为和原因。

参考：[TileLang 仓库](https://github.com/tile-ai/tilelang)、[最新审计 commit](https://github.com/tile-ai/tilelang/commit/6623b12d232b343648a5ba99992e3e6f0d6376d2)、[v0.1.13](https://github.com/tile-ai/tilelang/releases/tag/v0.1.13)。

重点审计入口：[layout_inference.cc](https://github.com/tile-ai/tilelang/blob/6623b12d232b343648a5ba99992e3e6f0d6376d2/src/transform/layout_inference.cc)、[layout.h](https://github.com/tile-ai/tilelang/blob/6623b12d232b343648a5ba99992e3e6f0d6376d2/src/layout/layout.h)、[cute_layout.h](https://github.com/tile-ai/tilelang/blob/6623b12d232b343648a5ba99992e3e6f0d6376d2/src/layout/cute_layout.h)。

### 3.3 Triton 与 MLIR 可复用能力

Triton 的 `RankedTensorType + EncodingAttr` 已证明适合表达寄存器分布；其 LinearLayout 将 `(register, lane, warp, block)` 映射到逻辑 tensor 坐标，并用 GF(2) 线性代数统一表达 transpose、swizzle、compose、inverse/pseudoinverse。这个方向应参考，但 Frisk 不应复制 Triton 大量特化 encoding 和 conversion 清理逻辑。

参考：[Triton LinearLayout](https://github.com/triton-lang/triton/blob/main/include/triton/Tools/LinearLayout.h)、[TritonGPU layout attributes](https://github.com/triton-lang/triton/blob/main/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td)、[RemoveLayoutConversions](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp)。

Frisk 当前使用的工具链二进制报告为 LLVM `21.0.0git`，本地 NVGPU 已有：

- `nvgpu.warpgroup.mma`、`nvgpu.warpgroup.generate.descriptor` 和 warpgroup accumulator；
- `nvgpu.tma.create.descriptor`、`tma.async.load/store`；
- `nvgpu.mbarrier.*`；
- 32B/64B/128B tensor-map swizzle；
- `nvgpu.ldmatrix` 和 warp-level `nvgpu.mma.sync`。

NVGPU 的定位正是高层 GPU/Vector 与低层 NVVM 之间的 NVIDIA 专用桥梁，所以 Frisk 应优先 lower 到 NVGPU，而不是在 Frisk Dialect 中长期保留 PTX 字符串或重新定义一套等价低层指令。参考：[MLIR NVGPU Dialect](https://mlir.llvm.org/docs/Dialects/NVGPU/)、[MLIR NVVM Dialect](https://mlir.llvm.org/docs/Dialects/NVVMDialect/)。

## 4. MLIR-native 双域总体架构

```text
Frisk 高层 Tile/Kernel IR
          |
          v
Layout IR Normalization
  Local/Register -> SSA RankedTensorType + Encoding
  Shared/Global  -> MemRef + layout_view/binding
          |
          v
Constraint Collection
  Op contracts + SSA def-use + alias/view + SM90 target rules
          |
          v
Unified Layout Solver
  hard propagation -> candidates -> cost selection -> relaxation
          |
          v
Materialization + Verification
  tensor encodings + storage bindings + frisk.convert_layout
          |
          v
Frisk -> Vector/MemRef/GPU/NVGPU -> NVVM/LLVM -> PTX/CUBIN
```

该架构有三条强制边界：

1. 通用层只理解 layout algebra、约束和 memory space，不硬编码 WGMMA 指令枚举。
2. SM90 target rule library 只提供合法布局候选、指令契约、资源限制和代价，不直接遍历/改写任意 IR。
3. Materializer 只消费已求解结果并改写类型/插入转换，不在改写时重新做隐式决策。

## 5. IR 与布局表示

### 5.1 Local/Register Tile：SSA Tensor Encoding

Local/Register Tile 使用：

```mlir
tensor<64x128xf32, #frisk.mma<kind = wgmma, ...>>
tensor<64x64xbf16, #frisk.distributed<map = #frisk.product<...>, ...>>
```

上面语法是设计示意，最终 assembly format 由 ODS 定义。关键语义是：

- Tensor 是不可变值，天然使用 SSA def-use。
- Encoding 是类型的一部分；两个不同 encoding 的 Tensor 不能在没有显式转换的情况下被当成同一物理分布。
- `frisk.convert_layout` 是显式 SSA 操作，便于 CSE、hoist、rematerialization、代价统计和 verifier 检查。
- `frisk.tile_load`/`frisk.tile_store` 首版只搬运静态 whole tile。load 的 MemRead 和
  store 的 MemWrite 均精确绑定到 MemRef operand。带 distributed encoding 的 Tensor
  carrier 必须针对其完整公共 RankedTensor shape 通过 encoding verifier；推断前也允许
  暂无 encoding 的 carrier。`tile_store` 对每个 logical element 只写一次；若输入布局有
  replicated owners，lowering 必须确定性选择一个 source owner，复制不代表允许多个线程
  对同一物理元素重复写入。
- identity `frisk.convert_layout` 是合法的输入 IR，并由 canonicalization 消除；布局
  materializer 不得生成 identity conversion。精确的 A→B→A conversion pair 可在结果
  类型安全且 canonical distributed map、topology 和 replication 相等时折叠。
- Scalar 不强制携带布局；标量广播到 Tile 时才产生 replicated candidate。

MLIR 的 `RankedTensorType` 原生支持 encoding attribute，因此无需自定义一个重复的 Frisk TensorType。参考：[MLIR Builtin Dialect](https://mlir.llvm.org/docs/Dialects/Builtin/)。

### 5.2 Shared/Global：MemRef + Storage Layout Binding

Shared/Global 仍是 MemRef，但不能把所有 shared XOR layout 直接塞进 MemRef type 的 layout 字段。MLIR 要求 `MemRefLayoutAttrInterface` 能转换为 semi-affine map，并且不能产生内部 alias；XOR swizzle 不属于一般 semi-affine map。参考：[MemRef layout 约束](https://mlir.llvm.org/docs/Dialects/Builtin/#memreftype)。

因此采用两层表示：

- MemRef type 表达 shape、element type、memory space，以及标准 strided/affine global layout。
- `frisk.layout_view` 或 layout-bearing allocation/access anchor 绑定 `StorageLayoutAttr`，表达 logical tile 到实际 shared/global 地址的非标准映射。

设计示意：

```mlir
%smem = frisk.alloc_buffer ... : memref<64x64xbf16, #frisk.shared>
%tile = frisk.layout_view %smem
  {layout = #frisk.storage<space = shared, map = #frisk.product<...>>}
  : memref<64x64xbf16, #frisk.shared>
```

`frisk.layout_view` 必须实现 `ViewLikeOpInterface`，不分配新内存；它为同一底层 MemRef 的不同逻辑 tile/view 提供唯一 SSA layout anchor。M2 沿零变换的 `layout_view` chain 合并 alias；M4 再由 AliasAnalysis 覆盖 `memref.cast`、`subview` 等带变换的 view，确保所有 alias 的 storage layout 一致。

Global MemRef 的默认物理布局直接复用 strided/affine layout；只有 tile permutation、packed/sub-byte 或特殊访问需要 `StorageLayoutAttr`。

### 5.3 布局映射方向

统一的 `LayoutMapAttr` 是带命名输入/输出维度和 extent 的有向映射。两类布局采用不同但可直接组合的自然方向：

- Distributed：
  `D(register, lane, warp, warp_group, cta) -> (logical_dim0, ..., logical_dimN)`。
- Storage：
  `S(logical_dim0, ..., logical_dimN) -> (byte_offset, bit_offset)`。

这样一次 register/shared copy 的每线程地址就是 `S(D(hardware_coordinate))`。Distributed 映射允许非单射，因为同一逻辑元素可能被多个线程复制；但必须满足所需逻辑 tile 的 coverage。Storage 映射对 live logical domain 必须单射，padding 只产生未使用的物理洞，不产生 alias。

该方向选择既保留了 Triton hardware → logical 对复制的良好表达，又让 MemRef 的 logical → address 语义保持自然。

### 5.4 Attribute 层次

建议新增以下属性及接口：

| 属性                        | 责任                                                                                                 |
| --------------------------- | ---------------------------------------------------------------------------------------------------- |
| `LayoutMapInterface`      | 通用 map 操作：compose、project、reshape、inverse/right-inverse、coverage、injectivity、canonicalize |
| `AffineLayoutMapAttr`     | 外层 tile、permutation、静态 quotient/remainder、padding、动态边界参数                               |
| `BitLinearLayoutMapAttr`  | GF(2) basis，表达 lane/register 位分布与 XOR swizzle                                                 |
| `ProductLayoutMapAttr`    | 仿射外层和位线性内层的组合/直和                                                                      |
| `DistributedEncodingAttr` | hardware carrier → logical tile，并包含 topology 与 replication 信息                                |
| `StorageLayoutAttr`       | logical tile → physical offset、alignment、vector granularity、memory space                         |
| `MmaEncodingAttr`         | WGMMA accumulator/result 指令契约，可规范化为 distributed map                                        |
| `DotOperandEncodingAttr`  | WGMMA A/B operand 契约，引用 parent MMA encoding 和 operand index                                    |
| `LayoutProvenanceAttr`    | 用户指定、指令强制、传播、默认选择、松弛转换等来源，默认只在 debug IR 保留                           |

所有具体 encoding 实现统一的 `LayoutEncodingInterface`，至少提供：

```cpp
FailureOr<Attribute> getCanonicalMap(ShapedType type) const;
LogicalResult verifyForType(ShapedType type, Location loc) const;
LayoutKind getKind() const; // distributed / storage / instruction
```

Encoding 展开不接收隐式 target 上下文：Distributed/Storage encoding 已携带确定的 map，
指令类 encoding 则必须先由 target rule library 选择并固化契约参数。这样 canonicalization
是纯函数，同一 attribute 不会因 pass 外部 target 状态不同而产生不同结果；返回
`Attribute` 是生成式 interface 的 ABI 边界，调用者必须再验证结果实现
`LayoutMapAttrInterface`。

`MmaEncodingAttr` 和 `DotOperandEncodingAttr` 是指令契约，而不是另一套独立数学系统；它们必须可展开成 canonical layout map。这样 target-specific 属性不会污染通用 compose/equality/verifier。

M1 固化的通用 encoding 文本与单位如下：

```mlir
#frisk.distributed<map = #map,
  topology = [register, lane, warp, warp_group, cta], replication = R>
#frisk.storage<map = #map,
  memory_space = #frisk<memory_space Shared>,
  alignment = A, vector_granularity = V>
```

- `topology` 中五项都是 carrier extent，不是 bit width；省略于 map 的 extent-1 carrier
  仍保留在 topology 中。对 BitLinear distributed map，`replication` 必须严格等于
  `product(topology) / 2^rank(B_D)`，因此零列/缺失 carrier 所代表的复制可被验证，不能
  只凭一个独立数字声明。
- Storage map 的输出名称和顺序固定为 `(byte_offset, bit_offset)`；Affine map 的
  `bit_offset` extent 固定为 8，BitLinear map 的对应 width 固定为 3。
- `alignment` 与 `vector_granularity` 均以 byte 为单位，必须是正的 2 次幂，且
  `vector_granularity <= alignment`。Storage 仅绑定 Shared/Global MemRef，live logical
  domain 上的地址必须可证明为单射，bit offset 必须位于 `[0, 8)`。
- M1 的 native Distributed type verifier 已使用 hardware → logical 的 BitLinear
  精确证明；legacy adapter 过渡期允许传入旧 Local MemRef 的 shape，M3 物化时公共
  carrier 将收紧为带 encoding 的 RankedTensor SSA。

### 5.5 M1 legacy adapter 的证明边界

`LegacyLayoutAdapter` 是迁移期语义桥，不是第二套布局系统。它作为独立
`FriskLegacyLayoutAdapter` 库依赖 `FriskIR`，避免让通用 `FriskAnalysis` 与 IR
形成循环依赖。当前支持集固定为：

- Distributed：静态、各逻辑 extent 为 2 次幂、legacy index/thread 均为单结果且
  无 symbol、replication 为正的 2 次幂；枚举 `(logical -> thread/register)` 后求逆，
  从 carrier basis 重建 GF(2) matrix，并对完整 domain 验证。replication 被编码为
  topology 中映射到零的 carrier bit，矩阵 kernel 大小必须与旧值一致。
- Storage：静态、无 thread map。legacy 多结果 physical tuple 先按逐维最大 extent
  作规范 row-major flatten，再转换为 `(byte_offset, bit_offset)`；2 次幂 domain
  优先从完整真值表恢复 GF(2) basis 并全域验证。只有旧 map 本身为单结果时才允许
  linear/padded Affine fallback。
- 单次迁移枚举最多 65,536 个 logical point。动态、超限、非 GF(2) 多结果 map、
  不可逆 ownership 或任何无法证明的情况都诊断
  `legacy layout cannot be represented by the canonical layout algebra`，不猜测布局。

M1 差分门覆盖当前 SM90 `sm90_ss` 与 `sm90_rs`：shared A/B 地址和 local A/C
ownership 均逐点比较。SM80 旧测试继续作为原生产路径回归保留，但不属于当前
SM90-only adapter 的支持承诺。

## 6. 创新核心：Affine × BitLinear 组合布局代数

### 6.1 为什么不能只用 AffineMap

MLIR AffineMap 很适合 permutation、常量 stride、`floordiv/mod` 和 padding，但不能直接表达通用 XOR。将 shared swizzle 拆成大量手写公式，会重现当前 `LayoutAttr` 的扩展瓶颈。

### 6.2 为什么不能只用 GF(2) LinearLayout

GF(2) 位线性映射非常适合 2 次幂内层 tile、lane/register 分布和 XOR，但动态外层、非 2 次幂 shape、ragged tile、一般 padding 和部分 dtype view 并不都是纯 GF(2) 线性问题。

### 6.3 组合表示

将每个逻辑维度分解为：

```text
logical_i = outer_i * inner_extent_i + inner_i
```

在 `ProductLayoutMapAttr` 的实际存储中，Affine outer 的输出采用已经缩放后的
`outer_base_i = outer_i * inner_extent_i`，因此组合求值固定为：

```text
logical_i = outer_base_i + inner_i
```

这种规范避免在 compose/lowering 的每一层重复猜测是否需要乘 split。
`split_extents[i]` 必须为正的 2 次幂，BitLinear inner 对应输出的 bit width 必须等于
`log2(split_extents[i])`，并且 verifier 必须证明 Affine outer 的结果恒为
`split_extents[i]` 的倍数。证明失败（包括动态条件不足）时不得构造 Product；查询动态
Affine map 的 coverage/injectivity 时返回 `Unknown`，不能把无法证明当成合法。

- `inner_i` 的 extent 是 SM90 指令原子决定的静态 2 次幂，使用 `BitLinearLayoutMapAttr`。
- `outer_i`、tile permutation、padding 和动态边界使用 `AffineLayoutMapAttr`。
- `ProductLayoutMapAttr` 负责 direct-sum、compose 和 named-dimension 对齐。
- 动态 shape 只作为 extent/bounds symbol 参与验证和 predicate，不允许动态值进入 encoding attribute 本体。

`BitLinearLayoutMapAttr` 的位序是规范的一部分：input/output name 数组从左到右
拼接各维 bit，维内按低位到高位排列；matrix 的列对应拼接后的 input bit，行对应
拼接后的 output bit，并使用 row-major `tensor<rows x cols x i1>` 存储。input 和
output 名称分别在各自集合内唯一，compose 通过名称及 bit width 对齐中间维度，
不得依赖 hash/指针顺序。

核心操作必须具备明确语义：

- `compose(A, B)`：按命名维度组合映射；
- `directProduct(A, B)`：组合独立 tile 层级；
- `project/drop/permute`：用于 reduce、broadcast、transpose；
- `reshape/factorize`：只在元素数和 bit-level storage 关系可证明时成功；
- `inverse/rightInverse`：仅在代数条件满足时返回，不能静默选择错误 inverse；
- `isEquivalent`：比较 canonical form，而不是比较构造路径；
- `isInjective/isSurjective/covers`：返回证明结果和反例；
- `enumerate`：仅作为小 tile 调试/测试 oracle，不能成为生产求解热路径。

位线性层通过 GF(2) 高斯消元计算 rank、kernel、image、inverse 和等价性，避免对常见 WGMMA/swizzle 调用 Z3。仿射层优先使用 MLIR Affine/Presburger；只有无法证明且确实需要支持的动态外层问题才进入可选 SMT fallback。

M1 的阶段性实现对完整 BitLinear domain 使用秩做精确证明；对静态
Affine/Product map 提供最多 65,536 个 domain point 的有界参考证明器，超过上限、
包含 symbol 或无法求值时保守返回 `Unknown`。该路径用于 Attr verifier、单元测试和
差分 oracle，不得放入候选求解的逐候选热循环；后续接入 Affine/Presburger 证明或
canonical proof cache 后，solver 才能消费相应结果。

### 6.4 性能与创新边界

这项创新的价值不是“数学形式更复杂”，而是：

- 同一个 canonical map 同时服务推断、验证、地址生成和差分测试；
- GF(2) 精确证明避免 ad-hoc swizzle 合并和错误等价判断；
- 外层不被强制补成 2 次幂，可减少不必要的 replication/padding；
- 求解阶段可以准确比较两个候选是否只是构造形式不同；
- lowering 可以从 map 生成索引，不需要为每种 layout 写独立地址公式。

若某个新布局不能生成与手写 SM90 规则等价或更好的地址代码，则不得默认启用。

## 7. 统一约束模型

### 7.1 基本对象

```cpp
struct LayoutVar {
  LayoutVarID id;
  LayoutKind kind;            // Distributed or Storage
  ShapedType shapedType;
  MemorySpace memorySpace;
  LogicalShape shape;
  SmallVector<LayoutCandidate> candidates;
  LayoutState state;
};

enum class LayoutState {
  Uninitialized,
  CandidateSet,
  Resolved,
  Conflict
};

struct LayoutConstraint {
  ConstraintKind kind;
  SmallVector<LayoutVarID> vars;
  ConstraintStrength strength; // Hard or Soft
  Operation *source;
  std::string reason;
  LayoutTransform relation;
};
```

每个 constraint 必须携带 source location、规则名和原因。冲突诊断至少输出从两个冲突 seed 到冲突值的 provenance chain；首版不必计算真正的 minimal unsat core。

约束种类至少包括：

| Constraint              | 语义                                        | 典型来源                              |
| ----------------------- | ------------------------------------------- | ------------------------------------- |
| `SameLayout`          | 两个值使用同一 canonical layout             | elementwise、loop-carried value       |
| `TransformLayout`     | 两个布局通过已知坐标变换关联                | transpose、reshape、broadcast、reduce |
| `RequireEncoding`     | 值必须属于指定 encoding/候选集合            | 用户注解、WGMMA result                |
| `InstructionContract` | 多个 operand/result 联合满足硬件契约        | GEMM/WGMMA、TMA、ldmatrix             |
| `StorageAccess`       | Distributed 与 Storage 可组合并满足访问条件 | copy/load/store                       |
| `AliasLayout`         | 多个 view 共享同一底层 physical mapping     | subview、layout_view、bitcast         |
| `Ownership`           | reader/writer/reducer 的 coverage 与唯一性  | fill、store、reduce                   |
| `ResourceLimit`       | register/shared/alignment/topology 上限     | SM90 target                           |
| `Preference`          | 可违反但计入 CostVector 的偏好              | coalescing、bank conflict、conversion |

### 7.2 Hard constraints

以下条件不可由代价模型违反：

- shape、rank、element/storage bit、memory space 一致性；
- WGMMA/MMA/TMA/ldmatrix 指令合法布局；
- SSA use 的 encoding 类型一致；
- view/reshape/transpose/bitcast 的逻辑坐标关系；
- alias group 的 storage mapping 一致性；
- storage layout 对 live domain 无内部 alias；
- distributed layout 覆盖所需 tile；
- reduce/store 具有唯一 writer 或显式原子/归约语义；
- warp-group 操作由完整 4 warps/128 threads 协作；
- TMA descriptor、swizzle、box dimension、alignment 和边界约束；
- 用户标记为 mandatory 的布局注解。

### 7.3 Soft preferences

- 全局内存合并访问与最大安全 vector width；
- shared bank conflict 数量与严重程度；
- 优先可直接 TMA/WGMMA lowering 的组合；
- 减少 `frisk.convert_layout` 数量和搬运字节数；
- 降低 replication、寄存器数量、shared padding 和 occupancy 损失；
- 保持生产者/消费者布局一致，优先 rematerialize 便宜计算而不是搬运大 tile；
- 用户标记为 preferred 的布局。

### 7.4 新 Op Interface

现有 `inferLayout(builder, DenseMap&)` 应废弃，替换为只收集语义、不修改全局状态的接口：

```cpp
class LayoutConstraintOpInterface {
public:
  LogicalResult collectLayoutConstraints(LayoutConstraintBuilder &builder);
};
```

Target-specific 合法集合通过独立接口注入：

```cpp
class LayoutTargetInterface {
public:
  void seedInstructionLayouts(Operation *op,
                              LayoutConstraintBuilder &builder);
  void enumerateCandidates(const LayoutDomain &domain,
                           SmallVectorImpl<LayoutCandidate> &out);
  FailureOr<CostVector> evaluate(const CandidateAssignment &assignment);
};
```

状态变化使用独立枚举，例如 `ChangeResult::{NoChange, Change}`；错误使用 `LogicalResult`/`FailureOr`，不能再用 `success(false)` 表达未变化。

### 7.5 Lattice、终止性与确定性

- 每个变量的语义状态是“允许候选集合”；hard propagation 只允许集合缩小，不能在同一阶段反复添加和删除候选。
- `Uninitialized` 在候选生成前表示尚无有限 domain；一旦生成 domain，后续只做交集、关系投影和剪枝。
- `Resolved` 是 singleton candidate set；空集合进入 `Conflict`。
- 候选生成、relaxation 和 commit 是不同阶段，保证 fixed-point 单调并终止。
- 对 affine/Presburger 证明返回 `true/false/unknown`；`unknown` 不能被当成满足 hard constraint，只能触发 guarded candidate、fallback 或 unsupported diagnostic。
- worklist、candidate 和 constraint 都按稳定 ID 排序；canonical hash 只用于查找，不决定最终选择，从而保证不同遍历顺序得到相同 IR。

## 8. 完整布局推断算法

### 8.1 总体伪代码

```text
inferLayouts(module, target):
  normalizeAndVerifyIR(module)

  graph = collectLayoutVariables(module)
  buildAliasAndViewGroups(graph)
  collectOpConstraints(graph)
  target.seedHardInstructionConstraints(graph)
  collectUserAnnotations(graph)

  strictWorklist = seedsAndAdjacentConstraints(graph)
  propagateUniqueHardFacts(strictWorklist)
  if conflict: emitProvenanceDiagnostic()

  propagateCommonConstraintsToFixedPoint(graph)
  if conflict: emitProvenanceDiagnostic()

  for each unresolved connected component:
    generateFiniteCandidates(component, target)
    pruneByHardConstraints(component)
    if empty: attemptControlledRelaxation(component)
    assignment = selectMinimumCostAssignment(component)
    if none: emitConflictOrUnsupportedDiagnostic()
    commit(assignment)

  verifySolvedLayoutGraph(graph, target)
  materializeEncodingsBindingsAndConversions(module, graph)
  canonicalizeConversionsAndCheapRematerialization(module)
  verifyMaterializedIR(module, target)
```

### 8.2 阶段 A：IR normalization 与前置验证

1. 为 module/kernel 建立 target 信息，使用 `#nvvm.target` 和/或 DLTI，而不是只依赖字符串 `frisk.target`。DLTI 用于查询 warp size、memory alignment、shared limit 等 target 属性。参考：[MLIR DLTI](https://mlir.llvm.org/docs/Dialects/DLTIDialect/)。
2. 修正所有 Frisk Op 的 MemoryEffect：
   - register-only elementwise 可以 `Pure`；
   - copy/fill/reduce/gemm 对 MemRef 的读写必须声明 effect；
   - alloc/view 使用 Allocate/View/Read/Write 对应语义。
   - kernel/parallel/block/for 等 region 容器使用 `RecursiveMemoryEffects`；
     无副作用的 `frisk.end` 显式标记 `Pure`，保证递归查询不会退化为 unknown。
3. 将等价 view/transpose/reshape 规范成少量 layout-aware op，保留 source location。
4. 将现有“用 fragment MemRef 表示寄存器值”的路径转换成过渡 SSA Tensor IR；前端不要求一次性全部重写。
5. 检查 rank、shape、dtype、memory space、region terminator 和 block argument 基本合法性。

### 8.3 阶段 B：变量、def-use 与 alias graph

- 每个 layout-bearing Tensor SSA value 建一个 Distributed `LayoutVar`。
- 每个 layout anchor（alloc、layout_view、kernel MemRef 参数）建一个 Storage `LayoutVar`。
- 通过 `ViewLikeOpInterface`、BufferViewFlowAnalysis/AliasAnalysis 建立 alias/view group。
- Region block argument、yield、loop-carried value 建立跨 region 等价边。
- 记录每个 use 对 operand layout 的要求，不在收集阶段选择布局。

MLIR DataFlow Framework 可以处理 region/branch/callgraph 的可达性和程序点，但本问题需要双向传播、候选集合与自由 root 选择，因此核心 solver 使用专用 worklist；DataFlow 只作为控制流传播基础，不强行套用单向 forward analysis。参考：[MLIR DataFlow Analysis](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/)。

### 8.4 阶段 C：硬布局 seed

按优先级建立 seed：

1. 用户 mandatory layout 注解；
2. WGMMA accumulator/result、A/B operand 和 shared descriptor 契约；
3. TMA source/destination descriptor 契约；
4. 已存在且通过 verifier 的 tensor encoding/storage binding；
5. `frisk.convert_layout` 的 source/target encoding；
6. 标量常量和 size-1 tile 的 fully-replicated seed。

两个 hard seed 冲突时立即失败，不允许通过插入隐式转换掩盖 IR 语义冲突；只有 consumer layout 不同但二者都合法时，才在后续 relaxation 中显式插入转换。

### 8.5 阶段 D：strict propagation

仅传播唯一确定的事实：

- `same-layout`、可逆 transpose、精确 reshape；
- WGMMA parent encoding 到 accumulator/dot operand；
- layout_view 和 alias anchor 的 storage layout；
- loop init/result/block argument/yield 的类型一致约束；
- 已知 copy 一端和唯一合法 vector/access contract 推导另一端。

本阶段不创建默认布局、不复制、不 padding，也不插入 conversion。

### 8.6 阶段 E：common fixed-point

按 constraint 依赖关系运行双向 worklist，直到 candidate set 不再缩小或新增：

- 从 producer 向 consumers 传播兼容布局；
- 从硬件 consumer 反向推导 producer 的优选布局；
- 对 alias group 传播 storage mapping；
- 对 region/loop 传播 join 后的候选交集；
- 对 elementwise/shape transform 传播经过坐标变换后的 candidate。

候选集合使用 canonical hash 去重；同一数学 layout 的不同构造路径只保留一个，并合并 provenance。

### 8.7 阶段 F：候选生成与剪枝

对仍未解析的每个连通分量，由 SM90 target rules 生成有限候选：

- Distributed：blocked、warp-striped、lane-striped、fully-replicated、MMA result/dot operand compatible；
- Shared：linear、transpose、32B/64B/128B swizzle、必要 padding；
- Copy：合法 vector width、TMA、cp.async、普通 vector load/store；
- Reduce：warp shuffle、warp-group/shared tree、replicated result 或 elected owner；
- GEMM：合法 WGMMA shape/dtype/major mode 的组合；
- Layout edge：保持共同布局、插入显式 conversion、对便宜 producer rematerialize。

生成后立即执行 hard pruning：shape 不整除时生成 guarded/ragged candidate，而不是错误地假设整除；指令内层 tile 不满足时删除候选；TMA alignment/box 条件不满足时保留 cp.async/SIMT fallback。

### 8.8 阶段 G：全局代价选择

以 layout constraint graph 的连通分量为优化单元。首版使用“传播剪枝 + 动态规划/beam search”，不引入通用 ILP。代价采用字典序向量，而不是一个难以校准的总分：

```text
CostVector = (
  illegal_or_unsupported,       // 必须为 0
  instruction_path_and_work,    // 结合实际 shape 的 WGMMA/TMA/cp.async/SIMT 成本
  estimated_memory_transactions,
  shared_bank_conflict_degree,
  conversion_bytes_and_sync_on_critical_path,
  spill_risk_and_registers,
  shared_bytes_and_occupancy,
  replication,
  code_size,
  deterministic_tiebreak_key
)
```

首先用 hard constraint 排除不合法候选，再联合评估指令路径、实际工作量、访存、bank conflict、conversion、寄存器和 occupancy。conversion 是强成本项，但不是绝对禁止项：如果一个受控 conversion 能让主计算命中明显更优的 WGMMA/TMA 路径，它可以优于零 conversion 的低性能方案。对无法可靠静态预测的近似相等候选使用稳定 tie-break；后续可接 autotuning，但不能让结果依赖遍历顺序。

### 8.9 阶段 H：受控 relaxation/free mode

显式 conversion 和 rematerialization 已经是阶段 F/G 的普通候选，不以“零转换方案不存在”为启用条件。只有普通候选域经过 hard pruning 后为空，才进入受控松弛，顺序为：

1. 允许局部 replication；
2. 对 shared storage 增加 padding 或降低 vector width；
3. 为 non-power-of-two/ragged 外层增加 guarded candidate；
4. 从 TMA 路径回退到 cp.async/vector copy；
5. 仅在 WGMMA 契约本身不合法时回退到 warp MMA/SIMT。

每次松弛必须记录原因和增加的 CostVector。禁止“为了让求解成功”静默改变 mandatory annotation、writer ownership 或 memory alias 语义。

### 8.10 阶段 I：求解结果验证

对每个 resolved layout 检查：

- Attr 本身格式和 extent 合法；
- Encoding 与 tensor shape/dtype 匹配；
- Distributed coverage 和允许的 replication；
- Store/Reduce owner 唯一性；
- Storage injectivity、bit offset、alignment 和 allocation size；
- 所有 alias/view 的映射可组合且一致；
- WGMMA/TMA/ldmatrix/copy contract 完整满足；
- 所有 layout-bearing value 均已解析；
- 每个 layout conversion 的 source/target 都合法且确实不同。

验证失败是 compiler error，不应靠 assertion 终止；诊断中输出 op location、冲突 layouts、坐标反例和 provenance。

### 8.11 阶段 J：物化与转换清理

- 将 Distributed 结果写入 `RankedTensorType` encoding。
- 将 Storage 结果写入 `frisk.layout_view`/allocation binding。
- 在 layout 边界插入 `frisk.convert_layout`。
- 对 SCF/Frisk region 的 block argument、yield 和 result 同步更新类型。
- 运行 type-safe canonicalization：删除 identity conversion、合并相邻 conversion、hoist loop-invariant conversion、对便宜 pure slice 做受控 rematerialization。
- debug 模式保留 provenance；release lowering 前移除非必要 debug attr。

转换清理不能承担“修复错误推断”的职责；它只优化已验证的显式转换。

M3 Task 16 已实现物化子阶段；转换清理与 lowering 不属于该阶段。实现仅接受
`ModuleOp` transaction root：按原始 `Value` 解析 encoding（不按 `Type` 缓存），
将 solver 选中的 consumer use 快照为 owner/operand index，在 detached module
中重建一致的 function/SCF/Tensor 类型和选定 conversion。原生 properties、属性、
location、scalar CFG 及 region captures 保留；DenseElements 常量属性随结果重塑。
完整 staged IR 通过 MLIR verifier 和布局关系验证后，才以一次 body transfer 提交；
任何失败均不修改原始 IR。graph/solution 为借用输入，其原始 SSA/use 身份仅在提交前
有效；提交后不可解引用其中的 `Value`、`Operation *`、`OpOperand *`，但容器可正常销毁。

物化后验证使用 `LayoutCollectionMode::RelationsOnly` 收集实际关系，不生成候选、
不传播求解。definition、synthetic consumer use、function result 均固定为 IR 中实际
存在的 encoding；不能用假想的后续 Convertible 边掩盖缺失 conversion。实现和回归记录见
[M3 Task 16 materialization](m3_task16_materialization.md)。

### 8.12 阶段 K：lowering

以下是长期完整流水线，而不是当前布局系统实施计划的全部交付范围：

```text
frisk-normalize-layout-ir
  -> frisk-infer-layouts
  -> frisk-verify-layouts
  -> frisk-materialize-sm90-pipeline
  -> convert-frisk-to-vector-memref-gpu-nvgpu
  -> convert-nvgpu-to-nvvm
  -> convert-gpu-to-nvvm
  -> LLVM IR / NVPTX
```

WGMMA/TMA/mbarrier 优先使用当前工具链已有的 NVGPU Op。若本地 LLVM fork 与 upstream API 不同，只在 `FriskToNVGPU` adapter 层处理版本差异，不把差异泄漏进 layout algebra/solver。

当前计划只实现 conversion 微基准所需的最小 lowering adapter，并验证 WGMMA/TMA 布局契约能被后续 adapter 消费；完整的 `frisk-materialize-sm90-pipeline` 及 Frisk → NVGPU/NVVM lowering 单独规划。

## 9. 各类操作的推断规则

### 9.1 Elementwise

- 相同 shape 的 Tensor operand/result 默认要求相同 Distributed layout。
- scalar operand 不建立 layout var；lowering 时按 result owner 本地广播。
- 不同 shape 仅在显式 broadcast 语义存在时允许，不做隐式猜测。
- 多 consumer 冲突时保留候选到全局选择；不在第一个 consumer 处锁定。

### 9.2 Broadcast

- 输出 layout 投影到非 broadcast 维必须等价于输入 layout。
- 新增维的 carrier bits 可以映射为 0，形成 replication。
- 若 replication 会显著增加寄存器，候选中加入 producer rematerialization 或 consumer-side broadcast。

### 9.3 Transpose/Permute

- 通过 logical output permutation 与输入 canonical map compose。
- transpose view 本身不产生搬运；只有 consumer 要求不同 carrier 分布时产生 conversion。
- Shared transpose 必须区分 logical view 和 physical storage transpose，二者不可混为同一个 attr。

### 9.4 Reshape/Expand/Collapse/View/Bitcast

- 先在逻辑元素序上建立可证明的 reshape relation，再 compose layout。
- 普通 reshape 要求元素数量一致；dtype-changing view 使用 storage bit 数而非 `dtype.bytes()`，正确支持 sub-byte 类型。
- 跨过不可整除/动态边界时必须生成 predicate 或失败，不能截断。
- Alias view 不改变底层 storage layout，只改变 logical → base logical 的 view transform。

### 9.5 Copy/Load/Store

Copy 同时连接一个 Distributed var 和一个 Storage var：

```text
thread/register -> logical -> physical address
```

规则同时验证：

- global 合并访问；
- shared bank conflict；
- vector width 和 alignment；
- 每个 logical element 的 reader/writer ownership；
- TMA/cp.async/普通 load-store 的可行性；
- ragged tile 的 predicate 和 OOB 行为。

TMA 是候选，不是默认假设。SM90 上 TMA 适合多维 global/shared tile 搬运，并使用 mbarrier 完成机制；参考：[CUDA TMA](https://docs.nvidia.com/cuda/cuda-programming-guide/04-special-topics/async-copies.html)。

### 9.6 Fill

- Fill 只约束目标 Storage layout，并产生 Write effect。
- scalar fill value 本身无 Distributed layout。
- 并行 fill 必须由目标 storage map 推导唯一/无重叠 owner；若多线程写同址且无等价幂等保证，视为错误。

### 9.7 GEMM/WGMMA

GEMM 是最强 layout anchor。SM90 规则库根据 dtype、M/N/K、transpose/major mode 和 source memory space 枚举合法 instruction contract：

- C/D accumulator 的 `MmaEncodingAttr` 是 hard seed；
- A/B shared operand 的 Storage layout 必须可生成合法 warpgroup descriptor；
- 支持 `.ss` 路径；`.rs` 路径在对应 dtype/shape 和寄存器预算满足时作为候选；
- 多条 WGMMA 拼接的大 tile 通过 outer affine tile × inner WGMMA atom 表示；
- accumulator 到后续 epilogue 的 layout 尽量保持不变，避免立刻转换；
- 非整 tile 用 predicate/边界策略，不改变内层指令 atom。

合法 shape/dtype 表必须由 SM90 PTX/NVGPU verifier 数据驱动，不把一份易过期的 `if/else` 散落在 `FriskOps.cpp`。PTX 对 WGMMA 提供不同 `m64nNk*` fragment 与 shared descriptor 规则；参考：[PTX WGMMA 目录](https://docs.nvidia.com/cuda/parallel-thread-execution/contents.html)。

### 9.8 Reduce

Reduce 规则由 `(input layout, axes, combiner, output shape)` 推导：

1. 投影掉 reduce axes，得到每个 carrier 对 output logical element 的 ownership。
2. 若同一 output 只在单线程内聚合，保持本地 reduce。
3. 若跨 lane，生成 warp shuffle tree candidate。
4. 若跨 warp/warp-group，生成 shared tree/mbarrier candidate；只有指令/代价允许时使用。
5. 输出默认选举唯一 owner；consumer 明确要求 replicated 时再产生 broadcast/replication。
6. 检查非交换 combiner 的顺序语义，不能任意重排。

当前 Reduce 的 `replicateSize` 应逐步替换为 canonical distributed map 的 kernel/replication 语义，避免重复维护一个独立数字。

### 9.9 Region、Parallel 与控制流

- `scf.if` 的各分支 yield 与 result 必须统一 encoding；否则在分支内部插入 conversion，不能修改 join 后类型逃避约束。
- `scf.for` 的 iter_arg/result 是 hard-equal slot；init/yield 的 consumer use
  可由 solver 选择 conversion 到该 slot，物化后四者类型一致。
- `scf.while` 分别约束两个 tuple：init/before arguments/after-region yield，
  以及 condition forwarded arguments/after arguments/results。两个 tuple 可有不同
  arity/type；condition predicate 不计入 forwarded slot，不能按位置强行合并两侧。
- `frisk.parallel` 提供 topology 和硬件坐标域，不递归手调每个具体 op 的 `inferLayout`。
- Region 内未知但只操作 scalar 的 op 可忽略；未知 tensor/memref layout-bearing op 若未实现接口，应给出 unsupported diagnostic。

### 9.10 Alloc 与 Alias

- Shared alloc 创建 Storage var，allocation size 由已选 storage map 的最大 physical offset + element storage bits 计算。
- 所有 subview/layout_view/transpose view 进入同一 alias group。
- 同一 alias group 可以有不同 logical view transform，但只能有一个一致的底层 physical mapping。
- dtype-changing alias 以 bit address 验证 alignment、覆盖和不重叠。

## 10. SM90 Target Rule Library

### 10.1 结构

建议将规则拆成数据表和少量生成器：

```text
SM90TargetInfo
  |- topology: warp=32, warp_group=4 warps
  |- WGMMAInstructionTable
  |- TMAConstraintTable
  |- SharedSwizzleTable
  |- VectorAccessTable
  |- ReductionStrategyTable
  `- ResourceCostModel
```

SM90 的 warp-group 固定为 4 warps/128 threads；一个 block 可以有一个或多个 warp-group，不能把整个 compiler 写死为“block 恰好 128 threads”。

### 10.2 Shared/TMA 规则

首版支持 `none/32B/64B/128B` swizzle，并验证：

- shared/global base alignment；
- inner box bytes；
- pattern repeat 与 base offset；
- TMA last dimension 16-byte granularity；
- descriptor rank/box/stride；
- mbarrier completion 和 async proxy 可见性。

NVIDIA 对 compute capability 9 的 TMA swizzle 给出了 32B/64B/128B 模式及 128-byte 基址对齐等要求；这些约束应来自 target table 并通过 verifier 覆盖。参考：[CUDA TMA swizzle requirements](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tensor-memory-accelerator-tma)。

### 10.3 Target 配置

- 使用 `#nvvm.target` 表达 chip/features，DLTI 表达可查询的设备属性。
- `sm_90` 与 `sm_90a` 作为 feature set，不在通用 layout attr 中写字符串比较。
- 编译器启动时验证本地 NVGPU adapter 是否具备请求的 op/type；缺失时明确回退或报 unsupported。
- 固定 Frisk 的 LLVM/MLIR toolchain commit；升级 toolchain 时运行完整 layout/lowering 回归。

## 11. 验证、测试与性能守门

### 11.1 布局代数单元测试

- GF(2) rank/kernel/image/inverse/pseudoinverse；
- compose/direct product/project/permute；
- Affine outer 的 quotient/remainder/padding；
- Product layout canonicalization 和等价性；
- injectivity/coverage 反例；
- dynamic bounds、ragged predicate；
- sub-byte dtype bit address。

### 11.2 Op 与 Solver 测试

- 每类 op 的正向、反向和双向推断；
- 多 consumer 冲突和 conversion placement；
- region/loop fixed-point；
- alias/view consistency；
- candidate traversal 顺序打乱后结果不变；
- hard conflict 的 provenance diagnostic；
- 未实现 layout-bearing op 必须失败。

### 11.3 SM90 契约与已支持 lowering 测试

- WGMMA accumulator/operand、TMA descriptor、swizzle、alignment 和 box 的布局契约；
- ldmatrix/cp.async/vector fallback 的候选合法性；
- `frisk.convert_layout` 最小 shuffle/shared-exchange lowering adapter；
- 对当前代码生成路径已经支持的 kernel 执行 FileCheck、PTX 结构检查和 runtime correctness；
- 完整 WGMMA/TMA/mbarrier lowering 纳入后续独立计划，在该路径实现前只进行布局映射与静态成本验收。

### 11.4 TileLang 差分测试

差分比较的是语义，不是 attr 文本：

- 给定 logical shape、dtype 和 op graph；
- 枚举小 tile 的 `(thread, register) -> logical` ownership；
- 枚举 `logical -> shared offset`；
- 比较 coverage、replication、writer ownership 和地址映射；
- 允许 Frisk 选不同但等价或成本更低的 canonical layout。

优先搬运 TileLang 新版本中已经暴露过 bug 的场景：non-power-of-two broadcast、ragged padding guard、dtype-changing view、unused fragment、owner compatibility、swizzled TMA 和高维 GEMM。

### 11.5 性能验收

每个重要 kernel 至少记录：

- runtime/吞吐；
- global memory transactions/coalescing；
- shared bank conflict；
- WGMMA/TMA 是否命中；
- `convert_layout` 次数与搬运字节；
- register count、spill、shared bytes、occupancy；
- 编译时间和 solver candidate 数。

性能守门建议：

- 正确性、race、OOB 为零容忍。
- 对等 WGMMA/TMA kernel 不允许因为新代数产生额外 layout conversion。
- 默认候选若相对固定手写 baseline 有稳定回退，应禁用该选择并保留基准数据，直到代价模型修正。
- 创新布局先通过 feature flag 和 benchmark 白名单，再成为默认。
- 固定环境下，现有 codegen 支持的 kernel 中位运行时间回退不得超过 3%。
- 常规连通分量候选上限为 256、首版 beam width 为 32；布局推断占完整编译时间应不超过 10%，超限时必须输出 component 和候选统计。
- 对尚无目标 lowering 的 WGMMA/TMA 路径，只能报告契约、bank/coalescing、conversion 和资源静态指标，不能宣称运行时性能通过。

## 12. 分阶段实施方案

实施采用方案 A：“基础能力 + 纵向切片 + 逐类扩展”。旧 `LayoutAttr + DenseMap` 在迁移期作为语义基线保留，但所有新增能力只进入新 Attr、constraint、solver 和 materializer；当 Copy、Fill、Gemm、Reduce 和基础 region 全部迁移后删除旧生产路径。

### M0：基础设施与现状冻结

工作：

- 将 `LayoutInfer.cpp` 加入 `FriskTransforms`，启用 Layout Interface TableGen。
- 增加 `frisk-opt`、MLIR lit 和 C++ unit test 入口。
- 修正 Gemm/Copy/Fill/Reduce/Alloc/View 的 MemoryEffect。
- 将 `LogicalResult` 与 `ChangeResult` 分离。
- 固定 target/toolchain 信息，将现有 Gemm/Reduce 测试冻结为 legacy baseline。

验收：pass 可从命令行运行；现有测试全过；DCE/重排不会删除或跨越内存写；unsupported layout op 有明确诊断。

### M1：组合布局代数与新 Attr

工作：

- 实现 `LayoutMapInterface`、Affine/BitLinear/Product attrs。
- 实现 canonicalization、compose、projection、inverse、coverage/injectivity。
- 实现 `DistributedEncodingAttr`、`StorageLayoutAttr` 及 verifier。
- 实现旧 `LayoutAttr` 到 canonical map 的 `LegacyLayoutAdapter`。

验收：代数单测和小 tile 枚举 oracle 全过；现有 SM90 fragment/swizzle 均可无损表达。

### M2：Storage 纵向切片

状态：已完成（2026-09-06）。`frisk-infer-layouts` 是 module pass，固定执行 collect、strict propagation、common fixed-point、bootstrap solve、solved verification、materialization 和 materialized verification；候选默认按 stable ordinal 决定，完整 CostVector/beam 选择仍按计划留在 M5。

工作：

- 新增 `frisk.layout_view` 和 Storage binding。
- 为 alloc/view/copy 建立 Storage LayoutVar、constraint 和 provenance。
- 实现 strict/common propagation、solved binding materialization 和 storage verifier 的最小闭环。

验收：linear、transpose、padding、shared XOR swizzle 可从输入 IR 推断并物化；alias 冲突输出来源和坐标反例。

### M3：Distributed Tensor 与 conversion 纵向切片

工作：

- Local/Register Tile 迁到 RankedTensor encoding。
- 新增 `frisk.convert_layout` 和必要的 tile load/store carrier。
- 为简单 elementwise、transpose 和 load/store 建立 Distributed constraints。
- 同步改写 Tensor type、SCF block argument、yield/result。
- 实现用于真实 conversion 微基准的最小 shuffle/shared-exchange lowering adapter。

验收：同一 Tensor 可被两个不同布局 consumer 使用；bootstrap solver 能选择共同布局或显式 conversion；单 warp shuffle 和单 CTA shared-exchange adapter 生成的 IR 通过 MLIR verifier。实际 CUDA runtime correctness 与性能统一在 M6 固定环境中验收。

### M4：完整约束传播与旧 IR 迁移

工作：

- 完成 Distributed/Storage/alias/region constraint graph。
- 完成 hard seed、strict propagation 和 common fixed-point。
- 迁移 Copy、Fill、Gemm、Reduce、Parallel 规则。
- 提供旧 local MemRef fragment 到 SSA Tensor 的 normalization。
- 所有上述 Op 迁移后删除旧 `inferLayout(builder, DenseMap&)` 生产路径。

验收：基础 Frisk kernel 不再依赖旧 map 推断；结果与遍历顺序无关；所有 layout-bearing value 均解析；冲突可追溯到 source op。

### M5：候选求解、SM90 契约与 conversion 优化

工作：

- 实现连通分量候选生成、hard pruning、canonical 去重和有上限的 beam search。
- 将 conversion/rematerialization 作为普通候选，加入联合 CostVector。
- 实现数据驱动的 SM90 WGMMA/TMA/shared-swizzle 布局契约和静态成本。
- 实现 identity elimination、conversion hoist 和受控 rematerialization。

验收：多 consumer、Gemm、Reduce 冲突图具有确定选择；不会为了零 conversion 放弃明显更优的合法 SM90 主路径；尚无 lowering 的指令路径只报告静态质量。

### M6：系统硬化与验收

工作：

- 建立 TileLang 差分 corpus、fuzz/property test 和失败用例最小化。
- 覆盖 ragged、non-power-of-two、sub-byte、alias、region 和 ownership。
- 统计 conversion、bank conflict、coalescing、资源、candidate 数和编译时间。
- 对当前 codegen 或最小 adapter 支持的路径执行运行时性能回归。

验收：正确性零容忍项全部通过；固定环境下受支持 kernel 中位运行时间回退不超过 3%；常规候选上限、beam width 和布局编译时间满足第 11.5 节；扩展新 target 不需要修改通用 solver 核心。

## 13. 建议的代码组织

```text
include/Dialect/Frisk/IR/
  FriskLayoutAttrs.td
  FriskLayoutInterfaces.td
  FriskLayoutOps.td

include/Dialect/Frisk/Analysis/
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

lib/Dialect/Frisk/IR/
  FriskLayoutAttrs.cpp
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
  TestLayoutConversionLowering.cpp

tools/frisk-opt/

benchmark/layout/
  conversion/
  solver/

test/Dialect/Frisk/layout/
unittests/Dialect/Frisk/Layout/
```

当前体量较大的 `FriskOps.cpp` 中，GEMM layout 构造、shared swizzle 和通用 IR 方法应按上述边界拆分，避免继续形成 target rule 与 Op 语义耦合的单文件实现。

## 14. 风险与控制措施

| 风险                           | 控制措施                                                                                  |
| ------------------------------ | ----------------------------------------------------------------------------------------- |
| SSA Tensor 迁移影响现有前端    | 先做 normalization adapter，保留旧 IR 输入一段过渡期                                      |
| Hybrid algebra 复杂度过高      | 先覆盖 SM90 静态内层；动态只进入外层；小 tile 枚举作 oracle                               |
| 候选组合爆炸                   | 连通分量、hard pruning、canonical 去重、beam 上限和统计诊断                               |
| MLIR/NVGPU API 变化            | 固定 toolchain；版本差异集中在 FriskToNVGPU adapter                                       |
| 代价模型误判造成性能回退       | 字典序成本、手写 baseline、硬性能守门、feature flag                                       |
| Alias/view 推断错误            | ViewLike + AliasAnalysis + bit-level verifier；禁止静默合并冲突 alias                     |
| 只推断不影响代码生成           | 每个纵向切片必须包含 materialization；已支持路径增加最小 lowering adapter 和 runtime 验收 |
| 过度参考 TileLang 导致架构绑定 | 只维护语义差分和规则审计，禁止依赖 TVM/TIRX 类型                                          |

## 15. 最终决策清单

- [X] 采用 MLIR-native 双域 IR 架构（原架构候选“方案 C”）。
- [X] 实施采用方案 A：基础能力 + 纵向切片 + 逐类扩展。
- [X] Local/Register 使用 SSA RankedTensor encoding。
- [X] Shared/Global 保持 MemRef。
- [X] 使用 layout_view/binding 表达 MemRef 无法承载的 XOR storage layout。
- [X] 采用统一 constraint graph，但区分 Distributed 和 Storage 变量。
- [X] 采用 Affine × BitLinear 组合代数。
- [X] 首个 target 仅为 SM90/SM90a。
- [X] WGMMA/TMA 等 target rules 数据驱动并与 solver 解耦。
- [X] 使用最新 TileLang 固定 commit 做参考，不使用旧本地版本，不追踪浮动 HEAD。
- [X] 后续完整 lowering 优先复用 MLIR GPU/NVGPU/NVVM 成熟 Dialect。
- [X] 当前计划交付推断、验证、物化、conversion 优化和受支持路径的性能回归；完整 WGMMA/TMA lowering 单独规划。

## 16. 推荐的首个实现切入点

首个可执行切片不应直接重写完整 GEMM，而应先完成 Storage 闭环，再接入 Distributed Tensor：

1. 修正 MemoryEffect、Transform CMake、Interface TableGen 和 pass 驱动。
2. 建立 `frisk-opt`、lit/unit test 和 legacy baseline。
3. 实现 Affine/BitLinear/Product map、StorageLayoutAttr 及 verifier。
4. 实现 `layout_view`，让 alloc/view/copy 完成 constraint → propagation → binding → verification。
5. 再实现 DistributedEncodingAttr、Tensor carrier 和 `frisk.convert_layout`。
6. 让一个双 consumer 用例选择共同布局或显式 conversion，并通过最小 lowering adapter 的正确性测试。

这两个连续切片依次验证 MemRef storage binding、组合布局代数、SSA encoding、约束求解和 conversion 物化，避免布局代数、IR 迁移与完整 GEMM 同时失控。完成后再迁移 Gemm/Reduce 并接入 WGMMA/TMA 布局契约。

## 17. M3 Task 15–16 当前实现边界

Distributed SSA/use graph、双向 transpose relation、候选闭包后单调裁剪、
以及按新增 conversion 数优先的有界求解已实现；细节和 Task 16 接口见
[M3 Task 15 analysis contract](m3_task15_analysis.md)。

- Bootstrap 上限仍为每连通分量 8 variables、每 domain 4 candidates；不静默截断。
- `StorageAccess` 的正确性关系为 `S(D(h))`，coalescing 是 soft preference；
  replicated store 的 lowering 必须选定唯一 deterministic owner。
- SameLayout/Keep 保留 exact SSA encoding equality；Transpose 按 tensor axis
  permutation 比较物理映射，并允许 source/destination 使用不同合法 output labels。
- 当前支持非零 rank、静态 power-of-two bit-linear tensor tile；逻辑 extent 1
  受 M1 禁止 named zero-bit dimension 的限制，rank 0 会在候选构造前明确拒绝。
- `analysis-only` 完成 collect/solve/verify 且不改 IR；Task 16 已实现 SCF
  if/for/while collection 和 ModuleOp-only detached transaction，一致物化
  Tensor/SCF/function type。借用 graph/solution 的原始身份只在提交前有效；
  staged `RelationsOnly` 验证将 consumer 固定为实际 operand encoding，不重新求解。
  详细契约见 [M3 Task 16 materialization](m3_task16_materialization.md)。当前未知
  tensor op、tensor call、external tensor signature 明确失败，不宣称完整 Distributed lowering。
- Conversion analysis 限 single CTA；Task 17 的测试 lowering 对不支持的
  execution-topology 组合明确诊断，不将 Convertible 当作可执行性证明。

## 18. M3 Task 17 conversion cleanup 与测试 adapter

`frisk-optimize-layout-conversions` 已实现 canonical map 检查下的 identity/
inverse elimination，以及 A→B→C 合并。所有 replacement 必须保持 exact SSA
result type；即使 canonical map 相同，encoding attribute 不同也不能直接
替换 SSA value。多 consumer 的中间 conversion 保留其其他用途。

`frisk-test-lower-layout-conversions` 是测试专用的静态 BitLinear adapter：

- 共享 planner 计算 `R = rightInverse(Dsrc) compose Ddst` 并逐 owner 验证。
  按 named input 独立编解码 register/lane/warp/warp_group；缺失 input 维度
  视为 replication。逻辑坐标按 tensor row-major 分配 scratch，不直接使用
  concatenated matrix bits 作为多维线性地址。
- source owner 优先 same-thread、其次 same-warp replica，物理 thread/register
  字典序打破平局；无本地 replica 时使用已证明的 R owner。Shared writer 是
  独立的每 logical element 唯一 global right-inverse owner，不按各 consumer
  的 local replica 选择重复写入。单线程 register permutation 不生成 shuffle。
- Shuffle 对每个候选 source register 在全部 lane 无条件执行，再根据目标
  lane 的 source-register 请求选择。i8/i16/i32/i64、f16/bf16/f32/f64 均通过
  bit-preserving i32 word 传输；窄类型 zero-extend/truncate，64-bit 拆两 word
  后重组，float 只 bitcast，不做数值转换。
- 跨 warp 使用 gpu.func workgroup attribution、唯一 owner 条件 store、
  无条件 gpu.barrier、每 destination logical slot load；使用原 dtype、自然
  alignment、独立 buffer。默认 49152-byte **adapter budget** 可配置，不代表
  SM90 硬件最大值。计入既有 attribution、buffer 间 padding 和全部新增 buffer；
  dynamic/non-attributed workgroup storage 明确拒绝，避免未计入的共享容量。
- 仅接受 single-block gpu.func kernel 的直接 entry-block conversion，要求
  已知 `[32*warp*warp_group,1,1]` block size、lane=32、CTA=1、两端 execution
  topology 相同（register extent 可不同）。拒绝 nested if/loop、CFG、缺失
  block metadata、跨 CTA/cluster、unsupported dtype 和 adapter 大小上限。
  单 CTA 是每 tile 的通信范围，不限制不同 block 独立执行同一 kernel。
- 所有计划/容量检查先于 IR mutation；失败不改变 module。公开 representation
  仍为 encoded RankedTensor。内部 `unrealized_conversion_cast` 的 Tensor↔
  per-thread Vector 桥接只定义测试 ABI；M6 runtime harness 必须以真实 tile
  load/store transport 替换并消除这些桥接，才能进入可执行 GPU lowering。

M3 只验收 ownership/payload oracle、合法 GPU-level IR 和明确 unsupported
diagnostics，不宣称已完成 GPU runtime correctness 或性能验证。动态 encoded
tensor 和缺 coverage map 已被 M1 verifier 先拒绝；planner 的 programmatic
negative tests 独立覆盖其拒绝边界。详细接口及命令见
[Task 17 implementation](m3_task17_conversions.md)。
