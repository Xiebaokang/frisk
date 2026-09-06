# Triton、TileLang 与 Frisk 布局推断对比及 Frisk 架构决策

> 文档性质：技术对比 + Frisk 架构决策记录
>
> Frisk 目标平台：NVIDIA SM90/SM90a
>
> 审阅日期：2026-08-16
>
> 关联文档：[Frisk 布局推断系统设计方案](./layout_inference_design.md)、[GF(2) 与组合布局说明](./gf2_layout_guide.md)

## 1. 结论摘要

三者的核心路线可以概括为：

```text
TileLang = 以 Buffer 为中心的兼容布局闭包
Triton   = 带 Encoding 的 Tensor SSA + 硬件锚点驱动的多轮布局改写
           + 显式 ConvertLayout 的插入、消除、提升与重计算
Frisk    = MLIR 类型化的 Distributed/Storage 双域布局
           + 全图约束传播 + 有限候选联合选择
           + 求解后显式物化 conversion
```

需要先修正两个容易产生误解的说法：

1. **TileLang 并不是在前端创建每个 Op 时立即确定布局。**每个 TileOperator 确实实现自己的 `InferLayout` 规则，但最新布局系统是在 pipeline planning、software pipeline rewrite 等变换之后，由独立的全局 `LayoutInference` pass 收集 TileOp，再执行 strict、common fixed-point、free-mode 和 alias finalize。
2. **Triton 也不是所有 Op 各自独立选择“自己的最优布局”。**TTIR 转成 TTGIR 时先赋予默认 encoding，随后 coalesce、matmul acceleration、dot operand optimization 等 pass 建立高价值布局锚点；`RemoveLayoutConversions` 再从锚点传播布局、解决冲突并重写 IR。普通 elementwise/shape Op 更多是在传播中跟随锚点，而不是各自运行一个完整代价搜索。

Frisk 的目标方案不是“以 Triton 为主、再复制 TileLang 的规则”，而是把三层明确分开：

| 层次 | Frisk 的选择 |
| --- | --- |
| IR 表示 | 采用 MLIR/Triton 风格的类型化 Tensor SSA、MemRef 和显式 conversion |
| 规则传播 | 吸收 TileLang 的 strict/common/free 分阶段思想，但改成候选集合上的单调传播 |
| 决策机制 | 使用 Frisk 自己的全局约束图，对布局、硬件路径和 conversion placement 联合选择 |
| 布局代数 | 使用“仿射外层 × GF(2) 位线性内层”，而不是迁移 TVM PrimExpr，也不只复制 Triton encoding 类层次 |
| 目标支持 | 首阶段只实现 SM90/SM90a，通用 solver 与 SM90 rule library 分离 |

因此，Frisk 的准确定位是：

> **表示层接近 Triton，传播阶段借鉴 TileLang，决策层与组合布局代数形成 Frisk 自己的架构。**

这里所说的“创新”首先是可落地、可验证的编译器架构创新。只有在完整实现、差分测试和性能评测之后，才能进一步宣称算法或性能上的研究创新。

## 2. 比较前必须统一的概念

### 2.1 “布局”不是单一概念

三套系统使用了不同 IR 载体。如果把 register、shared memory、global memory 都笼统称为“内存布局”，会掩盖关键差异。Frisk 应至少区分两个布局域：

#### Distributed layout

描述一个逻辑 Tensor 的元素如何分布到执行资源：

```text
(register, lane, warp, warp-group, CTA) -> logical tensor coordinates
```

它回答：

- 某个逻辑元素由哪个 lane 持有；
- 一个 lane 的第几个寄存器槽保存该元素；
- 元素是否被多个线程复制；
- 哪个线程拥有最终 store/reduce 的写权限。

“Register”在这里不是一种可以被 MemRef 任意寻址的普通内存，而是 SSA value 在 lowering 后的物理承载方式。因此 Frisk 目标架构使用：

```text
RankedTensorType + DistributedEncodingAttr
```

来表达寄存器分布，而不是继续用 `memref<..., local>` 模拟可寻址的寄存器数组。

#### Storage layout

描述逻辑坐标如何映射到真实可寻址存储：

```text
logical tensor coordinates -> physical bit/byte address -> bank/segment
```

它回答：

- shared memory 是否转置、padding 或 XOR swizzle；
- 地址、stride、alignment 和 allocation size；
- TMA box、descriptor 和 swizzle 是否合法；
- 同一 storage 的 view/alias 是否保持一致。

Frisk 使用 `MemRef` 保持可寻址存储语义，并在 allocation/layout view/binding 上关联 `StorageLayoutAttr`。首阶段它主要服务 shared memory；global memory 通常继续使用 MemRef 自身的 strided/affine layout，只有存在显式物理重排时才需要额外 storage binding。

### 2.2 Conversion 的含义

`convert_layout` 不是改变数学 Tensor 的值，而是改变同一逻辑值在执行资源上的分布。它可能 lower 成：

- register shuffle；
- register → shared → register；
- warp/warp-group 间交换；
- 重计算 producer，直接生成目标布局；
- 在特殊情况下退化为带同步的较重搬运。

因此 conversion 同时具有正确性语义和性能代价。它必须显式、可验证，也必须进入全局成本选择。

### 2.3 “推断时机”需要分成三个时刻

不能只用“早”或“晚”描述布局推断，应区分：

1. **布局规则何时可见**：高层 TileOp、TTIR Op 还是目标化 GPU Op；
2. **布局何时成为 IR 的一部分**：Buffer annotation、Tensor type encoding 或独立 binding；
3. **冲突何时被解决并物化**：推断过程中、专用 propagation pass，还是 lowering 前。

## 3. 三者总体流程

### 3.1 TileLang 总体流程

以已审阅的 TileLang commit `6623b12d` 为基线，其 CUDA 主流程的关键顺序为：

```text
High-level Tile IR
  -> warp specialization / Blackwell annotation preparation
  -> pipeline planning
  -> software pipeline rewrite
  -> LayoutInference
       0. floating fragment -> fully replicated
       1. strict inference
       2. common BFS fixed-point
       3. free-mode root search
       4. alias finalize
  -> ReducerPlanAndMaterialize
  -> LowerTileOp
  -> TIRX/CUDA lowerings
```

总体逻辑是：

> 收集 TileOp 对 Buffer 的布局要求，通过多轮兼容性传播，为每个 Buffer/alias group 得到一个最终布局；布局冻结后再降低 TileOp。

### 3.2 Triton 总体流程

Triton NVIDIA 后端的关键逻辑不是一个单独的“布局推断 pass”，而是多轮目标化和布局优化：

```text
TTIR: unencoded tensor SSA
  -> TTIR to TTGIR TypeConverter
       tensor -> default BlockedEncoding
       type mismatch -> ConvertLayout materialization
  -> Coalesce
       为昂贵 load/store 选择合并访存布局
  -> RemoveLayoutConversions
       锚点传播、冲突消解、IR 重写
  -> OptimizeThreadLocality
  -> AccelerateMatmul
       选择 MMA encoding / dot operand / shared contract
       插入必要 ConvertLayout
  -> RemoveLayoutConversions
  -> OptimizeDotOperands / descriptor optimizations
  -> scheduling / pipelining / TMA lowering
  -> RemoveLayoutConversions（后续再次清理）
  -> TTGIR to LLVM/PTX
```

总体逻辑是：

> 让 load/store、dot/MMA、descriptor 等高价值 Op 成为布局锚点，通过带 encoding 的 SSA 类型表达布局差异，以显式 conversion 保证不同局部最优布局可以共存，再通过传播、rematerialization 和 conversion cleanup 减少代价。

### 3.3 Frisk 目标流程

Frisk 目标方案采用“先分析、后选择、再物化”：

```text
Frisk high-level MLIR
  -> layout IR normalization + verifier
  -> 建立 Distributed value graph、Storage alias graph、region edges
  -> 收集 Op constraints 和 SM90 instruction contracts
  -> hard seed
  -> strict propagation
  -> common bidirectional fixed-point
  -> component candidate generation + hard pruning
  -> 布局 / 指令路径 / conversion edge 联合代价选择
  -> controlled relaxation（仅处理普通候选域无解的情况）
  -> solved graph verification
  -> materialize encodings, storage bindings, convert_layout
  -> conversion canonicalization / hoist / rematerialization
  -> SM90 pipeline materialization
  -> Vector/MemRef/GPU/NVGPU/NVVM lowering
```

总体逻辑是：

> Op 只声明关系、合法集合和性能偏好，不直接锁定最终布局。求解器在整个布局连通分量上同时考虑 producer、consumer、storage、硬件指令和 conversion 位置，最后一次性提交类型与 binding。

## 4. TileLang 的具体布局推断逻辑

### 4.1 IR 载体和推断对象

TileLang 的布局推断建立在 TVM/TIRX Buffer 和 TileOperator 之上：

- shared memory 使用一般 `Layout`；
- fragment/local buffer 使用 `Fragment`；
- 推断状态主要是 `Map<Buffer, Layout>`；
- 同一底层 `buffer->data` 的不同 Buffer view 被归入 alias group；
- parallel loop 的布局和 predicate 以 annotation 形式回写到 IR。

这是一种 **Buffer-centric** 模型。一个 Buffer 在最终 layout map 中对应一个确定布局，consumer 一般需要共同接受这个布局。

### 4.2 每个 Op 提供规则，全局 pass 负责调度

TileOperator 的 `InferLayout` 会根据：

- 当前已经推断的 Buffer layout；
- target 和线程范围；
- TVM Analyzer 的等价性证明；
- 当前 inference level；

返回对 Buffer layout map 的更新。

但调用顺序和收敛过程由全局 `LayoutInference` pass 控制，而不是前端创建 Op 时立即执行。因此它同时具有：

- **局部规则**：Copy、Gemm、Reduce、Parallel 等 Op 知道自己的布局关系；
- **全局调度**：use-list、BFS queue、connected component 和 alias propagation 决定传播顺序。

### 4.3 五个阶段

#### 阶段 0：floating fragment 复制

如果 fragment buffer 在 TileOp 之外被普通控制流或表达式访问，TileLang 无法从 TileOp 契约推断线程访问模式，于是将其设为 fully replicated，保证任意线程都可读取。

#### 阶段 1：strict inference

传播由硬件/语义唯一决定的布局事实，并形成 `strict_layout_map`。后续 common/free 推断不能随意破坏这些事实。

#### 阶段 2：common BFS fixed-point

从所有 TileOp 开始，通过 Buffer use-list 反复调用 `InferLayout(kCommon)`；新布局会把相关 users 重新加入队列，直到没有更新。

此阶段主要寻找兼容布局闭包。已有布局与新布局相遇时会：

- 接受数学上相同的布局；
- 对非 strict fragment，在可证明 containment 时采用包含更多映射的布局；
- 对 compatible shared swizzle 尝试合并到更小粒度；
- 无法兼容时报告布局冲突。

#### 阶段 3：free-mode

对仍未完全解析的连通分量，TileLang 尝试让不同 Op 作为推断 root，执行 free-level 传播，并比较得到的寄存器数量；选择 replication/寄存器占用更小、且具有确定性 tie-break 的方案。

这已经不是简单的“第一个 Op 决定一切”，但候选空间和成本仍主要围绕 Buffer 兼容与 fragment register count，并不是一个把 conversion edge 作为变量的通用全图优化器。

#### 阶段 4：alias finalize

对共享同一底层 storage Var 的 Buffer：

- shape 相同则传播同一布局；
- shape/dtype 不同则根据 storage bit ratio reshape；
- 保证 alias group 的物理 storage footprint 一致。

### 4.4 冲突处理

TileLang 主要通过“找到一个大家都能接受的 Buffer 布局”消除冲突。其常规机制并不是为同一个逻辑 Buffer 保留多个 SSA encoding，再在 use 边界插入通用 conversion。

典型结果为：

```text
兼容              -> 合并/传播
fragment containment -> 选择可包含的布局
shared swizzle 可合并 -> 合并粒度
free-mode 有可行 root -> 选择寄存器更少的 root
仍不兼容           -> 编译期冲突或 unsupported
```

### 4.5 优势与局限

优势：

- TileOp 语义强，Copy/Gemm/Reduce 的规则靠近高层调度信息；
- Buffer 模型与 TIR lowering 直接衔接；
- strict/common/free 机制直观，兼容性强；
- 对高性能手写 tile kernel，布局通常由少数强算子约束，求解开销较低。

局限：

- 一 Buffer 一布局使多 consumer 的不同局部最优难以同时表达；
- 通用冲突恢复能力弱，很多冲突只能合并或失败；
- 类型系统不会自动验证每一条 SSA use 的 encoding 一致性；
- 推断代码绑定 TVM Analyzer、PrimExpr、Buffer、TIRX 和 TileLang op hierarchy；
- 若直接迁移到 Frisk，会把 MLIR 降格为 TVM 数据结构的承载壳。

## 5. Triton 的具体布局推断与优化逻辑

### 5.1 IR 载体

TritonGPU IR 将 distributed layout 作为 `RankedTensorType` 的 encoding。典型 encoding 包括：

- `BlockedEncodingAttr`；
- MMA encoding；
- DotOperand encoding；
- Slice encoding；
- shared/memdesc 相关 encoding；
- 底层可归一到 LinearLayout 的映射。

这使布局成为 MLIR 类型的一部分：两个 shape/dtype 相同但 encoding 不同的 Tensor 是不同类型，不能在不转换的情况下错误连接。

shared memory 则通过可寻址的 memdesc/storage 类型和相应 encoding 表达，不能简单理解为“所有内容都用 Tensor encoding”。Triton 与 Frisk 真正值得复用的思想，是区分 **分布式值布局** 和 **存储布局**，并让转换边界在 IR 中显式存在。

### 5.2 TTIR 转 TTGIR：先给默认布局

TTIR tensor 尚未带 GPU encoding。TypeConverter 为未编码的 `RankedTensorType` 添加默认 `BlockedEncodingAttr`；Dialect Conversion 在期望类型与已有类型不一致时，可以物化 `ConvertLayoutOp`。

这一步建立的是合法、统一的初始 TTGIR，不代表已经得到整张图的最终最优布局。

### 5.3 目标化 pass 建立布局锚点

#### Coalesce

对昂贵 load/store，Triton 根据 axis/contiguity 信息选择适合 memory coalescing 和 vectorization 的 blocked layout。实现通常先：

1. 把 memory op operands 转到 coalesced layout；
2. 创建使用该布局的新 memory op；
3. 再把结果转回原布局；

随后由 `RemoveLayoutConversions` 尝试把这个好布局向前后传播，从而消除临时 conversion。

#### AccelerateMatmul

当目标支持 Tensor Core 时，该 pass 会：

- 为 accumulator/result 选择 MMA encoding；
- 为 A/B 建立 DotOperand 或 shared operand contract；
- 插入从旧 encoding 到新 encoding 的 `ConvertLayoutOp`；
- 在 SM90 路径上结合 shared descriptor、WGMMA/TMA 相关 lowering。

因此 dot/MMA 是强布局锚点，而不是普通 elementwise Op 自己决定布局。

#### 其他锚点

当前 `RemoveLayoutConversions` 源码把 descriptor、昂贵 load/store、dot、atomic、特定 gather/reshape、部分 target-specific Op 识别为 layout anchor。锚点表示“这个布局具有应当保留的硬件价值”。

### 5.4 RemoveLayoutConversions 的主算法

当前实现的注释明确给出四步：

1. 找出希望保留布局的 anchor ops；
2. 从每个 anchor 向 descendants 传播 encoding；
3. 一个 value 可能得到多个候选 encoding，随后解决冲突并在必要处插入 conversion；
4. 按 dominance/structured-region 顺序重写 IR。

需要注意：当前冲突选择仍包含启发式逻辑。例如在已审阅源码中，load/store 倾向 blocked encoding，其他冲突倾向 MMA encoding，并留下“需要更完善 heuristic”的 TODO。因此不能把 Triton 描述成已经拥有统一、精确的全局最优布局求解器。

### 5.5 Conversion 优化不是简单删除

Triton 的 conversion 优化包含：

- identity/冗余转换消除；
- backward rematerialization：重算便宜 producer，直接产生目标 encoding；
- 把 conversion 提升到 broadcast/extend 之前，使搬运 Tensor 更小；
- 将 conversion 移入条件分支，避免无条件执行；
- dot operand conversion hoist；
- 对 region/block argument/yield/result 进行一致重写。

这说明 Triton 的真正优势不只是“允许冲突”，而是：

> 把冲突显式化之后，能够使用 SSA、dominance、slice analysis 和 canonicalization 对 conversion 做系统优化。

### 5.6 优势与局限

优势：

- Tensor encoding 与 MLIR SSA 类型系统结合，错误连接能够被 verifier 捕获；
- 不同 consumer 可以拥有不同局部布局，冲突不必直接编译失败；
- coalescing、MMA、descriptor 等硬件优化可以独立迭代；
- conversion 可以利用 dominance、region 和 rematerialization 系统优化；
- 已经形成成熟的 NVIDIA lowering 与性能经验。

局限：

- 布局决策分散在多个 pass 中，pass 顺序本身承担隐式策略；
- 某个 pass 先建立局部最优，后续 pass 再修补 conversion，可能产生局部最优陷阱；
- 冲突消解仍存在 target/pass-specific heuristic；
- encoding 类型层次和转换清理代码复杂，新硬件特例容易扩散；
- conversion cleanup 能减少已有代价，但不能保证此前锚点选择是全局最优。

## 6. Frisk 的目标布局推断逻辑

### 6.1 当前实现状态与目标架构必须分开

截至本文审阅时，Frisk 当前代码已经有：

- 基于 `AffineMap` 的 `LayoutAttr`；
- `LayoutInterface::inferLayout`；
- Gemm、Reduce、Parallel 等局部推断代码；
- `DenseMap<Value, Attribute>` 形式的推断状态；
- 一部分 shared/fragment 布局构造工具。

但当前 `LayoutInfer.cpp` 仍基本是空 pass 壳，现有 register fragment 仍使用 MemRef 表达，也尚未具备本文所述的全图候选求解、类型化 Distributed encoding、Storage binding、显式 conversion 物化和完整 verifier。

因此本节描述的是 **Frisk 需要实现的目标架构**，不是对当前完成度的描述。

### 6.2 表示层：Distributed 与 Storage 分离

#### Register/local value

```mlir
tensor<64x64xf32, #frisk.distributed<...>>
```

使用 SSA Tensor 表示逻辑值，`DistributedEncodingAttr` 表达 lane/warp/register ownership。它不是“专门描述硬件寄存器编号”，而是描述逻辑元素到并行执行资源的分布；真实 PTX register allocation 仍由后端完成。

#### Shared/global storage

```mlir
memref<64x64xf16, ..., #gpu.address_space<workgroup>>
```

使用 MemRef 表示真实可寻址 storage，`StorageLayoutAttr` 通过 allocation/layout view/binding 表达 padding、transpose、XOR swizzle 和 bit address。global 通常使用标准 MemRef layout；shared 是首阶段 StorageLayoutAttr 的主要使用者。

#### 显式边界

- distributed → distributed：`frisk.convert_layout`；
- distributed ↔ storage：layout-aware load/store/copy；
- view/transpose/reshape：优先只改变坐标解释；只有 carrier distribution 或真实 storage mapping 改变时才产生搬运。

### 6.3 布局代数：仿射外层 × GF(2) 位线性内层

Frisk 不应让一个表达系统勉强承担所有问题，而应把布局分成：

```text
ProductLayout = AffineOuterLayout × BitLinearInnerLayout
```

#### 仿射外层

负责：

- tile/block/warp-group 的 quotient 和 remainder；
- 非 2 次幂 shape；
- dynamic/ragged boundary 和 predicate；
- padding、stride、base offset；
- logical view 与 storage view 的组合。

它可以复用 MLIR Affine、Presburger、MemRef layout 和 canonicalization 能力。

#### GF(2) 位线性内层

负责固定小 tile 内的位级分布：

- register/lane/warp bits 到 logical coordinate bits；
- XOR swizzle；
- transpose/permute/compose；
- rank、kernel、image、inverse/pseudoinverse；
- coverage、injectivity 和 replication 的精确证明。

#### 组合方式

先由仿射外层确定逻辑点属于哪个大 tile、边界是否有效，再由 GF(2) 内层确定这个 tile 内由哪个 lane/register 持有，或映射到哪个 shared bank/offset：

```text
logical coordinate
  -> affine quotient/remainder
  -> (outer tile coordinate, inner tile coordinate, predicate)
  -> GF(2) transform(inner coordinate / hardware bits)
  -> distributed owner or storage inner offset
  -> affine base + stride + transformed inner offset
```

相比 TileLang，这避免把全部证明绑定到 TVM PrimExpr/Analyzer；相比只使用 Triton LinearLayout，它不要求位线性代数同时承担动态边界、非 2 次幂和物理 padding。详细说明见 [GF(2) 与组合布局说明](./gf2_layout_guide.md)。

### 6.4 Op 不返回最终布局，而是声明约束

新的 Op interface 应返回 `LayoutConstraintSet`，而不是原地修改 `DenseMap<Value, Attribute>`。约束至少包括：

| 约束 | 含义 | 示例 |
| --- | --- | --- |
| `RequireEncoding` | 必须属于指定合法集合 | WGMMA result、用户 mandatory annotation |
| `SameDistributed` | 两个 SSA value 分布相同 | elementwise operand/result |
| `TransformDistributed` | 经过坐标变换后的分布关系 | transpose、reshape、broadcast |
| `StorageCompatible` | alias/view 必须保持物理一致 | shared view、subview |
| `InstructionContract` | 布局必须匹配目标指令 | WGMMA、TMA、ldmatrix |
| `UniqueOwner` | 写入或 reduce 结果拥有唯一 owner | store、reduce |
| `Convertible` | 可在边界插入显式 conversion | 多 consumer 冲突 |
| `Preference` | 可违反但需要计入代价 | coalescing、bank conflict、replication |

Op 可以提供候选生成 hook，但不能遍历任意 IR 或直接提交全局选择。SM90 rule library 负责返回合法候选、指令契约、资源限制和局部成本。

### 6.5 完整推断阶段

#### A. Normalization 与前置验证

- 规范 view/transpose/reshape；
- 修正 MemoryEffect；
- 建立 `#nvvm.target`/DLTI target 信息；
- 将 register fragment MemRef 逐步转为 Tensor SSA；
- 验证 shape、dtype、memory space、region terminator。

#### B. 建立图

- 每个 Tensor SSA value 建 Distributed `LayoutVar`；
- 每个 allocation/layout view/kernel storage 参数建 Storage `LayoutVar`；
- 建立 def-use、alias/view、SCF block argument/yield/loop-carried edges；
- 记录每个 use 的布局要求，不做最终选择。

#### C. Hard seeds

按语义优先级加入：

- 用户 mandatory annotation；
- WGMMA/TMA 的不可违反指令契约；
- 已存在且通过 verifier 的 encoding/binding；
- 显式 conversion 的 source/target；
- scalar/size-one 的 replicated seed。

#### D. Strict propagation

只传播唯一事实，例如 exact same-layout、可逆 transpose、WGMMA parent/operand 关系、alias storage、SCF 类型一致关系。不创建默认布局、不 padding，也不物化 conversion。

#### E. Common fixed-point

在候选集合上运行双向 worklist：

- producer → consumer 传播；
- WGMMA/TMA/store 等强 consumer → producer 反向传播；
- alias/view 传播 storage mapping；
- region join 传播候选交集；
- elementwise/shape transform 传播坐标变换后的候选。

候选 domain 一旦建立，只允许交集、投影和剪枝，保证阶段内单调、可终止、与遍历顺序无关。

#### F. 连通分量候选生成与 hard pruning

SM90 rule library 生成有限候选：

- distributed：blocked、lane/warp-striped、replicated、WGMMA-compatible；
- storage：linear、transpose、padding、32B/64B/128B swizzle；
- copy：vector load/store、cp.async、TMA；
- reduce：shuffle、shared tree、replicated result、elected owner；
- GEMM：合法 WGMMA shape/dtype/major mode；
- conversion edge：直接保留、显式转换、便宜 producer rematerialization。

随后删除违反 shape、alignment、coverage、injectivity、TMA box、WGMMA contract 或 ownership 的候选。

#### G. 联合代价选择

优化单元是 layout constraint graph 的连通分量。一个 assignment 同时包含：

```text
Distributed layout choices
+ Storage layout choices
+ instruction path choices
+ conversion/rematerialization choices
```

第一版采用 hard pruning + 动态规划/有上限的 beam search，不引入无界 ILP/SMT。评估维度包括：

```text
illegal_or_unsupported              // 必须为 0
instruction_path_and_estimated_work // WGMMA/TMA/cp.async/SIMT 与实际 shape
global_memory_transactions
shared_bank_conflicts
conversion_bytes_and_sync_on_critical_path
spill_risk_and_registers
shared_bytes_and_occupancy
replication_and_code_size
deterministic_tiebreak
```

成本选择不能把“conversion 数量最少”当作绝对目标。例如一个 conversion 如果能让主 GEMM 命中 WGMMA，通常优于零 conversion 的 SIMT 路径；反之，为收益很小的局部 coalescing 在热循环内引入重型 shared conversion，也可能得不偿失。

#### H. Controlled relaxation

**显式 conversion 本身不应只在“零转换无解”时才出现。**它是普通候选域中的一等决策变量。只有普通合法候选全部失败时，才进入 relaxation：

1. 增加受控 replication；
2. 增加 shared padding 或降低 vector width；
3. 采用 guarded/ragged fallback；
4. TMA 回退 cp.async/vector copy；
5. WGMMA 路径不合法时回退到 warp MMA/SIMT。

每次 relaxation 必须带 provenance 和新增成本，不能静默改变 mandatory annotation、writer ownership 或 alias 语义。

#### I. 求解结果验证

检查：

- Attr 格式与 canonical form；
- tensor shape/dtype 与 encoding；
- distributed coverage/replication/owner；
- storage injectivity/alignment/allocation size；
- alias/view 一致性；
- WGMMA/TMA/copy contract；
- conversion source/target 的合法性和必要性；
- 所有 layout-bearing value 已解析。

诊断应报告 source location、冲突候选、触发约束、传播链和一个具体坐标反例。

#### J. 物化与 conversion 优化

- 将 Distributed 结果写入 Tensor encoding；
- 将 Storage 结果写入 allocation/layout binding；
- 同步更新 region block arguments、yields 和 results；
- 在求解选择的边界插入 `frisk.convert_layout`；
- 删除 identity/相邻可合并 conversion；
- hoist loop-invariant conversion；
- 对便宜 pure slice 做受控 rematerialization。

conversion cleanup 只优化已经验证的选择，不能承担“修复错误推断”的职责。

#### K. SM90 lowering

目标流水线为：

```text
frisk-normalize-layout-ir
  -> frisk-infer-layouts
  -> frisk-verify-layouts
  -> frisk-materialize-sm90-pipeline
  -> convert-frisk-to-vector-memref-gpu-nvgpu
  -> convert-nvgpu-to-nvvm
  -> LLVM/PTX
```

通用 solver 不直接构造 WGMMA/TMA Op；SM90 adapter 把已经选定并验证的 instruction contract 转换到 NVGPU/NVVM 或必要的 target-specific op。

## 7. 各环节横向对比

| 环节 | TileLang | Triton | Frisk 目标方案 |
| --- | --- | --- | --- |
| 基础 IR | TVM/TIRX Buffer + TileOp | MLIR TTIR/TTGIR Tensor SSA + memdesc | MLIR Frisk Tensor SSA + MemRef |
| Register 表示 | local/fragment Buffer + Fragment layout | RankedTensorType encoding | RankedTensorType + DistributedEncodingAttr |
| Shared 表示 | Buffer + Layout | memdesc/storage encoding | MemRef + StorageLayoutAttr binding |
| 初始布局 | 用户 annotation、TileOp strict rule | TTIR→TTGIR 默认 BlockedEncoding | hard seeds；未解析值不急于给默认布局 |
| 推断时机 | pipeline rewrite 后、LowerTileOp 前 | TTGIR 转换及多个 GPU optimization pass 中反复调整 | normalization 后统一分析，SM90 pipeline materialization 前提交 |
| 局部规则 | 每个 TileOperator 的 `InferLayout` | coalesce、dot/MMA、descriptor 等 target pass | Op constraint interface + SM90 rule library |
| 全局状态 | `Map<Buffer, Layout>` | value → encoding candidates，类型直接携带结果 | LayoutVar domain + constraint/provenance graph |
| 传播方式 | strict + BFS common + free root search | anchor → descendants propagation，多轮 pass | strict + 双向 fixed-point + component candidate solve |
| 反向推断 | TileOp 可依据已知 buffer 推另一端 | anchor propagation/rematerialization 中发生 | 一等能力；consumer contract 可反推 producer/storage |
| Alias/view | 同 data Var 的 Buffer group finalize | MLIR SSA/memdesc view 规则 | MLIR AliasAnalysis/ViewLike + storage graph |
| 多 consumer | 尽量找到一个共同 Buffer 布局 | 可保留局部不同 encoding，以 conversion 分隔 | 联合比较共享布局、边界 conversion、rematerialization |
| 冲突处理 | containment、swizzle merge、free root；仍冲突则失败 | heuristic 选 encoding并插 conversion | hard conflict 诊断；soft conflict 进入全局候选选择 |
| Conversion | 不是通用核心抽象 | 显式 ConvertLayout，随后多轮优化 | 显式且是求解变量；求解后统一物化 |
| 成本重点 | 兼容性、replication/register count | pass-local hardware heuristic + conversion cleanup | 指令路径、访存、bank conflict、conversion、资源联合评估 |
| Region/loop | TIR loop annotation 与 lowering | structured MLIR region 重写 | SCF 类型一致约束 + region fixed-point |
| 验证 | Analyzer/ICHECK/专用 validator | MLIR verifier + target-specific checks | MLIR verifier + 数学证书 + provenance diagnostic |
| 代数 | PrimExpr/IndexMap/Layout/Fragment | encoding + LinearLayout/GF(2) 能力 | Affine outer × GF(2) inner canonical product |
| Target 耦合 | TileOp/layout 与 TVM CUDA lowering 紧密 | target-specific encoding/pass 较多 | 通用 solver 与 SM90 rule/adapter 分层 |
| 默认失败策略 | 找不到共同布局时冲突 | 插 conversion 或 target fallback | 先全局选成本最低合法解；无普通解才受控 fallback |

## 8. 同一个 SM90 WGMMA 数据流在三者中的处理

考虑：

```text
global A/B
  -> shared A/B（希望 TMA + 128B swizzle）
  -> WGMMA
  -> accumulator
      ├─ consumer 1: elementwise epilogue
      └─ consumer 2: coalesced global store
```

假设 WGMMA accumulator 的最佳 distributed layout 与最终 store 的最佳 coalesced layout 不同。

### 8.1 TileLang

1. TMA/Copy、Gemm、Parallel 等 TileOp 被全局 pass 收集；
2. Gemm strict rule 推出 accumulator fragment 以及 A/B shared layout 要求；
3. Copy rule 将 shared layout 与 global copy/thread mapping 关联；
4. epilogue 和 store consumer 尝试接受/传播同一 accumulator Buffer layout；
5. 如果 fragment 关系可以 containment/replication 兼容，则选共同布局；
6. free-mode 可以尝试不同 root，并倾向较少寄存器的结果；
7. 如果 store 所需布局与 accumulator 布局无法兼容，也没有 Op-specific lowering 处理该差异，则发生冲突或走专门 fallback。

这里的核心目标是：**尽量让一个 Buffer layout 同时满足整条链。**

### 8.2 Triton

1. 初始 TTGIR Tensor 具有 BlockedEncoding；
2. Coalesce 为 load/store 建立合并访存布局，并暂时插入 conversions；
3. AccelerateMatmul 为 dot result/operands 建立 MMA 相关 encoding，并插入 conversions；
4. MMA、昂贵 store、descriptor 成为 anchors；
5. `RemoveLayoutConversions` 从多个 anchor 传播 encoding；
6. accumulator/epilogue 可以继续使用 MMA-compatible encoding；
7. store 边界若仍需要不同 encoding，则保留 conversion；如果 epilogue 很便宜，可能 rematerialize 或移动 conversion；
8. lowering 将剩余 conversion 转成 register shuffle 或 shared-memory exchange。

这里的核心目标是：**允许多个硬件局部最优共存，再优化它们之间的显式边界。**

### 8.3 Frisk

Frisk 不在首次看到 WGMMA 或 store 时立即提交布局，而是建立一个联合约束分量：

```text
Storage(A_shared) --TMA/WGMMA contract--+
Storage(B_shared) --TMA/WGMMA contract--+--> Distributed(acc)
                                              | same/transform
                                              +--> epilogue
                                              |
                                              +--convertible--> store layout
```

候选至少包括：

- A：accumulator 和 epilogue 保持 WGMMA encoding，store 前转换；
- B：便宜 epilogue 在 store encoding 下 rematerialize；
- C：整段使用一个兼容布局，完全不转换，但可能降低 store coalescing；
- D：改变 shared swizzle/copy path，但仍满足 WGMMA contract；
- E：WGMMA/TMA 不合法时才考虑 cp.async 或更低性能计算路径。

solver 比较整条 critical path 的 WGMMA/TMA 命中、global transaction、bank conflict、conversion bytes/sync、register 和 occupancy 后选择。通常 WGMMA result 是强锚点，因此更可能选择 A 或 B，而不会为了“零 conversion”破坏主计算路径。

这里的核心目标是：

> **在一次联合决策中同时选择硬件路径、两类布局和转换位置，而不是先固定局部布局再完全依赖 cleanup 修补。**

## 9. Frisk 这样设计的好处

### 9.1 相比直接迁移 TileLang

1. **保留 MLIR 类型安全。**不同 distributed layout 是不同 Tensor type，错误 use 不能被一个无类型 `DenseMap` 静默掩盖。
2. **多 consumer 不再要求一 Buffer 一布局。**同一 SSA value 可以在不同 use 边界显式转换，不必为了兼容最弱 consumer 牺牲主路径。
3. **控制流处理更自然。**SCF block argument、yield、loop-carried value 和 result 的 encoding 通过类型一致性约束统一处理。
4. **alias 与 storage 语义分开。**MemRef/AliasAnalysis 处理真实 storage；Tensor SSA 处理 register ownership，避免把两类关系混在 Buffer map 中。
5. **复用成熟基础设施。**Dialect Conversion、DataFlow、Dominance、SideEffect、Affine、Presburger、Vector、GPU、NVGPU、NVVM 都可以直接接入。
6. **长期独立。**Frisk 只审计 TileLang 的语义、规则和测试，不依赖 TVM/TIRX ABI。

### 9.2 相比机械复制 Triton

1. **布局选择更集中。**锚点、storage、distributed、conversion 和资源成本在同一 constraint graph 中可见，减少 pass 顺序造成的隐式决策。
2. **候选冲突有明确 provenance。**不仅知道“插了 conversion”，还知道由哪两个 Op、哪条指令契约和哪次传播导致。
3. **组合代数覆盖边界语义。**GF(2) 负责精确位线性内层，Affine/Presburger 负责非 2 次幂、dynamic、padding 与 ragged，不让一种代数承担所有问题。
4. **target 特例受控。**SM90 rule library 产生候选和成本，通用 solver 不散布 `if (sm90)`。
5. **conversion 是前置决策变量。**不是只有布局都定完之后才被动清理，可以直接比较“共同布局”和“局部好布局 + conversion”的端到端成本。

### 9.3 对长期扩展的价值

- 新增 Op：实现 constraint interface 和 verifier，不必修改全局遍历逻辑；
- 新增硬件：增加 target rule library/adapter，不改核心布局图；
- 升级 MLIR：IR、analysis、conversion 和 lowering 仍遵循标准接口；
- 接 autotuning：有限候选和稳定 cost key 已经形成清晰入口；
- 做性能回归：可以按 candidate、conversion、register、bank conflict 输出统计，而不是只比较最终 PTX。

## 10. Frisk 的创新点

### 10.1 双域统一约束图

Frisk 既不把 register value 当成 Buffer，也不让 StorageLayoutAttr 同时承担线程 ownership。Distributed graph 和 Storage alias graph 分开建模，却通过 load/store/copy/WGMMA/TMA constraint 在同一个 component solve 中联合选择。

这是 Frisk 最重要的架构创新：**语义分离，决策联合。**

### 10.2 “仿射外层 × GF(2) 位线性内层”的规范化布局代数

创新不在于单独使用 Affine 或 GF(2)，而在于为两者规定清晰边界、组合规则和 canonical form：

- 外层处理不规则 shape 和物理地址结构；
- 内层处理硬件位级分布和 XOR；
- compose 后可以证明等价、coverage、injectivity 与 ownership；
- 同一数学布局的不同构造路径可 canonicalize 到稳定表示。

它同时避免 TileLang/TVM 表达的体系绑定，也避免 encoding class 不断特化却缺少统一数学等价判定。

### 10.3 布局、指令路径和 conversion placement 联合选择

传统实现容易分成：先选 layout，再插 conversion，再清理。Frisk 把以下变量放进同一次 component selection：

- WGMMA/TMA/cp.async/SIMT 路径；
- distributed ownership；
- shared swizzle/padding；
- conversion 放在哪条 SSA edge；
- 是否 rematerialize/replicate；
- register/shared/occupancy 成本。

这不是承诺求出理论全局最优，而是在有限、经过 hard pruning 的 SM90 候选域中做可控的全局比较。

### 10.4 可解释推断

每个候选和剪枝动作保留 provenance：

```text
候选从哪个 Op/target rule 产生
-> 经过哪些关系传播
-> 被哪条 hard constraint 删除
-> conversion 为什么保留
-> 最终 assignment 为什么胜出
```

验证失败输出实际坐标反例或冲突链，而不是只触发 assertion。这使布局系统可以测试、调试和长期演进。

### 10.5 语义差分而不是实现依赖

Frisk 固定 TileLang 参考 commit，将小 tile 的：

- `(thread, register) -> logical coordinate`；
- `logical coordinate -> shared bit offset`；
- coverage、replication、owner、bank；

做语义差分。Frisk 可以选择与 TileLang 文本不同但数学等价、或成本更低的布局，不迁移 TIRX 类型和内部 solver。

### 10.6 创新边界

以下内容不能只凭设计文档宣称已经实现：

- 全局选择一定优于 Triton heuristic；
- 组合代数一定产生更快代码；
- conversion 一定比 Triton 更少；
- 编译时间一定可控。

它们必须由 layout corpus、SM90 lowering 检查、runtime correctness、Nsight 指标和对照 benchmark 证明。Frisk 的创新目标是创造更好的决策空间和工程边界，而不是为了形式创新牺牲性能。

## 11. Frisk 架构决策

### 11.1 已决定采用

| 决策 | 结论 | 原因 |
| --- | --- | --- |
| Register carrier | SSA Tensor + DistributedEncodingAttr | 类型安全、def-use/region 可组合、显式 conversion |
| Storage carrier | MemRef + StorageLayoutAttr binding | 保留可寻址、alias、alignment 和 memory effect 语义 |
| Op interface | 返回 constraint/candidate，不直接改最终 map | 支持冲突、候选、代价和 provenance |
| 推断框架 | strict + common fixed-point + component solve | 兼顾确定事实传播和多 consumer 全局选择 |
| Conversion | 一等候选变量，求解后显式物化 | 不把零转换绝对化，也不依赖事后修补 |
| 布局代数 | Affine outer × GF(2) inner | 同时覆盖动态/不规则外层与位线性硬件内层 |
| Target 策略 | 首阶段只支持 SM90/SM90a | 收敛规则和性能验证范围 |
| TileLang 关系 | 最新审计 commit 作语义 oracle，不链接/不 vendor | 吸收经验，保持长期独立 |
| Triton 关系 | 参考类型、anchor、conversion/remat 机制，不复制 pass 树 | 复用 MLIR 思想，避免特例扩散 |

### 11.2 明确不采用

- 不迁移 TVM/TIRX/PrimExpr/Buffer 作为 Frisk layout ABI；
- 不继续以 `memref<..., local>` 作为长期 register value 模型；
- 不让 Op 访问并修改一个全局 `DenseMap<Value, Attribute>` 后立即锁定布局；
- 不要求整张图零 conversion；
- 不允许 conversion cleanup 修复不合法的推断结果；
- 不在通用 solver 中散布 SM90 intrinsic 构造；
- 首版不引入无界 ILP/SMT，也不追求任意 target 的抽象完备性。

### 11.3 Frisk 对 TileLang/Triton 的取舍

```text
保留 TileLang：
  - TileOp 强语义
  - strict/common/free 分阶段思想
  - alias/storage-bit 一致性
  - 语义差分 corpus

拒绝 TileLang：
  - TVM/TIRX 类型依赖
  - Buffer 单布局作为唯一公共模型
  - 冲突主要靠合并或失败

保留 Triton：
  - Tensor encoding
  - 显式 ConvertLayout
  - anchor propagation
  - dominance/region/rematerialization 优化

拒绝机械复制 Triton：
  - 大量特化 encoding/pass 树
  - 依赖 pass 顺序形成的隐式全局策略
  - 先局部定布局、再主要依靠 cleanup 修补
```

## 12. 风险、性能边界与验收标准

### 12.1 主要风险

| 风险 | 后果 | 控制方式 |
| --- | --- | --- |
| 候选组合爆炸 | 编译时间不可控 | 连通分量、hard pruning、canonical hash、beam 上限、统计诊断 |
| Cost model 判断错误 | 选择多余 conversion 或低性能路径 | 保守规则、benchmark 白名单、feature flag、稳定 fallback |
| Tensor SSA 改造范围大 | 影响前端与 lowering | 过渡 op/adapter，分阶段替换 local MemRef |
| GF(2)/Affine compose 错误 | ownership、OOB、bank mapping 错误 | property test、枚举小 tile、SM90 contract verifier |
| conversion lowering 过重 | shared traffic、同步和寄存器增加 | 估算 bytes/sync/critical path，hoist/remat，性能守门 |
| target rule 与 NVGPU 不匹配 | 合法布局无法 lowering | 固定 LLVM commit，SM90 adapter 测试，明确 unsupported |

### 12.2 正确性验收

- 所有 layout-bearing SSA value 和 storage anchor 均已解析；
- region/block argument/yield/result 类型一致；
- distributed coverage、replication、unique owner 正确；
- storage injectivity、allocation size、alignment、alias 正确；
- WGMMA/TMA/mbarrier contract 完整；
- ragged/non-power-of-two/sub-byte dtype 无 OOB/race；
- 推断结果与遍历顺序无关。

### 12.3 性能验收

每个关键 kernel 至少记录：

- WGMMA/TMA/cp.async 是否命中；
- runtime、吞吐与端到端延迟；
- global transactions/coalescing；
- shared bank conflict；
- `convert_layout` 数量、bytes、同步及所在循环层级；
- register、spill、shared bytes、occupancy；
- solver candidate 数与编译时间。

重要边界：

- 不为了追求“创新布局”破坏已验证的 WGMMA/TMA 主路径；
- 对等 kernel 不应无原因增加热循环 conversion；
- 与手写/TileLang/Triton baseline 稳定回退的候选不能默认开启；
- 任何性能结论必须在固定 shape、dtype、stage、clock/测量协议下比较。

## 13. 对现有设计文档的修订建议

[layout_inference_design.md](./layout_inference_design.md) 当前 Stage G 把 conversion 纳入 CostVector，但 Stage H 又写成“只有不存在零转换可行解时才允许松弛”。两者会导致策略歧义。

建议后续实现前统一为：

1. Stage F 同时生成无 conversion、显式 conversion 和 rematerialization 候选；
2. Stage G 对它们进行端到端联合选择；
3. conversion 是强成本项，但不是绝对禁止项；
4. Stage H 只负责 ordinary candidate domain 无解后的 replication/padding/instruction fallback；
5. Stage J 根据已选 edge 物化 conversion 并做 cleanup。

换言之：

```text
错误策略：先强制寻找零 conversion，失败后才考虑 conversion

推荐策略：零 conversion 与有 conversion 都是合法方案
          -> 在满足硬约束的前提下比较端到端成本
          -> 求解后才物化选中的 conversion
```

这个修订不会让 Frisk 退化成“到处插 conversion”的 Triton 简化版。相反，它使 compatibility 成为强偏好，同时保留为了 WGMMA/TMA、coalescing 或 bank-conflict 收益而进行受控转换的能力。

## 14. 最终结论

TileLang、Triton、Frisk 不应被简化成“早推断”和“晚推断”的区别。更准确的差异是：

- TileLang 以 Buffer 兼容闭包为中心，布局在高层 TileOp lowering 前冻结；
- Triton 以 encoded SSA 和硬件锚点为中心，在多个 TTGIR pass 中迭代布局，并用显式 conversion 连接局部最优；
- Frisk 应以 MLIR-native 的双域布局和全局约束图为中心，在物化前联合选择布局、指令路径和 conversion placement。

Frisk 的价值不是单独“比 TVM 更好”或“比 Triton 更新”，而是利用 MLIR 将以下能力组合成一套可长期扩展的系统：

```text
类型安全的 SSA
+ 明确的 storage/alias 语义
+ TileOp/SM90 强约束
+ Affine × GF(2) 精确代数
+ 全局有限候选选择
+ 显式且可优化的 conversion
+ 可解释验证与性能守门
```

在该架构下，Frisk 可以吸收 TileLang 和 Triton 的成熟经验，同时保持自己的 IR、求解器、目标规则库和长期演进路径。

## 15. 主要参考源码与文档

### TileLang

- [TileLang CUDA pipeline（固定审计 commit）](https://github.com/tile-ai/tilelang/blob/6623b12d232b343648a5ba99992e3e6f0d6376d2/tilelang/cuda/pipeline.py)
- [TileLang LayoutInference（固定审计 commit）](https://github.com/tile-ai/tilelang/blob/6623b12d232b343648a5ba99992e3e6f0d6376d2/src/transform/layout_inference.cc)
- [TileLang layout 实现目录](https://github.com/tile-ai/tilelang/tree/6623b12d232b343648a5ba99992e3e6f0d6376d2/src/layout)

### Triton

- [TTIR → TTGIR TypeConverter](https://github.com/triton-lang/triton/blob/main/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp)
- [Coalesce](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)
- [AccelerateMatmul](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp)
- [RemoveLayoutConversions](https://github.com/triton-lang/triton/blob/main/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp)
- [LinearLayout](https://github.com/triton-lang/triton/blob/main/include/triton/Tools/LinearLayout.h)
- [NVIDIA backend pipeline](https://github.com/triton-lang/triton/blob/main/third_party/nvidia/backend/compiler.py)

### MLIR

- [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [DataFlow Analysis](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/)
- [Affine Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
- [MemRef Dialect](https://mlir.llvm.org/docs/Dialects/MemRef/)
- [NVGPU Dialect](https://mlir.llvm.org/docs/Dialects/NVGPU/)
- [NVVM Dialect](https://mlir.llvm.org/docs/Dialects/NVVMDialect/)

### Frisk 当前代码

- [LayoutAttr](../include/Dialect/Frisk/IR/FriskAttributes.td)
- [LayoutInterface](../include/Dialect/Frisk/IR/FriskInterfaces.td)
- [当前 LayoutInfer pass](../lib/Dialect/Frisk/Transforms/LayoutInfer.cpp)
- [当前 Op 局部布局逻辑](../lib/Dialect/Frisk/IR/FriskOps.cpp)
- [当前 Reduce 布局逻辑](../lib/Dialect/Frisk/IR/FriskOps_Reduce.cpp)
