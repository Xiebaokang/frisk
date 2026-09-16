# Triton、TileLang 与 Frisk 布局推断对比及 Frisk 架构决策

> 文档性质：技术对比 + Frisk 架构决策记录
>
> Frisk 目标平台：NVIDIA SM90/SM90a
>
> 首次审阅：2026-08-16；本次源码审计：2026-09-12
>
> 状态：M0–M3 与 M4 Task 18 已实现；Task 18 独立复审和 30 lit / 104 unit / 4 CTest 通过，尚未提交/合入 main。其余 M4 迁移尚未实施，跨规范化的契约生命周期限制见 §10 与实现计划 §18.9。
>
> 关联文档：[Frisk 布局推断系统设计方案](./layout_inference_design.md)、[GF(2) 与组合布局说明](./gf2_layout_guide.md)

## 0. 审计基线、证据边界与维护约定

本轮按用户要求先审计与设计，再确认实施。下表保留审计起点；Frisk Task 18 的实现增量另见 §6.1、§13 和实现计划 §18.9。上游比较仅指固定快照，不表示今后 main 的永久状态。

| 项目 | 本轮源码快照 | 审计/验证范围 |
| --- | --- | --- |
| Frisk | `b120d06400a14a703a44dac1a37a0b38d8110935`，M3 合并后的 main | 读取新旧布局路径；本轮构建及 27 lit、74 unit、4 CTest 全部通过 |
| TileLang | `5e149e31674658f94779c7d0c6039549a1853123`，2026-09-12 | 官方 main 获取后核对 HEAD；读取推断、代数、Op、reducer verifier 和测试；未运行其测试 |
| Triton | `42c5e89c3871e1472968c92dd8e5c02d0b3dd40c`，2026-09-11（提交时区 -07:00） | 官方 main 获取后核对 HEAD；读取布局传播、SCF、代数、存储区域分析和测试；未运行其测试 |

历史记录：2026-08-16 比较采用 TileLang `6623b12d232b343648a5ba99992e3e6f0d6376d2`。本轮逐项核对旧版本，区分“旧版已有”和“本区间新增”；旧快照保留为迁移回归参考，不再代替新版语义。Triton 原文使用浮动 main，无法据此证明具体机制在哪次提交首次出现；本轮只陈述固定新版已具备什么。

维护规则：

- 每个相关实现任务同步本文件和[实现计划](./layout_inference_implementation_plan.md)：版本、源码证据、行为取舍、测试、未完成项缺一不可。
- “源码存在”“本轮测试通过”“设计目标”“性能/创新假设”分别标注；不得互相替代。
- 上游升级必须记录旧/新 SHA 和行为变化；不能只替换链接就宣称差分 corpus 已更新。
- 不采用“另一框架完全没有”作为创新证据；缺失结论限定到已审阅 pass/接口和支持子集。
- 本轮没有 TileLang/Triton 运行实验、GPU benchmark 或端到端性能结论。上表 27/74/4 是 M3 基线；Task 18 的新增测试结果单独记录，不能混用。

## 1. 结论摘要

三者的核心路线可以概括为：

```text
TileLang = 以 Buffer 为中心的兼容布局闭包
Triton   = 带 Encoding 的 Tensor SSA + 硬件锚点驱动的多轮布局改写
           + 显式 ConvertLayout 的插入、消除、提升与重计算
Frisk目标 = MLIR 类型化的 Distributed/Storage 双域布局
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

这里所说的“创新”是待检验的架构假设，不是已证明的独创性或性能优势；当前完成度见 §6.1，验证与否证条件见 §10。

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

以本轮 TileLang `5e149e3` 为基线，以下仅列 CUDA 布局相关关键阶段（不是完整 pass 清单）：

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

通过 Buffer use-list/worklist 调用 `InferLayout(kCommon)`；新布局及 reducer widening 会唤醒相关 users。不能据此断言所有事实更新都实现统一依赖唤醒：当前 containment 替换和 swizzle merge 分支并不统一重新入队全部 users，也未给所有 Layout 提供同一个有限格证明。

此阶段主要寻找兼容布局闭包。已有布局与新布局相遇时会：

- 接受数学上相同的布局；
- 对非 strict fragment，在可证明 containment 时采用包含更多映射的布局；
- 对 compatible shared swizzle 尝试合并到更小粒度；
- 无法兼容时报告布局冲突。

#### 阶段 3：free-mode

对未完全解析的连通分量尝试不同 root 的 native plan，并为符合条件的 reducer root 增加 scalar candidate。新版 `RunOneAttempt` 隔离队列、layout map 与 cloned op 状态，失败候选不能污染下一次尝试。

新版采用可插拔 `LayoutCostModel`：默认 `register-count` 先比较估计 spill traffic，再比较寄存器数；`tl.layout_cost_model="io-aware"` 为 opt-in，增加全局访问 bandwidth/issue 成本。I/O 模型利用 CuTe algebra 推导访问并采用 probe-then-prove，不能继续称 TileLang“只有寄存器启发式、没有访问代数或 I/O 成本”。它仍是有限 component attempt 评分，不是把任意 conversion placement 纳入的目标级全局最优求解。见 §15 固定源码。

#### 阶段 4：alias finalize

对共享同一底层 storage Var 的 Buffer：

- shape 相同则传播同一布局；
- shape/dtype 不同则根据 storage bit ratio reshape；
- 对支持的 full-buffer reshape/reinterpret 验证 storage bits 与映射兼容性。

这不是任意 offset/strided slice 的统一证明：前端 `T.view/T.reshape` 重建同 data 的 Tensor，检查总 storage bits，但不复制 source 的 strides/elem_offset。same-data 分组也不等于逻辑坐标恒等。位宽感知 reshape、widening 的同线程/连续且对齐槽位证明在旧 `6623b12` 就已存在。

### 4.4 冲突处理

TileLang 主要通过“找到一个大家都能接受的 Buffer 布局”消除冲突。其常规机制并不是为同一个逻辑 Buffer 保留多个 SSA encoding，再在 use 边界插入通用 conversion。

典型结果为：

```text
兼容              -> 合并/传播
fragment containment -> 选择可包含的布局
shared swizzle 可合并 -> 合并粒度
free-mode 有可行 root -> 按所选 cost policy 比较 attempt
仍不兼容           -> 编译期冲突或 unsupported
```

### 4.5 本轮确认的新机制与迁移影响

- **PartialFragment**：区分未归约的 addend lanes 与同值 copy groups；相同 storage algebra 不代表相同 reduction 语义。普通 Fragment 的 replica 处理不能直接复用。
- **Reducer 单调 widening**：`unset → narrow → wide`；strict annotation 冲突仍报错。dst-steering 决定哪个 Op 有首次提交无约束 dst 的权利，不是硬件 thread owner。
- **作用域与控制流**：过滤含外层自由变量的 open fragment；reducer epoch verifier 新支持 serial/pipelined loop 与受限 conditional refinement，检查 loop/then/else/while 语境。这不是通用 SSA block-argument/yield join 求解，但也不能写成“不支持控制流”。
- **Parallel**：free-mode 重访 frozen loop 时重新验证晚到约束；只读 broadcast 可退到 canonical fully replicated，written buffer 不能这样处理。
- **Copy/GEMM/Reduce**：Copy/GEMM 部分 positional 参数迁到 annotations；GEMM 新增 definite read-before-write region 语义。传统 Reduce 的主要 src→dst/containment 规则并未因 reducer 更新而全面双向化。旧 Frisk Op 迁移必须按当前 API 和语义重审。
- **逆映射**：新增 inverse round-trip 检查，可拒绝明确错误的逆；其有限域检查和 Unknown 处理有边界，不能称任意映射的完整反演证明。

证据与现有测试入口见 §15；这里只审阅测试源码，没有执行 TileLang 测试。

### 4.6 优势与局限

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

TypeConverter 保留已有 encoding，为未编码 `RankedTensorType` 添加默认 blocked encoding。当前 source materialization 可用 `UnrealizedConversionCastOp`，target materialization 使用 `ConvertLayoutOp`，二者不能混为同一路径。

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
2. 从 tensor function arguments 和 anchor 的 tensor results 向 descendants 传播 encoding（无 result 的 store 虽可被判定为 anchor，却不直接成为 forward seed）；
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

### 5.6 Region、存储分析与代数的实际边界

`RemoveLayoutConversions` 使用单调**增长**候选的 forward worklist；`SmallSetVector<Attribute, 8>` 中的 8 是 inline capacity，不是候选硬上限。随后还有独立的 rewrite、backward rematerialization/cleanup 循环，并非同一个有限域删减求解器。

`scf.while` 已支持两套不同 arity/type 的 tuple：init/after-yield 对 before args；condition-forwarded values 对 after args/results，跳过 predicate。反向 rematerialization 对 while block args 和部分 loop results 仍有限制。不能把局部限制描述成 Triton 没有 while 支持。`getValueAs` 的转换可放在定义后，并非一律紧贴 operand use。

当前 `BufferRegionAnalysis` 已区分 allocation frame、view offset 和精确 physical address sets；动态 index 可以形成合法 subbuffer 的 MAY 集合，未知 producer 提升为 Unknown。这些能力用于 memory effect、ConSan 与同步分析，**本轮未发现它参与 RemoveLayoutConversions 候选选择，不能将两者说成一个布局 solver**。

`LinearLayout` 本体为 GF(2) 映射，且允许非单射以表达复制；pseudoinverse 选择规范代表不等于证明唯一 writer。Triton 的 `PaddedSharedEncoding` 已组合 padding 与 linear component，不能把 GF(2) 本体限制扩大为“Triton 无法表达 padding”。固定源码与静态测试见 §15。

### 5.7 优势与局限

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

审计起点 Frisk `b120d06` 已完成 M0–M3，原文“LayoutInfer 是空壳”已经过时。下表加入用户确认后的 `feature/m4-task18` 增量，不代表 main 已合入：

| 已有部件/功能 | 已核对实现 | 当前边界 |
| --- | --- | --- |
| 布局代数与属性 | GF(2) 矩阵、Affine/BitLinear/Product、LayoutProof、Distributed/Storage attrs | 代数库能力不等于所有 map/shape 已接入 pass；Unknown 不可作为硬证明 |
| Storage 纵向切片 | layout_view、whole-tile Copy；Task 18 增加静态 cast/subview 坐标链、有限 origin 投影、全部同-root pairs 的 bit 区间证明 | 同 dtype/space、可静态证明；动态路径、reinterpret、一般 Product alias 投影保守拒绝；不引入隐藏 root assignment |
| Distributed 纵向切片 | Tensor encoding、transpose/broadcast 等关系、tile_load/store、显式 conversion | 静态受限形状；bootstrap 每 component 至多 8 vars、每域至多 4 candidates，不是 M5 的通用全局搜索 |
| 结构化控制流 | if/for/while 实际 use/slot 约束与转换物化；Task 18 显式带 kind/slot 的 region edges | 保留 while I/O 两套 tuple；回边仍允许 Convertible，不为制造缩域而改为相等约束 |
| 约束求解/物化 | stable IDs、hard constraints、provenance、确定性有限枚举；Task 18 stable FIFO 删减及真实 degree 上界统计；detached module 原子提交 | 完整 CostVector/SM90 指令路径联合优化未实现；候选有限初始化不是通用大图最优解证明 |
| 独立核验与测试 adapter | 从实际 IR 重建关系、singleton assignment 核验；受限 conversion lowering 测试 | 不是完整 WGMMA/TMA/mbarrier lowering，没有端到端性能优越性证据 |
| 旧 Op 规则 | FriskOps/FriskOps_Reduce 的 legacy 方法仍在，adapter 提供受限语义回归 | 新 pass 尚未通过 Op interface 全面迁移 Copy/Fill/Parallel/GEMM/Reduce；Task 19–22 待实现 |

源码锚点：`LayoutAliasAnalysis.cpp` 的 root/坐标/bit 证明；`StorageAliasCandidates.cpp` 的有限 origins；`LayoutPropagation.cpp` 的 strict/common FIFO；`LayoutRelations.cpp` 的 AliasLayout 证明缓存；`DistributedLayoutConstraints.cpp` 的 SCF 规则；`MaterializeLayouts.cpp` 的实际值重绑和事务提交。上游证据见 §15。

Task 18 对应的审计缺口已在实现分支处理：endpoint-owned 坐标 payload 随 ID remap；同-root 关系由邻接链改为全部 pairs；容量改查 root accessible bit span；RelationsOnly 在任何投影/枚举之前返回。此前 RelationsOnly seed 复制是模式契约缺口，不能倒推旧版已经发生错误验过；本次另有 actual verifier 非邻接冲突反例。

审计 baseline：27 lit、74 unit、4 CTest 全通过。Task 18 新增 30 个 alias/region/integration 单元和 3 份 lit，最终 Gate 为 30/30 lit、104/104 unit、4/4 CTest；命令及边界见实现计划 §18.9。§6.2–6.5 仍是目标分解，不能整体当作现状；完整硬件 pipeline 超出当前范围。

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

长期目标流水线为（不是当前可运行 pipeline，也不属于本计划完整指令 lowering 的交付承诺）：

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
| 反向推断 | TileOp 可依据已知 buffer 推另一端 | inferSrcEncoding / backward rematerialization；SCF 穿越有边界 | 一等能力；consumer contract 可反推 producer/storage |
| Alias/view | 同 data 分组、位宽感知 reshape，非任意 slice 证明 | memdesc view + 独立物理 BufferRegion 分析 | Task 18 显式坐标关系 + 全部同-root pairs 的 storage hard constraints；支持子集与 Unknown 明确 |
| 多 consumer | 尽量找到一个共同 Buffer 布局 | 可保留局部不同 encoding，以 conversion 分隔 | 联合比较共享布局、边界 conversion、rematerialization |
| 冲突处理 | containment、swizzle merge、free root；仍冲突则失败 | heuristic 选 encoding并插 conversion | hard conflict 诊断；soft conflict 进入全局候选选择 |
| Conversion | 不是通用核心抽象 | 显式 ConvertLayout，随后多轮优化 | 显式且是求解变量；求解后统一物化 |
| 成本重点 | 默认 spill/register；可选 I/O-aware component attempt 评分 | pass-local hardware heuristic + conversion cleanup | 指令路径、访存、bank conflict、conversion、资源联合评估（M5） |
| Region/loop | TIR loop/annotation、reducer epoch 语境验证；非通用 SSA join 图 | if/for/while 正向传播/重写；while 区分双 tuple，反向 remat 有边界 | M3 SCF 语义 + Task 18 显式边与有限域收敛统计 |
| 验证 | Analyzer/ICHECK/专用 validator | MLIR verifier + target-specific checks | MLIR verifier + 数学证书 + provenance diagnostic |
| 代数 | PrimExpr/IndexMap/Fragment/PartialFragment，部分 CuTe 访问代数 | encoding + LinearLayout；已有 padding/linear 组合 | Affine outer × GF(2) inner，实际支持范围见 §6.1 |
| Target 耦合 | TileOp/layout 与 TVM CUDA lowering 紧密 | target-specific encoding/pass 较多 | 通用 solver 与 SM90 rule/adapter 分层 |
| 默认失败策略 | 找不到共同布局时冲突 | 插 conversion 或 target fallback | 先全局选成本最低合法解；无普通解才受控 fallback |

## 8. 同一个 SM90 WGMMA 数据流在三者中的处理

本节是机制示意；尤其 §8.3 为 Frisk 未来设计，不是本轮编译/运行结果。

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
6. free-mode 比较 root attempts，按默认 spill/register 或 opt-in I/O-aware policy 评分；
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

## 10. 差异化假设与可核验创新边界

双域区分、GF(2)、padding、显式 conversion、alias/view、while、fixed-point、候选评分本身都不是 Frisk 独有。TileLang 的 PartialFragment/位宽视图与 Triton 的 BufferRegion/PaddedSharedEncoding 进一步说明：不能通过省略对方能力建立“创新”。

| 假设 | 与固定参考实现比较的具体点 | Frisk 当前状态 | 验证/否证条件 |
| --- | --- | --- | --- |
| H1：坐标化 alias 与 SSA region 关系在同一 hard-constraint 图内协作 | 对比 TileLang buffer 兼容闭包及 Triton 分离的布局传播/物理区域分析；比较作用范围，不说对方没有坐标分析 | Task 18 坐标 alias/显式边已编码，支持受限静态域；尚无性能结论 | 偏移/步长/降秩、三视图冲突、真实回边重调度；actual-only verifier 无隐藏 assignment。Tensor/storage 经访问关系协作，不代表任意跨域组合全覆盖 |
| H2：可审计的有限域终止与冲突解释 | 公开冻结候选数、实际删除数、queue pops、degree 上界，关联 seed→alias/region→冲突 | Task 18 worklist/stats、finite origins、proof cache 已编码 | 插入顺序扰动下 assignment/统计相同；重复 pass IR 相同；真实事件逐项对账，Unknown 拒绝。该性质是当前受限实现的工程差异，不是对上游整体的否定 |
| H3：组合代数扩大受支持布局域而保持证明一致 | 比较具体 Affine/GF(2) 组合域与 TileLang IndexMap/CuTe、Triton padded/linear 的交集及差集 | M1 代数已有，pass 子集受限 | 同坐标语义、coverage/injectivity/owner 与反例；非零 offset XOR 必须正确处理 carry。若只换一种文本表达，不构成新增能力 |
| H4：布局、指令路径和 conversion placement 联合选择改善结果 | 与固定 pipeline/attempt policy 比较合法候选及最终代价，不假定全局最优 | M3 只最少新增转换；M5 成本/路径选择未实现 | 固定硬件/输入/编译选项，报告 runtime、访存、bank、寄存器/共享内存和编译时间；超出计划回退阈值或无收益则不宣称性能优势 |

语义差分是验证方法，不是独创性结论。旧 Frisk/旧 TileLang 仅为已知正确子集的回归 oracle；新版机制、历史 bug 修正须用独立的坐标/owner/address 不变量及行为变更记录核验，不能为了“和旧版一致”保留错误。Task 28 将建立新版固定 corpus，目前尚未生成或运行。

本阶段实现 H1/H2 的受限 Task 18 切片并提供可复查测试，不扩展 GEMM/Reduce、cost solver、硬件 lowering，也不承诺本轮已经证明研究创新。whole-root alignment 前置契约若被 Pure/DCE 删除，后续 actual verifier 会保守拒绝；`infer → canonicalize → infer` 的证据生命周期限制已由回归记录，跨任意规范化的 durable contract 留待 Task 22。

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

## 13. 2026-09-12 审计决策记录

| 证据/缺口 | 决策 | 对应任务与状态 |
| --- | --- | --- |
| TileLang bit-aware alias/widening 旧版已有；same data 不足以表达任意 slice | 采用 storage-bit/owner 不变量，调整为显式坐标关系；拒绝直接 union 为布局相等 | Task 18 已编码；dtype reinterpret 本阶段拒绝 |
| TileLang 新 PartialFragment、dst-steering、epoch 语境扩展与 annotations ABI | 迁移前逐 Op 建新旧行为表；区分同值 replica 和 partial addend，区分读/写规则 | Task 19–21 后续设计输入，不在本轮实现 |
| Triton while 两套 tuple 已有，backward remat 有局限 | 保留 Frisk M3 两套 tuple；显式记录 region edges，不扩大转换移动权限 | Task 18 已编码，真实 SCF 回归保留 Convertible |
| Triton BufferRegion 已建模物理区域，但不是布局候选 solver | 借鉴 root/frame/地址集合与 Unknown；适配到 Frisk alias hard relation | Task 18 已编码；不照搬 memdesc 的专用 same-encoding 限制 |
| Frisk Attr==、邻接 alias 链、root capacity 与地址原点约定不足 | 共同 root 地址契约、逐对坐标/bit 区间证明，处理非相邻重叠和非预期碰撞 | Task 18 实现分支已修复，并有先失败后通过的集成用例 |
| Frisk 增长 closure 与多轮扫描尚无统计契约 | 有限候选初始化与单调删减分开；stable worklist、真实 degree 统计 | Task 18 已编码；Strict/Common 逐项账本、冲突与确定性测试 |
| 原比较称 pass 空壳、成本模型只有寄存器，或把目标写成完成 | 本轮修正完成度和固定证据，创新改成待检验假设 | 本文与实现计划已同步 |
| 旧文档曾对 conversion 普通候选/失败后 fallback 有歧义 | 维持计划现有决策：conversion 为普通候选，controlled fallback 仅处理普通域无解 | 历史问题已在现行设计/计划澄清，不再列为待修订缺陷 |

[Task 18 详细设计、验收矩阵与实现记录](./layout_inference_implementation_plan.md#task-18-完成-aliasview-与-region-graph)保存了先审计设计、用户确认、红灯测试、实现与复审的过程。复审补强向量地址保证、对齐依据 provenance 和 region/图 ID 不变量；长期文档、公共属性/Op 注释同步，push/merge 仍须用户另行指示。

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

- [TileLang CUDA pipeline（固定审计 commit）](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/tilelang/cuda/pipeline.py)
- [TileLang LayoutInference（固定审计 commit）](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/transform/layout_inference/layout_inference.cc)
- [TileLang layout 实现目录](https://github.com/tile-ai/tilelang/tree/5e149e31674658f94779c7d0c6039549a1853123/src/layout)

源码定位与静态测试（以下均已审阅，未执行上游测试）：

- [alias / reducer widening / attempt 调度](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/transform/layout_inference/layout_inference.cc#L311)：`propagate_alias`、`RunOneAttempt`。
- [PartialFragment](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/layout/layout.h#L256)、[reinterpret/reshape 与 inverse 检查](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/layout/layout.cc#L140)、[前端 view/reshape](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/tilelang/language/customize.py#L60)。
- [代价接口](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/transform/layout_inference/layout_cost_model.h#L38)、[代价实现](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/transform/layout_inference/layout_cost_model.cc)、[默认 policy](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/config.h#L38)。
- [epoch 语境验证](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/transform/verify_reducer_epoch.cc#L104)、[Parallel 规则](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/op/parallel.cc#L331)。
- [Copy 新 ABI](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/op/copy.cc#L619)、[CUDA Copy 区域规则](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/cuda/op/copy.cc#L584)、[GEMM 读写/契约](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/op/gemm.cc#L164)、[传统 Reduce](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/src/op/reduce.cc#L195)。
- [位宽 view 测试](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/language/test_tilelang_language_view.py)、[reducer v2 测试](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/language/test_tilelang_language_reducer_v2.py)、[attempt 隔离测试](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/transform/test_tilelang_transform_reducer_scalar_candidates.py)、[推断/成本测试](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/transform/test_tilelang_transform_layout_inference.py)、[inverse 测试](https://github.com/tile-ai/tilelang/blob/5e149e31674658f94779c7d0c6039549a1853123/testing/python/layout/test_tilelang_layout_inverse.py)。
- [历史 LayoutInference（6623b12，仅作新旧比较）](https://github.com/tile-ai/tilelang/blob/6623b12d232b343648a5ba99992e3e6f0d6376d2/src/transform/layout_inference.cc)。

### Triton

- [TTIR → TTGIR TypeConverter](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Conversion/TritonToTritonGPU/TritonGPUConversion.cpp)
- [Coalesce](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/Coalesce.cpp)
- [AccelerateMatmul](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/AccelerateMatmul.cpp)
- [RemoveLayoutConversions](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp)
- [LinearLayout](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/include/triton/Tools/LinearLayout.h)
- [NVIDIA backend pipeline](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/third_party/nvidia/backend/compiler.py)

- [while forward 与 conflict heuristic](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/RemoveLayoutConversions.cpp#L287)、[backward slice 限制](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/Transforms/Utility.cpp#L900)。
- [BufferRegion 契约](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/include/triton/Analysis/BufferRegion.h#L39)、[物理地址与 view 分析](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Analysis/BufferRegion.cpp)、[ConSan 使用点](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonInstrument/IR/Utility.cpp#L698)、[Membar 使用点](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Analysis/Membar.cpp#L38)。
- [PaddedSharedEncoding](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/include/triton/Dialect/TritonGPU/IR/TritonGPUAttrDefs.td#L208)、[memdesc verifier](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/lib/Dialect/TritonGPU/IR/Ops.cpp#L579)。
- [while/转换静态测试](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/test/TritonGPU/combine.mlir#L1896)、[物理布局 alias 测试](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/test/Analysis/test-buffer-region-layout-alias.mlir)、[动态/重解释 alias 测试](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/test/Analysis/test-buffer-region-alias.mlir)、[LinearLayout 单测](https://github.com/triton-lang/triton/blob/42c5e89c3871e1472968c92dd8e5c02d0b3dd40c/unittest/Tools/LinearLayoutTest.cpp)。

### MLIR

- [Dialect Conversion](https://mlir.llvm.org/docs/DialectConversion/)
- [DataFlow Analysis](https://mlir.llvm.org/docs/Tutorials/DataFlowAnalysis/)
- [Affine Dialect](https://mlir.llvm.org/docs/Dialects/Affine/)
- [MemRef Dialect](https://mlir.llvm.org/docs/Dialects/MemRef/)
- [NVGPU Dialect](https://mlir.llvm.org/docs/Dialects/NVGPU/)
- [NVVM Dialect](https://mlir.llvm.org/docs/Dialects/NVVMDialect/)

### Frisk 当前代码

- [LayoutAttr](../include/Dialect/Frisk/IR/FriskAttributes.td)
- [新布局 Op/Attr 接口入口](../include/Dialect/Frisk/IR/FriskLayoutInterfaces.h)（接口定义存在不等于全部 Op model 已迁移）
- [当前 LayoutInfer pass](../lib/Dialect/Frisk/Transforms/LayoutInfer.cpp)
- [当前 Op 局部布局逻辑](../lib/Dialect/Frisk/IR/FriskOps.cpp)
- [当前 Reduce 布局逻辑](../lib/Dialect/Frisk/IR/FriskOps_Reduce.cpp)

- [布局属性](../include/Dialect/Frisk/IR/FriskLayoutAttrs.td)、[代数/证明](../include/Dialect/Frisk/Analysis/LayoutAlgebra.h)。
- [约束图](../include/Dialect/Frisk/Analysis/LayoutConstraint.h)、[传播/alias root](../lib/Dialect/Frisk/Analysis/LayoutPropagation.cpp)、[关系核验](../lib/Dialect/Frisk/Analysis/LayoutRelations.cpp)。
- [SCF 与 Tensor 约束](../lib/Dialect/Frisk/Analysis/DistributedLayoutConstraints.cpp)、[求解 verifier](../lib/Dialect/Frisk/Analysis/LayoutVerifier.cpp)、[原子物化/实际 IR 核验](../lib/Dialect/Frisk/Transforms/MaterializeLayouts.cpp)。
- [SCF 回归](../test/Transforms/materialize-scf.mlir)、[传播单测](../unittests/Dialect/Frisk/Layout/LayoutPropagationTest.cpp)。

本地链接用于后续维护；上述“当前实现”结论固定到 Frisk `b120d06`，不能用之后的文件内容反推此次已实现范围。
