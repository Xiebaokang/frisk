# XOR、GF(2) 与 Frisk 布局组合详解

> 日期：2026-08-16
> 适用范围：Frisk 方案 C、SM90 Register/Shared Layout 设计
> 阅读方式：正文侧重直观理解和编译器实例，附录给出较严格的数学定义
> 关联文档：[Frisk SM90 MLIR-native 布局推断系统设计](./layout_inference_design.md)

## 1. 核心结论

可以先记住四句话：

1. XOR 是逐 bit 的无进位二进制加法。
2. GF(2) 是只包含 `0` 和 `1`，并把 XOR 当作加法的数学系统。
3. GPU 的 lane/register 分布和 Shared Memory swizzle 经常由 bit 选择、置换和 XOR 构成，所以可以用 GF(2) 矩阵表示。
4. 两个布局的组合，本质是把前一个布局的输出作为后一个布局的输入；在线性部分表现为 GF(2) 矩阵乘法。

Frisk 方案 C 中使用两种主要布局方向：

```text
Distributed Layout:
    hardware carrier -> logical coordinate

Storage Layout:
    logical coordinate -> physical address
```

因此一次 Register → Shared 的地址计算是：

```text
hardware carrier
    -> logical coordinate
    -> physical address
```

如果分别记为 `D` 和 `S`，组合结果为：

```text
A = S ∘ D
```

这里的 `S ∘ D` 表示先执行 `D`，再执行 `S`。

## 2. XOR 是什么

XOR 中文称为“异或”，常见写法是：

```text
a XOR b
a ^ b
a ⊕ b
```

它对两个 bit 逐位计算：

| a | b | a XOR b |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

规律是：

- 两个 bit 相同，结果为 `0`；
- 两个 bit 不同，结果为 `1`。

### 2.1 XOR 是无进位加法

普通二进制加法中：

```text
1 + 1 = 10
```

它会产生进位。XOR 忽略进位：

```text
1 XOR 1 = 0
```

例如：

```text
  0101   // 5
XOR
  0011   // 3
------
  0110   // 6
```

因此：

```text
5 XOR 3 = 6
```

### 2.2 XOR 的重要性质

#### 与 0 XOR 不改变值

```text
a XOR 0 = a
```

#### 与自身 XOR 得到 0

```text
a XOR a = 0
```

#### 同一个掩码使用两次会恢复原值

```text
(a XOR b) XOR b = a
```

例如：

```text
5 XOR 3 = 6
6 XOR 3 = 5
```

#### XOR 满足交换律和结合律

```text
a XOR b = b XOR a
(a XOR b) XOR c = a XOR (b XOR c)
```

这些性质使 XOR 很适合构造确定、可逆的索引排列。

## 3. XOR swizzle 是什么

假设 Shared Memory 中有一个逻辑 `4×4` Tile，并采用：

```text
physical_col = logical_col XOR row
```

对于固定的一行，列索引会被重新排列。

### 3.1 每一行的排列

当 `row = 0`：

```text
0 XOR 0 = 0
1 XOR 0 = 1
2 XOR 0 = 2
3 XOR 0 = 3
```

结果：

```text
0 1 2 3
```

当 `row = 1`：

```text
0 XOR 1 = 1
1 XOR 1 = 0
2 XOR 1 = 3
3 XOR 1 = 2
```

结果：

```text
1 0 3 2
```

完整排列是：

```text
row 0: 0 1 2 3
row 1: 1 0 3 2
row 2: 2 3 0 1
row 3: 3 2 1 0
```

该变换没有丢失元素，也没有把两个不同 column 映射到同一 column。对于固定的 `row`，它是一个可逆 permutation。

### 3.2 为什么能减少 Shared Memory bank conflict

如果多个线程按照不同 row、相同 column 访问 Shared Memory，普通 row-major layout 可能让这些地址集中到相同 bank。

XOR swizzle 使用 row 的某些 bit 翻转 column/address 的某些 bit，从而把原本集中在同一 bank 的访问重新分散。

需要注意：

- XOR swizzle 不是随机打乱；
- 相同输入永远得到相同输出；
- 是否真正减少 bank conflict，取决于线程访问模式、element size、bank width 和具体 swizzle 规则；
- 编译器仍然需要通过 target rule 或 bank-conflict 分析验证效果。

## 4. GF(2) 是什么

GF(2) 是只有两个元素的有限域：

```text
GF(2) = {0, 1}
```

其中：

- 加法是模 2 加法，等价于 XOR；
- 乘法与普通 `0/1` 乘法相同，等价于 AND。

### 4.1 GF(2) 加法

| a | b | a + b |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 1 |
| 1 | 0 | 1 |
| 1 | 1 | 0 |

因此：

```text
GF(2) addition = XOR
```

在 GF(2) 中：

```text
1 + 1 = 0
```

因为：

```text
2 mod 2 = 0
```

### 4.2 GF(2) 乘法

| a | b | a × b |
|---:|---:|---:|
| 0 | 0 | 0 |
| 0 | 1 | 0 |
| 1 | 0 | 0 |
| 1 | 1 | 1 |

因此：

```text
GF(2) multiplication = AND
```

### 4.3 为什么 GF(2) 是“域”

在这套规则下：

- `0` 是加法单位元；
- `1` 是乘法单位元；
- 每个元素都有加法逆元；
- 唯一的非零元素 `1` 有乘法逆元，仍然是 `1`。

因此可以像普通线性代数一样，在 GF(2) 上定义：

- 向量；
- 矩阵；
- 线性映射；
- rank；
- kernel；
- image；
- inverse；
- left/right inverse。

区别只是所有加法都变成 XOR，所有标量只有 `0/1`。

## 5. GF(2) 向量和矩阵

### 5.1 将 GPU 坐标拆成 bit

一个 warp 有 32 个 lane：

```text
lane_id ∈ [0, 31]
```

可以用 5 个 bit 表示：

```text
lane_id = [l4 l3 l2 l1 l0]
```

如果每线程持有 8 个逻辑 register slot：

```text
register_slot = [r2 r1 r0]
```

将这些 bit 组成输入向量：

```text
x = [l4 l3 l2 l1 l0 r2 r1 r0]^T
```

逻辑坐标也可以拆成 bit：

```text
m = [m5 m4 m3 m2 m1 m0]
n = [n3 n2 n1 n0]
```

布局可以规定：

```text
m0 = l2
m1 = l3 XOR r0
m2 = l4
n0 = l0
n1 = l1 XOR r1
```

每个输出 bit 都是若干输入 bit 的 XOR，因此可以写成：

```text
y = Bx
```

这里的矩阵乘法全部在 GF(2) 上进行。

### 5.2 一个小型矩阵例子

定义：

```text
y0 = x0 XOR x2
y1 = x1
```

矩阵形式为：

```text
    [1 0 1] [x0]
y = [0 1 0] [x1]
              [x2]
```

第一行表示：

```text
y0 = 1*x0 XOR 0*x1 XOR 1*x2
   = x0 XOR x2
```

第二行表示：

```text
y1 = x1
```

若：

```text
x = [1, 1, 0]^T
```

则：

```text
y0 = 1 XOR 0 = 1
y1 = 1
y  = [1, 1]^T
```

### 5.3 Basis 的直观含义

矩阵的每一列可以理解为一个 input bit 的 basis vector：

```text
设置该 input bit，会翻转哪些 output bits？
```

例如某一列是：

```text
[1, 0, 1]^T
```

表示这个 input bit 会同时翻转 output bit 0 和 output bit 2。

这与 GPU layout 中的描述非常一致：

- `lane bit 0` 控制 N 的最低位；
- `warp bit 1` 控制 M 的某一位；
- `register bit 0` 同时影响两个坐标位，形成 XOR permutation。

## 6. Frisk 中的两类布局映射

### 6.1 Distributed Layout

Distributed Layout 采用：

```text
D: H -> L
```

其中：

- `H` 是 hardware carrier domain；
- `L` 是 logical coordinate domain。

展开后：

```text
D(register_slot, lane, warp, warp_group, cta)
    -> (logical_dim0, ..., logical_dimN)
```

例如：

```text
D(reg=2, lane=17, warp=1) = C[35, 42]
```

`register_slot` 是当前线程的逻辑 fragment slot，不是物理寄存器编号。物理寄存器编号由 LLVM/NVPTX 的 register allocation 决定。

Distributed Layout 需要回答：

- 哪个线程持有哪些逻辑元素；
- 一个 warp/warp-group 是否覆盖整个 Tile；
- 是否存在 replication；
- Store/Reduce 时谁是唯一 writer；
- 两种 Register 分布之间是否需要 shuffle 或重排。

Distributed Layout 通常要求 coverage，但不要求 injective。多个 carrier 可以持有同一个逻辑元素，这表示 replication。

### 6.2 Storage Layout

Storage Layout 采用：

```text
S: L -> P
```

其中：

- `L` 是 logical coordinate；
- `P` 是 physical address/offset。

展开后：

```text
S(logical_dim0, ..., logical_dimN)
    -> (byte_offset, bit_offset)
```

它描述：

- stride；
- row-major/column-major；
- padding；
- XOR swizzle；
- alignment；
- vector granularity；
- sub-byte packing；
- TMA descriptor compatibility。

对于普通可变内存，Storage Layout 在 live logical domain 上必须 injective：

```text
l1 != l2  =>  S(l1) != S(l2)
```

否则两个不同逻辑元素会写入同一物理地址，形成非预期 alias。

Padding 可以产生没有逻辑元素对应的物理洞，因此 Storage Layout 不必覆盖整个物理地址空间。

## 7. 什么叫“组合两个布局”

### 7.1 函数组合的方向

假设：

```text
f: X -> Y
g: Y -> Z
```

那么可以组合：

```text
g ∘ f: X -> Z
```

含义是：

```text
(g ∘ f)(x) = g(f(x))
```

执行顺序是：

```text
先 f，后 g
```

虽然书写顺序是 `g ∘ f`，但数据流顺序是从右向左。

### 7.2 为什么不能任意组合

组合要求第一个映射的输出域与第二个映射的输入域兼容：

```text
f 的 codomain == g 的 domain
```

除了 rank 相同，还要检查：

- dimension name；
- dimension order；
- extent；
- element/storage bit width；
- dynamic bounds；
- view/transpose relation。

例如：

```text
f: carrier -> (M, N)
g: (K, N) -> address
```

不能直接组合，因为 `M` 与 `K` 的逻辑语义不同。必须先存在一个明确的 logical transform。

### 7.3 GF(2) 矩阵如何组合

若：

```text
f(x) = A x
g(y) = B y
```

则：

```text
(g ∘ f)(x)
    = g(Ax)
    = B(Ax)
    = (BA)x
```

所以组合矩阵是：

```text
B_composed = B * A
```

矩阵乘法中的加法使用 XOR，乘法使用 AND。

矩阵顺序不能写反：

```text
先 A，后 B  =>  B * A
```

通常：

```text
B * A != A * B
```

因此布局组合不是交换的。

### 7.4 四类组合速查

| 目标 | 已知映射 | 结果 | 关键前提 |
|---|---|---|---|
| 计算每个 carrier 的内存地址 | `D: H -> L`，`S: L -> P` | `S ∘ D: H -> P` | logical domain、shape 和 dtype 匹配 |
| Register layout 转换 | `D_src: H_src -> L`，`D_dst: H_dst -> L` | `rightInverse(D_src) ∘ D_dst: H_dst -> H_src` | Source 覆盖 Destination 所需元素 |
| Storage layout 转换 | `S_src: L -> P_src`，`S_dst: L -> P_dst` | `S_dst ∘ leftInverse(S_src): P_src_live -> P_dst` | Source 在 live logical domain 上 injective |
| 经过 transpose/view/reduce | layout + `T: L_src -> L_dst` | 按 domain/codomain 选择 `T ∘ layout` 或 `layout ∘ T` | 必须显式说明 transform 方向 |

表中的 inverse 只在其数学前提满足的范围内有效。实际 lowering 通常优先遍历 logical domain，而不是生成覆盖所有 physical holes 的物理到物理函数。

## 8. 组合一：Distributed → Storage

这是最重要、最直接的一类组合。

给定：

```text
D: H -> L
S: L -> P
```

组合得到：

```text
A = S ∘ D: H -> P
```

它回答：

> 每个 lane/warp 的每个 register slot，应访问哪个 Shared/Global 地址？

### 8.1 数据流

```text
(register, lane, warp)
          |
          | Distributed Layout D
          v
    logical coordinate
          |
          | Storage Layout S
          v
     physical address
```

### 8.2 一个完整的小型例子

假设使用 4 个 lane，每 lane 有 2 个 register slot：

```text
lane = [l1 l0]
reg  = [r0]
```

hardware 输入向量是：

```text
x = [l1, l0, r0]^T
```

逻辑 Tile 是 `2×4`，坐标 bit 为：

```text
y = [m0, n1, n0]^T
```

定义 Distributed Layout：

```text
m0 = l1
n1 = l0
n0 = l1 XOR r0
```

矩阵形式：

```text
      [1 0 0]
B_D = [0 1 0]
      [1 0 1]
```

也就是：

```text
y = B_D x
```

现在定义 Shared Storage Layout。物理 offset 使用 3 个 bit：

```text
p = [p2, p1, p0]^T
```

令：

```text
p2 = m0
p1 = m0 XOR n1
p0 = n0
```

这里 `p1 = m0 XOR n1` 是一个简化的 row-based swizzle。

Storage 矩阵是：

```text
      [1 0 0]
B_S = [1 1 0]
      [0 0 1]
```

组合矩阵：

```text
B_A = B_S * B_D
```

计算结果为：

```text
      [1 0 0]
B_A = [1 1 0]
      [1 0 1]
```

因此可以直接从 hardware bits 得到 physical address bits：

```text
p2 = l1
p1 = l1 XOR l0
p0 = l1 XOR r0
```

### 8.3 代入一个具体 carrier

令：

```text
l1 = 1
l0 = 0
r0 = 1
```

也就是：

```text
lane = binary 10 = 2
reg  = 1
```

先走 Distributed Layout：

```text
m0 = 1
n1 = 0
n0 = 1 XOR 1 = 0
```

得到逻辑坐标：

```text
(m, n) = (1, binary 00) = (1, 0)
```

再走 Storage Layout：

```text
p2 = 1
p1 = 1 XOR 0 = 1
p0 = 0
```

物理 offset：

```text
p = binary 110 = 6
```

直接使用组合矩阵也得到：

```text
p2 = l1 = 1
p1 = l1 XOR l0 = 1
p0 = l1 XOR r0 = 0
```

所以：

```text
A(lane=2, reg=1) = offset 6
```

### 8.4 组合地址不等于 Store 一定合法

`S ∘ D` 能计算地址，但还需要验证 ownership。

如果 Distributed Layout 存在 replication：

```text
D(h1) = D(h2), h1 != h2
```

那么：

```text
S(D(h1)) = S(D(h2))
```

两个 carrier 会访问同一个地址。

对于 Load：

- 可能允许多个线程重复读取；
- 但仍要计入访存代价。

对于 Store：

- 一般必须选举唯一 writer；
- 或者操作本身具有明确的 atomic/reduction 语义；
- 不能仅因为写入值“看起来相同”就默认允许 race。

因此 Store lowering 需要同时消费：

```text
address map + ownership policy
```

### 8.5 replication 的矩阵表现

如果某个 hardware input bit 不影响任何 logical output bit，它在矩阵中对应全零列。

例如增加一个 bit `q`：

```text
D(lane, reg, q) = D(lane, reg)
```

则 `q=0` 和 `q=1` 映射到相同逻辑元素。

更一般地，只要存在非零向量 `k` 满足：

```text
B_D k = 0
```

那么：

```text
B_D(x XOR k) = B_D x
```

`k` 属于 `B_D` 的 kernel，表示一种 replication 方向。

## 9. 组合二：Distributed → Distributed

给定两个 Register 分布：

```text
D_src: H_src -> L
D_dst: H_dst -> L
```

二者输出都是 logical coordinate，因此不能直接写：

```text
D_dst ∘ D_src
```

因为 `D_src` 的输出是 `L`，而 `D_dst` 的输入是 `H_dst`，domain 不匹配。

真正的问题通常是：

> 对于目标 carrier `h_dst` 所需要的逻辑元素，应从哪个源 carrier `h_src` 获取？

### 9.1 需要 inverse/right-inverse

目标 carrier 持有的逻辑元素是：

```text
l = D_dst(h_dst)
```

然后需要从 source layout 反向找到：

```text
h_src = chooseSourceCarrier(l)
```

如果 `D_src` 是双射，可以使用普通 inverse：

```text
h_src = D_src^{-1}(l)
```

所以 redistribution map 为：

```text
R = D_src^{-1} ∘ D_dst
```

方向是：

```text
R: H_dst -> H_src
```

它表示每个目标 carrier 应从哪个源 carrier 读取。

### 9.2 source 有 replication 时怎么办

如果 `D_src` 不是 injective，一个逻辑元素可能存在多个 source carrier：

```text
D_src(h_src_0) = l
D_src(h_src_1) = l
```

这时普通 inverse 不存在，但只要 `D_src` 覆盖全部所需逻辑元素，就可以构造 right-inverse：

```text
R_src: L -> H_src
```

并满足：

```text
D_src ∘ R_src = Identity_L
```

然后：

```text
R = R_src ∘ D_dst
```

right-inverse 不一定唯一。编译器应优先选择：

1. 同一 register slot；
2. 同一 lane；
3. 同一 warp；
4. warp shuffle；
5. Shared Memory staging。

因此 right-inverse 的选择不仅是数学问题，也是代价模型问题。

### 9.3 一个 transpose 式重排例子

假设 source carrier bits 是：

```text
h_src = [lane_bit, reg_bit]^T
```

Source Layout：

```text
m = lane_bit
n = reg_bit
```

即：

```text
B_src = [1 0]
        [0 1]
```

Destination Layout：

```text
m = reg_bit
n = lane_bit
```

即：

```text
B_dst = [0 1]
        [1 0]
```

`B_src` 是 identity，因此其 inverse 仍是 identity：

```text
B_R = B_src^{-1} * B_dst
    = B_dst
```

所以目标 carrier `[lane, reg]` 需要从源 carrier `[reg, lane]` 取值。

lowering 可能实现为：

- 同线程 register reorder；
- 跨 lane shuffle；
- 跨 warp 时 Shared Memory staging。

具体选择取决于 carrier bit 在 lane/reg/warp 哪一层发生变化。

### 9.4 coverage 不足时不能构造转换

如果 `D_src` 没有覆盖目标需要的某些逻辑元素：

```text
Image(D_dst) ⊄ Image(D_src)
```

那么不存在完整的 redistribution。

此时只能：

- 从更早的 producer 重新生成缺失元素；
- 从内存重新 Load；
- 修改目标候选 layout；
- 或报告 layout conflict。

不能通过任意 pseudoinverse 伪造不存在的数据。

## 10. 组合三：Storage → Storage

给定：

```text
S_src: L -> P_src
S_dst: L -> P_dst
```

它们都从 logical coordinate 出发，因此也不能直接相互 compose。

要回答：

> source 物理地址中的一个 live 元素，应搬到哪个 destination 物理地址？

需要先从 source address 恢复 logical coordinate，再应用 destination layout：

```text
M = S_dst ∘ S_src^{-1}
```

方向是：

```text
M: P_src_live -> P_dst
```

### 10.1 Storage inverse 的特殊性

Storage Layout 通常是 injective，但不一定覆盖所有物理地址。

例如带 padding 的 layout：

```text
logical elements:  e0 e1 e2 e3
physical storage:  e0 e1 pad e2 e3 pad
```

逻辑元素都对应唯一地址，所以 `S_src` 是 injective；但某些物理地址是 padding hole，没有 logical preimage。

因此 `S_src^{-1}` 只在 live physical image 上有定义：

```text
P_src_live = Image(S_src)
```

编译器不能把 padding 地址当成真实元素处理。

### 10.2 更常用的实现方式：以 logical domain 为循环域

实际 lowering 往往不显式生成 `P_src -> P_dst` 的完整函数，而是遍历 logical elements：

```text
for each logical coordinate l:
    src_addr = S_src(l)
    dst_addr = S_dst(l)
    copy(src_addr, dst_addr)
```

这种方式的优势是：

- 自动跳过 source/destination padding hole；
- 易于生成边界 predicate；
- logical ownership 更清楚；
- 可以与 Distributed Layout 组合，决定每个线程负责哪些 `l`。

### 10.3 Row-major → XOR swizzle

例如：

```text
S_src(row, col) = row * stride + col
S_dst(row, col) = row * stride + (col XOR row_phase)
```

物理到物理的概念映射为：

```text
dst_addr = S_dst(S_src^{-1}(src_addr))
```

但 lowering 更适合写成：

```text
for each (row, col):
    src = row * stride + col
    dst = row * stride + (col XOR row_phase)
```

### 10.4 Storage 非 injective 时的问题

如果：

```text
S_src(l1) = S_src(l2), l1 != l2
```

普通 Tensor storage 语义下说明两个逻辑元素 alias 到同一地址。

除非 IR 明确表示 broadcast view、overlapping view 或特殊 reduction storage，否则应由 verifier 拒绝。

## 11. 组合四：带 Logical Transform 的布局

Transpose、reshape、broadcast、reduce projection 会改变 logical coordinate domain。

定义 logical transform：

```text
T: L_src -> L_dst
```

### 11.1 Distributed Layout 经过 transpose

给定：

```text
D_src: H -> L_src
T: L_src -> L_dst
```

新布局为：

```text
D_dst = T ∘ D_src
```

它只是改变逻辑解释，不一定立即搬运数据。

只有后续 consumer 要求另一种 physical carrier 分布时，才需要真实 `convert_layout`。

### 11.2 Storage view

如果一个 view 使用：

```text
V: L_view -> L_base
```

底层 Storage Layout 为：

```text
S_base: L_base -> P
```

那么 view 的地址布局是：

```text
S_view = S_base ∘ V
```

这也是 `frisk.layout_view` 应表达的核心关系：

- view 改变 logical coordinate；
- 底层 allocation 的 physical mapping 不变；
- alias analysis 仍能识别相同 storage object。

### 11.3 Reduce projection

Reduce 将输入 logical domain 投影到输出 logical domain：

```text
P_reduce: L_input -> L_output
```

输入 Distributed Layout 为：

```text
D_input: H -> L_input
```

投影后的 ownership relation 为：

```text
P_reduce ∘ D_input: H -> L_output
```

多个 carrier 映射到同一个 output logical coordinate，表示它们需要参与同一 reduction，而不是普通 replication。

solver 随后选择：

- thread-local reduce；
- warp shuffle；
- warp-group/shared tree；
- 最终唯一 owner 或 replicated result。

## 12. 组合前必须检查什么

### 12.1 Dimension 名称和顺序

不要只比较 rank。

```text
(M, N)
```

与：

```text
(N, M)
```

需要显式 transpose transform，不能按位置自动视为相同。

### 12.2 Extent

例如：

```text
D output: M=64, N=64
S input:  M=64, N=128
```

不能直接组合。必须说明：

- D 只覆盖 S 的一个 slice；
- D 会 repeat；
- 或者 S 使用 subview。

### 12.3 Bit width 和 packing

对于 FP4/INT4：

```text
logical element -> (byte_offset, bit_offset)
```

必须使用 storage bit 数，不能简单使用向上取整后的 `dtype.bytes()`。

### 12.4 Injectivity、surjectivity 和 coverage

- Distributed producer 需要覆盖 consumer 所需逻辑元素；
- Distributed 可以有 replication；
- 普通 Storage 必须对 live logical domain injective；
- 带 padding Storage 不必覆盖所有 physical address。

### 12.5 Dynamic bounds

如果 extent 是动态的，而编译器无法证明组合关系：

- 可以生成带 predicate 的 guarded candidate；
- 可以回退到更保守的 layout；
- 或报告 unsupported；
- 不能把 `unknown` 当成 `true`。

### 12.6 Target contract

即使数学上可以组合，也可能不满足硬件：

- WGMMA fragment shape；
- TMA box/alignment；
- Shared swizzle alignment；
- ldmatrix access pattern；
- vector load/store alignment；
- warp-group topology。

因此：

```text
mathematically valid != target legal
```

## 13. “仿射外层 × GF(2) 位线性内层”

这里的“×”不是普通数值乘法，而是分层布局组合。

### 13.1 为什么需要分层

GF(2) 适合：

- 静态 2 次幂 tile；
- lane/register bit mapping；
- XOR swizzle；
- bit permutation；
- WGMMA fragment atom。

GF(2) 不适合独立处理：

- 非 2 次幂 extent；
- 动态 shape；
- ragged boundary；
- padding；
- 普通整数 stride；
- 带进位整数加法。

因此将逻辑坐标分解为：

```text
logical_i = outer_i * inner_extent_i + inner_i
```

### 13.2 WGMMA Tile 示例

假设大 Tile 为 `128×96`，内层原子为 `64×16`：

```text
m = 64 * m_outer + m_inner
n = 16 * n_outer + n_inner
```

范围为：

```text
m_outer ∈ [0, 2)
n_outer ∈ [0, 6)
m_inner ∈ [0, 64)
n_inner ∈ [0, 16)
```

其中：

- `m_inner/n_inner` 是 2 次幂范围，可以用 GF(2) bit matrix 表示；
- `n_outer` 的 extent 是 6，不是 2 次幂，适合用 Affine/Presburger 表示；
- 动态边界和 predicate 也由外层处理。

如果全部强制放入 GF(2)，`n_outer` 往往需要补到 8，导致逻辑宽度从 96 补到 128，并引入无效元素和额外 predicate。

### 13.3 Shared Storage 的分层地址

先把坐标拆成 outer/inner：

```text
row = R * row_outer + row_inner
col = C * col_outer + col_inner
```

Affine 外层计算 atom base：

```text
outer_base =
    row_outer * outer_row_stride
  + col_outer * atom_bytes
  + padding(row_outer)
```

GF(2) 内层计算 swizzled offset：

```text
inner_offset_bits = B_shared * inner_coordinate_bits
```

最终：

```text
byte_offset = outer_base + inner_offset
```

### 13.4 为什么要检查 alignment/no-overlap

GF(2) inner 通常假设自己只控制 atom 内的低地址 bit。Affine outer 控制 atom base 的高位。

如果 atom 是 128 bytes，理想情况是：

```text
outer_base % 128 == 0
0 <= inner_offset < 128
```

这样外层和内层地址范围不会互相覆盖。

如果 outer base 未正确对齐，普通整数加法可能产生 carry，并改变 GF(2) 内层以为自己独占的 bit。此时不能简单把两个矩阵拼在一起。

ProductLayout verifier 必须：

- 证明 base alignment；
- 证明 inner offset range；
- 验证 padding 后仍满足 target alignment；
- 无法证明时拒绝或使用更一般的表达。

### 13.5 outer bit 影响 inner swizzle 怎么办

某些 swizzle phase 或 base offset 可能依赖 outer coordinate 的低 bit。

有三种合法处理方式：

1. 将这些 outer bits 作为具名输入 bit 一并送入 BitLinear map；
2. 使用 bit-affine 形式 `y = Bx XOR b`，由 outer 层提供 bias `b`；
3. 如果依赖不是 bit-linear，保留为 Affine/特殊 target expression，不伪装成 GF(2) 线性映射。

不能假设 outer 和 inner 永远完全独立。

## 14. Bit-affine map 的组合

纯线性映射满足：

```text
f(0) = 0
```

但布局有时包含固定 origin/XOR bias：

```text
f(x) = A x XOR a
```

这叫 GF(2) affine map，本文称为 bit-affine map。

若：

```text
f(x) = A x XOR a
g(y) = B y XOR b
```

则：

```text
g(f(x))
  = B(Ax XOR a) XOR b
  = BAx XOR Ba XOR b
```

组合结果仍然是 bit-affine：

```text
linear part = B A
bias        = B a XOR b
```

实现上可以：

- 在 `BitLinearLayoutMapAttr` 外保存 origin/bias；
- 或把常量 `1` 作为额外输入维度，将 affine map 齐次化；
- 但 assembly 和 verifier 必须明确区分动态 bit 与常量 bit。

## 15. Frisk 中建议的组合流程

```text
composeLayouts(first, second):
  1. canonicalize named dimensions
  2. verify first outputs match second inputs
  3. bind shape/dtype/dynamic extent information
  4. split outer affine and inner bit-linear atoms
  5. compose inner matrices over GF(2)
  6. compose outer maps with Affine/Presburger
  7. propagate bit-affine bias and validity predicates
  8. reconstruct ProductLayout
  9. verify coverage/injectivity/ownership
 10. verify SM90 instruction and alignment contracts
 11. canonicalize and attach provenance
```

### 15.1 组合结果需要保存 provenance

例如最终 Register → Shared 地址布局应能解释为：

```text
result:
  derived from DistributedLayout of %value
  composed with StorageLayout of %shared_view
  constrained by CopyOp at location X
  selected because 128B swizzle avoids bank conflict
```

发生冲突时，应报告哪两个映射在什么维度、extent 或性质上不兼容，而不是只输出“layout inference failed”。

## 16. GF(2) 能帮助编译器证明什么

### 16.1 Coverage

矩阵的 image 是否覆盖全部逻辑输出 bit。

如果输出有 `m` 个独立 bit，则要求：

```text
rank(B) = m
```

这表示映射是 surjective。

### 16.2 Replication

kernel 中的非零向量表示改变某些 carrier bit 后 logical coordinate 不变。

```text
kernel(B) = {k | Bk = 0}
```

在完整的 2 次幂 domain 中，replication factor 与 kernel dimension 的关系为：

```text
replication factor = 2^(dim kernel(B))
```

实际 compiler 还要考虑裁剪后的 extent 和 validity predicate。

### 16.3 Storage injectivity

若 Storage inner matrix 的 kernel 只包含零向量：

```text
kernel(B) = {0}
```

则它在对应完整 bit domain 上是 injective。

带 outer/padding 的完整 Storage 仍需额外验证 outer map 和 live domain。

### 16.4 Inverse

方阵 `B` 满 rank时存在唯一 inverse：

```text
B^-1 B = I
B B^-1 = I
```

可通过 GF(2) 高斯消元计算。

### 16.5 Right-inverse

如果：

```text
B: H -> L
```

是 surjective，则存在：

```text
R: L -> H
```

满足：

```text
B R = I_L
```

它为每个 logical element 选择一个 carrier，适合处理 replicated Distributed Layout。

### 16.6 Left-inverse

如果：

```text
B: L -> P
```

是 injective，则存在：

```text
L_inv: P -> L
```

在 image 上满足：

```text
L_inv B = I_L
```

它适合从 live physical storage address 恢复 logical coordinate。

## 17. 常见误解

### 17.1 XOR 不是普通整数加法

```text
3 + 1 = 4
3 XOR 1 = 2
```

原因是 XOR 没有进位。

### 17.2 GF(2) 不是“所有数字对 2 取模”那么简单

在布局中，我们不是把整个 lane/address 当成一个模 2 数，而是把它拆成多个 bit，并把每个 bit 当成 GF(2) 元素。

一个 5-bit lane id 对应 GF(2) 上的 5 维向量，而不是 GF(2) 中的单个值。

### 17.3 LinearLayout 不等于普通线性地址

普通线性地址通常是：

```text
offset = row * stride + col
```

这里的“linear”是整数/仿射意义。

GF(2) LinearLayout 的“linear”是 bit-vector 意义：

```text
output_bits = B * input_bits
```

两者是不同的代数系统。

### 17.4 DistributedEncodingAttr 不分配物理寄存器

它描述：

```text
logical register slot/lane/warp -> logical element
```

真正的物理寄存器编号仍由后端 register allocator 决定。

### 17.5 StorageLayoutAttr 不只用于 Shared

它主要用于 Shared XOR/padding/TMA layout，但也可以描述：

- packed global layout；
- sub-byte storage；
- 特殊 global tile permutation。

普通 Global MemRef 应优先复用 MLIR builtin strided/affine layout。

### 17.6 数学上可逆不代表代码生成免费

两个 Distributed Layout 即使存在可逆映射，转换仍可能需要：

- register reorder；
- warp shuffle；
- cross-warp shared staging；
- barrier；
- 更多寄存器。

代价模型仍需判断是否应该转换，或者改选 producer/consumer layout。

## 18. 数学附录

### 18.1 GF(2) 向量空间

`GF(2)^n` 是所有长度为 `n` 的 bit vector 集合：

```text
GF(2)^n = {(x0, ..., x[n-1]) | xi ∈ {0,1}}
```

向量加法逐 bit XOR：

```text
x + y = x XOR y
```

它共有：

```text
2^n
```

个向量。

### 18.2 线性映射

映射：

```text
T: GF(2)^n -> GF(2)^m
```

如果满足：

```text
T(x XOR y) = T(x) XOR T(y)
```

就是 GF(2) 线性映射。

任意这样的映射都可以写成：

```text
T(x) = Bx
```

其中 `B` 是 `m×n` 的 0/1 矩阵。

### 18.3 Image

```text
Image(B) = {Bx | x ∈ GF(2)^n}
```

它表示所有能够被生成的输出。

对 Distributed Layout 来说，image 表示能被 carrier 覆盖的 logical coordinate bits。

### 18.4 Kernel

```text
Kernel(B) = {x | Bx = 0}
```

如果：

```text
Bx1 = Bx2
```

则：

```text
B(x1 XOR x2) = 0
```

所以 `x1 XOR x2` 属于 kernel。

这解释了为什么 kernel 对应 replication/alias 方向。

### 18.5 Rank-nullity

对于：

```text
B: GF(2)^n -> GF(2)^m
```

有：

```text
rank(B) + nullity(B) = n
```

其中：

- `rank(B) = dim Image(B)`；
- `nullity(B) = dim Kernel(B)`。

### 18.6 Injective 与 surjective

`B` injective 当且仅当：

```text
Kernel(B) = {0}
```

也就是：

```text
rank(B) = n
```

`B` surjective 当且仅当：

```text
rank(B) = m
```

### 18.7 Inverse、right-inverse 和 left-inverse

#### Inverse

当 `m=n` 且 `rank(B)=n` 时：

```text
B^-1 B = B B^-1 = I
```

#### Right-inverse

当 `B` surjective 时，存在 `R`：

```text
B R = I_m
```

如果 `B` 不是 injective，`R` 通常不唯一。

#### Left-inverse

当 `B` injective 时，存在 `L`：

```text
L B = I_n
```

如果 `B` 不是 surjective，`L` 在 `Image(B)` 外的取值可以有多种选择。

### 18.8 关于 pseudoinverse

编译器布局代码中的 pseudoinverse 通常不是实数线性代数中的 Moore-Penrose pseudoinverse。

它一般表示通过 GF(2) 高斯消元构造的、满足所需 compose 条件的确定性 generalized inverse，例如：

```text
B G B = B
```

或者在 surjective 情况下直接选择一个 right-inverse。

实现必须明确自己需要的是：

- inverse；
- right-inverse；
- left-inverse；
- 还是只在 image 上有效的 generalized inverse。

不要使用一个含糊的 `inverse()` API 隐藏这些不同前提。

### 18.9 GF(2) 高斯消元

计算 rank/inverse 的方法与普通高斯消元相似，但行操作变成：

- 交换两行；
- 使用一行 XOR 另一行；
- 不需要除以任意 pivot，因为非零 pivot 只能是 `1`。

例如消去某列：

```text
row_i = row_i XOR pivot_row
```

这使 GF(2) 矩阵分析非常适合使用 bitset/整数位运算实现。

## 19. 对 Frisk 实现的直接建议

1. `BitLinearLayoutMapAttr` 使用具名 input/output dimensions，避免只按矩阵位置解释 lane/warp/register。
2. 明确保存每个维度的 bit extent，禁止把非 2 次幂 domain 静默补齐后当作有效元素。
3. 区分纯线性 `Bx` 与 bit-affine `Bx XOR b`。
4. API 中分别提供 `inverse`、`rightInverse`、`leftInverseOnImage`，不要统一成无前提的 inverse。
5. compose 前验证 dimension name、extent、shape、dtype 和 target contract。
6. Distributed verifier 检查 coverage、replication 和 writer policy。
7. Storage verifier 检查 live-domain injectivity、padding hole、alignment 和 allocation size。
8. 小型 Tile 测试中同时使用矩阵算法和逐点枚举，两者互为 oracle。
9. 生产热路径使用 GF(2) 高斯消元，不对常见 WGMMA/swizzle 调用 Z3。
10. lowering 中将 compose 后的 map 与手写 SM90 地址公式、PTX fragment 表做差分验证。

## 20. 总结

XOR 是无进位的 bit 加法，GF(2) 是以 XOR 为加法的二元线性代数。它们适合 GPU 布局，不是因为 GPU 索引“都是特殊数学”，而是因为大量 fragment/swizzle 规则本来就是：

```text
选择某些 input bits
重新排列这些 bits
把若干 bits XOR 到一起
```

Frisk 中：

```text
Distributed Layout:
    carrier bits -> logical bits

Storage Layout:
    logical bits -> physical address bits

Register -> Shared:
    Storage ∘ Distributed

Register Layout Conversion:
    rightInverse(Source) ∘ Destination

Storage Layout Conversion:
    Destination ∘ leftInverseOnImage(Source)
```

GF(2) 负责静态、2 次幂、bit-linear 的硬件内层；Affine/Presburger 负责非 2 次幂、动态边界、padding 和普通整数地址外层。二者的分层组合，才构成完整的 Frisk Layout Algebra。
