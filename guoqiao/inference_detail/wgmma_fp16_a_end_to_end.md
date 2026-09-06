# SM90 WGMMA FP16 矩阵 A：Global Address → Swizzled Shared Memory → Register Fragment

> 日期：2026-08-22
> 指令原子：`wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16`，RS 形式
> 范围：只展开矩阵 A 的布局链路；B 的 shared-memory descriptor 和 D 的完整 store 布局不在本文展开

## 1. 结论先行

本文固定一个真实的 WGMMA 原子：

```text
D[64, 64]f32 = A[64, 16]f16 × B[16, 64]f16 + D[64, 64]f32
```

采用 WGMMA 的 RS 形式：

```text
A：register fragment
B：shared-memory descriptor
D：register accumulator fragment
```

矩阵 A 的完整数据流为：

```text
Global A[64,16]，row-major FP16，共 2048 B
  │
  │ TMA 将连续数据等价看作 G[16,64]f16
  │ 128B swizzle，16B 为一个 swizzle chunk
  ▼
Shared A：16 个 128B swizzle row，共 2048 B
  │
  │ 128 threads × 4 次 ld.shared.b32
  ▼
每线程 4 个 .f16x2 寄存器，共 8 个 FP16 元素
  │
  │ 128 × 8 = 1024 = 64 × 16
  ▼
PTX Figure 148 规定的 A register fragment
  │
  ▼
wgmma.mma_async.m64n64k16.f32.f16.f16
```

这个例子同时给出两个 GF(2) 映射：

1. `StorageLayout`：逻辑坐标 `(m, k)` 到 swizzled shared byte address；
2. `DistributedLayout`：`(warp, lane, register, subelement)` 到逻辑坐标 `(m, k)`。

二者组合后，可以直接得到每个线程每个寄存器应该读取的 shared address。组合结果还能证明：固定一个 warp 和一个寄存器编号时，32 个 lane 恰好访问 32 个不同的 shared-memory bank。

## 2. 先澄清“物理地址”的含义

CUDA kernel 中拿到的 global pointer 是设备地址空间中的地址，真正的 DRAM channel、partition 和物理页映射由硬件与驱动管理，PTX 不暴露这一级映射。

因此，本文所说的“物理地址”是布局系统可以观察和控制的：

```text
base address + byte offset
```

具体分为：

- Global Memory byte address：`A_base + global_byte_offset`；
- Shared Memory byte address：`A_smem_base + shared_byte_offset`；
- Shared bank：由 shared byte address 的低位决定。

本文不声称推断 DRAM 的真实物理地址。

## 3. 硬件契约

### 3.1 指令与 tile shape

使用：

```ptx
wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
```

它计算一个 `M=64, N=64, K=16` 的 warpgroup MMA。一个 warpgroup 由 4 个 warp、共 128 个线程组成。

根据 NVIDIA PTX ISA：

- A 可以位于 registers 或 shared memory；
- B 必须位于 shared memory；
- RS 形式的 A 是 register operand，B 是 64-bit shared descriptor；
- `.m64nNk16` 的 FP16 A fragment 中，每个线程持有 4 个 `.f16x2` 寄存器，即 8 个 FP16 元素；
- `N=64` 且 accumulator 为 FP32 时，每线程持有 `N/2 = 32` 个 FP32 accumulator registers；
- `.aligned` 要求 warpgroup 内所有线程执行相同的 WGMMA 指令，否则行为未定义；
- 当前 PTX 文档将该 WGMMA 路径标为需要 `sm_90a`。

对应官方章节：

- [WGMMA Register Fragments and Shared Memory Layouts](https://docs.nvidia.com/cuda/parallel-thread-execution/#register-fragments-and-shared-memory-matrix-layouts)
- [Matrix Fragments for wgmma.mma_async.m64nNk16](https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-wgmma-mma-async-m64nnk16)
- [wgmma.mma_async instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async)
- [NVIDIA Figure 148：FP16/BF16 A register fragment](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-A.png)

### 3.2 为什么选 RS，而不是 SS

SS 形式中，A 和 B 都由 shared descriptor 直接交给 WGMMA。A 不会形成程序员可见的 register fragment。

本文需要研究“最后到寄存器，并与真实硬件 register fragment 一致”，所以必须使用 RS：

```text
RS: {A registers}, B descriptor
```

需要特别注意：本文 A 的 128B swizzle 是 TMA staging layout，用来让显式 shared→register load 无 bank conflict。RS 指令本身不接收 A descriptor。

## 4. Global A 的地址布局

### 4.1 逻辑矩阵

固定 A 为：

```text
A : tensor<64x16xf16>
layout : row-major
```

坐标范围：

```text
0 <= m < 64
0 <= k < 16
sizeof(fp16) = 2 B
```

Global byte address 为：

```text
gmem_addr(m, k) = A_base + 2 × (16m + k)
```

整个 tile 大小：

```text
64 × 16 × 2 B = 2048 B
```

假设：

- `A_base` 至少 128B 对齐，以满足所选 TMA swizzle 路径；
- 当前例子是完整 tile，不处理越界和 ragged predicate。

### 4.2 为 TMA 重新分组，不改变 global address

A 的一行只有：

```text
16 × 2 B = 32 B
```

为了构造完整的 128B swizzle row，将连续 4 行组合为一个 128B group：

```text
4 × 32 B = 128 B
```

定义 TMA 视图：

```text
G[y, u] : tensor<16x64xf16>

y = floor(m / 4) = m >> 2
u = 16 × (m mod 4) + k
```

验证地址不变：

```text
2 × (64y + u)
= 2 × (64 × floor(m/4) + 16 × (m mod 4) + k)
= 2 × (16m + k)
```

所以 `A[64,16]` 与 `G[16,64]` 只是逻辑坐标分组不同，global byte address 完全相同，没有 transpose，也没有额外 copy。

## 5. TMA 128B Swizzle 到 Shared Memory

### 5.1 16B chunk 坐标

TMA swizzle 的粒度固定为 16B。对于 FP16，每个 16B chunk 含 8 个元素。

将 `u` 分解为：

```text
x = floor(u / 8)     // 128B row 内的 16B chunk，0 <= x < 8
e = u mod 8          // chunk 内的 FP16 element，0 <= e < 8
```

代入 `(m,k)`：

```text
x = 2 × (m mod 4) + floor(k / 8)
e = k mod 8
```

### 5.2 128B swizzle 公式

NVIDIA CUDA Programming Guide 对 128B 模式给出的 chunk index 关系是：

```text
x_swizzled = ((y + pattern_offset) mod 8) XOR x
```

其中：

```text
pattern_offset = (A_smem_base / 128) mod 8
```

本文让 shared buffer 以 1024B 对齐：

```text
A_smem_base mod 1024 = 0
pattern_offset = 0
```

因此：

```text
x_swizzled = (y mod 8) XOR x
```

最终 shared byte offset：

```text
smem_off(m, k)
  = 128 × y
  + 16 × x_swizzled
  + 2 × e

y           = m >> 2
x           = 2 × (m & 3) + (k >> 3)
x_swizzled  = (y & 7) XOR x
e           = k & 7
```

官方约束与模式说明：

- [CUDA Programming Guide: TMA Swizzle Modes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-swizzle-modes)
- 128B 模式以 16B 为 swizzle granularity；
- global alignment 为 128B；
- shared 至少 128B 对齐；
- 128B swizzle pattern 每 1024B 重复；本文用 1024B 对齐让 pattern offset 固定为 0。

### 5.3 Shared StorageLayout 的 GF(2) 形式

把坐标拆成 bit：

```text
m = [m5 m4 m3 m2 m1 m0]₂
k = [k3 k2 k1 k0]₂
```

将 shared byte offset 写成：

```text
smem_off = [b10 b9 b8 b7 b6 b5 b4 b3 b2 b1 b0]₂
```

因为元素是 2B 对齐：

```text
b0  = 0
b1  = k0
b2  = k1
b3  = k2
b4  = k3 XOR m2
b5  = m0 XOR m3
b6  = m1 XOR m4
b7  = m2
b8  = m3
b9  = m4
b10 = m5
```

这里的三个关键 XOR 正是 128B swizzle：

```text
chunk_bit_0' = k3 XOR m2
chunk_bit_1' = m0 XOR m3
chunk_bit_2' = m1 XOR m4
```

若输入 bit 列顺序固定为：

```text
[m0, m1, m2, m3, m4, m5, k0, k1, k2, k3]
```

输出列顺序固定为：

```text
[b1, b2, b3, b4, b5, b6, b7, b8, b9, b10]
```

则 GF(2) 矩阵为：

```text
          m0 m1 m2 m3 m4 m5 k0 k1 k2 k3
b1   = [  0  0  0  0  0  0  1  0  0  0 ]
b2   = [  0  0  0  0  0  0  0  1  0  0 ]
b3   = [  0  0  0  0  0  0  0  0  1  0 ]
b4   = [  0  0  1  0  0  0  0  0  0  1 ]
b5   = [  1  0  0  1  0  0  0  0  0  0 ]
b6   = [  0  1  0  0  1  0  0  0  0  0 ]
b7   = [  0  0  1  0  0  0  0  0  0  0 ]
b8   = [  0  0  0  1  0  0  0  0  0  0 ]
b9   = [  0  0  0  0  1  0  0  0  0  0 ]
b10  = [  0  0  0  0  0  1  0  0  0  0 ]
```

这是一个 `10×10` 满秩矩阵。逆映射可以直接写出：

```text
k0 = b1
k1 = b2
k2 = b3
m2 = b7
m3 = b8
m4 = b9
m5 = b10
k3 = b4 XOR b7
m0 = b5 XOR b8
m1 = b6 XOR b9
```

因此它在 2048B live shared domain 上是单射和满覆盖的：每个逻辑 A 元素对应唯一 2B slot，没有 alias，也没有洞。

### 5.4 为什么固定 1024B 对齐很重要

如果 `pattern_offset != 0`，公式包含：

```text
(y + pattern_offset) mod 8
```

普通整数加法可能产生 carry，它一般不是一个纯 GF(2) 线性变换。

Frisk 有三种正确处理方式：

1. 当前首版直接要求 1024B 对齐，使 offset 为 0；
2. 把固定 offset 表示为 bit-affine map；
3. 由 affine outer 处理带 carry 的 phase，再与 GF(2) inner 组合。

绝不能忽略非零 offset，然后仍套用 offset=0 的矩阵。

## 6. NVIDIA Figure 148 的真实 A Register Fragment

### 6.1 线程坐标

定义 warpgroup 内线程编号：

```text
tid = threadIdx.x mod 128

warp  = tid >> 5          // 0..3
lane  = tid & 31          // 0..31
group = lane >> 2         // 0..7
quad  = lane & 3          // 0..3
```

每线程有 4 个物理 32-bit A registers：

```text
reg = 0..3
```

每个 register 是 `.f16x2`，内部包含两个 FP16 subelements：

```text
sub = 0..1
element_number = 2 × reg + sub   // a0..a7
```

### 6.2 精确坐标公式

根据 NVIDIA Figure 148，线程 `(warp,lane)` 的 `reg/sub` 对应：

```text
m = 16 × warp
  + 8 × (reg & 1)
  + group

k = 8 × (reg >> 1)
  + 2 × quad
  + sub
```

等价分项表：

| Physical register | Fragment elements | `m`                  | `k`                 |
| ----------------- | ----------------- | ---------------------- | --------------------- |
| `reg0`          | `{a0,a1}`       | `16warp + group`     | `2quad + {0,1}`     |
| `reg1`          | `{a2,a3}`       | `16warp + 8 + group` | `2quad + {0,1}`     |
| `reg2`          | `{a4,a5}`       | `16warp + group`     | `8 + 2quad + {0,1}` |
| `reg3`          | `{a6,a7}`       | `16warp + 8 + group` | `8 + 2quad + {0,1}` |

这与 Figure 148 完全对应：

- warp 0 的 T0–T31 覆盖 A 的 row 0–15；
- warp 1 的 T32–T63 覆盖 row 16–31；
- warp 2 的 T64–T95 覆盖 row 32–47；
- warp 3 的 T96–T127 覆盖 row 48–63；
- `a0..a3` 覆盖 K 的 0–7；
- `a4..a7` 覆盖 K 的 8–15。

### 6.3 DistributedLayout 的 GF(2) 形式

将 carrier 拆成 bit：

```text
warp     = [w1 w0]₂
lane     = [l4 l3 l2 l1 l0]₂
register = [r1 r0]₂
sub      = [s]₂
```

逻辑坐标 bit 为：

```text
m0 = l2
m1 = l3
m2 = l4
m3 = r0
m4 = w0
m5 = w1

k0 = s
k1 = l0
k2 = l1
k3 = r1
```

输入列顺序取：

```text
[s, r0, r1, l0, l1, l2, l3, l4, w0, w1]
```

输出行顺序取：

```text
[m0, m1, m2, m3, m4, m5, k0, k1, k2, k3]
```

GF(2) 矩阵为：

```text
          s r0 r1 l0 l1 l2 l3 l4 w0 w1
m0   = [ 0  0  0  0  0  1  0  0  0  0 ]
m1   = [ 0  0  0  0  0  0  1  0  0  0 ]
m2   = [ 0  0  0  0  0  0  0  1  0  0 ]
m3   = [ 0  1  0  0  0  0  0  0  0  0 ]
m4   = [ 0  0  0  0  0  0  0  0  1  0 ]
m5   = [ 0  0  0  0  0  0  0  0  0  1 ]
k0   = [ 1  0  0  0  0  0  0  0  0  0 ]
k1   = [ 0  0  0  1  0  0  0  0  0  0 ]
k2   = [ 0  0  0  0  1  0  0  0  0  0 ]
k3   = [ 0  0  1  0  0  0  0  0  0  0 ]
```

这个矩阵只是 bit permutation，rank 为 10：

```text
4 warps × 32 lanes × 4 registers × 2 FP16/register
= 1024 FP16 values
= 64 × 16
```

每个逻辑元素恰好由一个 `(warp,lane,reg,sub)` 持有，没有 replication，也没有遗漏。

## 7. 组合 StorageLayout 与 DistributedLayout

### 7.1 直接得到 shared-load 地址

将寄存器 fragment 的 `(m,k)` 公式代入 shared swizzle 公式，得到：

```text
b0  = 0
b1  = s
b2  = l0
b3  = l1
b4  = r1 XOR l4
b5  = l2 XOR r0
b6  = l3 XOR w0
b7  = l4
b8  = r0
b9  = w0
b10 = w1
```

这就是组合：

```text
CarrierToShared
  = LogicalToShared compose CarrierToLogical
```

在 GF(2) 上等于两个矩阵相乘。

### 7.2 每个物理寄存器只需一次 32-bit load

对一个固定 `reg`，两个 subelements 的地址只在 `b1=s` 上不同：

```text
sub=0：offset
sub=1：offset + 2 B
```

因此二者在 shared 中连续且 4B 对齐，可以用一次 32-bit load 打包成 `.f16x2`：

```text
load_base(tid, reg) = smem_off(m(tid,reg), k(tid,reg,sub=0))
```

每线程执行 4 次：

```text
reg0 <- ld.shared.b32 [A_smem + load_base(tid, 0)]
reg1 <- ld.shared.b32 [A_smem + load_base(tid, 1)]
reg2 <- ld.shared.b32 [A_smem + load_base(tid, 2)]
reg3 <- ld.shared.b32 [A_smem + load_base(tid, 3)]
```

全 warpgroup 的读取量：

```text
128 threads × 4 registers/thread × 4 B/register = 2048 B
```

恰好读完整个 A tile。

### 7.3 GF(2) 证明 shared load 无 bank conflict

Hopper shared memory 有 32 个 bank，连续 32-bit words 映射到连续 bank。对一次 32-bit load：

```text
bank = (shared_byte_offset >> 2) mod 32
```

所以 bank 的 5 个 bit 是 `[b2,b3,b4,b5,b6]`：

```text
bank_bit0 = l0
bank_bit1 = l1
bank_bit2 = l4 XOR r1
bank_bit3 = l2 XOR r0
bank_bit4 = l3 XOR w0
```

对一次 warp load，`warp` 和 `reg` 固定，因此 `w0/r0/r1` 都是常量。lane 的 5 个 bit：

```text
[l0,l1,l2,l3,l4]
```

只是被置换，并与常量做 XOR。于是 lane→bank 是一个 rank=5 的可逆 GF(2) 映射：

```text
32 lanes → 32 distinct banks
```

结论：对 `reg0`、`reg1`、`reg2`、`reg3` 的每一轮 warp-wide `ld.shared.b32`，都不存在 bank conflict。

这比“枚举后看起来没有冲突”更强：编译器可以直接对 bank submatrix 求 rank；rank 为 5 就证明 32 个 lane 的 bank 唯一。

## 8. 完整数值样例：Thread 0

取：

```text
tid   = 0
warp  = 0
lane  = 0
group = 0
quad  = 0
```

### 8.1 Register 0：`{a0,a1}`

```text
m = 0
k = {0,1}
y = 0
x = 0
x_swizzled = 0 XOR 0 = 0
shared offsets = {0,2} B
```

因此：

```text
T0.reg0 = {A[0,0], A[0,1]}
ld.shared.b32 from A_smem + 0
```

### 8.2 Register 1：`{a2,a3}`

```text
m = 8
k = {0,1}
y = 2
x = 0
x_swizzled = 2 XOR 0 = 2
shared offsets = 128×2 + 16×2 + {0,2}
               = {288,290} B
```

因此：

```text
T0.reg1 = {A[8,0], A[8,1]}
ld.shared.b32 from A_smem + 288
```

### 8.3 Register 2：`{a4,a5}`

```text
m = 0
k = {8,9}
y = 0
x = 1
x_swizzled = 0 XOR 1 = 1
shared offsets = {16,18} B
```

因此：

```text
T0.reg2 = {A[0,8], A[0,9]}
ld.shared.b32 from A_smem + 16
```

### 8.4 Register 3：`{a6,a7}`

```text
m = 8
k = {8,9}
y = 2
x = 1
x_swizzled = 2 XOR 1 = 3
shared offsets = 128×2 + 16×3 + {0,2}
               = {304,306} B
```

因此：

```text
T0.reg3 = {A[8,8], A[8,9]}
ld.shared.b32 from A_smem + 304
```

汇总：

| Thread | Physical register | Figure 148 elements | Logical A coordinates | Shared load base | Bank |
| ------ | ----------------- | ------------------- | --------------------- | ---------------: | ---: |
| T0     | `reg0`          | `{a0,a1}`         | `{A[0,0], A[0,1]}`  |              0 B |    0 |
| T0     | `reg1`          | `{a2,a3}`         | `{A[8,0], A[8,1]}`  |            288 B |    8 |
| T0     | `reg2`          | `{a4,a5}`         | `{A[0,8], A[0,9]}`  |             16 B |    4 |
| T0     | `reg3`          | `{a6,a7}`         | `{A[8,8], A[8,9]}`  |            304 B |   12 |

这与 NVIDIA Figure 148 中 T0 的四组标记完全一致。

## 9. PTX 级数据流骨架

下面只表达寄存器数量、operand 形式和同步顺序。TMA tensor-map 创建、mbarrier phase 和 B descriptor 构造被省略，因此它不是一个可独立编译的完整 kernel。

```ptx
// A_smem 已由 TMA 以 128B swizzle 写入，并完成 mbarrier wait。

.reg .b32    rawA<4>;
.reg .f16x2  a<4>;
.reg .f32    d<32>;       // N=64, FP32 accumulator => 32 regs/thread
.reg .b64    descB;
.reg .pred   scaleD;

// 每个地址都由 CarrierToShared(tid, reg, sub=0) 计算。
ld.shared.b32 rawA0, [A_smem + off0];
ld.shared.b32 rawA1, [A_smem + off1];
ld.shared.b32 rawA2, [A_smem + off2];
ld.shared.b32 rawA3, [A_smem + off3];

// 示意：把相同的 32 bits 解释为 PTX .f16x2 operand。
mov.b32 a0, rawA0;
mov.b32 a1, rawA1;
mov.b32 a2, rawA2;
mov.b32 a3, rawA3;

// 在 WGMMA 读取 A/D registers 前建立寄存器访问顺序。
wgmma.fence.sync.aligned;

wgmma.mma_async.sync.aligned.m64n64k16.f32.f16.f16
  {d0, d1, d2, d3, d4, d5, d6, d7,
   d8, d9, d10, d11, d12, d13, d14, d15,
   d16, d17, d18, d19, d20, d21, d22, d23,
   d24, d25, d26, d27, d28, d29, d30, d31},
  {a0, a1, a2, a3},
  descB,
  scaleD, 1, 1, 0;

wgmma.commit_group.sync.aligned;
wgmma.wait_group.sync.aligned 0;
```

必须满足：

- warpgroup 128 个线程一致执行；
- TMA 写入 A_smem 后已通过正确的 mbarrier phase；
- shared→register loads 已完成；
- `wgmma.fence` 位于先前 A/D register accesses 与 WGMMA 之间；
- 使用 D 之前经过 `commit_group`/`wait_group`。

## 10. 在 Frisk 中应如何表示

### 10.1 Global A

Global A 仍使用普通 row-major MemRef：

```mlir
memref<64x16xf16>
```

其地址映射是普通 affine/strided layout：

```text
(m,k) -> 2 × (16m+k) bytes
```

不需要为 global row-major 地址创建 GF(2) Attr。

### 10.2 Shared A：StorageLayoutAttr

Shared A 应保持 MemRef/storage anchor，并绑定：

```text
StorageLayoutAttr {
  logical_shape       = [64,16],
  element_type        = f16,
  memory_space        = Shared,
  alignment_bytes     = 1024,
  swizzle_mode        = TMA_128B,
  vector_granularity  = 4 bytes,
  map                 = LogicalToSwizzledShared
}
```

其中 `map` 就是第 5.3 节的 10-bit GF(2) map。外层 TMA view：

```text
(m,k) -> (y=m>>2, u=16(m&3)+k)
```

可以作为 affine/product outer；128B XOR 作为 bit-linear inner。

### 10.3 Register A：DistributedEncodingAttr

Register A 是 SSA Tensor：

```mlir
tensor<64x16xf16, #a_wgmma_rs_encoding>
```

encoding 的逻辑内容为：

```text
DistributedEncodingAttr {
  topology = {
    registers_per_thread = 4,
    lanes_per_warp       = 32,
    warps_per_warpgroup  = 4
  },
  packing = {
    physical_register_bits = 32,
    element_bits           = 16,
    elements_per_register  = 2,
    order                  = low_to_high
  },
  map = CarrierToLogical
}
```

`CarrierToLogical` 是第 6.3 节的 GF(2) permutation。

### 10.4 这个真实例子暴露出的 Attr 要求

如果 `DistributedEncodingAttr` 只有：

```text
register/lane/warp -> logical coordinate
```

但没有 `packing/subelement`，会出现两种错误选择：

1. 把每线程描述成 8 个 registers——与硬件的 4 个 `.f16x2` 不一致；
2. 只描述 4 个 registers——无法指出每个 register 的低/高 16-bit 分别对应哪个逻辑元素。

因此 Frisk 的最终实现需要二选一：

- 推荐：给 Distributed encoding 增加显式 `RegisterPackingAttr`；
- 或者：让 Mma/DotOperand encoding 额外携带 physical packing contract。

无论选择哪种方式，通用 logical ownership 与 target-specific physical packing 都必须可同时验证。

### 10.5 InstructionContract

SM90 target rule 为该 Op 产生 hard contract：

```text
WGMMA_RS_FP16_M64N64K16_F32 {
  target                = sm_90a,
  participating_threads = 128,
  a_shape               = [64,16],
  a_type                = f16,
  a_location            = registers,
  a_registers_per_thread= 4,
  a_elements_per_reg    = 2,
  a_layout              = Figure148Map,
  b_location            = shared_descriptor,
  accumulator_type      = f32,
  accumulator_regs      = 32
}
```

这个 contract 是 hard constraint。代价模型不能为了减少一次 conversion 而选择一个不符合 Figure 148 的 A encoding。

### 10.6 Load relation

Shared A 与 Register A 通过一个 `StorageAccess` constraint 连接：

```text
SharedA.LogicalToPhysical
  compose
RegisterA.CarrierToLogical
  =
CarrierToSharedLoadAddress
```

candidate verifier 需要证明：

- 每个 register pair 的两个 FP16 元素连续且 4B 对齐；
- 1024 个逻辑元素 coverage 完整；
- writer/owner 唯一；
- 每轮 warp load 的 bank submatrix rank 为 5；
- shared offset 始终位于 `[0,2048)`；
- register map 与 WGMMA Figure 148 contract 数学等价。

## 11. 推断系统如何得到这个结果

这个布局不应由某一个 Op 在遍历时直接写死。推荐约束流程：

```text
1. TMA load collector
   - 创建 Shared A Storage LayoutVar
   - 生成 none/32B/64B/128B candidates
   - 加入 alignment、box size、memory-space hard constraints

2. WGMMA RS collector
   - 给 A Tensor 创建 Distributed LayoutVar
   - 加入 Figure148Map hard InstructionContract
   - 要求 4 physical registers × 2 packed FP16/thread

3. Shared→Register load relation
   - compose Storage map 与 Distributed map
   - 检查 vector width、alignment、coverage 和 bank rank

4. Candidate pruning
   - 删除不能形成连续 f16x2 pair 的 storage layout
   - 删除 bank rank < 5 的 layout
   - 删除 alignment/size 不合法的 TMA layout

5. Cost selection
   - 在仍合法的 layout 中比较 TMA path、transaction、bank、shared size

6. Materialization
   - Shared view 绑定 128B StorageLayoutAttr
   - A Tensor type 绑定 Figure148 DistributedEncodingAttr
   - lowering 生成 4 次 ld.shared.b32 和 RS WGMMA
```

在当前例子中，128B candidate 同时满足：

- TMA 128B transfer；
- 2048B 无 padding 存储；
- 4B vector load；
- 每轮 32-bank 无冲突；
- Figure 148 的精确 register ownership。

因此它是一个合法且高质量的候选。

## 12. 编译器验证清单

### 12.1 Global address 等价

对所有 `m∈[0,63]`、`k∈[0,15]` 验证：

```text
2 × (16m+k) == 2 × (64y+u)
```

### 12.2 Shared map 双射

枚举所有 1024 个 `(m,k)`：

- offset 为偶数；
- `0 <= offset < 2048`；
- 所有 offset/2 唯一；
- GF(2) matrix rank 为 10。

### 12.3 Register fragment coverage

枚举：

```text
warp 0..3
lane 0..31
reg  0..3
sub  0..1
```

验证 1024 个 `(m,k)` 唯一覆盖 `[64,16]`。

### 12.4 Figure 148 anchor cases

至少固定：

```text
T0   -> rows 0/8,   columns 0/1/8/9
T31  -> rows 7/15,  columns 6/7/14/15
T32  -> rows 16/24, columns 0/1/8/9
T127 -> rows 55/63, columns 6/7/14/15
```

并检查 `{a0,a1}`、`{a2,a3}`、`{a4,a5}`、`{a6,a7}` 顺序。

### 12.5 Bank conflict proof

对每个：

```text
warp ∈ [0,3]
reg  ∈ [0,3]
```

验证 32 个 lane 的 bank 集合恰好为 `{0..31}`，或者直接验证 bank GF(2) submatrix rank 为 5。

### 12.6 Compose consistency

对所有 carrier point 验证：

```text
CarrierToSharedDirect(carrier)
== LogicalToShared(CarrierToLogical(carrier))
```

这是测试 GF(2) compose 顺序最重要的 oracle。

## 13. 边界与不能混淆的事项

1. 本文是 A 的 RS 路径。B 仍需合法的 WGMMA shared descriptor；本文没有给出 B 的完整 canonical layout。
2. A 的 TMA swizzle descriptor 与 WGMMA matrix descriptor 不是同一个对象。A 在 RS WGMMA 中最终是 register operand。
3. 1024B 对齐是为了令 128B swizzle 的 pattern offset 为 0，从而得到纯 GF(2) 形式；并不表示所有 TMA buffer 的最低合法对齐都必须是 1024B。
4. `wgmma.mma_async` 的 `.sync` 不是等待计算完成；完成等待仍要使用 `commit_group`/`wait_group`。
5. `.aligned` 是 warpgroup 一致执行契约，不是内存地址对齐修饰符。
6. FP32 accumulation 的具体内部累加顺序、rounding 与 subnormal handling 由 PTX 标记为 unspecified；布局测试不应假定固定的逐项累加顺序。
7. 完整 kernel 还需要 B descriptor、mbarrier phase、pipeline stage 生命周期、D fragment/store 和边界处理。

## 14. 对 Frisk 布局系统的直接结论

这个真实 WGMMA A 例子说明，Frisk 不能只保存一个模糊的 `LayoutAttr`：

- Global address 是普通 affine/strided storage；
- Shared swizzle 是 `StorageLayoutAttr`，方向为 logical→physical byte address；
- Register fragment 是 `DistributedEncodingAttr`，方向为 hardware carrier→logical coordinate；
- `.f16x2` packing 是独立的 physical register contract；
- WGMMA Figure 148 是不可违反的 `InstructionContract`；
- Shared→Register 是两个布局的 compose，不是把两个 Attr 文本比较相等。

这里最适合 GF(2) 的原因非常具体：

```text
Shared swizzle：XOR + bit permutation
Register fragment：bit permutation
二者组合：GF(2) matrix multiplication
bank conflict：GF(2) submatrix rank
coverage/injectivity：GF(2) full rank
```

而 global tile 选择、ragged 边界、pipeline stage 和非零 swizzle phase 仍应由 affine outer、predicate 或 target-specific contract 处理。这正是“仿射外层 × GF(2) 位线性内层”在真实 SM90 WGMMA 路径中的落点。

## 15. 官方参考

- [NVIDIA PTX ISA 9.3](https://docs.nvidia.com/cuda/parallel-thread-execution/)
- [Register Fragments and Shared Memory Matrix Layouts](https://docs.nvidia.com/cuda/parallel-thread-execution/#register-fragments-and-shared-memory-matrix-layouts)
- [Matrix Fragments for wgmma.mma_async.m64nNk16](https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-fragments-for-wgmma-mma-async-m64nnk16)
- [Figure 148: WGMMA m64nNk16 register fragment layout for A](https://docs.nvidia.com/cuda/parallel-thread-execution/_images/wgmma-64N16-A.png)
- [Matrix Descriptor Format](https://docs.nvidia.com/cuda/parallel-thread-execution/#matrix-descriptor-format)
- [wgmma.mma_async instruction](https://docs.nvidia.com/cuda/parallel-thread-execution/#asynchronous-multiply-and-accumulate-instruction-wgmma-mma-async)
- [CUDA Programming Guide: TMA Swizzle](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#tma-swizzle)
- [CUDA Programming Guide: TMA Swizzle Modes](https://docs.nvidia.com/cuda/cuda-c-programming-guide/#the-swizzle-modes)
