# Fused Index Packing 与 GPU Tail-rANS 真实代码调研结果

## 1. 调研结论

本文基于当前仓库实现、CompressAI 1.2.8 实际源码和原生 rANS 扩展进行核对，对 `FUSED_CDF_GPU_RANS_FEASIBILITY_PLAN.md` 给出代码级可行性判断。

核心结论如下：

1. **CDF 表本身不是每张图动态生成的**。当前 `_quantized_cdf`、`_cdf_length` 和 `_offset` 已经在模型内缓存。真正反复执行的是 symbol 量化、scale-to-index 查找、checkerboard 打包和 CPU/GPU 传输。
2. **当前 `build_indexes()` 是最值得先验证的低层热点**。64 级 scale table 通过 63 次循环比较生成 index；完整 Hybrid codec 每张图调用 8 次，理论上约有 `8×63=504` 轮 tensor 比较/减法。
3. **`torch.searchsorted` 是应该优先测试的等价低风险替代**。对随机数、所有 scale-table 边界及相邻浮点数共 200,199 个值测试，与旧实现的 index 不一致数为 0。
4. **当前完整路径的主要问题是多次同步，而不是 rANS 算术本身**。编码端的 6 条前缀流和 2 个尾部阶段共调用 8 次原生入口，每次分别转移 symbols 和 indexes，合计可见 16 次 GPU→CPU tensor 转换。
5. **GPU Tail-rANS 技术上可实现，但 Batch 1 的收益风险较高**。单图只有 4 条尾流，每条约 8192 个符号；rANS 流内状态递推为串行依赖，4 个 lane 不足以充分利用 RTX 4090。
6. **建议的执行优先级是：阶段剖析 → searchsorted → 14 流统一提交 → fused packing → 条件性 GPU rANS**。不建议直接从完整 GPU coder 开始。

## 2. 核对范围与环境

调研对应仓库提交：

```text
483416aafae4fc6cf92a524287874efa48da475e
```

实际环境：

```text
Python       3.9.20
PyTorch      2.7.1+cu126
CUDA runtime 12.6
CompressAI   1.2.8
Triton       3.3.1
GPU          NVIDIA GeForce RTX 4090
```

核对文件：

- `并行优化/model/mambamodel_hybrid_parallel.py`
- `并行优化/native_parallel_rans/parallel_rans.cpp`
- `model/mamba/mambamodel_1_917.py`
- CompressAI `entropy_models/entropy_models.py`
- `并行优化/docs/FUSED_CDF_GPU_RANS_FEASIBILITY_PLAN.md`

## 3. 当前真实编码路径

### 3.1 概率参数与 index 生成

当前分组为：

```text
[16, 16, 32 | 64, 64, 64, 64]
```

编码和解码时均需要为以下阶段调用 `GaussianConditional.build_indexes()`：

```text
P1 Anchor
P1 Non-anchor
P2 Anchor
P2 Non-anchor
P3 Anchor
P3 Non-anchor
Tail Anchor batch
Tail Non-anchor batch
```

合计 8 次。CompressAI 1.2.8 的实际实现为：

```python
indexes = scales.new_full(scales.size(), len(self.scale_table) - 1).int()
for s in self.scale_table[:-1]:
    indexes -= (scales <= s).int()
```

本项目的 scale table 常量为：

```text
SCALES_MIN    = 0.11
SCALES_MAX    = 256
SCALES_LEVELS = 64
```

因此每次 index 生成有 63 轮循环。在 CUDA 上这不只是 63 次数值比较，还可能带来大量小 kernel 调度和中间 tensor。

### 3.2 CDF 表并没有每张图重建

当前 `_native_cdf_tables()` 使用以下指针作为缓存签名：

```text
_quantized_cdf.data_ptr()
_cdf_length.data_ptr()
_offset.data_ptr()
```

只有当这些表发生变化时才重新转换为 NumPy。因此建议把原计划中的 `Fused-CDF` 更准确地命名为：

```text
Fused Symbol Quantization, Scale-Index Search and Stream Packing
```

`_offset` 是每个 CDF row 的静态 symbol offset，不应与动态的 stream position/length 混为一个概念。

### 3.3 CPU/GPU 同步点

当前原生编码入口每次都执行：

```python
symbols_cpu = packed_symbols.int().cpu().numpy()
indexes_cpu = packed_indexes.int().cpu().numpy()
```

在 GPU 输入下，这两条 `.cpu()` 都会引入数据传输，并且可能作为之前 CUDA 工作的隐式同步点。

当前完整 y 编码路径有：

```text
6次前缀单流原生调用
2次尾部4流原生调用
= 8次 Python/C++ 边界调用
= 16次可见 GPU→CPU tensor 转换
```

解码入口每次有 indexes GPU→CPU，随后 decoded symbols CPU→GPU。前缀 6 阶段必须依次解码，但尾部 Anchor/Non-anchor 各阶段的 4 条流已经合并调用。

### 3.4 OpenMP 的实际作用范围

C++ 后端会执行：

```cpp
workers = std::max(1, std::min(workers, static_cast<int>(tasks.size())));
#pragma omp parallel for schedule(static) num_threads(workers)
```

这意味着：

- 前缀单流调用的 `tasks.size()==1`，即使配置 4 worker 也只有 1 个工作线程；
- 只有尾部 Anchor 和 Non-anchor 两次调用可以同时使用 4 worker；
- pybind 入口已经在 OpenMP 区域外释放 GIL，GIL 不是当前 1-worker/4-worker 差异很小的原因。

## 4. 已完成的可复现验证

### 4.1 `searchsorted` 的等价性

候选实现：

```python
scales = lower_bound_scale(scales)
indexes = torch.searchsorted(scale_table, scales, right=False)
indexes = indexes.clamp_max(len(scale_table) - 1).int()
```

测试数据包括：

- 200,000 个 `[0, 400)` 随机数；
- 64 个 scale-table 精确边界；
- 每个边界的前一个和后一个可表示浮点数；
- 小于 lower bound、大于 scale-table 上界及无穷大值。

实测输出：

```text
table_levels 64
values_tested 200199
mismatches 0
index_range 0 63
```

这证明候选表达式在本次测试范围内与 CompressAI 循环实现等价。正式替换前仍需对 30 张真实图像的全部 index 逐元素比较。

### 4.2 CPU 同形状微基准

| Tensor shape | 循环实现 | `searchsorted` | CPU speedup |
|---|---:|---:|---:|
| `(1,16,16,8)` | 0.9215 ms | 0.0378 ms | 24.39× |
| `(1,32,16,8)` | 1.1451 ms | 0.0429 ms | 26.68× |
| `(4,64,16,8)` | 3.5791 ms | 0.1568 ms | 22.82× |

该结果只证明算法级替换具有潜力，**不能直接外推为 GPU 或完整 codec 加速比**。GPU 结论必须使用 CUDA Event 单独测量。

### 4.3 原生 rANS 微基准

使用与真实尾部流相同的 4 任务、每任务 8192 symbols，300 次重复：

| Workers | Encode | Decode |
|---:|---:|---:|
| 1 | 0.5050 ms | 0.1674 ms |
| 4 | 0.1991 ms | 0.0818 ms |

纯 rANS 内核的 4-worker 编码/解码分别约为 `2.54×/2.05×`。OpenMP 确实在工作，但绝对时间只有零点几毫秒，因此无法显著改变 35–38 ms 的完整 codec 时间。

## 5. 对原可行性方案的修正

### 5.1 不建议直接从 Triton/CUDA fused operator 开始

建议先增加 A0.5：

```text
A0.5 = searchsorted scale-index + 现有 native rANS
```

原因：

- 不增加自定义编译扩展；
- 不改变 CDF 或码流协议；
- 可以先单独确认 scale-index 是否为实际 GPU 热点；
- 只有 A0.5 仍受中间 tensor 和布局限制时，才有必要开发 Triton/CUDA 融合算子。

### 5.2 Fused packing 不应只关注尾部

只修改尾部可以将 8 次 `build_indexes()` 中的 2 次替换掉，但前缀仍保留 6 次。建议：

1. 先在尾部做最小正确性原型；
2. 通过后将相同 scale-index 逻辑扩展到全部前缀阶段；
3. 编码端将所有已生成的 symbols/indexes 打包后一次传输。

### 5.3 编码端可以延迟熵编码，解码端不可以完全延迟

编码端的前缀概率预测有依赖，但量化重建可以直接由 `round(y-mean)+mean` 得到，不需要等待码流写入完成。因此可以：

```text
依次计算概率参数与量化重建
→ 保存全部 symbols/indexes
→ 最后一次提交6条前缀流+8条尾流
```

这可以把编码端 8 次原生调用合并为 1 次，并让 OpenMP 对所有 14 条流调度。

解码端不能照搬，因为后一阶段的概率参数依赖前一阶段已解出的 latent。解码端只能保持前缀 6 阶段依次执行，尾部保持 2 个批量阶段。

## 6. GPU Tail-rANS 的可行性边界

### 6.1 为什么 Batch 1 风险高

rANS 的状态在单条流内不能对所有 symbol 做普通的无依赖并行。当前 Batch 1 尾部每个阶段只有 4 条独立流，即使每条流对应一个 GPU lane，也远不足以填满 GPU。

要提高 GPU 并行度，通常需要：

- 每组切分多个 tile/substream；
- 或者使用多状态 interleaved rANS；
- 或者增加 batch 与同时码流数。

这些方法都可能改变码流布局，并增加 substream length/header 开销。因此必须报告真实 bitstream BPP，不能只比较解码正确性。

### 6.2 GPU coder 还需要明确的实现问题

原方案在正式实现前需要补充：

1. CDF table、CDF length 和 symbol offset 在 GPU 上的常驻缓冲区；
2. 编码输出的最大安全缓冲区或两遍输出策略；
3. 每条子流的实际 byte length 及布局元数据；
4. bypass symbol 与当前 CPU `Rans64` 语义的一致性；
5. 解码端的边界检查、非法码流处理和确定性；
6. 若不追求字节完全相同，新旧 bitstream 版本的标识与兼容性。

### 6.3 建议的 GPU rANS 启动条件

只有在阶段剖析表明下列任一条件成立时，才建议开发最小 GPU rANS 原型：

- 尾部 `transfer + rANS` 占完整 Enc. 或 Dec. 时间至少 10%；
- 尾部 CPU↔GPU 往返绝对时间至少约 2 ms；
- Batch 8 是明确的部署场景，而不只是为了显示吞吐数据。

如果不满足上述条件，应停止 GPU coder 开发，把资源转向概率网络、数据布局或 Hybrid-V2 的依赖深度优化。

## 7. 修正后的执行顺序

```text
P0  当前 Native-1T 准确阶段剖析
    - CUDA Event 测 GPU 阶段
    - CPU timer 测 rANS
    - 阶段和完整时间误差不超过5%

P1  searchsorted 等价替换
    - 全部前缀和尾部阶段
    - 30张图全量 index/symbol/latent 比较
    - 先确认完整时间收益

P2  编码端14流统一提交
    - 8次原生调用合并为1次
    - symbols/indexes合并传输
    - 比较 Native 1T/4T

P3  Fused symbol/index/packing
    - 只在 P1/P2 仍显示中间 tensor 为瓶颈时开发
    - 优先 Triton 原型，通过后再考虑 C++/CUDA

P4  条件性 GPU Tail-rANS
    - 必须先满足第6.3节启动条件
    - Batch 1与Batch 8分开报告
    - 不得用吞吐改善代替单图延迟结论

P5  若低层收益到顶，转Hybrid-V2
    - [16,48 | 64,64,64,64]
    - 理论依赖深度8→6
```

## 8. Go/No-Go 判定表

| 阶段 | Go 条件 | No-Go 后处理 |
|---|---|---|
| P0 阶段剖析 | 阶段和能解释完整时间，误差≤5% | 修正 CUDA 异步计时，不进入优化 |
| P1 searchsorted | index/symbol/latent 全部一致，完整时间下降≥3% | 恢复原路径，转 P2 |
| P2 14流提交 | bitstream BPP 与 latent 一致，Enc. 下降≥3% | 保留当前稳定实现 |
| P3 fused packing | 局部阶段下降≥30%，完整时间下降≥5% | 不扩大融合范围 |
| P4 GPU Tail-rANS | Batch 1 完整 Enc. 或 Dec. 下降≥5%且绝对下降≥2 ms | 停止 GPU coder，转 P5 |
| P5 Hybrid-V2 | 完整 codec 明显下降，RD 代价可接受 | 保留当前 Hybrid 主版本 |

## 9. 对原方案的最终评价

`FUSED_CDF_GPU_RANS_FEASIBILITY_PLAN.md` 的实验原则是正确的：固定 checkpoint 和图像、分阶段计时、严格正确性检查、设置停止条件，并且不用理论并行性代替完整 codec 实测。

但结合真实代码后，需要做三项关键修正：

1. 将“融合 CDF”修正为“融合 symbol、scale-index 与码流布局”；
2. 在自定义 CUDA/Triton 和 GPU rANS 之前，先做 `searchsorted` 和编码端 14 流统一提交；
3. GPU rANS 必须由阶段占比触发，不应作为默认必做项。

因此，该方向值得继续，但应先使用低风险等价替换和真实阶段数据筛选优化项，避免在仅占零点几毫秒的 rANS 内核上过早投入大量 GPU coder 开发工作。
