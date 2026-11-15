# 并行熵模型创新点验证与论文实验推进方案

## 1. 实验目标

当前论文需要分别证明两类贡献：

1. **结构贡献**：前期非均匀串行分组与后期共享前缀并行分组，是否缩短真实解码依赖路径，并保持可接受的率失真性能。
2. **执行贡献**：多流原生熵编码后端是否把结构并行性转化为可复现的完整编解码加速。

实验必须避免把 `searchsorted`、C++ rANS、OpenMP 调度等底层收益全部归因于并行熵模型。结构与后端使用独立消融验证。

## 2. 当前已经完成的结果

当前优化路径包括：

```text
Searchsorted
+ 编码端14流统一提交
+ 直接反向rANS
+ OpenMP动态任务调度
```

Landsat8、`lambda=16384`、5张预热和30张正式图片的最新结果：

| Method | Enc. mean | Dec. mean | BPP | PSNR | MS-SSIM(dB) |
|---|---:|---:|---:|---:|---:|
| Initial Native 1-worker | 37.86 ms | 35.63 ms | 0.551530 | 45.3345 | 23.0796 |
| Optimized Native 1-worker | 25.194 ms | 26.107 ms | 0.551530 | 45.3345 | 23.0796 |
| Optimized Native 4-worker | **24.200 ms** | **25.759 ms** | 0.551530 | 45.3345 | 23.0796 |

已有正确性证据：

- 码流字节级一致；
- 编码端与解码端 latent 逐元素一致；
- BPP、PSNR和MS-SSIM保持一致；
- 随机极值与 bypass symbol 测试通过；
- 编码阶段计时残差约0.58%，低于5%。

已有停止结论：

- Fused symbol packing 完整编码收益约1.21%，不进入默认路径；
- GPU→CPU传输约0.52 ms，暂不开发GPU Tail-rANS；
- 不继续通过增加Python/OpenMP线程追求主要收益。

## 3. 必须完成的核心结构消融

### 3.1 模型设置

所有模型使用相同的Encoder、Decoder、Hyperprior、训练数据、`lambda`、训练预算和优化后端。

| ID | Channel grouping | Tail dependency | Purpose |
|---|---|---|---|
| S0 | `[16,16,32,64,192]` | 五组完整串行 | 旧版串行基线 |
| S1 | `[16,16,32,64,64,64,64]` | 七组依次串行 | 排除仅重新分组的影响 |
| S2 | `[16,16,32 | 64,64,64,64]` | 四个尾组共享前缀、相互独立 | 当前并行模型 |

理论Anchor/Non-anchor依赖阶段：

| Model | Sequential stages |
|---|---:|
| S0 | 10 |
| S1 | 14 |
| S2 | 8 |

### 3.2 最小训练量

核心消融只使用：

- Dataset：Landsat8；
- `lambda=16384`；
- 一个随机种子；
- 每个新增结构训练一次。

训练安排：

| Model | New training required |
|---|---:|
| S0 | 0，优先复用相同训练配置的旧版checkpoint |
| S1 | **1次** |
| S2 | 0，复用当前Hybrid checkpoint |

只有当S0与S2的训练数据、`lambda`或训练预算无法对齐时，才补训S0。六个码率点属于最终主实验，不属于结构消融。

### 3.3 验证问题

- S0 vs. S1：将192通道拆成四个64通道组本身对RD和时间的影响。
- S1 vs. S2：取消尾组间依赖、改为共享前缀后带来的真实加速与RD变化。
- S0 vs. S2：当前完整并行结构相对旧版模型的最终收益。

### 3.4 必报指标

- 实际码流BPP；
- PSNR；
- MS-SSIM与MS-SSIM(dB)；
- 完整`compress()`时间；
- 完整`decompress()`时间；
- Parameters、FLOPs和GPU峰值显存；
- 依赖阶段数与独立码流数量；
- 编码端与解码端latent精确一致性。

### 3.5 Go/No-Go标准

S2满足以下条件后，再扩展更多码率点：

- 相对S0或S1的完整解码时间稳定下降至少5%；
- 完整编码时间无明显退化；
- BPP变化处于可接受范围；
- PSNR与MS-SSIM无明显下降；
- latent逐元素精确恢复。

如果单点通过，可补充`lambda={4096,16384,65536}`三个代表点。无需为消融直接训练全部六个码率点。

## 4. 可选模块消融

优先验证Tail Group Embedding：

| ID | Configuration | Training |
|---|---|---:|
| M0 | S2 without Tail Group Embedding | 1次，可选 |
| M1 | S2 with Tail Group Embedding | 复用当前checkpoint |

只使用Landsat8、`lambda=16384`。关注BPP、PSNR、MS-SSIM、各尾组平均比特数、参数量与完整时间。

四个独立尾组预测器与共享预测器的对比暂不作为必做项。若页面或训练资源允许，再增加一个代表码率点；否则通过共享结构、参数量和FLOPs说明。

因此：

- 核心消融最少只新增训练S1，共 **1次**；
- 加入Group Embedding消融后共 **2次**。

## 5. 无需重新训练的后端消融

固定同一个S2 checkpoint，只切换执行开关：

| ID | Execution path | Validation target |
|---|---|---|
| E0 | Python serial entropy coding | 原始执行基线 |
| E1 | Initial Native path | 原生接口收益 |
| E2 | E1 + Searchsorted | Scale-index优化 |
| E3 | E2 + deferred 14-stream submission | 批量提交收益 |
| E4 | E3 + direct reverse rANS | rANS数据结构优化 |
| E5 | E4 + dynamic 4-worker scheduling | 多流线程并行收益 |
| E6 | E5 + fused packing | 负收益消融，说明不采用原因 |

必报指标：

- Enc./Dec. mean、median、standard deviation和P90；
- symbol/index准备时间；
- GPU→CPU传输与同步时间；
- Native rANS时间；
- 完整编解码时间；
- BPP、bitstream exact与latent exact。

解释边界：

- E2属于通用底层优化；
- E3开始利用当前多流码流布局；
- E4属于原生熵编码实现优化；
- E5表示线程级并行的额外收益；
- 不得把E0到E5的全部收益都写成结构并行收益。

## 6. 分组合理性分析

固定现有checkpoint，无需重新训练。对每个尾组计算：

\[
\Delta R_i=
-\log_2p_{\mathrm{shared}}(\hat y_i)
+\log_2p_{\mathrm{full}}(\hat y_i).
\]

其中：

- `full`使用完整前序尾组条件；
- `shared`只使用前三组共享前缀。

报告：

- 每组平均比特数；
- 去除前一尾组条件后的码率增量；
- 前缀组和尾组的能量或方差；
- 四个尾组的码流长度、rANS时间和负载不均衡程度。

该实验用于说明：前期组提供主要条件信息，而尾组之间的新增条件贡献相对较弱，因此可以共享公共前缀并独立编码。

## 7. 统一速度测试协议

- Dataset：Landsat8与Sentinel-2，正文不再使用`RS`名称；
- 相同CPU、GPU、checkpoint和图片顺序；
- Batch 1；
- 5张预热、30张正式计时图片；
- `compress()`和`decompress()`边界前后同步CUDA；
- 正序和反序交叉执行；
- 最终结果至少重复3次；
- 报告mean、median、standard deviation和P90；
- 使用真实码流时间，不用forward时间代替；
- Batch 8只能作为吞吐量补充，不得替代单图延迟结论。

## 8. 最终主实验

只有核心结构消融通过后，才完成S2在两个数据集上的六个码率点：

\[
\lambda=\{1024,2048,4096,8192,16384,65536\}.
\]

最终报告：

- PSNR RD曲线；
- MS-SSIM RD曲线；
- PSNR和MS-SSIM BD-Rate；
- 实际码流BPP；
- 完整编解码时间；
- 参数量、FLOPs和显存；
- Landsat8与Sentinel-2可视化结果。

HPCM不重新加入论文图、表和正文对比。仓库中的HPCM复测结果只用于内部判断。

## 9. 论文实验章节顺序

1. Datasets and training configurations；
2. Compared methods；
3. Rate-distortion performance；
4. Coding time and complexity；
5. Visual quality；
6. Structural ablation：S0/S1/S2；
7. Module ablation：Group Embedding；
8. Backend ablation：E0–E6；
9. Stage-wise profiling and dependency analysis。

论文中的并行贡献应表述为：

> 前期非均匀串行分组保留主要条件信息，后期均匀尾组共享同一前缀并形成独立熵码流；与该依赖结构匹配的原生多流后端将结构并行性转化为可复现的完整编解码收益。

不得表述为四线程线性加速，也不得把全部底层优化收益归因于尾组条件解耦。

## 10. 当前执行顺序

```text
1. 固定并复测E0–E6后端结果
        ↓
2. 复用S0与S2，新增训练一次S1
        ↓
3. 完成S0/S1/S2结构消融
        ↓
4. 完成无需训练的组间条件贡献分析
        ↓
5. 若结构消融通过，选择性训练一次M0
        ↓
6. 补充低/中/高三个lambda点
        ↓
7. 完成两个数据集的六码率最终主实验
        ↓
8. 将最终数值写入论文
```

当前最高优先级是S0/S1/S2相同后端下的结构消融；它决定并行熵模型能否被严格认定为模型创新，而不是单纯的实现加速。
