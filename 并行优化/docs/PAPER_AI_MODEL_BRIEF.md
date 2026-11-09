# 论文撰写事实简报：Hybrid-Parallel Mamba 熵模型

> 用途：本文档可直接交给论文 AI。其中“已验证结果”和“待完成实验”必须严格区分，不得把理论并行深度写成已验证的实际加速比。

## 1. 研究问题

当前工作以 MambaTLIC/现有 Mamba 学习式遥感图像压缩模型为基线，研究如何降低自回归通道上下文导致的编解码串行延迟。研究重点是真实 `compress()`/`decompress()` 流程，而不只是训练阶段的并行 forward。

方法不删除 latent，不做 token pruning，不引入教师模型或知识蒸馏，不改变分辨率。模型使用原有的率失真目标端到端训练。

## 2. 基线模型

基线保留以下主体：

- Mamba analysis transform/encoder；
- Mamba synthesis transform/decoder；
- HyperEncoder 和 HyperDecoder；
- EntropyBottleneck 用于超先验 latent `z`；
- GaussianConditional 用于主 latent `y`；
- checkerboard Anchor → Non-anchor 空间上下文。

主 latent 通道数为 `M=320`，原始通道分组为：

```text
[16, 16, 32, 64, 192]
```

五个通道组依次使用已解码的前缀通道作为条件，每个组内都需要 Anchor 和 Non-anchor 两个依赖阶段。因此理论串行依赖深度为：

```text
5 groups × 2 checkerboard stages = 10 stages
```

## 3. Hybrid-Parallel 熵模型

### 3.1 混合通道分组

新模型把 320 个通道重新分为：

```text
[16, 16, 32 | 64, 64, 64, 64]
```

竖线左侧是三个不均匀串行前缀组，右侧是四个均匀并行尾部组。前缀共 64 通道，尾部共 256 通道。

### 3.2 串行前缀

前三组保留精确通道依赖：

```text
P1: Hyperprior → Anchor → Non-anchor
P2: Hyperprior + P1 → Anchor → Non-anchor
P3: Hyperprior + P1 + P2 → Anchor → Non-anchor
```

这一设计用少量串行计算保留先解码通道的高质量概率估计。完整重建的 64 通道前缀经 `tail_prefix_transform` 生成一个共享 `PrefixContext`。

### 3.3 条件独立的并行尾部

四个尾部组 `T1–T4` 之间不再相互依赖。它们共同依赖：

```text
Hyperprior + shared PrefixContext + Tail Group Embedding
```

对于每个尾部组 `Tk`：

```text
Tk Anchor     <- Hyperprior + PrefixContext + GroupEmbedding(k)
Tk Non-anchor <- Hyperprior + PrefixContext + GroupEmbedding(k) + Tk Anchor
```

`Tk Non-anchor` 只依赖本组 Anchor，不依赖其他尾部组。因此四个 Anchor 可以在同一阶段中计算，四个 Non-anchor 可以在第二阶段中计算。

理论串行依赖深度由 10 降为：

```text
3 prefix groups × 2 stages + 2 shared tail stages = 8 stages
```

注意：`10 → 8` 是依赖图分析，不等于 1.25倍或 4倍实际加速。

### 3.4 共享预测器与组标识

四个尾部组共享同一套 checkerboard context model 和概率参数预测网络。为避免四组被当作完全相同的分布，引入可学习的 `Tail Group Embedding`，当前 embedding 维数为 16。

训练和概率参数预测时，四个尾部 tensor 沿 batch 维堆叠，因而四组 Anchor 只需要一次共享网络调用，四组 Non-anchor 也只需要一次共享网络调用。

### 3.5 真实并行熵码流

方法不只实现 forward 并行。每个尾部组拥有独立的 Anchor 码流和 Non-anchor 码流，因此尾部共有 8 条可独立编解码的 y 码流。

原生后端使用 C++/OpenMP 实现 batched multi-stream rANS：

- 一次 Python/C++ 边界调用提交多条码流；
- 后端释放 Python GIL；
- 多条独立 rANS 任务可用 OpenMP 工作线程执行；
- CDF 表在模型内缓存，避免重复转换；
- 解码端按相同的两阶段依赖恢复 Anchor 和 Non-anchor。

这使得方法是“码流级真实可并行”，而不是仅在 GPU 上同时预测概率参数。

## 4. 训练目标和数据

训练使用标准率失真目标：

```text
L = Rz + Rprefix + Rtail + lambda * D
```

Landsat8 实验设置：

- 7 波段、16-bit 数据；
- 像素归一化为 `x / 65535.0`；
- 1000 张训练图像，200 张训练验证图像；
- 最终全量测试集为 1700 张；
- `N=192, M=320, base_channel=7`；
- Hybrid 模型参数量约 20.07M；
- lambda 取值为 1024、2048、4096、8192、16384、32768、65536。

已完成的第一阶段为 400 epochs。当前正在从可恢复的 epoch 395 续训至 600 epochs，最后训练 epoch 为 599。最终论文数据应使用 600-epoch 流水线完成后的重测结果。

## 5. 可以作为论文贡献的创新点

### 创新点 1：不均匀串行前缀与均匀并行尾部的混合熵模型

方法不是将所有通道粗暴地设为独立组。它在前缀保留通道自回归建模，利用小通道组逐步建立信息丰富的共享上下文；对占大部分通道的尾部改用条件独立建模，以换取并行性。其本质是率失真性能与解码依赖深度之间的结构化折中。

### 创新点 2：由共享前缀条件化的尾部并行分支

四个尾部组都由同一个 `PrefixContext` 条件化，取消尾部组间依赖，同时保留组内 Anchor → Non-anchor 的空间建模。与完全去除上下文的并行方案相比，该设计仍然使用前缀通道和组内空间信息。

### 创新点 3：通过 Group Embedding 实现的组感知共享预测器

尾部概率网络共享权重，用于批量计算和控制参数量；同时利用可学习组嵌入区分不同通道组的统计特性。这是“权重共享”与“分布个性化”之间的折中。

### 创新点 4：从概率图到真实码流的端到端并行实现

训练图中的 batch-stacked 并行和真实编解码中的独立码流具有一致的依赖关系。通过原生 batched rANS/OpenMP 后端，方法解决了 Python GIL 和多次跨语言调用带来的额外开销。这一点应作为“模型结构—码流组织—执行后端”协同设计来描述。

## 6. 当前已验证的中间结果

### 6.1 400-epoch 全量 RD 结果（将被 600-epoch 结果替换）

Landsat8 Hybrid 在 1700 张完整测试图像上的范围：

```text
BPP:        0.0689 -- 1.1460
PSNR:       39.702 -- 48.121 dB
MS-SSIM:    16.082 -- 26.590 dB
```

此处 BPP 由 likelihood 估计，用于与其他方法的全量 RD 协议对齐；不是实际码流字节数。与原始串行 Mamba 相比，当前 400-epoch Hybrid 的 PSNR BD-Rate 约为 `+24.41%`，说明并行化付出了明显的 RD 代价。这个数字不应被隐藏，但应等待 600-epoch 全量重测后再用于最终结论。

### 6.2 真实 codec 延迟先导实验

当前 Landsat8 `lambda=16384` 上的先导试验使用 1 张预热和 5 张正式计时图像。时间包含完整 `compress()` 和 `decompress()` 调用，计时前后执行 CUDA synchronize。

| 运行方式 | 压缩时间 | 解压时间 | 真实 BPP | PSNR | MS-SSIM(dB) |
|---|---:|---:|---:|---:|---:|
| Hybrid Python serial entropy | 0.11065 s | 0.11885 s | 0.80430 | 41.894 | 21.085 |
| Hybrid native rANS, 1 worker | 0.06639 s | 0.06621 s | 0.80430 | 41.894 | 21.085 |
| Hybrid native rANS, 4 workers | 0.06609 s | 0.06607 s | 0.80430 | 41.894 | 21.085 |
| Original serial Mamba | 0.18803 s | 0.12630 s | 0.72617 | 42.924 | 21.852 |
| HPCM | 0.02499 s | 0.02348 s | 0.81231 | 42.583 | 21.640 |

目前可得出的事实：

- Native 4-worker 相对 Hybrid Python 熵编码完整路径，压缩约 `1.67x`，解压约 `1.80x`；
- Native 4-worker 与 Native 1-worker 的完整时间几乎相同，仅约 `1.00x`；
- Hybrid Native 4-worker 相对原始 Mamba 的完整压缩/解压时间分别约为 `2.85x/1.91x`，但两者 RD 点不一致，不能将此写成同等质量下的严格加速比；
- 已验证 Hybrid Python、Native 1-worker 和 Native 4-worker 三者的 BPP、PSNR 和 MS-SSIM 完全一致；
- 已通过编码端/解码端量化 latent 逐元素一致性检查；
- HPCM 当前仍明显更快，因此暂不能声称整体编解码速度达到 SOTA。

4-worker 没有继续快于 1-worker，说明当前主要收益来自 batched native API、减少 Python/C++ 边界开销和数据整理，而不是四条 rANS 线程本身带来的线性加速。小码流粒度、CPU 线程调度、GPU 概率参数预测、CPU/GPU 数据转换和前后处理都会限制并行收益。

## 7. 论文中应该如何表述

推荐的核心表述：

> We propose a hybrid serial–parallel entropy model for learned remote-sensing image compression. The model preserves accurate autoregressive modeling in a compact non-uniform channel prefix, while factorizing the remaining channels into uniformly sized tail groups that share the same decoded prefix and can be processed in two group-parallel checkerboard stages. A group-aware shared predictor and independent tail bitstreams connect the parallel probability model to a native batched multi-stream rANS implementation, enabling end-to-end parallelizable entropy coding without pruning latent representations or introducing a teacher model.

对中文论文可表述为：

> 本文提出一种面向学习式遥感图像压缩的混合串并行熵模型。该模型在紧凑的不均匀通道前缀中保留精确的通道自回归建模，并将剩余通道拆分为依赖同一已解码前缀的均匀尾部组，从而将尾部概率参数预测归约为两个组并行 checkerboard 阶段。通过组感知共享预测器、独立尾部码流和原生批量多码流 rANS 后端，方法将概率模型中的并行性延伸到真实编解码流程，且无需删除 latent 或引入教师模型。

## 8. 不得夸大的结论

论文 AI 必须遵守以下限制：

1. 不得把依赖深度 `10 → 8` 直接写成 20%、1.25倍或 4倍实际加速。
2. 不得宣称 OpenMP 4-worker 已获得四倍熵编码加速；当前 4-worker 与 1-worker 几乎相同。
3. 不得把 likelihood BPP 与真实 bitstream BPP 混在同一个数值协议中。
4. 不得在 RD 点不匹配的情况下，宣称与基线的速度比是“同等质量下”的加速。
5. 不得宣称整体速度 SOTA；当前 HPCM 在先导测试中更快。
6. 400-epoch 结果是中间数据；最终表格和摘要必须等待 600-epoch 续训、1700 张全量 RD 重测和 30 张真实 codec 重测完成。

## 9. 论文实验表格建议

最终主表至少包含：

| Method | BPP | PSNR | MS-SSIM | Compress time | Decompress time | Encode speedup | Decode speedup | Real bitstream |
|---|---:|---:|---:|---:|---:|---:|---:|---|

建议消融项：

1. Original serial Mamba：`[16,16,32,64,192]`；
2. Uniform regrouping but serial tail：`[16,16,32,64,64,64,64]`，尾部仍逐组依赖；
3. Hybrid dependency graph + Python serial entropy coder；
4. Hybrid + native batched rANS, 1 worker；
5. Hybrid + native batched rANS, 4 workers。

当前已有第 1、3、4、5 项的主要实现和先导时间数据；第 2 项“只重分组但仍串行”尚未形成完整的最终消融实验，不应虚构其结果。

## 10. 最终数据位置

600-epoch 流水线完成后，论文 AI 应优先读取：

```text
/data/huzhenghao/ml/h/并行优化/final_results/landsat8_hybrid_600/
```

其中：

- `landsat8_full_rd.csv`：1700 张全量 RD 结果；
- `landsat8_full_test_rd.png`：最终 BPP–PSNR 和 BPP–MS-SSIM 曲线；
- `landsat8_hybrid_codec_30_images.csv`：真实编解码时间和真实码流 BPP；
- `result_manifest.json`：测试协议和数据来源。
