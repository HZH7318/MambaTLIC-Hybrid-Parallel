# 融合 CDF 与 GPU 常驻尾组 rANS 可行性实验

## 1. 目标

在不重新训练、不改变模型权重和概率模型的前提下，分别验证以下两项底层优化是否能降低完整编解码时间：

1. **Fused-CDF**：用一个融合算子完成尾部四组的 symbol、CDF index、offset 和输出布局生成。
2. **GPU Tail-rANS**：让尾部四条独立码流在 GPU 上完成熵编码与解码，减少 GPU/CPU 往返。

本阶段只判断可行性，不直接替换当前稳定实现。

## 2. 固定测试条件

- 模型：当前 Hybrid Parallel 600-epoch checkpoint。
- 数据集：Landsat8。
- 码率点：`lambda=16384`。
- 图像：沿用当前固定的 30 张真实码流测试图。
- 输入：batch 1，原始测试尺寸；批量吞吐实验额外使用 batch 8。
- 预热：5 次。
- 统计：30 张图逐图计时，同时报告 mean、median、P90。
- 基线：当前 `Native 1-worker`，现有参考值约为 Enc. 37.86 ms、Dec. 35.63 ms。
- 固定随机种子、设备、线程数和 checkpoint；测试期间不得更换模型权重。
- GPU 阶段使用 CUDA Event；CPU/rANS 阶段使用高精度 CPU timer。只在明确的阶段边界同步，避免计时遗漏异步计算。

## 3. 先做阶段剖析

对当前 Native 1-worker 分别记录：

1. transform/hyperprior；
2. 前缀组概率预测；
3. 尾组概率预测；
4. symbol/CDF index/offset 准备；
5. GPU→CPU 与 CPU→GPU 传输及同步；
6. 纯 rANS；
7. synthesis transform；
8. 完整 `compress()` 与 `decompress()`。

所有阶段时间之和应能解释完整时间；若误差超过 5%，检查异步计时和隐式同步。

## 4. 实验 A：Fused-CDF

### A0：当前实现

- 当前 PyTorch 多算子路径生成 symbol、CDF index 和 offset。
- 当前 Native 1-worker rANS。

### A1：融合实现

仅修改四个并行尾组：

\[
(y,\mu,\sigma,\mathrm{mask},\mathrm{group\_id})
\rightarrow
(\mathrm{symbols},\mathrm{cdf\_indexes},\mathrm{offsets},\mathrm{stream\_positions})
\]

要求：

- 一个融合 CUDA/Triton/C++ 算子完成 clamp、scale-table 查找、symbol 量化、mask/gather 和连续输出布局；
- 不生成不必要的完整中间张量；
- 前缀组和 Native 1-worker rANS 保持不变；
- A0 与 A1 使用完全相同的 CDF 表、取整规则和码流顺序。

### 正确性检查

对全部测试图逐元素比较：

- symbols；
- CDF indexes；
- offsets；
- 解码 latent；
- 重建图像。

理想情况下 A0 与 A1 码流字节完全一致；若码流布局未改变，这是强制要求。

### 可行性判定

满足以下条件则进入实验 B：

- latent 与重建结果逐元素一致；
- BPP、PSNR、MS-SSIM 完全一致；
- symbol/CDF 准备加传输时间至少下降 30%；
- 完整编码或解码时间下降约 10% 或以上。

如果阶段时间下降明显但完整时间下降不足 5%，记录结果并停止继续扩大融合范围，重新检查真正瓶颈。

## 5. 实验 B：GPU Tail-rANS

实验 B 必须建立在正确的 A1 上。串行前缀继续使用当前 CPU Native rANS，只替换四个尾组。

### B0：CPU 尾组基线

- Fused-CDF；
- 尾组使用 CPU Native 1-worker rANS。

### B1：GPU 尾组 rANS

- Fused-CDF 输出直接保留在 GPU；
- 四个尾组分别维护独立的 rANS state/stream；
- GPU 并行完成四条尾组码流的编码与解码；
- 编码端仅将最终码流传回 CPU；
- 解码端码流一次送入 GPU，尾组 latent 留在 GPU，直接进入后续重建；
- 前缀依赖顺序不得改变。

第一版只做四条尾组码流，不继续切分空间 tile。只有 GPU 利用率不足时，再测试每组 2/4 个 tile 的多 lane 版本。

### 测试场景

1. **Batch 1**：关注单图 Enc./Dec. latency。
2. **Batch 8**：关注 images/s 和平均单图时间，判断较大任务量是否能摊薄 GPU 启动成本。

### 正确性检查

- 解码 latent 必须逐元素一致；
- 重建图像必须逐元素一致；
- PSNR 和 MS-SSIM 必须一致；
- 报告每条子流大小、码流头开销和总 BPP；
- 如果 GPU rANS 改变码流排列，可以不要求字节相同，但总符号序列必须无损恢复。

### 可行性判定

满足任意一项即可认为 GPU Tail-rANS 值得继续：

- Batch 1 完整编码或解码时间下降 10% 以上；
- Batch 8 吞吐量提高 1.5 倍以上；
- 尾组熵编码阶段提高 2 倍以上，且完整时间存在可测下降。

如果纯 GPU rANS 明显更快而完整时间没有下降，结论应写为“rANS 已非主要瓶颈”，停止扩展 GPU coder，转向概率预测、同步或数据布局优化。

## 6. 必须输出的结果

### 6.1 汇总 CSV

建议字段：

```text
method,batch_size,image_count,
enc_mean_ms,enc_median_ms,enc_p90_ms,
dec_mean_ms,dec_median_ms,dec_p90_ms,
prob_prefix_ms,prob_tail_ms,cdf_prepare_ms,
transfer_sync_ms,rans_encode_ms,rans_decode_ms,
throughput_img_s,bpp,psnr,msssim_db,
symbol_exact,cdf_index_exact,latent_exact,recon_exact,
peak_gpu_mem_mib,notes
```

方法至少包含：

- `native_1w_current`
- `fused_cdf_cpu_rans`
- `fused_cdf_gpu_tail_rans`

### 6.2 汇总表

| Method | Batch | CDF Prepare | Transfer/Sync | rANS Enc. | rANS Dec. | Full Enc. | Full Dec. | img/s | BPP | Exact |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|:---:|
| Native-1T | 1 | | | | | | | | | |
| Fused-CDF | 1 | | | | | | | | | |
| Fused-CDF + GPU Tail-rANS | 1 | | | | | | | | | |
| Fused-CDF + GPU Tail-rANS | 8 | | | | | | | | | |

### 6.3 简短结论

结论必须分别回答：

1. Fused-CDF 是否减少了 CDF 准备、数据传输和完整延迟？
2. GPU Tail-rANS 是否改善单图延迟，还是只改善批量吞吐？
3. 当前完整编解码的主要瓶颈最终位于概率网络、数据通路还是 rANS？
4. 两个方案是否值得进入完整 Landsat8 与 Sentinel-2 多码率实验？

## 7. 执行顺序

```text
当前 Native-1T 阶段剖析
        ↓
实现并验证 Fused-CDF
        ↓
A1 通过正确性与速度门槛
        ↓
实现四流 GPU Tail-rANS
        ↓
Batch 1 / Batch 8 对比
        ↓
决定继续优化或停止
```

不得同时实现 A1 和 B1 后只给出一个总时间，否则无法区分收益来源。
