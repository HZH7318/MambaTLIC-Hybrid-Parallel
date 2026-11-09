# `/data/huzhenghao/ml/test` 结果审计

## 数据源优先级

1. `summary_all_results.csv`：2026-04-12 生成的逐图完整结果，是当前 RD 主表的可靠来源。
2. `log/<dataset>/`：逐图运行日志，可用于追溯，但体积大，不作为直接绘图入口。
3. `out/<dataset>/`：每个 λ 只有一条汇总，但当前 `test.py` 在第一张图后 `break`，且时间来自 forward 内部分段；只能做冒烟检查。
4. `compression_data.csv` 和 `bd_results.csv`：2026-03-15 的旧聚合，数值与最新逐图 CSV 显著不一致，不再用于论文主表。

## 完整度

- Landsat8：7 个方法，每个方法 7 个 λ，每个点 1700 张图，无重复文件。
- RS：7 个方法；各方法有 6–12 个 λ，每个点 1217 张图，无重复文件。
- HPCM 的 `out/` 文件为空，但其完整逐图结果已存在于 `summary_all_results.csv`。
- 新 hybrid 的训练验证集只有 200 张，且文件名与 `/ml/test/testdata` 完全不重合，因此不能直接与上述方法比较。

## 当前测试脚本的时间问题

`/data/huzhenghao/ml/test/test.py` 调用的是 `net(img_batch)`，没有调用真实 `compress()` / `decompress()`；随后立即 `break`。日志中的 Encode/Decode 是模型 forward 内人为划分的网络计算时间，BPP 是 likelihood 估计值，不是码流字节。因此：

- 可使用完整逐图 CSV 比较 RD；
- 不可使用 `out/` 的 Encode/Decode 宣称编解码加速；
- 真实时间必须使用 CUDA 同步包围完整 codec API，并由实际字节计算 BPP。

## 各方法真实码流能力

| 方法 | 真实 codec | 处理方式 |
|---|---|---|
| ELIC | 支持 | 直接适配 `compress/decompress` |
| RWK | 支持 | 继承 CompressAI `Elic2022Official` codec API |
| MLIC | 支持 | 直接适配 `compress/decompress` |
| HPCM | 支持 | 直接适配自带 rANS 扩展 |
| FTIC | 支持但需适配 | 接口使用文件输出；每张图使用隔离临时目录 |
| HL | 不支持 | 只有 forward 与 likelihood BPP；真实时间和真实码流 BPP 标记 N/A |
| Our/Base Mamba | 支持 | 已由 `benchmark_real_codec.py` 验证 |
| Hybrid | 支持 | 已验证 15 条独立码流及编码/解码 latent 严格一致 |

## 已发现的数据异常

- RS-ELIC checkpoint 中存在 `52488.pth.tar`。该点的 BPP/PSNR 显著高于随后 λ=65536 的点，破坏 λ 单调性，很可能原意是 `524288` 或 checkpoint 被错误命名。正式 BD-Rate 前必须核对来源。
- RS-HL 和 RS-MLIC 在高码率处也有轻微 PSNR 非单调，应以 BPP 排序绘图，并在 BD-Rate 计算前检查曲线重叠区和异常点。
- 用户之前手工绘图的数据、旧 `compression_data.csv` 和最新逐图 CSV 来自不同时间或测试口径，不能混合计算 BD-Rate。

## 新增统一测试

`evaluate_hybrid_full_test.py` 会在与其他方法完全相同的 `/ml/test/testdata` 上评估 hybrid likelihood-RD；`run_full_comparison_eval.py` 负责遍历所有 λ。结果写入：

`并行优化/experiments/full_test_comparison/`

该结果用于完整 RD/BD-Rate；真实压缩时间仍使用：

`并行优化/experiments/real_codec_landsat_rs_30/`
