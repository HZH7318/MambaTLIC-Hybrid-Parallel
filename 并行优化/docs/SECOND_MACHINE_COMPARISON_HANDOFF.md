# 第二台机器对比实验交接说明

## 给第二台机器 Codex 的任务

> 请先完整阅读本文档。你要把 server2 上的 Landsat8 对比实验迁移到当前机器，保持 `/data/huzhenghao/ml/h` 和 `/data/huzhenghao/ml/新方法` 路径结构。先拉取代码、按 manifest 拉取精简数据集、同步 latest/best 断点，然后执行环境和断点完整性检查。检查通过后，按文档分配启动两张 GPU。不要训练 Landsat8 `lambda=131072`。启动前报告实际拉取到的 checkpoint epoch/loss和两张 GPU 状态；启动后再核对两个 tmux 会话的显存占用和当前 lambda。

## 1. 目标和路径约定

第二台机器接手 Landsat8 对比实验，使用两张与 server2 规格相同的 GPU。两台机器保持相同的绝对路径：

```text
/data/huzhenghao/ml/h
/data/huzhenghao/ml/新方法
```

`h` 是 Git 仓库，远程为：

```text
https://github.com/HZH7318/MambaTLIC-Hybrid-Parallel.git
```

`新方法` 是本地对比实验工作区，它不是独立 Git 仓库，需要从 server2 用 rsync 同步。`h/并行优化/vendor/rwk_compressai126` 被 Git 忽略，GLIC 依赖它，迁移脚本会单独同步。

## 2. 数据协议

| 数据集 | train/val/test | 通道 | 归一化 | 精简数据体积 |
|---|---:|---:|---:|---:|
| Landsat8 | 1000/200/200 | 7 | 65535 | 约 1.20 GB |
| RS | 1000/200/200 | 10 | 32767 | 约 1.84 GB |
| FMoW | 1000/300/200 | 8 | 4096 | 约 1.47 GB |

不使用 valid mask。Landsat8 本轮只用 `1024, 2048, 4096, 8192, 16384, 32768, 65536`，不训练 `131072`。RD 测试用 likelihood BPP，codec 时间测试才用实际码流字节。

manifest 内已经是与目标机器一致的绝对路径。迁移脚本会按 manifest 精确复制文件，不复制原始全量数据。

## 3. server2 当前状态和已做改动

动态状态以下面文件为准：

```text
/data/huzhenghao/ml/新方法/experiments/pipelines/landsat8_chwdta_glic_100/status.json
```

2026-11-14 15:23 的交接快照：

- ChWDTA 七个 lambda 已全部完成 50 epochs。
- GLIC `1024, 2048, 4096, 8192, 16384` 已完成 100 epochs。
- GLIC 正在训练 `32768`，之后还有 `65536`。
- GLIC `16384` 的 latest 在第 100 轮发散，final loss 为 `811.8885`；best 在 epoch 80，loss 为 `1.4543`。最终 RD 评估使用 best，latest 只用于训练边界和追溯。

已做的训练工程修改：

- GLIC 每个 epoch 更新续训断点，训练结束强制验证并保存精确的最终 checkpoint。
- ChWDTA 续训保留历史 best loss，不再在断点后重置。
- ChWDTA 续训时先在 CPU 读取 latest，恢复后立即释放 checkpoint 对象，避免第一个 batch 因 2.27 GB 断点重复占用 GPU 显存。
- ChWDTA 默认关闭每 5 轮的 2.27 GB 编号权重，保留 `latest` 和 `best`。这会避免第二台机器再产生约 160 GB 重复中间权重。
- 流水线新增 `--phases`、`--lambdas` 和 `--pipeline-name`，可以在两张 GPU 上安全分片。

## 4. 迁移量

`新方法` 源目录约 214 GB，其中大部分是不需要的历史权重。实际迁移内容：

- 对比方法代码、协议、manifest、脚本：约 32 MB。
- GLIC/ChWDTA 的 `latest + best`：约 44 GB。
- Landsat8 精简数据：约 1.20 GB。
- 若拉取三个数据集：共约 4.51 GB。

## 5. 目标机器的执行顺序

将 `SOURCE_SSH` 替换为第二台机器能访问 server2 的 SSH 名称，通常是 `huzhenghao@server2`。

### 5.1 先拉取交接包

```bash
mkdir -p /data/huzhenghao/ml/h/并行优化/second_machine
rsync -a -s huzhenghao@server2:/data/huzhenghao/ml/h/并行优化/second_machine/ \
  /data/huzhenghao/ml/h/并行优化/second_machine/
mkdir -p /data/huzhenghao/ml/h/并行优化/docs
rsync -a -s huzhenghao@server2:/data/huzhenghao/ml/h/并行优化/docs/SECOND_MACHINE_COMPARISON_HANDOFF.md \
  /data/huzhenghao/ml/h/并行优化/docs/
```

### 5.2 同步代码、Landsat8 数据和必要断点

```bash
bash /data/huzhenghao/ml/h/并行优化/second_machine/pull_from_server2.sh \
  huzhenghao@server2 landsat8
```

若需要三个精简数据集，将最后一个参数改为 `all`。如果目标机器已有 `h`，脚本只执行 `git fetch`，不会覆盖它现有的本地改动。只有显式设置 `UPDATE_H=1` 才执行 fast-forward pull。

### 5.3 环境和完整性检查

```bash
bash /data/huzhenghao/ml/h/并行优化/second_machine/setup_second_machine_env.sh

/data/huzhenghao/ml/新方法/.venvs/chwdta/bin/python \
  /data/huzhenghao/ml/h/并行优化/second_machine/verify_second_machine_assets.py
```

环境基准为 Python 3.9.20、PyTorch 2.4.1+cu121、torchvision 0.19.1+cu121、CompressAI 1.1.5。GLIC 另外使用已审计的 vendor CompressAI 快照。环境检查不会自动更换目标机器的 CUDA/PyTorch。

### 5.4 启动两张 GPU

```bash
bash /data/huzhenghao/ml/h/并行优化/second_machine/start_second_machine_jobs.sh
```

分配如下：

| GPU | 任务 |
|---|---|
| GPU 0 | 先补齐所有未完成的 GLIC 100 轮，再训 ChWDTA `1024/2048/4096` 到 100 轮 |
| GPU 1 | ChWDTA `8192/16384/32768/65536` 从 50 续训到 100 轮 |

已达到目标的 checkpoint 会自动跳过。两张卡不会同时写入同一 lambda 目录。

tmux 会话：

```text
second_glic_then_chwdta_low
second_chwdta_high
```

## 6. 切换时的注意事项

checkpoint 文件很大，不要在 server2 正在覆写某个 `checkpoint_latest.pth.tar` 时复制该文件。推荐切换步骤：

1. 让 server2 当前 epoch 结束并停止原流水线。
2. 在目标机器执行同步脚本。
3. 执行完整性检查，确保 `torch.load` 能读取每个断点。
4. 启动第二台机器的两个 tmux 任务。

不要同时在两台机器续训同一个 method/lambda，否则后续很难判定哪条训练轨迹是正式结果。

## 7. 给第二台机器 Codex 的验收要求

启动前报告：

- 两张 GPU 型号、显存和空闲状态。
- Landsat8 manifest 是否完整为 1000/200/200。
- 七个 ChWDTA latest 是否都为 epoch 49。
- GLIC 每个 latest/best 的 epoch、loss 和是否完成 100 轮。
- 两个 tmux 会话是否分别占用 GPU 0/1。

完成后保留 `latest`、`best`、训练日志和 pipeline `status.json`，再同步回 server2 统一做 RD 测试和绘图。
