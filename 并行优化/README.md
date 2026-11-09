# 混合并行熵模型

本目录是仓库的主实验区，实现“三个不均匀串行前缀组 + 四个均匀并行尾部组”。

```text
并行优化/
├── model/                         # Hybrid 模型
├── native_parallel_rans/          # C++/OpenMP 多码流 rANS
├── config/                        # Landsat8/RS 配置
├── docs/                          # 方法、实验和论文说明
├── results/                       # 可提交数据、图和 Notebook
├── final_results/                 # 600 轮流水线最终输出
├── run_mamba_hybrid_parallel.py   # 训练入口
├── run_full_comparison_eval.py    # 全测试集 RD
├── benchmark_codec_time_method.py # 真实 compress/decompress
└── run_landsat8_to_600.py          # 当前收尾流水线
```

## 核心方法

原始分组为 `[16,16,32,64,192]`；Hybrid 分组为
`[16,16,32 | 64,64,64,64]`。前缀保留 Anchor → Non-anchor 串行依赖，
四个尾部组共享 PrefixContext，用 Group Embedding 区分分布，并使用
独立码流进行真实并行编解码。详见
[`docs/PAPER_AI_MODEL_BRIEF.md`](docs/PAPER_AI_MODEL_BRIEF.md)。

## 环境和数据

已验证环境是 `mambah` + CUDA。Git 仓库不包含数据集和权重。配置中的
`dateset_path`/`train_val_path` 需指向本地 `.npy` 数据。像素按 16-bit
`x / 65535.0` 归一化。

安装 Python 依赖后构建原生 rANS：

```bash
pip install -r requirements.txt
cd 并行优化/native_parallel_rans
python setup.py build_ext --inplace
```

## 运行

```bash
# 单个 Landsat8 配置
python 并行优化/run_mamba_hybrid_parallel.py \
  --config 并行优化/config/auto_to_train/16384.ini

# 完整 Landsat8 Hybrid 测试
python 并行优化/run_full_comparison_eval.py landsat8 --device cuda:0

# 真实码流时间
bash 并行优化/run_codec_benchmark.sh \
  --method hybrid_native_4w --dataset landsat8 --lambda-value 16384
```

## 结果

`results/data/full_test_rd_all_methods.csv` 是 Landsat8 和 RS 的全量 RD 表；
`results/data/codec_time_comparison.csv` 是真实编解码先导表。两个数据集的
可执行 Notebook 位于 `results/notebooks/`。

当前正在运行 Landsat8 600-epoch 收尾流水线。仓库中现有 Hybrid RD
为 400-epoch 中间结果；流水线完成后应用 `final_results/landsat8_hybrid_600/`
替换 Landsat8 Hybrid 行。
