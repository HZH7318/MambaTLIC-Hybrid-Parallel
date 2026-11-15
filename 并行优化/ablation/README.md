# S0/S1/S2 independent ablation workspace

This directory is isolated from both production model files:

- `model/mamba/mambamodel_1_917.py` remains the historical S0 baseline;
- `并行优化/model/mambamodel_hybrid_parallel.py` remains the S2 model;
- all new S0/S1 training code and outputs stay below `并行优化/ablation/`.

## Structural definitions

| ID | Groups | Dependency |
|---|---|---|
| S0 | `[16,16,32,64,192]` | historical serial first-plus-previous rule |
| S1 | `[16,16,32,64,64,64,64]` | same serial rule, only regrouped |
| S2 | `[16,16,32 | 64,64,64,64]` | shared-prefix independent tail |

The historical implementation does **not** concatenate every preceding group.
For group `i>1` it conditions on group 0 and group `i-1`.  S1 intentionally
keeps this exact rule so S0 versus S1 isolates regrouping rather than silently
changing the context model.

S0 and S1 use the same searchsorted, deferred multi-stream submission, direct
reverse rANS and dynamic OpenMP scheduling as S2 for real codec evaluation.
Their training `forward()` retains the historical serial probability model.

## Why S0 must be retrained

The available historical S0 checkpoint was trained with a different dataset
protocol.  It cannot be used as the final controlled S0 row against the current
1000/200 Landsat8 subset.  Final structural ablation therefore requires S0 and
S1 to be trained with the configs in this directory.  S2 can reuse its current
400-epoch checkpoint because it used the current subset and seed.

## Training

Run only after the current comparison queues release a GPU:

```bash
cd /data/huzhenghao/ml/h

/data/huzhenghao/anaconda3/envs/mambah/bin/python \
  并行优化/ablation/run_train.py --variant s0 \
  --config 并行优化/ablation/config/s0_landsat8_16384.ini

/data/huzhenghao/anaconda3/envs/mambah/bin/python \
  并行优化/ablation/run_train.py --variant s1 \
  --config 并行优化/ablation/config/s1_landsat8_16384.ini
```

The committed configs target 400 epochs for the final controlled comparison.
For a pilot, copy each config outside Git, set `train_max_epoch=150`, and keep
all other fields identical.  Do not compare a 150-epoch S0/S1 checkpoint with
the 400-epoch S2 result as a final paper row.

`run_train.py` automatically finds the newest timestamped checkpoint directory
under the selected variant and resumes from the following epoch.  The first
run starts from scratch; no INI editing is required after an interruption.

## Required checks before long training

1. Construct S0 and strictly load a historical S0 state dict to verify key and
   tensor-shape compatibility.
2. Run one real forward/backward step for S0 and S1.
3. Run one real `compress()`/`decompress()` round trip and verify latent exact.
4. Measure peak memory before choosing batch size; do not copy another model's
   batch size without testing.
5. Confirm both configs resolve exactly 1000 training and 200 validation files.

Checkpoints, logs and experimental outputs are intentionally ignored by Git.
