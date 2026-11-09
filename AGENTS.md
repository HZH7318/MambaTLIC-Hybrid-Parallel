# Repository guide for coding agents

The active research target is `并行优化/`. Read its `README.md` and
`docs/PAPER_AI_MODEL_BRIEF.md` before changing the model or describing results.

Key rules:

- Keep `model/mamba/mambamodel_1_917.py` as the serial baseline.
- Implement Hybrid changes in `并行优化/model/mambamodel_hybrid_parallel.py`.
- Do not commit datasets, checkpoints, logs, native binaries or `experiments/`.
- Full-test RD uses likelihood BPP; codec timing uses actual bitstream bytes.
  Never merge the two BPP protocols without labeling them.
- `results/` contains the committed 400-epoch snapshot. The final Landsat8
  600-epoch outputs belong in `并行优化/final_results/landsat8_hybrid_600/`.
- Do not claim four-thread or SOTA speedups without complete-codec measurements.

The original server paths embedded in INI/evaluation scripts are provenance,
not portable defaults. Update dataset/checkpoint paths for a new machine.
