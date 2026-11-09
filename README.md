# Hybrid-Parallel Mamba Learned Image Compression

This is the cleaned research repository for a serial-prefix/parallel-tail
entropy model built on the current MambaTLIC codebase. It contains source,
configuration, result tables, figures and executable notebooks. Datasets,
checkpoints, logs and compiled binaries are deliberately excluded.

Start with:

- [`并行优化/README.md`](并行优化/README.md) for reproduction;
- [`并行优化/docs/PAPER_AI_MODEL_BRIEF.md`](并行优化/docs/PAPER_AI_MODEL_BRIEF.md) for the method and paper-writing facts;
- [`并行优化/results/`](并行优化/results/) for Landsat8/RS RD and codec results.

The active 600-epoch Landsat8 experiment runs outside Git tracking. Its final
tables replace the interim 400-epoch Hybrid rows after completion.
