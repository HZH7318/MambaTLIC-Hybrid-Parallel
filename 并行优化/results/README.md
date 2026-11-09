# Reproducible results

- `data/full_test_rd_all_methods.csv`: Landsat8 (1700 images) and RS (1217 images), likelihood-based full-test RD.
- `data/codec_time_comparison.csv`: real-bitstream pilot timing (5 measured images after 1 warm-up).
- `notebooks/landsat8_results.ipynb`: Landsat8 table and BPP–PSNR/BPP–MS-SSIM plots.
- `notebooks/rs_results.ipynb`: RS table and BPP–PSNR/BPP–MS-SSIM plots.
- `notebooks/codec_time_rd_results.ipynb`: complete-codec timing analysis.

The committed Hybrid RD rows are the completed 400-epoch results. The active
600-epoch Landsat8 run will produce the final replacement under
`../final_results/landsat8_hybrid_600/`.
