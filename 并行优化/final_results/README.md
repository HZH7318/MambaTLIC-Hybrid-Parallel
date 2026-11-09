# Final experiment outputs

Only publication-facing results are collected under this directory. Existing
training checkpoints, diagnostic notebooks, smoke tests, and intermediate
files remain in their original locations so experiments stay reproducible.

`landsat8_hybrid_600/` is populated automatically after the 600-epoch resume:

- `landsat8_full_rd.csv`: complete Landsat8 BPP/PSNR/MS-SSIM results;
- `landsat8_full_test_rd.png`: final full-test RD curves;
- `landsat8_hybrid_codec_30_images.csv`: real Hybrid encode/decode latency;
- `codec_time_comparison_pilot_5_images.csv`: existing same-sample method comparison;
- `result_manifest.json`: protocol and provenance;
- `status.json`: live pipeline state;
- `logs/`: training/evaluation logs, kept out of the main result tables.
