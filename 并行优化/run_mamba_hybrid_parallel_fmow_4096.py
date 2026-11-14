"""FMoW Hybrid training entry using the dataset's 4096 normalization.

The actual normalization is read from the INI.  This named entry prevents the
FMoW experiment from being confused with the Landsat8/RS 65535 protocol.
"""

from archive.fmow import run_mamba_hybrid_parallel_fmow_stable as stable


if __name__ == "__main__":
    stable.training.main()
