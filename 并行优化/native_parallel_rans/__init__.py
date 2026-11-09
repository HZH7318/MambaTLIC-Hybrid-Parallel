"""Native OpenMP rANS functions used by the hybrid tail codec."""

from ._parallel_rans import decode_batch, encode_batch, openmp_enabled

__all__ = ["encode_batch", "decode_batch", "openmp_enabled"]
