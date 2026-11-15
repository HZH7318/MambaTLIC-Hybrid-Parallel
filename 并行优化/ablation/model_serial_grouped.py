"""Serial grouped entropy models used only by the S0/S1 ablation.

This file deliberately does not modify the historical serial baseline or the
production Hybrid model.  S0 and S1 share the original first-plus-previous
channel-context rule; only their channel grouping differs.  Their real codec
uses the same optimized native multi-stream backend as S2.
"""

from __future__ import annotations

from typing import List, Sequence, Tuple

import torch
from torch import Tensor, nn

from model.mamba.mambamodel_1_917 import (
    CheckboardMaskedConv2d,
    LossyCompressor as HistoricalSerialCompressor,
    RateDistortion,
    conv,
    conv1x1,
)
from mambamodel_hybrid_parallel import HybridParallelLossyCompressor


S0_GROUPS: Tuple[int, ...] = (16, 16, 32, 64, 192)
S1_GROUPS: Tuple[int, ...] = (16, 16, 32, 64, 64, 64, 64)


class SerialGroupedLossyCompressor(HistoricalSerialCompressor):
    """Historical serial dependency graph with configurable channel groups."""

    # Reuse the exact checkerboard layout and optimized native entropy helpers
    # from S2 so structure comparisons do not mix in backend differences.
    _pack_anchor = staticmethod(HybridParallelLossyCompressor._pack_anchor)
    _unpack_anchor = staticmethod(HybridParallelLossyCompressor._unpack_anchor)
    _pack_non_anchor = staticmethod(HybridParallelLossyCompressor._pack_non_anchor)
    _unpack_non_anchor = staticmethod(HybridParallelLossyCompressor._unpack_non_anchor)
    _build_indexes = HybridParallelLossyCompressor._build_indexes
    _compress_native_tasks = HybridParallelLossyCompressor._compress_native_tasks
    _decompress_independent_groups = (
        HybridParallelLossyCompressor._decompress_independent_groups
    )
    _native_cdf_tables = HybridParallelLossyCompressor._native_cdf_tables

    def __init__(
        self,
        numch_n: int,
        numch_m: int,
        base_channel: int,
        groups: Sequence[int],
        entropy_workers: int = 4,
    ) -> None:
        groups = tuple(int(value) for value in groups)
        if sum(groups) != numch_m:
            raise ValueError(f"groups sum to {sum(groups)}, expected M={numch_m}")
        super().__init__(numch_n, numch_m, base_channel)
        self.groups = [0, *groups]
        self.entropy_workers = max(1, int(entropy_workers))
        self.use_native_entropy = True
        self.use_searchsorted_indexes = True
        self.defer_native_encode = True
        self.fuse_symbol_packing = False
        self._native_cdf_cache = None

        count = len(groups)
        self.cc_transforms = nn.ModuleList(
            nn.Sequential(
                conv(
                    self.groups[min(1, index) if index > 0 else 0]
                    + self.groups[index if index > 1 else 0],
                    224,
                    stride=1,
                    kernel_size=5,
                ),
                nn.ReLU(inplace=True),
                conv(224, 128, stride=1, kernel_size=5),
                nn.ReLU(inplace=True),
                conv(128, self.groups[index + 1] * 2, stride=1, kernel_size=5),
            )
            for index in range(1, count)
        )
        self.context_prediction = nn.ModuleList(
            CheckboardMaskedConv2d(
                self.groups[index + 1],
                2 * self.groups[index + 1],
                kernel_size=5,
                padding=2,
                stride=1,
            )
            for index in range(count)
        )
        self.ParamAggregation = nn.ModuleList(
            nn.Sequential(
                conv1x1(
                    640
                    + self.groups[index + 1 if index > 0 else 0] * 2
                    + self.groups[index + 1] * 2,
                    640,
                ),
                nn.ReLU(inplace=True),
                conv1x1(640, 512),
                nn.ReLU(inplace=True),
                conv1x1(512, self.groups[index + 1] * 2),
            )
            for index in range(count)
        )

    def _serial_support(
        self,
        index: int,
        y_hat_slices: Sequence[Tensor],
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Tensor:
        hyper = torch.cat([latent_means, latent_scales], dim=1)
        if index == 0:
            return hyper
        if index == 1:
            channel_input = y_hat_slices[0]
        else:
            # This is the dependency rule implemented by the historical S0:
            # the first slice and the immediately preceding slice.
            channel_input = torch.cat(
                [y_hat_slices[0], y_hat_slices[index - 1]], dim=1
            )
        channel_context = self.cc_transforms[index - 1](channel_input)
        return torch.cat([channel_context, hyper], dim=1)

    def _prepare_group(
        self,
        index: int,
        y_slice: Tensor,
        y_hat_slices: Sequence[Tensor],
        latent_means: Tensor,
        latent_scales: Tensor,
    ):
        support = self._serial_support(
            index, y_hat_slices, latent_means, latent_scales
        )
        empty_context = support.new_zeros(
            support.shape[0], 2 * y_slice.shape[1], *support.shape[2:]
        )
        means_anchor, scales_anchor = self.ParamAggregation[index](
            torch.cat([empty_context, support], dim=1)
        ).chunk(2, dim=1)

        anchor_values = self._pack_anchor(y_slice)
        anchor_means = self._pack_anchor(means_anchor)
        anchor_scales = self._pack_anchor(scales_anchor)
        anchor_indexes = self._build_indexes(anchor_scales)
        anchor_hat_packed = torch.round(anchor_values - anchor_means) + anchor_means
        anchor_hat = self._unpack_anchor(anchor_hat_packed)

        masked_context = self.context_prediction[index](anchor_hat)
        means_non_anchor, scales_non_anchor = self.ParamAggregation[index](
            torch.cat([masked_context, support], dim=1)
        ).chunk(2, dim=1)
        non_anchor_values = self._pack_non_anchor(y_slice)
        non_anchor_means = self._pack_non_anchor(means_non_anchor)
        non_anchor_scales = self._pack_non_anchor(scales_non_anchor)
        non_anchor_indexes = self._build_indexes(non_anchor_scales)
        non_anchor_hat_packed = (
            torch.round(non_anchor_values - non_anchor_means) + non_anchor_means
        )
        y_hat = anchor_hat + self._unpack_non_anchor(non_anchor_hat_packed)
        return y_hat, [
            (anchor_values, anchor_indexes, anchor_means),
            (non_anchor_values, non_anchor_indexes, non_anchor_means),
        ]

    def compress(self, x: Tensor, return_debug: bool = False):
        y = self.encoder(x)
        z = self.hyperencoder(y)
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.shape[-2:])
        latent_means, latent_scales = self.hyperdecoder(z_hat)

        y_hat_slices: List[Tensor] = []
        tasks = []
        for index, y_slice in enumerate(torch.split(y, self.groups[1:], dim=1)):
            y_hat, group_tasks = self._prepare_group(
                index,
                y_slice,
                y_hat_slices,
                latent_means,
                latent_scales,
            )
            y_hat_slices.append(y_hat)
            tasks.extend(group_tasks)
        encoded = self._compress_native_tasks(tasks)
        y_strings = [
            [encoded[2 * index], encoded[2 * index + 1]]
            for index in range(len(self.groups) - 1)
        ]
        output = {"strings": [y_strings, z_strings], "shape": z.shape[-2:]}
        if return_debug:
            output["debug_y_hat"] = torch.cat(y_hat_slices, dim=1)
        return output

    def decompress(self, strings, shape, return_debug: bool = False):
        y_strings, z_strings = strings
        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        latent_means, latent_scales = self.hyperdecoder(z_hat)
        y_hat_slices: List[Tensor] = []
        for index, group_strings in enumerate(y_strings):
            support = self._serial_support(
                index, y_hat_slices, latent_means, latent_scales
            )
            channels = self.groups[index + 1]
            empty_context = support.new_zeros(
                support.shape[0], 2 * channels, *support.shape[2:]
            )
            means_anchor, scales_anchor = self.ParamAggregation[index](
                torch.cat([empty_context, support], dim=1)
            ).chunk(2, dim=1)
            anchor_means = self._pack_anchor(means_anchor)
            anchor_indexes = self._build_indexes(self._pack_anchor(scales_anchor))
            anchor_hat_packed = self._decompress_independent_groups(
                (group_strings[0],), (anchor_indexes,), (anchor_means,)
            )[0]
            anchor_hat = self._unpack_anchor(anchor_hat_packed)

            masked_context = self.context_prediction[index](anchor_hat)
            means_non_anchor, scales_non_anchor = self.ParamAggregation[index](
                torch.cat([masked_context, support], dim=1)
            ).chunk(2, dim=1)
            non_anchor_means = self._pack_non_anchor(means_non_anchor)
            non_anchor_indexes = self._build_indexes(
                self._pack_non_anchor(scales_non_anchor)
            )
            non_anchor_hat_packed = self._decompress_independent_groups(
                (group_strings[1],), (non_anchor_indexes,), (non_anchor_means,)
            )[0]
            y_hat_slices.append(
                anchor_hat + self._unpack_non_anchor(non_anchor_hat_packed)
            )
        y_hat = torch.cat(y_hat_slices, dim=1)
        output = {"x_hat": self.decoder(y_hat).clamp_(0, 1)}
        if return_debug:
            output["debug_y_hat"] = y_hat
        return output


class S0Net(nn.Sequential):
    def __init__(self, numch_n: int, numch_m: int, base_channel: int):
        super().__init__()
        self.net = SerialGroupedLossyCompressor(
            numch_n, numch_m, base_channel, S0_GROUPS
        )


class S1Net(nn.Sequential):
    def __init__(self, numch_n: int, numch_m: int, base_channel: int):
        super().__init__()
        self.net = SerialGroupedLossyCompressor(
            numch_n, numch_m, base_channel, S1_GROUPS
        )


__all__ = [
    "RateDistortion",
    "S0Net",
    "S1Net",
    "S0_GROUPS",
    "S1_GROUPS",
    "SerialGroupedLossyCompressor",
]
