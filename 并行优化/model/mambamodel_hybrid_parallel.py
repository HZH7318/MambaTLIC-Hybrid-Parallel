"""Hybrid serial-prefix / parallel-tail entropy model.

This module intentionally leaves ``mambamodel_1_917.py`` untouched.  It
reuses the original analysis/synthesis transforms and hyperprior, but replaces
the five-slice entropy model

    [16, 16, 32, 64, 192]

with three exact serial prefix slices and four conditionally-independent tail
slices

    [16, 16, 32 | 64, 64, 64, 64].

The four tail slices share one context model.  A learned group embedding tells
the shared model which tail distribution it is predicting.  During training,
the four tails are stacked on the batch dimension, so anchor prediction and
non-anchor prediction each require one network invocation.  During real
coding, every tail owns an independent rANS bitstream. The four streams are
submitted to a native OpenMP rANS backend which releases the GIL and processes
the streams concurrently without repeated Python/C++ boundary calls.
"""

from __future__ import annotations

import time
from typing import List, Sequence, Tuple

import numpy as np
import torch
from torch import Tensor, nn

from native_parallel_rans import (
    decode_batch as native_rans_decode_batch,
    encode_batch as native_rans_encode_batch,
    openmp_enabled as native_rans_openmp_enabled,
)

from model.mamba.mambamodel_1_917 import (
    CheckboardMaskedConv2d,
    LossyCompressor,
    Quantizer,
    RateDistortion,
    conv,
    conv1x1,
)


PREFIX_GROUPS: Tuple[int, ...] = (16, 16, 32)
TAIL_GROUPS: Tuple[int, ...] = (64, 64, 64, 64)
HYBRID_GROUPS: Tuple[int, ...] = PREFIX_GROUPS + TAIL_GROUPS


class HybridParallelLossyCompressor(LossyCompressor):
    """Mamba codec with a serial prefix and native-parallel tail streams."""

    def __init__(
        self,
        numch_n: int,
        numch_m: int,
        base_channel: int,
        entropy_workers: int = 4,
        use_native_entropy: bool = True,
        tail_embedding_dim: int = 16,
        *args,
        **kwargs,
    ) -> None:
        if numch_m != sum(HYBRID_GROUPS):
            raise ValueError(
                "Hybrid grouping is defined for numch_m=320, "
                f"but received numch_m={numch_m}."
            )

        # Build and retain the original encoder, decoder, hyperprior and entropy
        # bottlenecks.  The context modules below replace the parent's modules.
        super().__init__(numch_n, numch_m, base_channel, *args, **kwargs)

        self.groups = [0, *HYBRID_GROUPS]
        self.num_prefix_groups = len(PREFIX_GROUPS)
        self.num_tail_groups = len(TAIL_GROUPS)
        self.tail_channels = TAIL_GROUPS[0]
        self.entropy_workers = max(1, int(entropy_workers))
        self.use_native_entropy = bool(use_native_entropy)
        if self.use_native_entropy and not native_rans_openmp_enabled:
            raise RuntimeError("The native tail entropy backend lacks OpenMP support")
        self._native_cdf_cache = None
        self.tail_embedding_dim = int(tail_embedding_dim)

        # Prefix slices preserve the dependency pattern of the original model:
        # P1 <- hyperprior, P2 <- P1, P3 <- (P1, P2).
        self.cc_transforms = nn.ModuleList(
            self._make_channel_context(
                sum(PREFIX_GROUPS[:target_index]),
                2 * PREFIX_GROUPS[target_index],
            )
            for target_index in range(1, self.num_prefix_groups)
        )
        self.context_prediction = nn.ModuleList(
            CheckboardMaskedConv2d(
                channels,
                2 * channels,
                kernel_size=5,
                padding=2,
                stride=1,
            )
            for channels in PREFIX_GROUPS
        )
        self.ParamAggregation = nn.ModuleList(
            self._make_param_aggregation(
                2 * self.M
                + (0 if index == 0 else 2 * channels)
                + 2 * channels,
                2 * channels,
            )
            for index, channels in enumerate(PREFIX_GROUPS)
        )

        # The complete reconstructed prefix has 64 channels.  It is projected
        # once and reused by every tail group.
        self.tail_prefix_transform = self._make_channel_context(
            sum(PREFIX_GROUPS), 2 * self.tail_channels
        )
        self.tail_context_prediction = CheckboardMaskedConv2d(
            self.tail_channels,
            2 * self.tail_channels,
            kernel_size=5,
            padding=2,
            stride=1,
        )
        self.tail_group_embedding = nn.Embedding(
            self.num_tail_groups, self.tail_embedding_dim
        )
        nn.init.normal_(self.tail_group_embedding.weight, mean=0.0, std=0.02)

        tail_param_in = (
            2 * self.M
            + 2 * self.tail_channels  # shared PrefixContext
            + 2 * self.tail_channels  # anchor/non-anchor spatial context
            + self.tail_embedding_dim
        )
        self.tail_param_aggregation = self._make_param_aggregation(
            tail_param_in, 2 * self.tail_channels
        )
        self.quantizer = Quantizer()

    @staticmethod
    def _make_channel_context(in_channels: int, out_channels: int) -> nn.Sequential:
        return nn.Sequential(
            conv(in_channels, 224, stride=1, kernel_size=5),
            nn.ReLU(inplace=True),
            conv(224, 128, stride=1, kernel_size=5),
            nn.ReLU(inplace=True),
            conv(128, out_channels, stride=1, kernel_size=5),
        )

    @staticmethod
    def _make_param_aggregation(
        in_channels: int, out_channels: int
    ) -> nn.Sequential:
        return nn.Sequential(
            conv1x1(in_channels, 640),
            nn.ReLU(inplace=True),
            conv1x1(640, 512),
            nn.ReLU(inplace=True),
            conv1x1(512, out_channels),
        )

    @staticmethod
    def _checkerboard_masks(y: Tensor) -> Tuple[Tensor, Tensor]:
        anchor_mask = torch.zeros_like(y)
        anchor_mask[:, :, 0::2, 0::2] = 1
        anchor_mask[:, :, 1::2, 1::2] = 1
        return anchor_mask, 1 - anchor_mask

    @classmethod
    def _checkerboard_split(cls, y: Tensor) -> Tuple[Tensor, Tensor]:
        anchor_mask, non_anchor_mask = cls._checkerboard_masks(y)
        return y * anchor_mask, y * non_anchor_mask

    @staticmethod
    def _pack_anchor(x: Tensor) -> Tensor:
        if x.shape[-1] % 2:
            raise ValueError("The latent width must be even for checkerboard coding.")
        packed = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2)
        packed[:, :, 0::2, :] = x[:, :, 0::2, 0::2]
        packed[:, :, 1::2, :] = x[:, :, 1::2, 1::2]
        return packed

    @staticmethod
    def _unpack_anchor(x: Tensor) -> Tensor:
        unpacked = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * 2)
        unpacked[:, :, 0::2, 0::2] = x[:, :, 0::2, :]
        unpacked[:, :, 1::2, 1::2] = x[:, :, 1::2, :]
        return unpacked

    @staticmethod
    def _pack_non_anchor(x: Tensor) -> Tensor:
        if x.shape[-1] % 2:
            raise ValueError("The latent width must be even for checkerboard coding.")
        packed = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3] // 2)
        packed[:, :, 0::2, :] = x[:, :, 0::2, 1::2]
        packed[:, :, 1::2, :] = x[:, :, 1::2, 0::2]
        return packed

    @staticmethod
    def _unpack_non_anchor(x: Tensor) -> Tensor:
        unpacked = x.new_zeros(x.shape[0], x.shape[1], x.shape[2], x.shape[3] * 2)
        unpacked[:, :, 0::2, 1::2] = x[:, :, 0::2, :]
        unpacked[:, :, 1::2, 0::2] = x[:, :, 1::2, :]
        return unpacked

    @staticmethod
    def _merge_checkerboard_parameters(
        means_anchor: Tensor,
        scales_anchor: Tensor,
        means_non_anchor: Tensor,
        scales_non_anchor: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        means = torch.zeros_like(means_anchor)
        scales = torch.zeros_like(scales_anchor)
        means[:, :, 0::2, 0::2] = means_anchor[:, :, 0::2, 0::2]
        means[:, :, 1::2, 1::2] = means_anchor[:, :, 1::2, 1::2]
        means[:, :, 0::2, 1::2] = means_non_anchor[:, :, 0::2, 1::2]
        means[:, :, 1::2, 0::2] = means_non_anchor[:, :, 1::2, 0::2]
        scales[:, :, 0::2, 0::2] = scales_anchor[:, :, 0::2, 0::2]
        scales[:, :, 1::2, 1::2] = scales_anchor[:, :, 1::2, 1::2]
        scales[:, :, 0::2, 1::2] = scales_non_anchor[:, :, 0::2, 1::2]
        scales[:, :, 1::2, 0::2] = scales_non_anchor[:, :, 1::2, 0::2]
        return means, scales

    def _prefix_support(
        self,
        index: int,
        y_hat_prefix: Sequence[Tensor],
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Tensor:
        hyper = torch.cat([latent_means, latent_scales], dim=1)
        if index == 0:
            return hyper
        previous = torch.cat(list(y_hat_prefix), dim=1)
        channel_context = self.cc_transforms[index - 1](previous)
        return torch.cat([channel_context, hyper], dim=1)

    def _prefix_anchor_parameters(
        self, index: int, support: Tensor
    ) -> Tuple[Tensor, Tensor]:
        channels = PREFIX_GROUPS[index]
        empty_context = support.new_zeros(
            support.shape[0], 2 * channels, support.shape[2], support.shape[3]
        )
        return self.ParamAggregation[index](
            torch.cat([empty_context, support], dim=1)
        ).chunk(2, dim=1)

    def _tail_embedding_map(
        self, batch: int, height: int, width: int, reference: Tensor
    ) -> Tensor:
        group_ids = torch.arange(self.num_tail_groups, device=reference.device)
        embeddings = self.tail_group_embedding(group_ids).to(dtype=reference.dtype)
        return (
            embeddings[:, None, :, None, None]
            .expand(-1, batch, -1, height, width)
            .reshape(self.num_tail_groups * batch, self.tail_embedding_dim, height, width)
        )

    def _tail_support(
        self,
        prefix_context: Tensor,
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Tensor:
        batch, _, height, width = latent_means.shape
        repeats = (self.num_tail_groups, 1, 1, 1)
        return torch.cat(
            [
                prefix_context.repeat(*repeats),
                latent_means.repeat(*repeats),
                latent_scales.repeat(*repeats),
                self._tail_embedding_map(batch, height, width, latent_means),
            ],
            dim=1,
        )

    def _tail_parameters(
        self, spatial_context: Tensor, tail_support: Tensor
    ) -> Tuple[Tensor, Tensor]:
        return self.tail_param_aggregation(
            torch.cat([spatial_context, tail_support], dim=1)
        ).chunk(2, dim=1)

    def _compress_independent_groups(
        self,
        values: Sequence[Tensor],
        indexes: Sequence[Tensor],
        means: Sequence[Tensor],
    ) -> List[List[bytes]]:
        if len(values) <= 1 or not self.use_native_entropy:
            return [
                self.gaussian_conditional.compress(value, index, means=mean)
                for value, index, mean in zip(values, indexes, means)
            ]

        # Convert all tasks once, then encode them concurrently in native C++.
        # The pybind entry point releases the GIL and OpenMP owns the workers.
        group_batch = values[0].shape[0]
        packed_values = torch.cat(list(values), dim=0)
        packed_indexes = torch.cat(list(indexes), dim=0)
        packed_means = torch.cat(list(means), dim=0)
        packed_symbols = self.gaussian_conditional.quantize(
            packed_values, "symbols", packed_means
        )
        symbols_cpu = packed_symbols.reshape(packed_symbols.shape[0], -1).int().cpu().numpy()
        indexes_cpu = packed_indexes.reshape(packed_indexes.shape[0], -1).int().cpu().numpy()
        symbol_arrays = [symbols_cpu[index] for index in range(symbols_cpu.shape[0])]
        index_arrays = [indexes_cpu[index] for index in range(indexes_cpu.shape[0])]
        cdf, cdf_lengths, offsets = self._native_cdf_tables()
        packed_strings = native_rans_encode_batch(
            symbol_arrays,
            index_arrays,
            cdf,
            cdf_lengths,
            offsets,
            self.entropy_workers,
        )
        return [
            packed_strings[start : start + group_batch]
            for start in range(0, len(packed_strings), group_batch)
        ]

    def _decompress_independent_groups(
        self,
        strings: Sequence[Sequence[bytes]],
        indexes: Sequence[Tensor],
        means: Sequence[Tensor],
    ) -> List[Tensor]:
        if len(strings) <= 1 or not self.use_native_entropy:
            return [
                self.gaussian_conditional.decompress(stream, index, means=mean)
                for stream, index, mean in zip(strings, indexes, means)
            ]

        group_batch = indexes[0].shape[0]
        packed_strings = [item for group in strings for item in group]
        packed_indexes = torch.cat(list(indexes), dim=0)
        indexes_cpu = packed_indexes.reshape(packed_indexes.shape[0], -1).int().cpu().numpy()
        index_arrays = [indexes_cpu[index] for index in range(indexes_cpu.shape[0])]
        cdf, cdf_lengths, offsets = self._native_cdf_tables()
        decoded = native_rans_decode_batch(
            packed_strings,
            index_arrays,
            cdf,
            cdf_lengths,
            offsets,
            self.entropy_workers,
        )
        decoded_cpu = np.stack([np.asarray(sample) for sample in decoded], axis=0)
        packed_symbols = torch.from_numpy(decoded_cpu).reshape_as(packed_indexes).to(
            device=means[0].device, dtype=torch.int32
        )
        packed_means = torch.cat(list(means), dim=0)
        packed_output = self.gaussian_conditional.dequantize(
            packed_symbols, packed_means
        )
        return list(torch.split(packed_output, group_batch, dim=0))

    def _native_cdf_tables(self):
        conditional = self.gaussian_conditional
        signature = (
            conditional._quantized_cdf.data_ptr(),
            conditional._cdf_length.data_ptr(),
            conditional._offset.data_ptr(),
        )
        if self._native_cdf_cache is None or self._native_cdf_cache[0] != signature:
            tables = tuple(
                tensor.detach().contiguous().cpu().numpy().astype(
                    np.int32, copy=False
                )
                for tensor in (
                    conditional._quantized_cdf,
                    conditional._cdf_length,
                    conditional._offset,
                )
            )
            self._native_cdf_cache = (signature, tables)
        return self._native_cdf_cache[1]

    def _forward_prefix_group(
        self,
        index: int,
        y_slice: Tensor,
        y_hat_prefix: Sequence[Tensor],
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Tuple[Tensor, Tensor]:
        anchor, non_anchor = self._checkerboard_split(y_slice)
        anchor_mask, non_anchor_mask = self._checkerboard_masks(y_slice)
        support = self._prefix_support(
            index, y_hat_prefix, latent_means, latent_scales
        )
        means_anchor, scales_anchor = self._prefix_anchor_parameters(index, support)

        y_anchor_hat = (
            self.quantizer.quantize(anchor - means_anchor, "ste") + means_anchor
        ) * anchor_mask
        masked_context = self.context_prediction[index](y_anchor_hat)
        means_non_anchor, scales_non_anchor = self.ParamAggregation[index](
            torch.cat([masked_context, support], dim=1)
        ).chunk(2, dim=1)

        means, scales = self._merge_checkerboard_parameters(
            means_anchor, scales_anchor, means_non_anchor, scales_non_anchor
        )
        _, likelihood = self.gaussian_conditional(y_slice, scales, means=means)
        y_non_anchor_hat = (
            self.quantizer.quantize(non_anchor - means_non_anchor, "ste")
            + means_non_anchor
        ) * non_anchor_mask
        return y_anchor_hat + y_non_anchor_hat, likelihood

    def forward(self, input: Tensor):
        y = self.encoder(input)
        z = self.hyperencoder(y)
        z_hat, z_likelihoods = self.entropy_bottleneck(z)
        latent_means, latent_scales = self.hyperdecoder(z_hat)

        y_slices = torch.split(y, self.groups[1:], dim=1)
        prefix_slices = y_slices[: self.num_prefix_groups]
        tail_slices = y_slices[self.num_prefix_groups :]

        y_hat_prefix: List[Tensor] = []
        prefix_likelihoods: List[Tensor] = []
        for index, y_slice in enumerate(prefix_slices):
            y_hat_slice, likelihood = self._forward_prefix_group(
                index,
                y_slice,
                y_hat_prefix,
                latent_means,
                latent_scales,
            )
            y_hat_prefix.append(y_hat_slice)
            prefix_likelihoods.append(likelihood)

        prefix_context = self.tail_prefix_transform(torch.cat(y_hat_prefix, dim=1))
        tail_support = self._tail_support(
            prefix_context, latent_means, latent_scales
        )
        tail_batch = torch.cat(list(tail_slices), dim=0)
        tail_anchor, tail_non_anchor = self._checkerboard_split(tail_batch)
        anchor_mask, non_anchor_mask = self._checkerboard_masks(tail_batch)

        empty_spatial_context = tail_batch.new_zeros(
            tail_batch.shape[0],
            2 * self.tail_channels,
            tail_batch.shape[2],
            tail_batch.shape[3],
        )
        means_anchor, scales_anchor = self._tail_parameters(
            empty_spatial_context, tail_support
        )
        tail_anchor_hat = (
            self.quantizer.quantize(tail_anchor - means_anchor, "ste")
            + means_anchor
        ) * anchor_mask

        masked_context = self.tail_context_prediction(tail_anchor_hat)
        means_non_anchor, scales_non_anchor = self._tail_parameters(
            masked_context, tail_support
        )
        means, scales = self._merge_checkerboard_parameters(
            means_anchor, scales_anchor, means_non_anchor, scales_non_anchor
        )
        _, tail_likelihood_batch = self.gaussian_conditional(
            tail_batch, scales, means=means
        )
        tail_non_anchor_hat = (
            self.quantizer.quantize(tail_non_anchor - means_non_anchor, "ste")
            + means_non_anchor
        ) * non_anchor_mask
        tail_hat_batch = tail_anchor_hat + tail_non_anchor_hat

        batch = input.shape[0]
        tail_hat_slices = list(torch.split(tail_hat_batch, batch, dim=0))
        tail_likelihoods = list(
            torch.split(tail_likelihood_batch, batch, dim=0)
        )
        y_hat = torch.cat(y_hat_prefix + tail_hat_slices, dim=1)
        y_likelihoods = torch.cat(prefix_likelihoods + tail_likelihoods, dim=1)

        return {
            "x_hat": self.decoder(y_hat),
            "img_likelihoods": {"y": y_likelihoods, "z": z_likelihoods},
        }

    def _compress_prefix_group(
        self,
        index: int,
        y_slice: Tensor,
        y_hat_prefix: Sequence[Tensor],
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Tuple[Tensor, List[List[bytes]]]:
        support = self._prefix_support(
            index, y_hat_prefix, latent_means, latent_scales
        )
        means_anchor, scales_anchor = self._prefix_anchor_parameters(index, support)

        y_anchor = self._pack_anchor(y_slice)
        means_anchor_packed = self._pack_anchor(means_anchor)
        scales_anchor_packed = self._pack_anchor(scales_anchor)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_packed)
        anchor_strings = self.gaussian_conditional.compress(
            y_anchor, indexes_anchor, means=means_anchor_packed
        )
        anchor_hat_packed = torch.round(y_anchor - means_anchor_packed) + means_anchor_packed
        anchor_hat = self._unpack_anchor(anchor_hat_packed)

        masked_context = self.context_prediction[index](anchor_hat)
        means_non_anchor, scales_non_anchor = self.ParamAggregation[index](
            torch.cat([masked_context, support], dim=1)
        ).chunk(2, dim=1)
        y_non_anchor = self._pack_non_anchor(y_slice)
        means_non_anchor_packed = self._pack_non_anchor(means_non_anchor)
        scales_non_anchor_packed = self._pack_non_anchor(scales_non_anchor)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(
            scales_non_anchor_packed
        )
        non_anchor_strings = self.gaussian_conditional.compress(
            y_non_anchor,
            indexes_non_anchor,
            means=means_non_anchor_packed,
        )
        non_anchor_hat_packed = (
            torch.round(y_non_anchor - means_non_anchor_packed)
            + means_non_anchor_packed
        )
        y_hat = anchor_hat + self._unpack_non_anchor(non_anchor_hat_packed)
        return y_hat, [anchor_strings, non_anchor_strings]

    def compress(self, x: Tensor, return_debug: bool = False):
        y_enc_start = time.perf_counter()
        y = self.encoder(x)
        y_enc = time.perf_counter() - y_enc_start

        z_enc_start = time.perf_counter()
        z = self.hyperencoder(y)
        z_enc = time.perf_counter() - z_enc_start
        z_strings = self.entropy_bottleneck.compress(z)
        z_hat = self.entropy_bottleneck.decompress(z_strings, z.size()[-2:])

        z_dec_start = time.perf_counter()
        latent_means, latent_scales = self.hyperdecoder(z_hat)
        z_dec = time.perf_counter() - z_dec_start

        params_start = time.perf_counter()
        y_slices = torch.split(y, self.groups[1:], dim=1)
        prefix_slices = y_slices[: self.num_prefix_groups]
        tail_slices = y_slices[self.num_prefix_groups :]

        y_hat_prefix: List[Tensor] = []
        y_strings: List[List[List[bytes]]] = []
        for index, y_slice in enumerate(prefix_slices):
            y_hat_slice, group_strings = self._compress_prefix_group(
                index,
                y_slice,
                y_hat_prefix,
                latent_means,
                latent_scales,
            )
            y_hat_prefix.append(y_hat_slice)
            y_strings.append(group_strings)

        # Tail anchor stage: one shared network pass, four independent streams.
        prefix_context = self.tail_prefix_transform(torch.cat(y_hat_prefix, dim=1))
        tail_support = self._tail_support(
            prefix_context, latent_means, latent_scales
        )
        tail_batch = torch.cat(list(tail_slices), dim=0)
        batch = x.shape[0]
        empty_spatial_context = tail_batch.new_zeros(
            tail_batch.shape[0],
            2 * self.tail_channels,
            tail_batch.shape[2],
            tail_batch.shape[3],
        )
        means_anchor, scales_anchor = self._tail_parameters(
            empty_spatial_context, tail_support
        )
        y_anchor_packed = self._pack_anchor(tail_batch)
        means_anchor_packed = self._pack_anchor(means_anchor)
        scales_anchor_packed = self._pack_anchor(scales_anchor)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_packed)

        anchor_values = torch.split(y_anchor_packed, batch, dim=0)
        anchor_indexes = torch.split(indexes_anchor, batch, dim=0)
        anchor_means = torch.split(means_anchor_packed, batch, dim=0)
        tail_anchor_strings = self._compress_independent_groups(
            anchor_values, anchor_indexes, anchor_means
        )

        anchor_hat_packed = (
            torch.round(y_anchor_packed - means_anchor_packed)
            + means_anchor_packed
        )
        anchor_hat = self._unpack_anchor(anchor_hat_packed)

        # Tail non-anchor stage: each branch depends only on its own anchor.
        masked_context = self.tail_context_prediction(anchor_hat)
        means_non_anchor, scales_non_anchor = self._tail_parameters(
            masked_context, tail_support
        )
        y_non_anchor_packed = self._pack_non_anchor(tail_batch)
        means_non_anchor_packed = self._pack_non_anchor(means_non_anchor)
        scales_non_anchor_packed = self._pack_non_anchor(scales_non_anchor)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(
            scales_non_anchor_packed
        )
        non_anchor_values = torch.split(y_non_anchor_packed, batch, dim=0)
        non_anchor_indexes = torch.split(indexes_non_anchor, batch, dim=0)
        non_anchor_means = torch.split(means_non_anchor_packed, batch, dim=0)
        tail_non_anchor_strings = self._compress_independent_groups(
            non_anchor_values, non_anchor_indexes, non_anchor_means
        )

        y_strings.extend(
            [list(pair) for pair in zip(tail_anchor_strings, tail_non_anchor_strings)]
        )
        params_time = time.perf_counter() - params_start

        output = {
            "strings": [y_strings, z_strings],
            "shape": z.size()[-2:],
            "time": {
                "y_enc": y_enc,
                "z_enc": z_enc,
                "z_dec": z_dec,
                "params": params_time,
            },
        }
        if return_debug:
            non_anchor_hat_packed = (
                torch.round(y_non_anchor_packed - means_non_anchor_packed)
                + means_non_anchor_packed
            )
            tail_hat_batch = anchor_hat + self._unpack_non_anchor(
                non_anchor_hat_packed
            )
            tail_hat_slices = list(torch.split(tail_hat_batch, batch, dim=0))
            output["debug_y_hat"] = torch.cat(
                y_hat_prefix + tail_hat_slices, dim=1
            )
        return output

    def _decompress_prefix_group(
        self,
        index: int,
        group_strings: Sequence[Sequence[bytes]],
        y_hat_prefix: Sequence[Tensor],
        latent_means: Tensor,
        latent_scales: Tensor,
    ) -> Tensor:
        support = self._prefix_support(
            index, y_hat_prefix, latent_means, latent_scales
        )
        means_anchor, scales_anchor = self._prefix_anchor_parameters(index, support)
        means_anchor_packed = self._pack_anchor(means_anchor)
        scales_anchor_packed = self._pack_anchor(scales_anchor)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_packed)
        anchor_hat_packed = self.gaussian_conditional.decompress(
            group_strings[0], indexes_anchor, means=means_anchor_packed
        )
        anchor_hat = self._unpack_anchor(anchor_hat_packed)

        masked_context = self.context_prediction[index](anchor_hat)
        means_non_anchor, scales_non_anchor = self.ParamAggregation[index](
            torch.cat([masked_context, support], dim=1)
        ).chunk(2, dim=1)
        means_non_anchor_packed = self._pack_non_anchor(means_non_anchor)
        scales_non_anchor_packed = self._pack_non_anchor(scales_non_anchor)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(
            scales_non_anchor_packed
        )
        non_anchor_hat_packed = self.gaussian_conditional.decompress(
            group_strings[1],
            indexes_non_anchor,
            means=means_non_anchor_packed,
        )
        return anchor_hat + self._unpack_non_anchor(non_anchor_hat_packed)

    def decompress(self, strings, shape, return_debug: bool = False):
        if not isinstance(strings, list) or len(strings) != 2:
            raise ValueError("strings must be [y_strings, z_strings].")
        y_strings, z_strings = strings
        if len(y_strings) != len(HYBRID_GROUPS):
            raise ValueError(
                f"Expected {len(HYBRID_GROUPS)} y groups, got {len(y_strings)}."
            )

        z_hat = self.entropy_bottleneck.decompress(z_strings, shape)
        latent_means, latent_scales = self.hyperdecoder(z_hat)

        y_hat_prefix: List[Tensor] = []
        for index in range(self.num_prefix_groups):
            y_hat_prefix.append(
                self._decompress_prefix_group(
                    index,
                    y_strings[index],
                    y_hat_prefix,
                    latent_means,
                    latent_scales,
                )
            )

        prefix_context = self.tail_prefix_transform(torch.cat(y_hat_prefix, dim=1))
        tail_support = self._tail_support(
            prefix_context, latent_means, latent_scales
        )
        batch, _, height, width = latent_means.shape
        tail_batch_size = self.num_tail_groups * batch

        # Decode four independent anchor streams in one batched API call.
        empty_spatial_context = latent_means.new_zeros(
            tail_batch_size, 2 * self.tail_channels, height, width
        )
        means_anchor, scales_anchor = self._tail_parameters(
            empty_spatial_context, tail_support
        )
        means_anchor_packed = self._pack_anchor(means_anchor)
        scales_anchor_packed = self._pack_anchor(scales_anchor)
        indexes_anchor = self.gaussian_conditional.build_indexes(scales_anchor_packed)
        anchor_indexes = torch.split(indexes_anchor, batch, dim=0)
        anchor_means = torch.split(means_anchor_packed, batch, dim=0)
        tail_strings = y_strings[self.num_prefix_groups :]
        anchor_strings = [group[0] for group in tail_strings]
        anchor_hat_groups = self._decompress_independent_groups(
            anchor_strings, anchor_indexes, anchor_means
        )
        anchor_hat = self._unpack_anchor(torch.cat(anchor_hat_groups, dim=0))

        # Decode four independent non-anchor streams after the anchor stage.
        masked_context = self.tail_context_prediction(anchor_hat)
        means_non_anchor, scales_non_anchor = self._tail_parameters(
            masked_context, tail_support
        )
        means_non_anchor_packed = self._pack_non_anchor(means_non_anchor)
        scales_non_anchor_packed = self._pack_non_anchor(scales_non_anchor)
        indexes_non_anchor = self.gaussian_conditional.build_indexes(
            scales_non_anchor_packed
        )
        non_anchor_indexes = torch.split(indexes_non_anchor, batch, dim=0)
        non_anchor_means = torch.split(means_non_anchor_packed, batch, dim=0)
        non_anchor_strings = [group[1] for group in tail_strings]
        non_anchor_hat_groups = self._decompress_independent_groups(
            non_anchor_strings, non_anchor_indexes, non_anchor_means
        )
        non_anchor_hat = self._unpack_non_anchor(
            torch.cat(non_anchor_hat_groups, dim=0)
        )

        tail_hat_slices = list(
            torch.split(anchor_hat + non_anchor_hat, batch, dim=0)
        )
        y_hat = torch.cat(y_hat_prefix + tail_hat_slices, dim=1)

        y_dec_start = time.perf_counter()
        x_hat = self.decoder(y_hat).clamp_(0, 1)
        y_dec = time.perf_counter() - y_dec_start
        output = {"x_hat": x_hat, "time": {"y_dec": y_dec}}
        if return_debug:
            output["debug_y_hat"] = y_hat
        return output


class HNet(nn.Sequential):
    """Drop-in wrapper matching the existing training script API."""

    def __init__(
        self,
        numch_n: int,
        numch_m: int,
        base_channel: int,
        entropy_workers: int = 4,
        use_native_entropy: bool = True,
    ) -> None:
        super().__init__()
        self.net = HybridParallelLossyCompressor(
            numch_n,
            numch_m,
            base_channel,
            entropy_workers=entropy_workers,
            use_native_entropy=use_native_entropy,
        )

    def forward(self, x: Tensor):
        return self.net(x)


__all__ = [
    "HNet",
    "HybridParallelLossyCompressor",
    "RateDistortion",
    "PREFIX_GROUPS",
    "TAIL_GROUPS",
    "HYBRID_GROUPS",
]
