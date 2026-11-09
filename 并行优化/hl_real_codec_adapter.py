"""Real arithmetic-coding adapter for the repository's HL model.

The original HL implementation exposes likelihood estimates only.  This
adapter adds a decodable checkerboard stream without modifying the old model.
It must be reported as an adapter result, not as an author-provided codec.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import torch


@dataclass
class TensorMeta:
    shape: tuple[int, ...]
    minimum: int
    maximum: int


def _normalize_cdf(cdf: torch.Tensor) -> torch.Tensor:
    cdf = cdf - cdf[..., :1]
    denominator = cdf[..., -1:].clamp_min(1e-9)
    return (cdf / denominator).clamp_(0.0, 1.0).float()


def _factorized_cdf(estimator, meta: TensorMeta, device: torch.device) -> torch.Tensor:
    # L boundaries for symbols [minimum, maximum].
    boundaries = torch.arange(
        meta.minimum - 0.5,
        meta.maximum + 1.0,
        device=device,
        dtype=torch.float32,
    )
    channels = meta.shape[1]
    values = boundaries[:, None, None, None].expand(-1, channels, 1, 1)
    cdf = estimator(values).squeeze(-1).squeeze(-1).transpose(0, 1)
    cdf = cdf[None, :, None, None, :].expand(*meta.shape, -1)
    return _normalize_cdf(cdf)


def _laplace_cdf(value: torch.Tensor, mean: torch.Tensor, scale: torch.Tensor) -> torch.Tensor:
    centered = value - mean
    scale = scale.clamp_min(1e-6)
    return torch.where(
        centered < 0,
        0.5 * torch.exp(centered / scale),
        1.0 - 0.5 * torch.exp(-centered / scale),
    )


def _mixture_cdf(
    means: torch.Tensor,
    scales: torch.Tensor,
    weights: torch.Tensor,
    minimum: int,
    maximum: int,
) -> torch.Tensor:
    boundaries = torch.arange(
        minimum - 0.5,
        maximum + 1.0,
        device=means.device,
        dtype=means.dtype,
    )
    values = boundaries.view(1, 1, 1, 1, 1, -1)
    component_cdf = _laplace_cdf(
        values,
        means.unsqueeze(-1),
        scales.unsqueeze(-1),
    )
    cdf = (weights.unsqueeze(-1) * component_cdf).sum(dim=-2)
    return _normalize_cdf(cdf)


def _checkerboard_mask(shape: tuple[int, ...], device: torch.device) -> torch.Tensor:
    height, width = shape[-2:]
    rows = torch.arange(height, device=device)[:, None]
    columns = torch.arange(width, device=device)[None, :]
    spatial = (rows + columns).remainder(2).eq(0)
    return spatial.view(1, 1, height, width).expand(shape)


class HLRealCodecAdapter:
    def __init__(self, model: torch.nn.Module):
        self.model = model

    @staticmethod
    def _torchac():
        import torchac

        return torchac

    def _encode_tensor(self, tensor: torch.Tensor, cdf: torch.Tensor) -> tuple[bytes, TensorMeta]:
        torchac = self._torchac()
        quantized = torch.round(tensor).to(torch.int32)
        minimum = int(quantized.min().item())
        maximum = int(quantized.max().item())
        meta = TensorMeta(tuple(quantized.shape), minimum, maximum)
        symbols = (quantized - minimum).to(torch.int16).cpu()
        stream = torchac.encode_float_cdf(cdf.detach().cpu(), symbols)
        return stream, meta

    def _decode_tensor(self, stream: bytes, cdf: torch.Tensor, meta: TensorMeta) -> torch.Tensor:
        torchac = self._torchac()
        symbols = torchac.decode_float_cdf(cdf.detach().cpu(), stream)
        return (symbols.to(cdf.device, torch.float32) + meta.minimum).reshape(meta.shape)

    def _encode_selected(
        self,
        tensor: torch.Tensor,
        cdf: torch.Tensor,
        selected: torch.Tensor,
        minimum: int,
    ) -> bytes:
        torchac = self._torchac()
        symbols = (tensor[selected].to(torch.int32) - minimum).to(torch.int16).cpu()
        chosen_cdf = cdf[selected].detach().cpu()
        return torchac.encode_float_cdf(chosen_cdf, symbols)

    def _decode_selected(
        self,
        stream: bytes,
        cdf: torch.Tensor,
        selected: torch.Tensor,
        minimum: int,
    ) -> torch.Tensor:
        torchac = self._torchac()
        values = torchac.decode_float_cdf(cdf[selected].detach().cpu(), stream)
        return values.to(cdf.device, torch.float32) + minimum

    @torch.inference_mode()
    def compress(self, image: torch.Tensor) -> dict[str, Any]:
        x_l, x_h = self.model.Encoder(image)
        z_l, z_h = self.model.priorEncoder(x_l, x_h)
        z_l_hat, z_h_hat = torch.round(z_l), torch.round(z_h)

        z_l_meta = TensorMeta(tuple(z_l_hat.shape), int(z_l_hat.min()), int(z_l_hat.max()))
        z_h_meta = TensorMeta(tuple(z_h_hat.shape), int(z_h_hat.min()), int(z_h_hat.max()))
        z_l_cdf = _factorized_cdf(self.model.bitEstimator_z, z_l_meta, z_l.device)
        z_h_cdf = _factorized_cdf(self.model.bitEstimator_h, z_h_meta, z_h.device)
        z_l_stream, _ = self._encode_tensor(z_l_hat, z_l_cdf)
        z_h_stream, _ = self._encode_tensor(z_h_hat, z_h_cdf)

        hyper_l, hyper_h = self.model.priorDecoder(z_l_hat, z_h_hat)
        y_l_hat, y_h_hat = torch.round(x_l), torch.round(x_h)
        y_l_meta = TensorMeta(tuple(y_l_hat.shape), int(y_l_hat.min()), int(y_l_hat.max()))
        y_h_meta = TensorMeta(tuple(y_h_hat.shape), int(y_h_hat.min()), int(y_h_hat.max()))
        anchor_l = _checkerboard_mask(y_l_meta.shape, y_l_hat.device)
        anchor_h = _checkerboard_mask(y_h_meta.shape, y_h_hat.device)

        zero_l, zero_h = torch.zeros_like(y_l_hat), torch.zeros_like(y_h_hat)
        params_l, params_h = self.model.entropy(hyper_l, hyper_h, zero_l, zero_h)
        cdf_l = _mixture_cdf(*params_l, y_l_meta.minimum, y_l_meta.maximum)
        cdf_h = _mixture_cdf(*params_h, y_h_meta.minimum, y_h_meta.maximum)
        y_l_anchor_stream = self._encode_selected(y_l_hat, cdf_l, anchor_l, y_l_meta.minimum)
        y_h_anchor_stream = self._encode_selected(y_h_hat, cdf_h, anchor_h, y_h_meta.minimum)

        partial_l = torch.where(anchor_l, y_l_hat, zero_l)
        partial_h = torch.where(anchor_h, y_h_hat, zero_h)
        params_l, params_h = self.model.entropy(hyper_l, hyper_h, partial_l, partial_h)
        cdf_l = _mixture_cdf(*params_l, y_l_meta.minimum, y_l_meta.maximum)
        cdf_h = _mixture_cdf(*params_h, y_h_meta.minimum, y_h_meta.maximum)
        y_l_non_stream = self._encode_selected(y_l_hat, cdf_l, ~anchor_l, y_l_meta.minimum)
        y_h_non_stream = self._encode_selected(y_h_hat, cdf_h, ~anchor_h, y_h_meta.minimum)

        return {
            "strings": [
                z_l_stream, z_h_stream,
                y_l_anchor_stream, y_h_anchor_stream,
                y_l_non_stream, y_h_non_stream,
            ],
            "meta": {"z_l": z_l_meta, "z_h": z_h_meta, "y_l": y_l_meta, "y_h": y_h_meta},
            "debug_y_hat": (y_l_hat, y_h_hat),
        }

    @torch.inference_mode()
    def decompress(self, strings: list[bytes], meta: dict[str, TensorMeta]) -> dict[str, Any]:
        device = next(self.model.parameters()).device
        z_l_cdf = _factorized_cdf(self.model.bitEstimator_z, meta["z_l"], device)
        z_h_cdf = _factorized_cdf(self.model.bitEstimator_h, meta["z_h"], device)
        z_l_hat = self._decode_tensor(strings[0], z_l_cdf, meta["z_l"])
        z_h_hat = self._decode_tensor(strings[1], z_h_cdf, meta["z_h"])
        hyper_l, hyper_h = self.model.priorDecoder(z_l_hat, z_h_hat)

        y_l = torch.zeros(meta["y_l"].shape, device=device)
        y_h = torch.zeros(meta["y_h"].shape, device=device)
        anchor_l = _checkerboard_mask(meta["y_l"].shape, device)
        anchor_h = _checkerboard_mask(meta["y_h"].shape, device)
        params_l, params_h = self.model.entropy(hyper_l, hyper_h, y_l, y_h)
        cdf_l = _mixture_cdf(*params_l, meta["y_l"].minimum, meta["y_l"].maximum)
        cdf_h = _mixture_cdf(*params_h, meta["y_h"].minimum, meta["y_h"].maximum)
        y_l[anchor_l] = self._decode_selected(strings[2], cdf_l, anchor_l, meta["y_l"].minimum)
        y_h[anchor_h] = self._decode_selected(strings[3], cdf_h, anchor_h, meta["y_h"].minimum)

        params_l, params_h = self.model.entropy(hyper_l, hyper_h, y_l, y_h)
        cdf_l = _mixture_cdf(*params_l, meta["y_l"].minimum, meta["y_l"].maximum)
        cdf_h = _mixture_cdf(*params_h, meta["y_h"].minimum, meta["y_h"].maximum)
        y_l[~anchor_l] = self._decode_selected(strings[4], cdf_l, ~anchor_l, meta["y_l"].minimum)
        y_h[~anchor_h] = self._decode_selected(strings[5], cdf_h, ~anchor_h, meta["y_h"].minimum)
        reconstruction = self.model.Decoder(y_l, y_h).clamp_(0.0, 1.0)
        return {"x_hat": reconstruction, "debug_y_hat": (y_l, y_h)}

