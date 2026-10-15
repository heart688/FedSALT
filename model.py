"""Standalone forecasting backbone and local adaptation used by FedSALT.

Inputs are city-standardized speed histories, not raw physical speeds.
No dataset loader, graph, federated runtime, or compression codec is required.
"""
from __future__ import annotations

import math
from typing import Mapping

import torch
from torch import Tensor, nn
from torch.nn import functional as F


class FeedForward(nn.Module):
    def __init__(self, dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(dim, ff_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(ff_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x: Tensor) -> Tensor:
        return self.net(x)


class FactorLinear(nn.Module):
    """Frozen base projection plus B @ A; both factors are locally trainable."""

    def __init__(self, base: nn.Linear, rank: int):
        super().__init__()
        self.base = base
        self.A = nn.Parameter(torch.randn(rank, base.in_features) / math.sqrt(base.in_features))
        self.B = nn.Parameter(torch.zeros(base.out_features, rank))

    def forward(self, x: Tensor) -> Tensor:
        return self.base(x) + F.linear(F.linear(x, self.A), self.B)


class ZeroCalendar(nn.Module):
    """Parameter-free placeholder retaining the original module structure."""

    def __init__(self, dim: int):
        super().__init__()
        self.dim = dim

    def forward(self, x: Tensor) -> Tensor:
        return x.new_zeros(x.shape[0], self.dim)


class KernelBlock(nn.Module):
    def __init__(self, dim: int, routing_dim: int, ff_dim: int, dropout: float):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.value = nn.Linear(dim, dim)
        self.out = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.ffn = FeedForward(dim, ff_dim, dropout)
        self.drop = nn.Dropout(dropout)
        self.spatial_scale = nn.Parameter(torch.tensor(-2.0))
        self.kernel_query = nn.Linear(dim, routing_dim)
        self.kernel_key = nn.Linear(dim, routing_dim)

    def forward(self, x: Tensor, modulation: tuple[Tensor, Tensor] | None = None) -> Tensor:
        z = self.norm1(x)
        if modulation is not None:
            scale, shift = modulation
            z = z * (1 + scale) + shift
        q = F.elu(self.kernel_query(z)) + 1
        k = F.elu(self.kernel_key(z)) + 1
        moment = torch.einsum("bnr,bnd->brd", k, self.value(z))
        mass = k.sum(1)
        context = torch.einsum("bnr,brd->bnd", q, moment) / torch.einsum(
            "bnr,br->bn", q, mass
        ).unsqueeze(-1).clamp_min(1e-6)
        x = x + torch.sigmoid(self.spatial_scale) * self.drop(self.out(context))
        return x + self.ffn(self.norm2(x))


class KernelNetwork(nn.Module):
    def __init__(self, history: int, horizon: int, dim: int, routing_dim: int,
                 ff_dim: int, layers: int, dropout: float):
        super().__init__()
        self.history_embed = nn.Sequential(nn.Linear(history, dim), nn.GELU(), nn.LayerNorm(dim))
        self.calendar_embed = ZeroCalendar(dim)
        self.blocks = nn.ModuleList([
            KernelBlock(dim, routing_dim, ff_dim, dropout) for _ in range(layers)
        ])
        self.final_norm = nn.LayerNorm(dim)
        self.head = nn.Linear(dim, horizon)


def _speed_history(x: Tensor, history: int) -> Tensor:
    """Accept [B,N,L] or the original [B,N,L,2] compatibility layout."""
    if not x.is_floating_point():
        raise ValueError("Input must be a floating-point tensor.")
    if x.ndim == 3 and x.shape[-1] == history:
        speed = x
    elif x.ndim == 4 and tuple(x.shape[-2:]) == (history, 2):
        # Preserve the original transpose/cat materialization and reduction layout.
        data = x.transpose(1, 2)
        data = torch.cat((data, torch.zeros_like(data[..., :1])), -1)
        speed = data[..., 0].transpose(1, 2)
    else:
        raise ValueError(f"Expected [B,N,{history}] or [B,N,{history},2].")
    if min(speed.shape[:2]) < 1:
        raise ValueError("Batch and sensor dimensions must be nonempty.")
    return speed


def state_features(history: Tensor) -> Tensor:
    """Recent mean, population std, last speed, short slope, and full-window slope."""
    return torch.stack((
        history.mean(-1), history.std(-1, unbiased=False), history[..., -1],
        (history[..., -1] - history[..., -3]) / 2,
        (history[..., -1] - history[..., 0]) / (history.shape[-1] - 1),
    ), -1)


class TrafficBackbone(nn.Module):
    """Source-stage predictor, without low-rank factors or local conditioner.

    Output [B,N,H] has the same city-standardized units as the speed input.
    Window normalization is inverted inside forward; city normalization is not.
    """

    def __init__(self, history: int = 12, horizon: int = 6, dim: int = 128,
                 routing_dim: int = 16, ff_dim: int = 256, layers: int = 3,
                 dropout: float = 0.1):
        super().__init__()
        if min(history, horizon, dim, routing_dim, ff_dim, layers) < 1:
            raise ValueError("Model dimensions must be positive.")
        if history < 3:
            raise ValueError("At least three history steps are required.")
        self.history = history
        self.network = KernelNetwork(history, horizon, dim, routing_dim, ff_dim, layers, dropout)

    def forward(self, x: Tensor, conditioner: nn.Module | None = None) -> Tensor:
        speed = _speed_history(x, self.history)
        mean = speed.mean(-1, keepdim=True)
        std = speed.std(-1, keepdim=True, unbiased=False).clamp_min(1e-4)
        normalized = (speed - mean) / std
        hidden = self.network.history_embed(normalized)
        hidden = hidden + hidden.new_zeros(hidden.shape[0], 1, hidden.shape[-1])
        state = state_features(speed) if conditioner is not None else None
        for block in self.network.blocks:
            # One local conditioner is reused across all blocks.
            modulation = None if conditioner is None else conditioner(state).chunk(2, -1)
            hidden = block(hidden, modulation)
        return self.network.head(self.network.final_norm(hidden)) * std + mean


class FedSALT(nn.Module):
    """Selective routing adapters and optional final-local traffic conditioning.

    Default dimensions match the final forecasting model. Parameter names retain
    the research checkpoint layout: base.network.* and personalization.film.*.
    Random construction is not a pretrained model or a reproduction of a table.
    """

    def __init__(self, local_adaptation: bool = True, history: int = 12,
                 horizon: int = 6, dim: int = 128, routing_dim: int = 16,
                 ff_dim: int = 256, layers: int = 3, rank: int = 4,
                 dropout: float = 0.1):
        super().__init__()
        if rank < 1:
            raise ValueError("Adapter rank must be positive.")
        self.base = TrafficBackbone(history, horizon, dim, routing_dim, ff_dim, layers, dropout)
        for block in self.base.network.blocks:
            block.kernel_query = FactorLinear(block.kernel_query, rank)
            block.kernel_key = FactorLinear(block.kernel_key, rank)
        self.personalization = nn.ModuleDict()
        if local_adaptation:
            self.personalization["film"] = nn.Sequential(
                nn.Linear(5, 16), nn.GELU(), nn.Linear(16, 2 * dim)
            )
            nn.init.zeros_(self.personalization["film"][-1].weight)
            nn.init.zeros_(self.personalization["film"][-1].bias)
        self.configure_stage("local" if local_adaptation else "collaboration")

    def forward(self, x: Tensor) -> Tensor:
        conditioner = self.personalization["film"] if "film" in self.personalization else None
        return self.base(x, conditioner)

    def configure_stage(self, stage: str) -> None:
        """Apply the collaboration or final-local trainable mask.

        Source pretraining uses TrafficBackbone instead. Insert the local
        conditioner after collaboration, as in the original training schedule.
        """
        if stage not in {"collaboration", "local"}:
            raise ValueError("Stage must be 'collaboration' or 'local'.")
        if stage == "collaboration" and "film" in self.personalization:
            raise ValueError("Use local_adaptation=False during collaboration.")
        for name, parameter in self.named_parameters():
            frozen = ((".kernel_query.base." in name or ".kernel_key.base." in name)
                      or (stage == "local" and (".out." in name or name.endswith(".spatial_scale"))))
            parameter.requires_grad_(not frozen)

    def shared_parameter_names(self) -> tuple[str, ...]:
        """Only output-side Q/K factors B belong to the target sharing interface."""
        return tuple(name for name, _ in self.named_parameters()
                     if name.endswith(".B") and (".kernel_query." in name or ".kernel_key." in name))

    def load_source_backbone(self, state: Mapping[str, Tensor]) -> None:
        """Install a TrafficBackbone state and reset Q/K adapter updates to zero.

        Source keys are network.*; the six Q/K weight/bias pairs are mapped to
        their wrapped base projections. Existing local factors A are retained.
        """
        current = self.base.state_dict()
        expected = set()
        for name in current:
            if name.endswith((".A", ".B")):
                continue
            expected.add(name.replace(".kernel_query.base.", ".kernel_query.")
                         .replace(".kernel_key.base.", ".kernel_key."))
        if set(state) != expected:
            raise ValueError("Source backbone keys do not match this model configuration.")
        for name in current:
            if name.endswith(".A"):
                continue
            if name.endswith(".B"):
                current[name] = torch.zeros_like(current[name])
            else:
                source_name = name.replace(".kernel_query.base.", ".kernel_query.")
                source_name = source_name.replace(".kernel_key.base.", ".kernel_key.")
                current[name] = state[source_name]
        self.base.load_state_dict(current, strict=True)

    def capture_value_reference(self) -> dict[str, Tensor]:
        """Call once after collaboration, before local optimization.

        References are detached clones, deliberately outside model state_dict.
        Save them separately if resuming local adaptation. Capture after moving
        the model to its intended device/dtype.
        """
        return {name: parameter.detach().clone() for name, parameter in self.named_parameters()
                if name.startswith("base.network.blocks.") and ".value." in name}

    def value_anchor_penalty(self, reference: Mapping[str, Tensor], fit_std: float,
                             coefficient: float = 0.05) -> Tensor:
        """Block-relative value-weight+bias anchor, in physical speed units.

        Add this term to MAE computed after city inverse-standardization.
        fit_std is a positive physical-speed standard deviation fitted on the
        designated target support prefix, not on forecast targets or test data.
        """
        if not math.isfinite(float(fit_std)) or float(fit_std) <= 0:
            raise ValueError("fit_std must be positive and finite.")
        if not math.isfinite(coefficient) or coefficient < 0:
            raise ValueError("coefficient must be nonnegative and finite.")
        parameters = dict(self.named_parameters())
        names = tuple(name for name in parameters
                      if name.startswith("base.network.blocks.") and ".value." in name)
        if set(reference) != set(names):
            raise ValueError("Reference must contain exactly the value weights and biases.")
        for name in names:
            ref, live = reference[name], parameters[name]
            if (ref.requires_grad or ref.shape != live.shape or ref.dtype != live.dtype
                    or ref.device != live.device or ref.data_ptr() == live.data_ptr()):
                raise ValueError("Reference must be detached, independent, and match the model.")
        ratios = []
        for i in range(len(self.base.network.blocks)):
            block_names = [name for name in names if name.startswith(f"base.network.blocks.{i}.value.")]
            numerator = sum((parameters[name] - reference[name]).square().sum() for name in block_names)
            denominator = sum(reference[name].square().sum() for name in block_names).clamp_min(1e-8)
            ratios.append(numerator / denominator)
        return coefficient * float(fit_std) * torch.stack(ratios).mean()


__all__ = ["TrafficBackbone", "FedSALT", "state_features"]
