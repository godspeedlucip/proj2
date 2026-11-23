from collections.abc import Mapping

import torch
from dynamic_network_architectures.architectures.unet import PlainConvUNet
from torch import Tensor, nn
from torch.nn.functional import scaled_dot_product_attention

from nnunetv2.utilities.get_network_from_plans import get_network_from_plans
from nnunetv2.utilities.paired_tam_contract import PHASES, validate_context_phases


_ATTENTION_STAGE_COUNT = {"C1": 0, "C3": 1, "C4": 2, "C5": 3}
_NUM_HEADS = 8


class TAM3D(nn.Module):
    """Fuse ordered temporal references independently for each case."""

    def __init__(self, channels: int, embed_dim: int, num_heads: int = _NUM_HEADS):
        super().__init__()
        if num_heads <= 0 or embed_dim <= 0 or embed_dim % num_heads != 0:
            raise ValueError("TAM embedding width must be positive and divisible by the number of heads")
        self.embed_dim = embed_dim
        self.num_heads = num_heads
        self.head_dim = embed_dim // num_heads
        self.query = nn.Conv3d(channels, embed_dim, kernel_size=1)
        self.key = nn.Conv3d(channels, embed_dim, kernel_size=1)
        self.value = nn.Conv3d(channels, embed_dim, kernel_size=1)
        self.gate = nn.Sequential(nn.Conv3d(embed_dim, embed_dim, kernel_size=1), nn.Sigmoid())
        self.fusion = nn.Sequential(
            nn.Conv3d(embed_dim + channels, embed_dim, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.BatchNorm3d(embed_dim),
        )
        self.projection = nn.Conv3d(embed_dim, channels, kernel_size=1)

    def _attention_tokens(self, projected: Tensor, batch_size: int, phases: int) -> Tensor:
        # SDPA's fused backends require a contiguous head dimension.
        tokens = projected.flatten(2).transpose(1, 2)
        tokens = tokens.reshape(batch_size, phases, -1, self.num_heads, self.head_dim)
        return tokens.permute(0, 1, 3, 2, 4).contiguous()

    def forward(self, features: Tensor) -> Tensor:
        batch_size, phases, _, depth, height, width = features.shape
        if phases < 2:
            raise ValueError("TAM requires at least two temporal frames")
        flattened = features.flatten(0, 1)
        queries = self._attention_tokens(self.query(flattened), batch_size, phases)
        keys = self._attention_tokens(self.key(flattened), batch_size, phases)
        values = self._attention_tokens(self.value(flattened), batch_size, phases)

        outputs = []
        for target in range(phases):
            contributions = []
            for reference in range(phases):
                if reference == target:
                    continue
                attended = scaled_dot_product_attention(
                    queries[:, reference], keys[:, target], values[:, target]
                )
                attended = attended.transpose(-1, -2).reshape(batch_size, self.embed_dim, depth, height, width)
                gated = attended * self.gate(attended)
                contributions.append(self.fusion(torch.cat((gated, features[:, target]), dim=1)))
            outputs.append(torch.stack(contributions).mean(dim=0))
        fused = torch.stack(outputs, dim=1).flatten(0, 1)
        return self.projection(fused).unflatten(0, (batch_size, phases))


def _get_embedding_overrides(tam_kwargs, n_stages):
    if not isinstance(tam_kwargs, Mapping) or set(tam_kwargs) - {"embed_dim_by_stage", "context_phases"}:
        raise ValueError("TAM plans accept an embed_dim_by_stage mapping and context_phases")
    overrides = tam_kwargs.get("embed_dim_by_stage", {})
    if not isinstance(overrides, Mapping):
        raise ValueError("TAM embed_dim_by_stage must map stage indices to embedding widths")

    # A single plan can carry overrides for the full ablation, including in C1.
    c5_stage_count = _ATTENTION_STAGE_COUNT["C5"]
    eligible = {str(stage) for stage in range(max(0, n_stages - c5_stage_count), n_stages)}
    resolved = {}
    for stage, width in overrides.items():
        stage = str(stage)
        if stage not in eligible or stage in resolved:
            raise ValueError(f"Invalid or duplicate TAM embed_dim_by_stage index {stage}")
        if type(width) is not int or width <= 0 or width % _NUM_HEADS != 0:
            raise ValueError(f"TAM embedding width for stage {stage} must be a positive multiple of {_NUM_HEADS}")
        resolved[stage] = width
    return resolved


class PairedPlainConvUNet(PlainConvUNet):
    def __init__(self, input_channels, num_classes, tam_preset, tam_kwargs, **architecture_kwargs):
        if input_channels != 1:
            raise ValueError("Paired TAM requires one input channel per phase")
        if architecture_kwargs.get("conv_op") is not nn.Conv3d:
            raise ValueError("Paired TAM requires a Conv3d PlainConvUNet")
        if tam_preset not in _ATTENTION_STAGE_COUNT:
            raise ValueError(f"Unknown paired TAM preset {tam_preset!r}. Expected C1, C3, C4, or C5")

        n_stages = architecture_kwargs["n_stages"]
        stage_count = _ATTENTION_STAGE_COUNT[tam_preset]
        if n_stages < max(2, stage_count):
            raise ValueError(f"PlainConvUNet has too few encoder stages for paired TAM preset {tam_preset}")
        overrides = _get_embedding_overrides(tam_kwargs, n_stages)
        context_phases = validate_context_phases(tam_kwargs.get("context_phases", ()))
        super().__init__(input_channels=input_channels, num_classes=num_classes, **architecture_kwargs)
        self.input_phase_count = len(PHASES) + bool(context_phases)
        # Initialize the shared backbone before drawing any TAM parameters.
        self.apply(self.initialize)

        stages = list(range(n_stages - stage_count, n_stages))
        embedding_dims = {}
        self.tam = nn.ModuleDict()
        for stage in stages:
            stage_key = str(stage)
            channels = self.encoder.output_channels[stage]
            embed_dim = overrides.get(stage_key, channels)
            if embed_dim <= 0 or embed_dim % _NUM_HEADS != 0:
                raise ValueError(f"TAM embedding width for stage {stage} must be a positive multiple of {_NUM_HEADS}")
            self.tam[stage_key] = TAM3D(channels, embed_dim)
            embedding_dims[stage_key] = int(embed_dim)
        self.tam.apply(self.initialize)
        self.tam_metadata = {
            "preset": tam_preset,
            "stages": stages,
            "num_heads": _NUM_HEADS,
            "embed_dim_by_stage": embedding_dims,
            "phase_order": list(PHASES),
        }
        if context_phases:
            self.tam_metadata["context_phases"] = list(context_phases)

    def forward(self, x: Tensor) -> Tensor | list[Tensor]:
        if x.ndim != 6 or x.shape[1:3] != (self.input_phase_count, 1):
            raise ValueError(f"Paired TAM expects [B, {self.input_phase_count}, 1, D, H, W], got {tuple(x.shape)}")
        case_and_phase = x.shape[:2]
        x = x.flatten(0, 1)
        skips = []
        for stage_index, stage in enumerate(self.encoder.stages):
            x = stage(x)
            stage_key = str(stage_index)
            if stage_key in self.tam:
                x = self.tam[stage_key](x.unflatten(0, case_and_phase)).flatten(0, 1)
            skips.append(x)

        logits = self.decoder(skips)
        if self.decoder.deep_supervision:
            return [output.unflatten(0, case_and_phase) for output in logits]
        return logits.unflatten(0, case_and_phase)


def build_paired_network(
    architecture_class_name,
    arch_init_kwargs,
    arch_init_kwargs_req_import,
    num_input_channels,
    num_output_channels,
    enable_deep_supervision,
    preset,
) -> PairedPlainConvUNet:
    """Build the plans-defined spatial backbone and the preset's attention stages."""
    if architecture_class_name != f"{PlainConvUNet.__module__}.{PlainConvUNet.__name__}":
        raise ValueError("Paired TAM requires a PlainConvUNet plans architecture")
    kwargs = dict(arch_init_kwargs)
    kwargs["tam_kwargs"] = kwargs.pop("tam", {})
    kwargs["tam_preset"] = preset
    return get_network_from_plans(
        f"{PairedPlainConvUNet.__module__}.{PairedPlainConvUNet.__name__}",
        kwargs,
        arch_init_kwargs_req_import,
        num_input_channels,
        num_output_channels,
        allow_init=False,
        deep_supervision=enable_deep_supervision,
    )
