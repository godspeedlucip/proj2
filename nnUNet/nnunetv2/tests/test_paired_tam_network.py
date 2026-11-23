import json

import pytest
import torch
from torch import nn

from nnunetv2.network_architecture.paired_tam import TAM3D, build_paired_network
from nnunetv2.utilities.get_network_from_plans import get_network_from_plans


PLAIN_UNET = "dynamic_network_architectures.architectures.unet.PlainConvUNet"
IMPORTS = ["conv_op", "norm_op", "nonlin"]


@pytest.fixture(scope="module", autouse=True)
def limit_cpu_threads():
    previous = torch.get_num_threads()
    torch.set_num_threads(1)
    yield
    torch.set_num_threads(previous)


@pytest.fixture
def architecture_kwargs():
    return {
        "n_stages": 6,
        "features_per_stage": [8, 8, 8, 8, 16, 24],
        "conv_op": "torch.nn.Conv3d",
        "kernel_sizes": [[3, 3, 3]] * 6,
        "strides": [[1, 1, 1], [2, 2, 2], [2, 2, 2], [2, 2, 2], [2, 1, 1], [1, 2, 1]],
        "n_conv_per_stage": 1,
        "n_conv_per_stage_decoder": 1,
        "conv_bias": True,
        "norm_op": "torch.nn.InstanceNorm3d",
        "norm_op_kwargs": {"affine": True},
        "nonlin": "torch.nn.LeakyReLU",
    }


@pytest.fixture
def paired_input():
    generator = torch.Generator().manual_seed(42)
    return torch.randn(2, 2, 1, 16, 16, 16, generator=generator)


def build(kwargs, preset, deep_supervision=False):
    return build_paired_network(PLAIN_UNET, kwargs, IMPORTS, 1, 3, deep_supervision, preset)


@pytest.mark.parametrize(
    "preset, stages, widths",
    [
        ("C1", [], {}),
        ("C3", [5], {"5": 24}),
        ("C4", [4, 5], {"4": 16, "5": 24}),
        ("C5", [3, 4, 5], {"3": 16, "4": 16, "5": 24}),
    ],
)
def test_preset_builds_only_its_stages_and_records_resolved_widths(architecture_kwargs, preset, stages, widths):
    architecture_kwargs["tam"] = {"embed_dim_by_stage": {"3": 16}}
    network = build(architecture_kwargs, preset)

    assert list(network.tam) == [str(stage) for stage in stages]
    assert network.tam_metadata == json.loads(json.dumps(network.tam_metadata)) == {
        "preset": preset,
        "stages": stages,
        "num_heads": 8,
        "embed_dim_by_stage": widths,
        "phase_order": ["phase1", "phase5"],
    }
    for stage, width in widths.items():
        assert network.tam[stage].query.out_channels == width
    assert architecture_kwargs["tam"] == {"embed_dim_by_stage": {"3": 16}}


def test_c5_stage_indices_follow_encoder_depth(architecture_kwargs):
    architecture_kwargs.update(n_stages=4, features_per_stage=[8] * 4)
    architecture_kwargs["strides"] = architecture_kwargs["strides"][:4]
    architecture_kwargs["kernel_sizes"] = architecture_kwargs["kernel_sizes"][:4]

    assert build(architecture_kwargs, "C5").tam_metadata["stages"] == [1, 2, 3]


def test_c1_matches_initialized_plain_unet_per_phase(architecture_kwargs, paired_input):
    torch.manual_seed(23)
    baseline = get_network_from_plans(PLAIN_UNET, architecture_kwargs, IMPORTS, 1, 3, deep_supervision=True).eval()
    torch.manual_seed(23)
    network = build(architecture_kwargs, "C1", deep_supervision=True).eval()

    assert network.state_dict().keys() == baseline.state_dict().keys()
    for name, weight in baseline.state_dict().items():
        torch.testing.assert_close(network.state_dict()[name], weight, atol=0, rtol=0)
    with torch.inference_mode():
        expected_by_phase = [baseline(paired_input[:, phase]) for phase in range(2)]
        actual = network(paired_input)
    for logits, logit_by_phase in zip(actual, zip(*expected_by_phase), strict=True):
        torch.testing.assert_close(logits, torch.stack(logit_by_phase, dim=1))


def test_c5_passes_augmented_stage_features_to_decoder_and_next_stage(architecture_kwargs, paired_input):
    torch.manual_seed(31)
    architecture_kwargs["tam"] = {"embed_dim_by_stage": {"3": 16}}
    network = build(architecture_kwargs, "C5", deep_supervision=True)
    tam_outputs, next_inputs, decoder_skips = {}, {}, []

    def capture_tam(stage):
        def hook(module, args, output):
            tam_outputs[stage] = output.flatten(0, 1)
        return hook

    def capture_next_input(stage):
        def hook(module, args):
            next_inputs[stage] = args[0]
        return hook

    def capture_decoder_input(module, args):
        decoder_skips.extend(args[0])

    hooks = [network.tam[str(stage)].register_forward_hook(capture_tam(stage)) for stage in (3, 4, 5)]
    hooks.extend(
        network.encoder.stages[stage + 1].register_forward_pre_hook(capture_next_input(stage))
        for stage in (3, 4)
    )
    hooks.append(network.decoder.register_forward_pre_hook(capture_decoder_input))
    try:
        logits = network(paired_input)
    finally:
        for hook in hooks:
            hook.remove()

    assert list(tam_outputs) == [3, 4, 5]
    for stage, output in tam_outputs.items():
        torch.testing.assert_close(decoder_skips[stage], output, atol=0, rtol=0)
        if stage in next_inputs:
            torch.testing.assert_close(next_inputs[stage], output, atol=0, rtol=0)
    assert [tuple(output.shape) for output in logits] == [
        (2, 2, 3, 16, 16, 16),
        (2, 2, 3, 8, 8, 8),
        (2, 2, 3, 4, 4, 4),
        (2, 2, 3, 2, 2, 2),
        (2, 2, 3, 1, 2, 2),
    ]
    sum(output.square().mean() for output in logits).backward()
    for stage in ("3", "4", "5"):
        for parameter in network.tam[stage].parameters():
            assert parameter.grad is not None and torch.isfinite(parameter.grad).all()
        assert torch.count_nonzero(network.tam[stage].query.weight.grad) > 0


def test_c5_eval_couples_phases_within_the_same_case(architecture_kwargs, paired_input):
    torch.manual_seed(37)
    network = build(architecture_kwargs, "C5").eval()
    modified_input = paired_input.clone()
    modified_input[0, 1] = torch.roll(modified_input[0, 1], shifts=3, dims=-1)

    with torch.inference_mode():
        original = network(paired_input)
        modified = network(modified_input)
        reversed_phases = network(paired_input.flip(1))

    assert original.shape == (2, 2, 3, 16, 16, 16)
    assert (original[0, 0] - modified[0, 0]).abs().max() > 1e-4
    torch.testing.assert_close(original[1], modified[1], atol=0, rtol=0)
    torch.testing.assert_close(original, reversed_phases.flip(1))


def test_attention_uses_reference_queries_target_keys_and_values_and_averages_references():
    tam = TAM3D(channels=2, embed_dim=2, num_heads=2).eval()

    def set_identity(conv):
        nn.init.dirac_(conv.weight)
        if conv.bias is not None:
            nn.init.zeros_(conv.bias)

    features = torch.tensor([[[1., 2.], [-2., 1.]], [[3., -1.], [2., 0.]], [[4., 1.], [-1., 2.]]])
    with torch.no_grad():
        for conv in (tam.query, tam.key, tam.value, tam.projection):
            set_identity(conv)
        tam.gate[0].weight.zero_()
        tam.gate[0].bias.zero_()
        tam.fusion[0].weight.zero_()
        tam.fusion[0].bias.zero_()
        for channel in range(2):
            tam.fusion[0].weight[channel, channel, 1, 1, 1] = 1
            tam.fusion[0].weight[channel, channel + 2, 1, 1, 1] = 1
        tam.fusion[2].bias.fill_(0.25)
        actual = tam(features.reshape(1, 3, 2, 1, 1, 2)).reshape_as(features)

    expected = []
    for target_index, target in enumerate(features):
        contributions = []
        for reference_index, reference in enumerate(features):
            if reference_index != target_index:
                weights = (reference.unsqueeze(-1) * target.unsqueeze(-2)).softmax(dim=-1)
                attention = (weights * target.unsqueeze(-2)).sum(dim=-1)
                normalized_fusion = (target + attention / 2).relu() / (1 + tam.fusion[2].eps) ** 0.5 + 0.25
                contributions.append(normalized_fusion)
        expected.append(torch.stack(contributions).mean(dim=0))

    torch.testing.assert_close(actual, torch.stack(expected))


@pytest.mark.parametrize(
    "tam",
    [
        {"embed_dim_by_stage": {"4": 10}},
        {"embed_dim_by_stage": {"2": 16}},
        {"embed_dim_by_stage": {"3": 0}},
        {"num_heads": 4},
        {"embed_dim_by_stage": {3: 16, "3": 8}},
    ],
)
def test_invalid_tam_configuration_is_rejected(architecture_kwargs, tam):
    architecture_kwargs["tam"] = tam

    with pytest.raises(ValueError, match="TAM|embed_dim_by_stage"):
        build(architecture_kwargs, "C5")


def test_width_must_be_divisible_by_eight_at_active_stage(architecture_kwargs):
    architecture_kwargs["features_per_stage"][5] = 12

    with pytest.raises(ValueError, match="stage 5.*8"):
        build(architecture_kwargs, "C3")
    assert build(architecture_kwargs, "C1").tam_metadata["stages"] == []


def test_builder_rejects_invalid_task_or_architecture(architecture_kwargs):
    with pytest.raises(ValueError, match="one input channel"):
        build_paired_network(PLAIN_UNET, architecture_kwargs, IMPORTS, 2, 3, False, "C1")
    with pytest.raises(ValueError, match="PlainConvUNet"):
        build_paired_network(
            PLAIN_UNET.replace("PlainConv", "ResidualEncoder"), architecture_kwargs, IMPORTS, 1, 3, False, "C5"
        )
    with pytest.raises(ValueError, match="preset"):
        build(architecture_kwargs, "C2")
    architecture_kwargs["conv_op"] = "torch.nn.Conv2d"
    with pytest.raises(ValueError, match="Conv3d"):
        build(architecture_kwargs, "C1")


def test_paired_network_rejects_input_missing_the_two_phase_axis(architecture_kwargs, paired_input):
    network = build(architecture_kwargs, "C1")

    with pytest.raises(ValueError, match="B, 2, 1, D, H, W"):
        network(paired_input[:, 0])
    with pytest.raises(ValueError, match="B, 2, 1, D, H, W"):
        network(paired_input[:, :1])
