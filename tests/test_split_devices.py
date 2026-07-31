from __future__ import annotations

import unittest
import warnings
from unittest import mock

import torch

import ideogram4.pipeline_ideogram4 as pipeline_module
from ideogram4.constants import LLM_TOKEN_INDICATOR, OUTPUT_IMAGE_INDICATOR
from ideogram4.pipeline_ideogram4 import (
  Ideogram4Pipeline,
  Ideogram4PipelineConfig,
  _append_image_feature_zeros,
  _move_tensor_to_device,
)


def _bare_pipeline(
  *,
  diffusion_device: torch.device | None = None,
  text_device: torch.device | None = None,
) -> Ideogram4Pipeline:
  diffusion_device = diffusion_device or torch.device("cpu")
  text_device = text_device or torch.device("cpu")
  pipeline = Ideogram4Pipeline.__new__(Ideogram4Pipeline)
  pipeline.config = Ideogram4PipelineConfig()
  pipeline.diffusion_device = diffusion_device
  pipeline.text_device = text_device
  pipeline.device = diffusion_device
  return pipeline


class SplitDeviceTests(unittest.TestCase):
  def test_constructor_defaults_text_device_to_diffusion_device(self) -> None:
    device = torch.device("cuda:0")
    shift = mock.Mock()
    scale = mock.Mock()
    with (
      mock.patch.object(
        pipeline_module,
        "get_latent_norm",
        return_value=(shift, scale),
      ),
      mock.patch.object(pipeline_module, "CaptionVerifier"),
    ):
      pipeline = Ideogram4Pipeline(
        conditional_transformer=mock.Mock(),
        unconditional_transformer=mock.Mock(),
        text_encoder=mock.Mock(),
        text_tokenizer=mock.Mock(),
        autoencoder=mock.Mock(),
        config=Ideogram4PipelineConfig(),
        device=device,
        dtype=torch.bfloat16,
      )

    self.assertEqual(pipeline.device, device)
    self.assertEqual(pipeline.diffusion_device, device)
    self.assertEqual(pipeline.text_device, device)
    shift.to.assert_called_once_with(device)
    scale.to.assert_called_once_with(device)

  def test_from_pretrained_routes_components_to_role_devices(self) -> None:
    diffusion_device = torch.device("cpu")
    text_device = torch.device("meta")
    conditional = mock.Mock()
    unconditional = mock.Mock()

    with (
      mock.patch.object(
        pipeline_module,
        "_load_indexed_or_single_state_dict",
        side_effect=[{}, {}],
      ),
      mock.patch.object(pipeline_module, "hf_hub_download", return_value="vae"),
      mock.patch.object(
        pipeline_module,
        "_build_transformer",
        side_effect=[conditional, unconditional],
      ) as build_transformer,
      mock.patch.object(
        pipeline_module,
        "_load_qwen3_vl",
        return_value=(mock.Mock(), mock.Mock()),
      ) as load_qwen,
      mock.patch.object(
        pipeline_module, "_load_autoencoder", return_value=mock.Mock()
      ) as load_autoencoder,
      mock.patch.object(
        pipeline_module,
        "get_latent_norm",
        return_value=(torch.zeros(1), torch.ones(1)),
      ),
      mock.patch.object(pipeline_module, "CaptionVerifier"),
      mock.patch.object(pipeline_module, "_log_device_memory"),
    ):
      pipeline = Ideogram4Pipeline.from_pretrained(
        device=diffusion_device,
        text_device=text_device,
      )

    self.assertEqual(pipeline.diffusion_device, diffusion_device)
    self.assertEqual(pipeline.text_device, text_device)
    self.assertEqual(build_transformer.call_args_list[0].args[2], diffusion_device)
    self.assertEqual(build_transformer.call_args_list[1].args[2], diffusion_device)
    self.assertEqual(load_qwen.call_args.args[1], text_device)
    self.assertEqual(load_autoencoder.call_args.args[1], diffusion_device)

  def test_build_inputs_assigns_tensors_to_role_devices(self) -> None:
    pipeline = _bare_pipeline(text_device=torch.device("meta"))
    pipeline._tokenize = mock.Mock(return_value=(torch.tensor([7, 8]), 2))

    inputs = pipeline._build_inputs(["prompt"], height=16, width=16)

    for key in ("token_ids", "text_position_ids", "text_indicator"):
      self.assertEqual(inputs[key].device, torch.device("meta"))
    for key in ("position_ids", "segment_ids", "indicator"):
      self.assertEqual(inputs[key].device, torch.device("cpu"))

  def test_encode_text_trims_trailing_image_positions(self) -> None:
    pipeline = _bare_pipeline()
    recorded: dict[str, int] = {}

    def fake_embeddings(token_ids, attention_mask, pos_2d):
      recorded["sequence_length"] = token_ids.shape[1]
      hidden = torch.ones(token_ids.shape[0], token_ids.shape[1], 2)
      return [hidden for _ in pipeline_module.QWEN3_VL_ACTIVATION_LAYERS]

    pipeline._get_qwen3_vl_embeddings = fake_embeddings
    max_text_tokens = 37
    total_length = max_text_tokens + 4096
    token_ids = torch.zeros(1, total_length, dtype=torch.long)
    position_ids = torch.zeros(1, total_length, 3, dtype=torch.long)
    text_indicator = torch.zeros(1, total_length, dtype=torch.long)
    text_indicator[:, :max_text_tokens] = LLM_TOKEN_INDICATOR

    features = pipeline._encode_text(
      token_ids, position_ids, text_indicator, max_text_tokens
    )

    self.assertEqual(recorded["sequence_length"], max_text_tokens)
    self.assertEqual(features.shape[:2], (1, max_text_tokens))

  def test_append_image_feature_zeros_restores_full_shape(self) -> None:
    text_features = torch.randn(2, 5, 7)
    features = _append_image_feature_zeros(
      text_features, num_image_tokens=11, destination=torch.device("cpu")
    )

    self.assertEqual(features.shape, (2, 16, 7))
    torch.testing.assert_close(features[:, :5], text_features)
    self.assertEqual(torch.count_nonzero(features[:, 5:]).item(), 0)

  def test_left_padding_stays_in_common_text_block(self) -> None:
    pipeline = _bare_pipeline()
    tokenized = {
      "short": (torch.tensor([11, 12]), 2),
      "long": (torch.tensor([21, 22, 23, 24]), 4),
    }
    pipeline._tokenize = mock.Mock(side_effect=lambda prompt: tokenized[prompt])

    inputs = pipeline._build_inputs(["short", "long"], height=16, width=32)
    max_text_tokens = inputs["max_text_tokens"]
    self.assertEqual(max_text_tokens, 4)
    torch.testing.assert_close(inputs["token_ids"][0, :4], torch.tensor([0, 0, 11, 12]))
    torch.testing.assert_close(
      inputs["token_ids"][1, :4], torch.tensor([21, 22, 23, 24])
    )
    torch.testing.assert_close(
      inputs["text_indicator"][0, :4],
      torch.tensor([0, 0, LLM_TOKEN_INDICATOR, LLM_TOKEN_INDICATOR]),
    )
    self.assertTrue(
      torch.all(inputs["indicator"][:, max_text_tokens:] == OUTPUT_IMAGE_INDICATOR)
    )

  def test_copy_fallback_stages_through_cpu(self) -> None:
    destination = torch.device("cuda:1")

    class FakeTensor:
      def __init__(self, device: str, *, fail_direct: bool = False):
        self.device = torch.device(device)
        self.fail_direct = fail_direct

      def to(self, device):
        resolved = torch.device(device)
        if self.fail_direct and resolved == destination:
          raise RuntimeError("peer copy unavailable")
        return FakeTensor(str(resolved))

    source = FakeTensor("cuda:0", fail_direct=True)
    with warnings.catch_warnings(record=True) as caught:
      moved = _move_tensor_to_device(source, destination)

    self.assertEqual(moved.device, destination)
    self.assertEqual(len(caught), 1)
    self.assertIn("staging through CPU", str(caught[0].message))

  def test_copy_fallback_does_not_swallow_staging_errors(self) -> None:
    destination = torch.device("cuda:1")

    class BrokenTensor:
      device = torch.device("cuda:0")

      def to(self, device):
        if torch.device(device) == destination:
          raise RuntimeError("peer copy unavailable")
        raise ValueError("CPU staging failed")

    with (
      self.assertRaisesRegex(ValueError, "CPU staging failed"),
      warnings.catch_warnings(),
    ):
      warnings.simplefilter("ignore")
      _move_tensor_to_device(BrokenTensor(), destination)


if __name__ == "__main__":
  unittest.main()
