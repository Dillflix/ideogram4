from __future__ import annotations

import json
import time
import warnings
from collections.abc import Sequence
from dataclasses import dataclass
from posixpath import dirname as _posix_dirname
from posixpath import join as _posix_join
from typing import Optional

import torch
from huggingface_hub import hf_hub_download
from huggingface_hub.errors import EntryNotFoundError
from PIL import Image
from safetensors.torch import load_file
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.masking_utils import create_causal_mask

from ideogram4.autoencoder import (
  AutoEncoder,
  AutoEncoderParams,
  convert_diffusers_state_dict,
)
from ideogram4.caption_verifier import CaptionVerifier
from ideogram4.constants import (
  IMAGE_POSITION_OFFSET,
  LLM_TOKEN_INDICATOR,
  OUTPUT_IMAGE_INDICATOR,
  QWEN3_VL_ACTIVATION_LAYERS,
  SEQUENCE_PADDING_INDICATOR,
)
from ideogram4.latent_norm import get_latent_norm
from ideogram4.modeling_ideogram4 import Ideogram4Config, Ideogram4Transformer
from ideogram4.quantized_loading import (
  FP8_TEXT_ENCODER_CONFIG_FLAG,
  is_bnb4bit_state_dict,
  is_fp8_state_dict,
  load_bnb4bit_state_dict,
  load_fp8_state_dict,
  swap_linears_to_bnb4bit,
  swap_linears_to_fp8,
)
from ideogram4.scheduler import (
  LogitNormalSchedule,
  get_schedule_for_resolution,
  make_step_intervals,
)


def _device_name(device: torch.device) -> str:
  """Return a useful device name without allowing diagnostics to fail."""
  try:
    if device.type == "cuda" and device.index is not None:
      return torch.cuda.get_device_name(device.index)
  except Exception:  # noqa: BLE001, S110 - diagnostics must never fail inference
    pass
  return str(device)


def _device_description(device: torch.device) -> str:
  return f"{device} ({_device_name(device)})"


def _synchronize_device(device: torch.device) -> None:
  try:
    if device.type == "cuda":
      torch.cuda.synchronize(device)
  except Exception:  # noqa: BLE001, S110 - timing must never fail inference
    pass


def _log_device_memory(devices: Sequence[torch.device]) -> None:
  seen: set[str] = set()
  for device in devices:
    key = str(device)
    if key in seen:
      continue
    seen.add(key)
    try:
      if device.type != "cuda":
        continue
      allocated = torch.cuda.memory_allocated(device) / 1024**2
      reserved = torch.cuda.memory_reserved(device) / 1024**2
      print(
        f"Ideogram4 memory: device={_device_description(device)}, "
        f"allocated={allocated:.1f} MiB, reserved={reserved:.1f} MiB",
        flush=True,
      )
    except Exception as exc:  # noqa: BLE001 - diagnostics must never fail inference
      warnings.warn(
        f"Unable to read Ideogram4 memory diagnostics for {device}: {exc}",
        stacklevel=2,
      )


def _validate_device_configuration(
  diffusion_device: torch.device, text_device: torch.device
) -> None:
  if diffusion_device == text_device:
    return
  if diffusion_device.type == "cuda" and text_device.type == "cuda":
    if diffusion_device.index is None or text_device.index is None:
      raise ValueError(
        "Separate Ideogram4 CUDA/ROCm devices must use explicit indexes; got "
        f"diffusion_device={diffusion_device}, text_device={text_device}"
      )
    try:
      device_count = torch.cuda.device_count()
    except Exception as exc:
      raise RuntimeError(
        "Unable to inspect CUDA/ROCm devices for split Ideogram4 placement: "
        f"diffusion_device={diffusion_device}, text_device={text_device}"
      ) from exc
    highest_index = max(diffusion_device.index, text_device.index)
    if device_count <= highest_index:
      raise RuntimeError(
        "Split Ideogram4 placement requested unavailable CUDA/ROCm devices: "
        f"diffusion_device={diffusion_device}, text_device={text_device}, "
        f"visible_device_count={device_count}"
      )


def _move_tensor_to_device(
  tensor: torch.Tensor, destination: torch.device
) -> torch.Tensor:
  """Move one tensor directly, with a CPU-staged fallback for GPU peers."""
  if tensor.device == destination:
    return tensor
  try:
    return tensor.to(destination)
  except RuntimeError as direct_error:
    warnings.warn(
      f"Direct tensor transfer {tensor.device} -> {destination} failed; "
      f"staging through CPU. {direct_error}",
      stacklevel=2,
    )
    return tensor.to("cpu").to(destination)


def _append_image_feature_zeros(
  text_features: torch.Tensor,
  num_image_tokens: int,
  destination: torch.device,
) -> torch.Tensor:
  """Rebuild full diffusion conditioning after compact text encoding."""
  batch_size, _, feature_dim = text_features.shape
  image_feature_zeros = torch.zeros(
    batch_size,
    num_image_tokens,
    feature_dim,
    dtype=text_features.dtype,
    device=destination,
  )
  return torch.cat([text_features, image_feature_zeros], dim=1)


def _load_subfolder_state_dict(
  repo_id: str, subfolder: str, basename: str
) -> dict[str, torch.Tensor]:
  """Download a component's weights, whether sharded (index) or a single file.

  ``basename`` is the safetensors stem (``model`` for transformers components,
  ``diffusion_pytorch_model`` for diffusers ones).
  """
  prefix = f"{subfolder}/" if subfolder else ""
  index_filename = f"{prefix}{basename}.safetensors.index.json"
  try:
    return _load_sharded_state_dict(repo_id, index_filename)
  except EntryNotFoundError:
    single_path = hf_hub_download(
      repo_id=repo_id, filename=f"{prefix}{basename}.safetensors"
    )
    return load_file(single_path)


def _load_fp8_text_encoder(
  repo_id: str,
  device: torch.device,
  dtype: torch.dtype,
  *,
  text_encoder_subfolder: str,
):
  """Rebuild the text encoder from its config and load weight-only FP8 weights.

  transformers' ``from_pretrained`` can't read our float8 layout, so we
  instantiate the architecture with ``from_config`` (which also computes the
  non-persistent buffers such as rotary caches), swap the quantized Linears, and
  load the FP8 state dict with ``assign=True``.
  """
  config = AutoConfig.from_pretrained(
    repo_id, subfolder=text_encoder_subfolder, trust_remote_code=True
  )
  model = AutoModel.from_config(config, trust_remote_code=True)
  state_dict = _load_subfolder_state_dict(repo_id, text_encoder_subfolder, "model")
  swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
  # assign=True so unquantized params take the loaded dtype and the computed
  # rotary buffers (absent from the checkpoint) survive; tied weights, if any,
  # surface as benign missing keys.
  load_fp8_state_dict(
    model, state_dict, device=device, dtype=dtype, assign=True, strict=False
  )
  model.eval()
  return model


def _load_qwen3_vl(
  repo_id: str,
  device: torch.device,
  dtype: torch.dtype,
  *,
  tokenizer_subfolder: str | None = None,
  text_encoder_subfolder: str | None = None,
):
  """Load the Qwen3-VL tokenizer + model, optionally from named subfolders of ``repo_id``.

  When the weights are published in diffusers layout the tokenizer lives at ``tokenizer/``
  and the model at ``text_encoder/`` within the same repo as the transformer weights, so
  there is no need to fetch them from a separate upstream repo.

  If the saved ``text_encoder/config.json`` carries a ``quantization_config`` (e.g.
  a bitsandbytes 4-bit checkpoint), transformers handles the bnb placement via
  ``device_map`` and we skip the explicit ``.to(device)`` move afterwards.
  """
  tokenizer_kwargs = {"subfolder": tokenizer_subfolder} if tokenizer_subfolder else {}
  model_kwargs = {"subfolder": text_encoder_subfolder} if text_encoder_subfolder else {}
  tokenizer = AutoTokenizer.from_pretrained(repo_id, **tokenizer_kwargs)

  cfg_path = hf_hub_download(
    repo_id=repo_id,
    filename=f"{text_encoder_subfolder}/config.json"
    if text_encoder_subfolder
    else "config.json",
  )
  with open(cfg_path) as f:
    cfg_data = json.load(f)
  is_quantized = "quantization_config" in cfg_data
  is_fp8 = bool(cfg_data.get(FP8_TEXT_ENCODER_CONFIG_FLAG, False))

  if is_fp8:
    model = _load_fp8_text_encoder(
      repo_id,
      device,
      dtype,
      text_encoder_subfolder=text_encoder_subfolder or "",
    )
  elif is_quantized:
    model = AutoModel.from_pretrained(
      repo_id,
      torch_dtype=dtype,
      trust_remote_code=True,
      device_map={"": device},
      **model_kwargs,
    )
    model.eval()
  else:
    model = AutoModel.from_pretrained(
      repo_id, torch_dtype=dtype, trust_remote_code=True, **model_kwargs
    )
    model.to(device)
    model.eval()
  return tokenizer, model


def _build_transformer(
  transformer_config: "Ideogram4Config",
  state_dict: dict[str, torch.Tensor],
  device: torch.device,
  dtype: torch.dtype,
) -> "Ideogram4Transformer":
  model = Ideogram4Transformer(transformer_config)
  if is_bnb4bit_state_dict(state_dict):
    if device.type != "cuda":
      raise ValueError(f"bnb 4-bit weights require a CUDA device, got device={device}")
    swap_linears_to_bnb4bit(model, compute_dtype=dtype)
    load_bnb4bit_state_dict(model, state_dict, device=device, dtype=dtype)
  elif is_fp8_state_dict(state_dict):
    # Weight-only FP8: cast the unquantized params to the compute dtype first,
    # then swap in Fp8Linear layers (which keep their weights as float8).
    model.to(dtype)
    swap_linears_to_fp8(model, state_dict, compute_dtype=dtype)
    load_fp8_state_dict(model, state_dict, device=device, dtype=dtype)
  else:
    model.load_state_dict(state_dict)
    model.to(device=device, dtype=dtype)
  model.eval()
  return model


def _load_autoencoder(weights_path: str, device: torch.device, dtype: torch.dtype):
  ae = AutoEncoder(AutoEncoderParams())
  state_dict = convert_diffusers_state_dict(load_file(weights_path))
  ae.load_state_dict(state_dict)
  ae.to(device=device, dtype=dtype)
  ae.eval()
  return ae


def _load_sharded_state_dict(
  repo_id: str, index_filename: str
) -> dict[str, torch.Tensor]:
  """Download a sharded safetensors checkpoint and merge it into one state dict.

  ``index_filename`` is the path of the safetensors index file inside the repo
  (e.g. ``conditional_model/model.safetensors.index.json``). Shard filenames in
  the index are interpreted relative to that index's directory, matching the
  layout written by ``huggingface_hub.save_torch_state_dict``.
  """
  index_path = hf_hub_download(repo_id=repo_id, filename=index_filename)
  with open(index_path) as f:
    index = json.load(f)
  weight_map: dict[str, str] = index["weight_map"]
  shard_dir = _posix_dirname(index_filename)
  shard_filenames = sorted(set(weight_map.values()))

  state_dict: dict[str, torch.Tensor] = {}
  for shard in shard_filenames:
    shard_repo_path = _posix_join(shard_dir, shard) if shard_dir else shard
    shard_path = hf_hub_download(repo_id=repo_id, filename=shard_repo_path)
    state_dict.update(load_file(shard_path))
  return state_dict


def _load_indexed_or_single_state_dict(
  repo_id: str, index_filename: str
) -> dict[str, torch.Tensor]:
  """Load a component whether published as a sharded index or a single file.

  Some repos publish each component as a single ``.safetensors`` file rather
  than a sharded checkpoint with an ``.index.json``. Try the index first and
  fall back to the single file (the index filename with ``.index.json``
  dropped) when it isn't present.
  """
  try:
    return _load_sharded_state_dict(repo_id, index_filename)
  except EntryNotFoundError:
    single_filename = index_filename.removesuffix(".index.json")
    single_path = hf_hub_download(repo_id=repo_id, filename=single_filename)
    return load_file(single_path)


@dataclass
class Ideogram4PipelineConfig:
  weights_repo: str = "ideogram-ai/ideogram-4-nf4"
  conditional_index_filename: str = (
    "transformer/diffusion_pytorch_model.safetensors.index.json"
  )
  unconditional_index_filename: str = (
    "unconditional_transformer/diffusion_pytorch_model.safetensors.index.json"
  )
  autoencoder_filename: str = "vae/diffusion_pytorch_model.safetensors"
  text_encoder_subfolder: str = "text_encoder"
  tokenizer_subfolder: str = "tokenizer"
  patch_size: int = 2
  ae_scale_factor: int = 8
  max_text_tokens: int = 2048


class Ideogram4Pipeline:
  """Ideogram 4 text-to-image pipeline."""

  def __init__(
    self,
    conditional_transformer: Ideogram4Transformer,
    unconditional_transformer: Ideogram4Transformer,
    text_encoder,
    text_tokenizer,
    autoencoder,
    config: Ideogram4PipelineConfig,
    device: torch.device,
    dtype: torch.dtype,
    text_device: torch.device | None = None,
  ) -> None:
    self.conditional_transformer = conditional_transformer
    self.unconditional_transformer = unconditional_transformer
    self.text_encoder = text_encoder
    self.text_tokenizer = text_tokenizer
    self.autoencoder = autoencoder
    self.config = config
    self.diffusion_device = torch.device(device)
    self.text_device = (
      torch.device(text_device) if text_device is not None else self.diffusion_device
    )
    self.device = self.diffusion_device
    self.dtype = dtype
    self.caption_verifier = CaptionVerifier()

    shift, scale = get_latent_norm()
    self.latent_shift = shift.to(self.diffusion_device)
    self.latent_scale = scale.to(self.diffusion_device)

  @classmethod
  def from_pretrained(
    cls,
    *,
    config: Optional[Ideogram4PipelineConfig] = None,
    device: str | torch.device = "cuda",
    text_device: str | torch.device | None = None,
    dtype: torch.dtype = torch.bfloat16,
    transformer_config: Optional[Ideogram4Config] = None,
  ) -> "Ideogram4Pipeline":
    config = config or Ideogram4PipelineConfig()
    transformer_config = transformer_config or Ideogram4Config()
    diffusion_device = torch.device(device)
    resolved_text_device = (
      torch.device(text_device) if text_device is not None else diffusion_device
    )
    _validate_device_configuration(diffusion_device, resolved_text_device)
    print(
      "Ideogram4 devices: "
      f"diffusion_device={_device_description(diffusion_device)}, "
      f"text_device={_device_description(resolved_text_device)}",
      flush=True,
    )

    conditional_state_dict = _load_indexed_or_single_state_dict(
      config.weights_repo, config.conditional_index_filename
    )
    unconditional_state_dict = _load_indexed_or_single_state_dict(
      config.weights_repo, config.unconditional_index_filename
    )
    autoencoder_weights = hf_hub_download(
      repo_id=config.weights_repo, filename=config.autoencoder_filename
    )

    conditional_transformer = _build_transformer(
      transformer_config, conditional_state_dict, diffusion_device, dtype
    )
    del conditional_state_dict
    unconditional_transformer = _build_transformer(
      transformer_config, unconditional_state_dict, diffusion_device, dtype
    )
    del unconditional_state_dict

    text_tokenizer, text_encoder = _load_qwen3_vl(
      config.weights_repo,
      resolved_text_device,
      dtype,
      tokenizer_subfolder=config.tokenizer_subfolder,
      text_encoder_subfolder=config.text_encoder_subfolder,
    )
    autoencoder = _load_autoencoder(autoencoder_weights, diffusion_device, dtype)

    pipeline = cls(
      conditional_transformer=conditional_transformer,
      unconditional_transformer=unconditional_transformer,
      text_encoder=text_encoder,
      text_tokenizer=text_tokenizer,
      autoencoder=autoencoder,
      config=config,
      device=diffusion_device,
      dtype=dtype,
      text_device=resolved_text_device,
    )
    _log_device_memory([diffusion_device, resolved_text_device])
    return pipeline

  def _tokenize(self, prompt: str) -> tuple[torch.Tensor, int]:
    """Build chat-formatted token ids for a single prompt."""
    messages = [{"role": "user", "content": [{"type": "text", "text": prompt}]}]
    text = self.text_tokenizer.apply_chat_template(
      messages, add_generation_prompt=True, tokenize=False
    )
    encoded = self.text_tokenizer(text, return_tensors="pt", add_special_tokens=False)
    token_ids = encoded["input_ids"][0]
    num_text_tokens = int(token_ids.shape[0])
    if num_text_tokens > self.config.max_text_tokens:
      raise ValueError(
        f"prompt has {num_text_tokens} tokens, exceeds max_text_tokens={self.config.max_text_tokens}"
      )
    return token_ids, num_text_tokens

  def _build_inputs(
    self,
    prompts: list[str],
    height: int,
    width: int,
  ) -> dict[str, torch.Tensor]:
    """Build the packed sequence (text tokens + image tokens) for one batch."""
    tokenized = [self._tokenize(p) for p in prompts]
    batch_size = len(prompts)

    patch = self.config.patch_size * self.config.ae_scale_factor
    if height % patch != 0 or width % patch != 0:
      raise ValueError(
        f"height/width must be divisible by patch_size*ae_scale_factor={patch}"
      )
    grid_h = height // patch
    grid_w = width // patch
    num_image_tokens = grid_h * grid_w

    max_text_tokens = max(num_text for _, num_text in tokenized)
    total_seq_len = max_text_tokens + num_image_tokens

    # Image position ids (t=0, h, w) offset to keep them disjoint from text positions.
    h_idx = torch.arange(grid_h).view(-1, 1).expand(grid_h, grid_w).reshape(-1)
    w_idx = torch.arange(grid_w).view(1, -1).expand(grid_h, grid_w).reshape(-1)
    t_idx = torch.zeros_like(h_idx)
    image_pos = torch.stack([t_idx, h_idx, w_idx], dim=1) + IMAGE_POSITION_OFFSET

    token_ids = torch.zeros(batch_size, total_seq_len, dtype=torch.long)
    text_position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)
    position_ids = torch.zeros(batch_size, total_seq_len, 3, dtype=torch.long)

    segment_ids = torch.full(
      (batch_size, total_seq_len), SEQUENCE_PADDING_INDICATOR, dtype=torch.long
    )
    indicator = torch.zeros(batch_size, total_seq_len, dtype=torch.long)

    for b, (toks, num_text) in enumerate(tokenized):
      pad_len = max_text_tokens - num_text
      total_unpadded = num_text + num_image_tokens

      # Layout: [pad_len zeros] [text tokens] [image tokens]
      offset = pad_len
      token_ids[b, offset : offset + num_text] = toks
      # Image token slots stay at 0.

      text_pos = torch.arange(num_text)
      text_pos_3d = torch.stack([text_pos, text_pos, text_pos], dim=1)
      text_position_ids[b, offset : offset + num_text] = text_pos_3d
      position_ids[b, offset : offset + num_text] = text_pos_3d
      position_ids[b, offset + num_text :] = image_pos

      indicator[b, offset : offset + num_text] = LLM_TOKEN_INDICATOR
      indicator[b, offset + num_text :] = OUTPUT_IMAGE_INDICATOR

      # Segment id 1 for the (text+image) sample, padding stays at 0.
      segment_ids[b, offset : offset + total_unpadded] = 1

    return {
      "token_ids": token_ids.to(self.text_device),
      "text_position_ids": text_position_ids.to(self.text_device),
      "text_indicator": indicator.to(self.text_device),
      "position_ids": position_ids.to(self.diffusion_device),
      "segment_ids": segment_ids.to(self.diffusion_device),
      "indicator": indicator.to(self.diffusion_device),
      "num_image_tokens": num_image_tokens,  # type: ignore[dict-item]
      "grid_h": grid_h,  # type: ignore[dict-item]
      "grid_w": grid_w,  # type: ignore[dict-item]
      "max_text_tokens": max_text_tokens,  # type: ignore[dict-item]
    }

  def _get_qwen3_vl_embeddings(
    self,
    token_ids: torch.Tensor,
    attention_mask: torch.Tensor,
    pos_2d: torch.Tensor,
  ) -> list[torch.Tensor]:
    language_model = self.text_encoder.language_model

    inputs_embeds = language_model.embed_tokens(token_ids)

    position_ids_4d = pos_2d[None, ...].expand(4, pos_2d.shape[0], -1)
    text_position_ids = position_ids_4d[0]
    mrope_position_ids = position_ids_4d[1:]

    causal_mask = create_causal_mask(
      config=language_model.config,
      inputs_embeds=inputs_embeds,
      attention_mask=attention_mask,
      past_key_values=None,
      position_ids=text_position_ids,
    )
    position_embeddings = language_model.rotary_emb(inputs_embeds, mrope_position_ids)

    tap_set = set(QWEN3_VL_ACTIVATION_LAYERS)
    captured: dict[int, torch.Tensor] = {}
    hidden_states = inputs_embeds
    for layer_idx, decoder_layer in enumerate(language_model.layers):
      hidden_states = decoder_layer(
        hidden_states,
        attention_mask=causal_mask,
        position_ids=text_position_ids,
        past_key_values=None,
        position_embeddings=position_embeddings,
      )
      if layer_idx in tap_set:
        captured[layer_idx] = hidden_states

    return [captured[i] for i in QWEN3_VL_ACTIVATION_LAYERS]

  def _encode_text(
    self,
    token_ids: torch.Tensor,
    text_position_ids: torch.Tensor,
    text_indicator: torch.Tensor,
    max_text_tokens: int,
  ) -> torch.Tensor:
    """Run Qwen3-VL and stack hidden states from the activation layers.

    Returns text-position features on the diffusion device as a
    (B, max_text_tokens, hidden_size * num_layers) float32 tensor.
    """
    token_ids = token_ids[:, :max_text_tokens].contiguous()
    text_position_ids = text_position_ids[:, :max_text_tokens].contiguous()
    text_indicator = text_indicator[:, :max_text_tokens].contiguous()
    batch_size, seq_len = token_ids.shape

    # Real text positions are exactly the LLM_TOKEN_INDICATOR positions.
    attention_mask = (text_indicator == LLM_TOKEN_INDICATOR).to(torch.long)

    pos_2d = text_position_ids[..., 0].contiguous()

    print(
      "Ideogram4 text encoding started: "
      f"sequence_length={seq_len}, device={_device_description(self.text_device)}",
      flush=True,
    )
    _synchronize_device(self.text_device)
    encode_started = time.perf_counter()
    with torch.no_grad():
      selected = self._get_qwen3_vl_embeddings(token_ids, attention_mask, pos_2d)
    stacked = torch.stack(selected, dim=0)  # (num_taps, B, L, H)
    stacked = torch.permute(stacked, (1, 2, 3, 0))
    stacked = stacked.reshape(batch_size, seq_len, -1)

    # Zero out non-LLM positions (left padding) so the transformer only sees real
    # text features at LLM_TOKEN_INDICATOR positions.
    text_mask = attention_mask.to(stacked.dtype).unsqueeze(-1)
    stacked = stacked * text_mask
    stacked = stacked.to(torch.float32)
    _synchronize_device(self.text_device)
    encode_elapsed = time.perf_counter() - encode_started

    transfer_mib = stacked.numel() * stacked.element_size() / 1024**2
    copy_started = time.perf_counter()
    text_features = _move_tensor_to_device(stacked, self.diffusion_device)
    _synchronize_device(self.diffusion_device)
    copy_elapsed = time.perf_counter() - copy_started
    print(
      "Ideogram4 text encoding finished: "
      f"feature_shape={tuple(text_features.shape)}, dtype={text_features.dtype}, "
      f"transfer={transfer_mib:.1f} MiB, source={self.text_device}, "
      f"destination={self.diffusion_device}, encode_time={encode_elapsed:.3f}s, "
      f"copy_time={copy_elapsed:.3f}s",
      flush=True,
    )
    return text_features

  def _verify_prompts(
    self, prompts: list[str], *, raise_on_issues: bool = True
  ) -> None:
    """Run each prompt through the caption verifier.

    Raises ``ValueError`` if any prompt has issues. When ``raise_on_issues``
    is False, issues are emitted as warnings instead.
    """
    messages: list[str] = []
    for i, prompt in enumerate(prompts):
      issues = self.caption_verifier.verify_raw(prompt)
      if not issues:
        continue
      messages.append(f"caption verifier flagged prompt[{i}]:\n" + "\n".join(issues))
    if not messages:
      return
    combined = "\n".join(messages)
    if raise_on_issues:
      raise ValueError(combined)
    warnings.warn(combined, stacklevel=2)

  @torch.no_grad()
  def __call__(
    self,
    prompts: str | list[str],
    *,
    height: int = 1024,
    width: int = 1024,
    num_steps: int = 128,
    guidance_scale: float = 7.0,
    guidance_schedule: Optional[Sequence[float] | torch.Tensor] = None,
    mu: float = 0.5,
    std: float = 1.0,
    seed: Optional[int] = None,
    schedule: Optional[LogitNormalSchedule] = None,
    raise_on_caption_issues: bool = True,
  ) -> list[Image.Image]:
    """Generate images for the given prompts."""
    if isinstance(prompts, str):
      prompts = [prompts]

    self._verify_prompts(prompts, raise_on_issues=raise_on_caption_issues)

    schedule = schedule or get_schedule_for_resolution(
      (height, width), known_mean=mu, std=std
    )
    step_intervals = make_step_intervals(num_steps).to(self.diffusion_device)

    if guidance_schedule is not None:
      gw_per_step = torch.as_tensor(
        guidance_schedule, dtype=torch.float32, device=self.diffusion_device
      )
      if gw_per_step.shape != (num_steps,):
        raise ValueError(
          f"guidance_schedule must have shape ({num_steps},), "
          f"got {tuple(gw_per_step.shape)}"
        )
    else:
      gw_per_step = torch.full(
        (num_steps,),
        float(guidance_scale),
        dtype=torch.float32,
        device=self.diffusion_device,
      )

    inputs = self._build_inputs(prompts, height=height, width=width)
    batch_size = len(prompts)
    num_image_tokens = inputs["num_image_tokens"]
    grid_h, grid_w = inputs["grid_h"], inputs["grid_w"]
    max_text_tokens = inputs["max_text_tokens"]
    latent_dim = self.conditional_transformer.config.in_channels

    llm_features = self._encode_text(
      inputs["token_ids"],
      inputs["text_position_ids"],
      inputs["text_indicator"],
      max_text_tokens,
    )
    llm_features = _append_image_feature_zeros(
      llm_features, num_image_tokens, self.diffusion_device
    )

    # Negative branch is image-only (asymmetric CFG) with zeroed conditioning.
    neg_position_ids = inputs["position_ids"][:, max_text_tokens:]
    neg_segment_ids = inputs["segment_ids"][:, max_text_tokens:]
    neg_indicator = inputs["indicator"][:, max_text_tokens:]
    neg_llm_features = torch.zeros(  # type: ignore[call-overload]
      batch_size,
      num_image_tokens,
      llm_features.shape[-1],
      dtype=llm_features.dtype,
      device=self.diffusion_device,
    )

    generator = torch.Generator(device=self.diffusion_device)
    if seed is not None:
      generator.manual_seed(seed)
    z = torch.randn(  # type: ignore[call-overload]
      batch_size,
      num_image_tokens,
      latent_dim,
      dtype=torch.float32,
      device=self.diffusion_device,
      generator=generator,
    )

    text_z_padding = torch.zeros(  # type: ignore[call-overload]
      batch_size,
      max_text_tokens,
      latent_dim,
      dtype=torch.float32,
      device=self.diffusion_device,
    )

    for i in range(num_steps - 1, -1, -1):
      t_val = float(schedule(step_intervals[i + 1].unsqueeze(0)).item())
      s_val = float(schedule(step_intervals[i].unsqueeze(0)).item())
      t = torch.full(
        (batch_size,), t_val, dtype=torch.float32, device=self.diffusion_device
      )

      pos_z = torch.cat([text_z_padding, z], dim=1)
      pos_out = self.conditional_transformer(
        llm_features=llm_features,
        x=pos_z,
        t=t,
        position_ids=inputs["position_ids"],
        segment_ids=inputs["segment_ids"],
        indicator=inputs["indicator"],
      )
      pos_v = pos_out[:, max_text_tokens:]

      neg_v = self.unconditional_transformer(
        llm_features=neg_llm_features,
        x=z,
        t=t,
        position_ids=neg_position_ids,
        segment_ids=neg_segment_ids,
        indicator=neg_indicator,
      )

      gw_i = gw_per_step[i]
      v = gw_i * pos_v + (1.0 - gw_i) * neg_v
      delta = s_val - t_val
      z = z + v * delta

      # Do not retain the final step's branch outputs while the VAE allocates
      # its decode workspace. These tensors are recreated on every step.
      del pos_z, pos_out, pos_v, neg_v, v, t

    # Full text-plus-image conditioning is needed only by the diffusion models.
    # At 1024x1024 the positive and negative float32 feature tensors occupy
    # roughly 1.7 GiB together, enough to crowd out the VAE's peak workspace on
    # a 20 GiB diffusion device if their references survive into _decode().
    del (
      llm_features,
      neg_llm_features,
      text_z_padding,
      neg_position_ids,
      neg_segment_ids,
      neg_indicator,
      inputs,
      gw_per_step,
      step_intervals,
      generator,
    )
    _synchronize_device(self.diffusion_device)
    if self.diffusion_device.type == "cuda":
      torch.cuda.empty_cache()
    _log_device_memory([self.diffusion_device])

    return self._decode(z, grid_h=grid_h, grid_w=grid_w)  # type: ignore[arg-type]

  def _decode(self, z: torch.Tensor, *, grid_h: int, grid_w: int) -> list[Image.Image]:
    """Unpatch and run the autoencoder decoder."""
    batch_size = z.shape[0]
    patch = self.config.patch_size

    z = z * self.latent_scale + self.latent_shift

    ae_channels = z.shape[-1] // (patch * patch)
    z = z.view(batch_size, grid_h, grid_w, patch, patch, ae_channels)
    z = z.permute(0, 5, 1, 3, 2, 4).contiguous()
    z = z.view(batch_size, ae_channels, grid_h * patch, grid_w * patch)

    z = z.to(self.dtype)
    decoded = self.autoencoder.decoder(z)

    decoded = decoded.float().clamp(-1.0, 1.0)
    decoded = ((decoded + 1.0) * 127.5).round().to(torch.uint8)
    decoded = decoded.permute(0, 2, 3, 1).cpu().numpy()
    return [Image.fromarray(arr) for arr in decoded]
