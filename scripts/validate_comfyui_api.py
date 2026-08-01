from __future__ import annotations

import argparse
import json
import time
import urllib.error
import urllib.parse
import urllib.request
import uuid
from pathlib import Path
from typing import Any


def _parse_args() -> argparse.Namespace:
  parser = argparse.ArgumentParser(
    description="Submit and verify the Ideogram 4 ComfyUI acceptance workflow."
  )
  parser.add_argument("--base-url", default="http://127.0.0.1:8189")
  parser.add_argument("--caption-file", type=Path, required=True)
  parser.add_argument("--startup-timeout", type=float, default=300.0)
  parser.add_argument("--generation-timeout", type=float, default=3600.0)
  return parser.parse_args()


def _request_json(
  url: str,
  *,
  payload: dict[str, Any] | None = None,
  timeout: float = 30.0,
) -> dict[str, Any]:
  data = None
  headers = {}
  if payload is not None:
    data = json.dumps(payload).encode("utf-8")
    headers["Content-Type"] = "application/json"
  request = urllib.request.Request(url, data=data, headers=headers)
  try:
    with urllib.request.urlopen(request, timeout=timeout) as response:
      body = response.read().decode("utf-8")
  except urllib.error.HTTPError as exc:
    body = exc.read().decode("utf-8", errors="replace")
    raise RuntimeError(f"ComfyUI HTTP {exc.code} from {url}: {body}") from exc
  result = json.loads(body)
  if not isinstance(result, dict):
    raise TypeError(f"Expected a JSON object from {url}, got {type(result).__name__}")
  return result


def _wait_for_server(base_url: str, timeout: float) -> None:
  deadline = time.monotonic() + timeout
  last_error: Exception | None = None
  while time.monotonic() < deadline:
    try:
      _request_json(f"{base_url}/system_stats", timeout=5.0)
      print(f"ComfyUI API ready: {base_url}", flush=True)
      return
    except (OSError, RuntimeError, ValueError) as exc:
      last_error = exc
      time.sleep(1.0)
  raise TimeoutError(f"ComfyUI did not become ready within {timeout:g}s: {last_error}")


def _workflow(caption: str) -> dict[str, Any]:
  return {
    "1": {
      "class_type": "Ideogram4PipelineLoader",
      "inputs": {"model_weights": "4.0 NF4"},
    },
    "2": {
      "class_type": "Ideogram4Generate",
      "inputs": {
        "pipeline": ["1", 0],
        "prompt": caption,
        "width": 1024,
        "height": 1024,
        "sampler_preset": "4.0 Default 20",
        "num_steps": 20,
        "guidance_scale": 7.0,
        "mu": 0.0,
        "std": 1.75,
        "seed": 0,
      },
    },
    "3": {
      "class_type": "SaveImage",
      "inputs": {
        "images": ["2", 0],
        "filename_prefix": "ideogram4-split-validation",
      },
    },
  }


def _error_details(entry: dict[str, Any]) -> str:
  status = entry.get("status", {})
  messages = status.get("messages", []) if isinstance(status, dict) else []
  for message in reversed(messages):
    if (
      isinstance(message, list)
      and len(message) >= 2
      and message[0] == "execution_error"
    ):
      return json.dumps(message[1], indent=2)
  return json.dumps(status, indent=2)


def _wait_for_result(base_url: str, prompt_id: str, timeout: float) -> dict[str, Any]:
  deadline = time.monotonic() + timeout
  while time.monotonic() < deadline:
    history = _request_json(f"{base_url}/history/{prompt_id}")
    entry = history.get(prompt_id)
    if isinstance(entry, dict):
      status = entry.get("status", {})
      if isinstance(status, dict) and status.get("completed"):
        if status.get("status_str") != "success":
          raise RuntimeError("ComfyUI workflow failed:\n" + _error_details(entry))
        return entry
    time.sleep(2.0)
  raise TimeoutError(f"ComfyUI workflow {prompt_id} exceeded {timeout:g}s")


def _saved_images(entry: dict[str, Any]) -> list[dict[str, Any]]:
  outputs = entry.get("outputs", {})
  save_output = outputs.get("3", {}) if isinstance(outputs, dict) else {}
  images = save_output.get("images", []) if isinstance(save_output, dict) else []
  return [image for image in images if isinstance(image, dict)]


def _verify_image(base_url: str, image: dict[str, Any]) -> int:
  query = urllib.parse.urlencode(
    {
      "filename": image.get("filename", ""),
      "subfolder": image.get("subfolder", ""),
      "type": image.get("type", "output"),
    }
  )
  with urllib.request.urlopen(f"{base_url}/view?{query}", timeout=30.0) as response:
    data = response.read()
  if not data:
    raise RuntimeError(f"ComfyUI returned an empty saved image: {image}")
  return len(data)


def main() -> None:
  args = _parse_args()
  base_url = args.base_url.rstrip("/")
  caption = args.caption_file.read_text(encoding="utf-8")
  json.loads(caption)

  _wait_for_server(base_url, args.startup_timeout)
  queued = _request_json(
    f"{base_url}/prompt",
    payload={"prompt": _workflow(caption), "client_id": str(uuid.uuid4())},
  )
  prompt_id = queued.get("prompt_id")
  if not isinstance(prompt_id, str) or not prompt_id:
    raise RuntimeError(f"ComfyUI did not return a prompt_id: {queued}")
  if queued.get("node_errors"):
    raise RuntimeError(
      "ComfyUI rejected workflow nodes: " + json.dumps(queued["node_errors"], indent=2)
    )
  print(f"ComfyUI workflow queued: {prompt_id}", flush=True)

  entry = _wait_for_result(base_url, prompt_id, args.generation_timeout)
  images = _saved_images(entry)
  if not images:
    raise RuntimeError("ComfyUI workflow completed without a SaveImage output")
  size = _verify_image(base_url, images[0])
  print(f"ComfyUI saved image: {json.dumps(images[0], sort_keys=True)}", flush=True)
  print(f"ComfyUI saved image bytes: {size}", flush=True)


if __name__ == "__main__":
  main()
