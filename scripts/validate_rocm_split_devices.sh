#!/usr/bin/env bash

set -Eeuo pipefail

usage() {
  cat <<'EOF'
Validate Ideogram 4 split-device inference on the target Fedora/ROCm host.

Usage:
  scripts/validate_rocm_split_devices.sh --caption FILE [options]

Required:
  --caption FILE              Structured Ideogram JSON caption stored locally.

Options:
  --python PATH               Python from the ROCm/ComfyUI environment.
  --comfyui-dir DIR           ComfyUI checkout; also selects its .venv by default.
  --output-dir DIR            Logs and images directory (default: timestamped).
  --diffusion-device DEVICE   Diffusion/VAE device (default: cuda:0).
  --text-device DEVICE        Qwen3-VL device (default: cuda:1).
  --visible-devices LIST      ROCR_VISIBLE_DEVICES value (default: existing or 0,1).
  --weights-repo REPO         Hugging Face weights repo.
  --expected-diffusion TEXT   Required substring in diffusion device name.
                              Default: 7900 XT
  --expected-text TEXT        Required substring in text device name.
                              Default: 8060S
  --allow-name-mismatch       Warn instead of failing when names do not match.
  --skip-small                Skip the 256x256 two-step smoke test.
  --skip-1024                 Skip the 1024x1024 V4_DEFAULT_20 test.
  --launch-comfyui            Launch ComfyUI after core tests (foreground).
  --comfyui-port PORT         ComfyUI port (default: 8189).
  -h, --help                  Show this help.

The script never calls a hosted Magic Prompt API. It installs this core checkout
editable with --no-deps so it cannot replace the existing ROCm PyTorch build.
EOF
}

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" && pwd)"
CORE_DIR="$(cd -- "${SCRIPT_DIR}/.." && pwd)"
[[ -f "$CORE_DIR/src/ideogram4/pipeline_ideogram4.py" ]] || fail \
  "run this script from the modified ideogram4 checkout"
CAPTION_FILE=""
PYTHON_BIN=""
COMFYUI_DIR=""
OUTPUT_DIR=""
DIFFUSION_DEVICE="cuda:0"
TEXT_DEVICE="cuda:1"
VISIBLE_DEVICES="${ROCR_VISIBLE_DEVICES:-0,1}"
WEIGHTS_REPO="ideogram-ai/ideogram-4-nf4"
EXPECTED_DIFFUSION="7900 XT"
EXPECTED_TEXT="8060S"
ALLOW_NAME_MISMATCH=0
SKIP_SMALL=0
SKIP_1024=0
LAUNCH_COMFYUI=0
COMFYUI_PORT=8189

while (($#)); do
  case "$1" in
    --caption)
      (($# >= 2)) || fail "$1 requires a value"
      CAPTION_FILE="$2"
      shift 2
      ;;
    --python)
      (($# >= 2)) || fail "$1 requires a value"
      PYTHON_BIN="$2"
      shift 2
      ;;
    --comfyui-dir)
      (($# >= 2)) || fail "$1 requires a value"
      COMFYUI_DIR="$2"
      shift 2
      ;;
    --output-dir)
      (($# >= 2)) || fail "$1 requires a value"
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --diffusion-device)
      (($# >= 2)) || fail "$1 requires a value"
      DIFFUSION_DEVICE="$2"
      shift 2
      ;;
    --text-device)
      (($# >= 2)) || fail "$1 requires a value"
      TEXT_DEVICE="$2"
      shift 2
      ;;
    --visible-devices)
      (($# >= 2)) || fail "$1 requires a value"
      VISIBLE_DEVICES="$2"
      shift 2
      ;;
    --weights-repo)
      (($# >= 2)) || fail "$1 requires a value"
      WEIGHTS_REPO="$2"
      shift 2
      ;;
    --expected-diffusion)
      (($# >= 2)) || fail "$1 requires a value"
      EXPECTED_DIFFUSION="$2"
      shift 2
      ;;
    --expected-text)
      (($# >= 2)) || fail "$1 requires a value"
      EXPECTED_TEXT="$2"
      shift 2
      ;;
    --allow-name-mismatch)
      ALLOW_NAME_MISMATCH=1
      shift
      ;;
    --skip-small)
      SKIP_SMALL=1
      shift
      ;;
    --skip-1024)
      SKIP_1024=1
      shift
      ;;
    --launch-comfyui)
      LAUNCH_COMFYUI=1
      shift
      ;;
    --comfyui-port)
      (($# >= 2)) || fail "$1 requires a value"
      COMFYUI_PORT="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      fail "unknown argument: $1"
      ;;
  esac
done

[[ "$(uname -s)" == "Linux" ]] || fail "this script must run on Linux"
[[ -n "$CAPTION_FILE" ]] || fail "--caption is required"
[[ -r "$CAPTION_FILE" ]] || fail "caption file is not readable: $CAPTION_FILE"
CAPTION_FILE="$(readlink -f -- "$CAPTION_FILE")"

if [[ -n "$COMFYUI_DIR" ]]; then
  COMFYUI_DIR="$(readlink -f -- "$COMFYUI_DIR")"
  [[ -f "$COMFYUI_DIR/main.py" ]] || fail "ComfyUI main.py not found under $COMFYUI_DIR"
fi

if [[ -z "$PYTHON_BIN" && -n "$COMFYUI_DIR" ]]; then
  if [[ -x "$COMFYUI_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$COMFYUI_DIR/.venv/bin/python"
  elif [[ -x "$COMFYUI_DIR/venv/bin/python" ]]; then
    PYTHON_BIN="$COMFYUI_DIR/venv/bin/python"
  fi
fi
PYTHON_BIN="${PYTHON_BIN:-python3}"
PYTHON_BIN="$(command -v -- "$PYTHON_BIN" || true)"
[[ -n "$PYTHON_BIN" && -x "$PYTHON_BIN" ]] || fail "Python executable not found"

if [[ -z "$OUTPUT_DIR" ]]; then
  OUTPUT_DIR="$HOME/ideogram4-rocm-validation-$(date +%Y%m%d-%H%M%S)"
fi
mkdir -p -- "$OUTPUT_DIR"
OUTPUT_DIR="$(readlink -f -- "$OUTPUT_DIR")"

export ROCR_VISIBLE_DEVICES="$VISIBLE_DEVICES"
export IDEOGRAM4_DIFFUSION_DEVICE="$DIFFUSION_DEVICE"
export IDEOGRAM4_TEXT_DEVICE="$TEXT_DEVICE"
export IDEOGRAM4_REPO="$CORE_DIR"
export PYTHONUNBUFFERED=1

SUMMARY_LOG="$OUTPUT_DIR/summary.log"
SMALL_LOG="$OUTPUT_DIR/core-256.log"
FULL_LOG="$OUTPUT_DIR/core-1024.log"
GPU_LOG="$OUTPUT_DIR/gpu-monitor.log"
MONITOR_PID=""

stop_monitor() {
  if [[ -n "$MONITOR_PID" ]] && kill -0 "$MONITOR_PID" 2>/dev/null; then
    kill "$MONITOR_PID" 2>/dev/null || true
    wait "$MONITOR_PID" 2>/dev/null || true
  fi
}

on_exit() {
  status=$?
  stop_monitor
  if ((status == 0)); then
    echo "Validation command completed. Artifacts: $OUTPUT_DIR"
  else
    echo "Validation failed with exit code $status. Logs: $OUTPUT_DIR" >&2
  fi
}
trap on_exit EXIT
trap 'exit 130' INT
trap 'exit 143' TERM

start_monitor() {
  if ! command -v rocm-smi >/dev/null 2>&1; then
    echo "rocm-smi not found; relying on PyTorch peak-memory reporting" | tee -a "$SUMMARY_LOG"
    return
  fi
  (
    while true; do
      date --iso-8601=seconds
      rocm-smi --showproductname --showuse --showmemuse || true
      sleep 1
    done
  ) >"$GPU_LOG" 2>&1 &
  MONITOR_PID=$!
  echo "GPU monitor PID=$MONITOR_PID log=$GPU_LOG" | tee -a "$SUMMARY_LOG"
}

{
  echo "Ideogram 4 ROCm split-device validation"
  echo "started=$(date --iso-8601=seconds)"
  echo "host=$(hostname)"
  echo "kernel=$(uname -srmo)"
  echo "core_dir=$CORE_DIR"
  echo "python=$PYTHON_BIN"
  echo "caption=$CAPTION_FILE"
  echo "output_dir=$OUTPUT_DIR"
  echo "ROCR_VISIBLE_DEVICES=$ROCR_VISIBLE_DEVICES"
  echo "diffusion_device=$DIFFUSION_DEVICE"
  echo "text_device=$TEXT_DEVICE"
  echo "weights_repo=$WEIGHTS_REPO"
  echo "HF_HOME=${HF_HOME:-<default>}"
  df -h -- "$OUTPUT_DIR"
} | tee "$SUMMARY_LOG"

"$PYTHON_BIN" - "$DIFFUSION_DEVICE" "$TEXT_DEVICE" \
  "$EXPECTED_DIFFUSION" "$EXPECTED_TEXT" "$ALLOW_NAME_MISMATCH" <<'PY' | tee -a "$SUMMARY_LOG"
import sys

try:
  import torch
except ImportError as exc:
  raise SystemExit(
    "PyTorch is missing from the selected environment. Install the ROCm build first."
  ) from exc

diffusion = torch.device(sys.argv[1])
text = torch.device(sys.argv[2])
expected_diffusion = sys.argv[3].casefold()
expected_text = sys.argv[4].casefold()
allow_mismatch = bool(int(sys.argv[5]))

print("torch:", torch.__version__)
print("HIP:", torch.version.hip)
print("CUDA/ROCm available:", torch.cuda.is_available())
print("device count:", torch.cuda.device_count())

if torch.version.hip is None:
  raise SystemExit("Selected Python does not contain a ROCm PyTorch build (torch.version.hip is None)")
if not torch.cuda.is_available():
  raise SystemExit("ROCm devices are not available through torch.cuda")

for index in range(torch.cuda.device_count()):
  props = torch.cuda.get_device_properties(index)
  print(
    f"cuda:{index}: {torch.cuda.get_device_name(index)} "
    f"VRAM={props.total_memory / 1024**3:.2f} GiB"
  )

if diffusion.type != "cuda" or text.type != "cuda":
  raise SystemExit(f"Expected CUDA/ROCm roles, got {diffusion} and {text}")
if diffusion.index is None or text.index is None:
  raise SystemExit("Split-device validation requires explicit cuda:N indexes")
highest = max(diffusion.index, text.index)
if torch.cuda.device_count() <= highest:
  raise SystemExit(
    f"Requested {diffusion} and {text}, but only {torch.cuda.device_count()} device(s) are visible"
  )
if diffusion == text:
  raise SystemExit("Diffusion and text devices must differ for this validation")

diffusion_name = torch.cuda.get_device_name(diffusion.index)
text_name = torch.cuda.get_device_name(text.index)
diffusion_vram_gib = torch.cuda.get_device_properties(diffusion.index).total_memory / 1024**3
text_vram_gib = torch.cuda.get_device_properties(text.index).total_memory / 1024**3
mismatches = []
if expected_diffusion and expected_diffusion not in diffusion_name.casefold():
  generic_amd_name = diffusion_name.casefold() == "amd radeon graphics"
  expected_7900_memory = 18.0 <= diffusion_vram_gib <= 24.0
  if expected_diffusion == "7900 xt" and generic_amd_name and expected_7900_memory:
    print(
      "WARNING: ROCm reported the diffusion GPU with the generic name "
      f"{diffusion_name!r}; accepting it as the expected 7900 XT based on "
      f"its {diffusion_vram_gib:.2f} GiB VRAM."
    )
  else:
    mismatches.append(
      f"diffusion device {diffusion} is {diffusion_name!r} "
      f"with {diffusion_vram_gib:.2f} GiB VRAM, expected substring {sys.argv[3]!r}"
    )
if expected_text and expected_text not in text_name.casefold():
  mismatches.append(
    f"text device {text} is {text_name!r} with {text_vram_gib:.2f} GiB VRAM, "
    f"expected substring {sys.argv[4]!r}"
  )
if mismatches:
  message = "\n".join(mismatches)
  if allow_mismatch:
    print("WARNING: device-name mismatch allowed:\n" + message)
  else:
    raise SystemExit(
      "Device mapping does not match the requested roles:\n"
      + message
      + "\nCorrect the device arguments or pass --allow-name-mismatch intentionally."
    )

print(f"resolved diffusion: {diffusion} ({diffusion_name})")
print(f"resolved text: {text} ({text_name})")
print(
  f"peer access {diffusion}->{text}:",
  torch.cuda.can_device_access_peer(diffusion.index, text.index),
)
print(
  f"peer access {text}->{diffusion}:",
  torch.cuda.can_device_access_peer(text.index, diffusion.index),
)
PY

echo "Installing editable core without changing environment dependencies..." | tee -a "$SUMMARY_LOG"
"$PYTHON_BIN" -m pip install --no-deps -e "$CORE_DIR" 2>&1 | tee -a "$SUMMARY_LOG"

"$PYTHON_BIN" - <<'PY' | tee -a "$SUMMARY_LOG"
import importlib
import ideogram4

required = (
  "accelerate",
  "bitsandbytes",
  "einops",
  "huggingface_hub",
  "PIL",
  "safetensors",
  "sentencepiece",
  "transformers",
)
missing = []
for name in required:
  try:
    importlib.import_module(name)
  except ImportError:
    missing.append(name)
if missing:
  raise SystemExit("Missing dependencies in selected environment: " + ", ".join(missing))

print("ideogram4 import:", ideogram4.__file__)
try:
  from huggingface_hub import get_token
  print("Hugging Face authentication available:", bool(get_token()))
except Exception as exc:
  print("WARNING: unable to inspect Hugging Face authentication:", exc)
PY

start_monitor

if ((SKIP_SMALL == 0)); then
  echo "Running 256x256 two-step core smoke test..." | tee -a "$SUMMARY_LOG"
  "$PYTHON_BIN" "$CORE_DIR/scripts/smoke_split_devices.py" \
    --caption-file "$CAPTION_FILE" \
    --output "$OUTPUT_DIR/ideogram4-256.png" \
    --weights-repo "$WEIGHTS_REPO" \
    --diffusion-device "$DIFFUSION_DEVICE" \
    --text-device "$TEXT_DEVICE" \
    --height 256 --width 256 --num-steps 2 \
    2>&1 | tee "$SMALL_LOG"
fi

if ((SKIP_1024 == 0)); then
  echo "Running 1024x1024 V4_DEFAULT_20 acceptance test..." | tee -a "$SUMMARY_LOG"
  "$PYTHON_BIN" "$CORE_DIR/scripts/smoke_split_devices.py" \
    --caption-file "$CAPTION_FILE" \
    --output "$OUTPUT_DIR/ideogram4-1024.png" \
    --weights-repo "$WEIGHTS_REPO" \
    --diffusion-device "$DIFFUSION_DEVICE" \
    --text-device "$TEXT_DEVICE" \
    --height 1024 --width 1024 \
    --sampler-preset V4_DEFAULT_20 \
    2>&1 | tee "$FULL_LOG"
fi

if grep -Fq "staging through CPU" "$SMALL_LOG" "$FULL_LOG" 2>/dev/null; then
  COPY_PATH="CPU-staged fallback"
else
  COPY_PATH="direct device transfer (no CPU-staging warning observed)"
fi
echo "feature_copy_path=$COPY_PATH" | tee -a "$SUMMARY_LOG"
echo "core_validation=PASS" | tee -a "$SUMMARY_LOG"

if ((LAUNCH_COMFYUI == 1)); then
  [[ -n "$COMFYUI_DIR" ]] || fail "--launch-comfyui requires --comfyui-dir"
  WRAPPER_DIR="$COMFYUI_DIR/custom_nodes/ComfyUI-Ideogram4"
  if [[ ! -f "$WRAPPER_DIR/nodes.py" ]]; then
    fail "ComfyUI-Ideogram4 is not installed under $COMFYUI_DIR/custom_nodes"
  fi
  if ! grep -Fq "IDEOGRAM4_TEXT_DEVICE" "$WRAPPER_DIR/nodes.py"; then
    fail "installed ComfyUI-Ideogram4 does not contain the split-device wrapper change"
  fi
  echo "Launching ComfyUI on port $COMFYUI_PORT." | tee -a "$SUMMARY_LOG"
  echo "Queue the existing Pipeline Loader -> Generate -> Save Image workflow." | tee -a "$SUMMARY_LOG"
  echo "Use 4.0 NF4, 1024x1024, V4_DEFAULT_20, batch size 1." | tee -a "$SUMMARY_LOG"
  cd -- "$COMFYUI_DIR"
  "$PYTHON_BIN" main.py \
    --listen 0.0.0.0 \
    --port "$COMFYUI_PORT" \
    --disable-pinned-memory \
    2>&1 | tee "$OUTPUT_DIR/comfyui.log"
else
  if [[ -n "$COMFYUI_DIR" ]]; then
    cat <<EOF | tee -a "$SUMMARY_LOG"

Core validation passed. To launch ComfyUI with the same routing:
  export ROCR_VISIBLE_DEVICES='$ROCR_VISIBLE_DEVICES'
  export IDEOGRAM4_REPO='$CORE_DIR'
  export IDEOGRAM4_DIFFUSION_DEVICE='$DIFFUSION_DEVICE'
  export IDEOGRAM4_TEXT_DEVICE='$TEXT_DEVICE'
  '$PYTHON_BIN' '$COMFYUI_DIR/main.py' --listen 0.0.0.0 --port '$COMFYUI_PORT' --disable-pinned-memory
EOF
  else
    echo "Core validation passed. Re-run with --comfyui-dir DIR --launch-comfyui for integration." \
      | tee -a "$SUMMARY_LOG"
  fi
fi
