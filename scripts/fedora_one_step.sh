#!/usr/bin/env bash

set -Eeuo pipefail

CORE_REPO="https://github.com/Dillflix/ideogram4.git"
CORE_BRANCH="feature/split-text-diffusion-devices"
WRAPPER_REPO="https://github.com/Dillflix/ComfyUI-Ideogram4.git"
WRAPPER_BRANCH="feature/multigpu-device-routing"
REMOTE_NAME="dillflix-validation"
STATE_ROOT="${XDG_DATA_HOME:-$HOME/.local/share}/ideogram4-split-validation"
CORE_DIR="$STATE_ROOT/ideogram4"

fail() {
  echo "ERROR: $*" >&2
  exit 1
}

find_comfyui() {
  local candidate
  if [[ -n "${COMFYUI_DIR:-}" ]]; then
    candidate="$COMFYUI_DIR"
    [[ -f "$candidate/main.py" ]] || fail "COMFYUI_DIR does not contain main.py: $candidate"
    readlink -f -- "$candidate"
    return
  fi

  for candidate in "$PWD" "$HOME/ComfyUI" "$HOME/comfyui"; do
    if [[ -f "$candidate/main.py" ]]; then
      readlink -f -- "$candidate"
      return
    fi
  done
  fail "ComfyUI was not found. Run from its directory or prefix the command with COMFYUI_DIR=/path/to/ComfyUI"
}

checkout_branch() {
  local repo_url="$1"
  local branch="$2"
  local destination="$3"

  if [[ ! -d "$destination/.git" ]]; then
    if [[ -e "$destination" && -n "$(find "$destination" -mindepth 1 -maxdepth 1 -print -quit 2>/dev/null)" ]]; then
      fail "cannot clone into non-empty directory: $destination"
    fi
    mkdir -p -- "$(dirname -- "$destination")"
    git clone --branch "$branch" --single-branch "$repo_url" "$destination"
    return
  fi

  if [[ -n "$(git -C "$destination" status --porcelain)" ]]; then
    fail "existing checkout has local changes; refusing to switch it: $destination"
  fi

  if git -C "$destination" remote get-url "$REMOTE_NAME" >/dev/null 2>&1; then
    git -C "$destination" remote set-url "$REMOTE_NAME" "$repo_url"
  else
    git -C "$destination" remote add "$REMOTE_NAME" "$repo_url"
  fi
  git -C "$destination" fetch "$REMOTE_NAME" "$branch"

  if git -C "$destination" show-ref --verify --quiet "refs/heads/$branch"; then
    git -C "$destination" switch "$branch"
  else
    git -C "$destination" switch --create "$branch" --track "$REMOTE_NAME/$branch"
  fi
  git -C "$destination" merge --ff-only "$REMOTE_NAME/$branch"
}

command -v git >/dev/null 2>&1 || fail "git is required"
[[ "$(uname -s)" == "Linux" ]] || fail "this launcher must run on Linux"

COMFYUI_DIR="$(find_comfyui)"
WRAPPER_DIR="$COMFYUI_DIR/custom_nodes/ComfyUI-Ideogram4"
mkdir -p -- "$STATE_ROOT"

echo "ComfyUI: $COMFYUI_DIR"
echo "Validation workspace: $STATE_ROOT"
echo "Preparing modified core..."
checkout_branch "$CORE_REPO" "$CORE_BRANCH" "$CORE_DIR"
echo "Preparing modified ComfyUI wrapper..."
checkout_branch "$WRAPPER_REPO" "$WRAPPER_BRANCH" "$WRAPPER_DIR"

CAPTION_FILE="$STATE_ROOT/validation-caption.json"
cat >"$CAPTION_FILE" <<'JSON'
{
  "high_level_description": "A clean technical poster celebrating a dual-GPU image generation workstation.",
  "style_description": {
    "aesthetics": "precise, modern, polished, high contrast",
    "lighting": "soft studio lighting with subtle cyan and amber rim light",
    "medium": "graphic_design",
    "art_style": "minimal technical poster, crisp geometric forms, premium product visualization",
    "color_palette": ["#101820", "#00AEEF", "#FFB000", "#F5F7FA"]
  },
  "compositional_deconstruction": {
    "background": "A deep charcoal studio backdrop with a faint grid and restrained cyan highlights.",
    "elements": [
      {
        "type": "obj",
        "bbox": [180, 120, 820, 880],
        "desc": "Two elegant abstract GPU modules connected by a single luminous data arc, arranged symmetrically as a premium technical product composition."
      },
      {
        "type": "text",
        "bbox": [70, 180, 180, 820],
        "text": "SPLIT COMPUTE",
        "desc": "Large crisp uppercase geometric sans-serif title in white."
      }
    ]
  }
}
JSON

echo "Starting automated ROCm validation."
echo "This downloads gated NF4 weights if they are not already cached."
echo "After both core tests pass, ComfyUI will remain running in the foreground."

VALIDATOR_ARGS=(
  --caption "$CAPTION_FILE"
  --comfyui-dir "$COMFYUI_DIR"
  --launch-comfyui
)

if [[ -n "${IDEOGRAM4_VALIDATION_OUTPUT:-}" ]]; then
  VALIDATOR_ARGS+=(--output-dir "$IDEOGRAM4_VALIDATION_OUTPUT")
fi
if [[ -n "${IDEOGRAM4_DIFFUSION_DEVICE:-}" ]]; then
  VALIDATOR_ARGS+=(--diffusion-device "$IDEOGRAM4_DIFFUSION_DEVICE")
fi
if [[ -n "${IDEOGRAM4_TEXT_DEVICE:-}" ]]; then
  VALIDATOR_ARGS+=(--text-device "$IDEOGRAM4_TEXT_DEVICE")
fi

exec "$CORE_DIR/scripts/validate_rocm_split_devices.sh" "${VALIDATOR_ARGS[@]}"
