#!/usr/bin/env bash
set -euo pipefail

# V4-iter1 Colab GPU launcher with strict EXIT/INT/TERM trap stopping ultra-4 GPU immediately.
export COLAB_ACCOUNT="${COLAB_ACCOUNT:-ultra-4}"
COLAB_BIN="/Users/eligo/.local/bin/colab"
PREFERRED_GPU="${PREFERRED_GPU:-L4}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_TENSORS="${LOCAL_TENSORS:-${CODE_ROOT}/artifacts/nfl_10hz_tensors.npz}"
LOCAL_OOF="${LOCAL_OOF:-${CODE_ROOT}/artifacts/v4_positional_attn_gpu_oof.npz}"
LOCAL_META="${LOCAL_META:-${CODE_ROOT}/artifacts/v4_colab_gpu_metadata.json}"
POS_ATTN_PY="${CODE_ROOT}/src/nfl_bdb2027/positional_attn.py"
TRAIN_PY="${SCRIPT_DIR}/train_v4_colab_gpu.py"

SESSION="nfl2027-v4-posattn-$(date +%s)"

cleanup() {
  echo "[COLAB-TRAP] Stopping ${SESSION} on COLAB_ACCOUNT=${COLAB_ACCOUNT} ..."
  COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" stop -s "${SESSION}" >/dev/null 2>&1 || true
  COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" sessions || true
  COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" usage || true
}
trap cleanup EXIT INT TERM

echo "[V4-COLAB] Account=${COLAB_ACCOUNT} | Session=${SESSION} | GPU=${PREFERRED_GPU}"
if [[ "${PREFERRED_GPU}" == "A100" ]]; then
  COLAB_CLI_ASSIGN_RETRIES=0 COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" new -s "${SESSION}" --gpu A100 \
    || COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" new -s "${SESSION}" --gpu L4
else
  COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" new -s "${SESSION}" --gpu "${PREFERRED_GPU}"
fi

COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" ssh -s "${SESSION}" -- \
  'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' </dev/null

COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" upload -s "${SESSION}" "${LOCAL_TENSORS}" /content/nfl_10hz_tensors.npz
COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" upload -s "${SESSION}" "${POS_ATTN_PY}" /content/positional_attn.py
COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" upload -s "${SESSION}" "${TRAIN_PY}" /content/train_v4_colab_gpu.py

LAUNCHER="$(mktemp /tmp/nfl2027_v4_launch.XXXXXX.py)"
cat > "${LAUNCHER}" <<PYEOF
import os, subprocess, sys
os.environ["COLAB_ACCOUNT"] = "${COLAB_ACCOUNT}"
cmd = [
    sys.executable,
    "/content/train_v4_colab_gpu.py",
    "--mode", "train",
    "--tensors-npz", "/content/nfl_10hz_tensors.npz",
    "--out-npz", "/content/v4_positional_attn_gpu_oof.npz",
    "--out-meta", "/content/v4_colab_gpu_metadata.json",
]
print("[V4-LAUNCHER] Executing:", " ".join(cmd), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
assert p.stdout is not None
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
if rc != 0:
    raise RuntimeError(f"V4 GPU training failed with rc={rc}")
PYEOF

COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" exec -s "${SESSION}" --timeout 0 -f "${LAUNCHER}"
rm -f "${LAUNCHER}"

COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" download -s "${SESSION}" /content/v4_positional_attn_gpu_oof.npz "${LOCAL_OOF}"
COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" download -s "${SESSION}" /content/v4_colab_gpu_metadata.json "${LOCAL_META}"

echo "[V4-COLAB] Explicitly stopping GPU runtime..."
COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" stop -s "${SESSION}"
echo "[V4-COLAB] GPU training & artifact retrieval complete!"
