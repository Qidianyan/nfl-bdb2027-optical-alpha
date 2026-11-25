#!/usr/bin/env bash
set -euo pipefail

# Strict Colab GPU dispatcher with guaranteed EXIT trap stop & zero-leakage artifact retrieval.
export COLAB_ACCOUNT="${COLAB_ACCOUNT:-ultra-4}"
COLAB_BIN="/Users/eligo/.local/bin/colab"
PREFERRED_GPU="${PREFERRED_GPU:-L4}"
EPOCHS="${EPOCHS:-72}"
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CODE_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
LOCAL_TENSORS="${LOCAL_TENSORS:-${CODE_ROOT}/artifacts/nfl_10hz_tensors.npz}"
LOCAL_OOF="${LOCAL_OOF:-${CODE_ROOT}/artifacts/colab_temporal_cnn_oof.npz}"
LOCAL_META="${LOCAL_META:-${CODE_ROOT}/artifacts/colab_gpu_run_metadata.json}"
TRAIN_SCRIPT="${TRAIN_SCRIPT:-${SCRIPT_DIR}/train_temporal_cnn_colab.py}"

SESSION="nfl2027-cnn-$(date +%s)"

cleanup() {
  echo "[COLAB-TRAP] Stopping session ${SESSION} on account ${COLAB_ACCOUNT} ..."
  COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" stop -s "${SESSION}" >/dev/null 2>&1 || true
  echo "[COLAB-TRAP] Verifying active sessions & billing rate:"
  COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" sessions || true
  COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" usage || true
}
trap cleanup EXIT

echo "[COLAB] Account=${COLAB_ACCOUNT} | Session=${SESSION} | Preferred GPU=${PREFERRED_GPU}"

if [[ "${PREFERRED_GPU}" == "A100" ]]; then
  COLAB_CLI_ASSIGN_RETRIES=0 COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" new -s "${SESSION}" --gpu A100 \
    || COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" new -s "${SESSION}" --gpu L4
else
  COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" new -s "${SESSION}" --gpu L4
fi

echo "[COLAB] Checking remote GPU hardware..."
COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" ssh -s "${SESSION}" -- 'nvidia-smi --query-gpu=name,memory.total --format=csv,noheader' </dev/null

echo "[COLAB] Uploading 10 Hz optical sequence tensor bundle & training script..."
COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" upload -s "${SESSION}" "${LOCAL_TENSORS}" /content/nfl_10hz_tensors.npz
COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" upload -s "${SESSION}" "${TRAIN_SCRIPT}" /content/train_temporal_cnn_colab.py

echo "[COLAB] Running Multi-Scale 1D-CNN + Attention 5-fold + Temporal LOYO training (--timeout 0)..."
LAUNCHER="$(mktemp /tmp/nfl2027_colab_launch.XXXXXX.py)"
cat > "${LAUNCHER}" <<PYEOF
import subprocess, sys
cmd = [
    sys.executable,
    "/content/train_temporal_cnn_colab.py",
    "--mode", "train",
    "--tensors-npz", "/content/nfl_10hz_tensors.npz",
    "--out-npz", "/content/colab_temporal_cnn_oof.npz",
    "--out-meta", "/content/colab_gpu_run_metadata.json",
    "--epochs", "${EPOCHS}",
]
print("[LAUNCHER] Executing:", " ".join(cmd), flush=True)
p = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True, bufsize=1)
assert p.stdout is not None
for line in p.stdout:
    print(line, end="", flush=True)
rc = p.wait()
if rc != 0:
    raise RuntimeError(f"Training script failed with exit code {rc}")
PYEOF
COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" exec -s "${SESSION}" --timeout 0 -f "${LAUNCHER}"
rm -f "${LAUNCHER}"

echo "[COLAB] Downloading trained OOF embeddings & GPU run metadata..."
COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" download -s "${SESSION}" /content/colab_temporal_cnn_oof.npz "${LOCAL_OOF}"
COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" download -s "${SESSION}" /content/colab_gpu_run_metadata.json "${LOCAL_META}"

echo "[COLAB] Explicitly stopping GPU runtime..."
COLAB_ACCOUNT="${COLAB_ACCOUNT}" "${COLAB_BIN}" stop -s "${SESSION}"
echo "[COLAB] Done!"
