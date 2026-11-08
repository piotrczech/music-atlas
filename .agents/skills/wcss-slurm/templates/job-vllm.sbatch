#!/bin/bash
# Serve or run an LLM on one Lem GPU node. For several nodes, read the
# multi-node section of references/large-models.md first: the network interface
# names and the Ray start pattern are both site-specific and both fail silently.

#SBATCH --job-name=EXP-001-llm
#SBATCH --account=YOUR_SERVICE_ACCOUNT
#SBATCH --partition=lem-gpu-short           # 3 d; lem-gpu-normal is 7 d
#SBATCH --gres=gpu:hopper:4,storage:local:400g
#SBATCH --cpus-per-task=32                  # <= 16 per GPU
#SBATCH --mem=250gb                         # <= 250 GB per GPU
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0    # the venv links against this module's libpython
module load CUDA/13.0.0                     # or deep_gemm dies on import: no CUDA_HOME (LLM-21)

PD="/lustre/pd03/${SLURM_JOB_ACCOUNT}"
MODEL="$PD/models/YOUR_MODEL"

# Every cache defaults into $HOME, which has a 50 GB quota. Move them all.
export HF_HOME="$TMPDIR/hf" PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
export DG_JIT_CACHE_DIR="$PD/llm-stack/dg-cache"   # one export per line: a second
export DG_CACHE_DIR="$PD/llm-stack/dg-cache"       # assignment cannot read the first
export TRITON_CACHE_DIR="$PD/llm-stack/triton-cache" VLLM_CACHE_ROOT="$PD/llm-stack/vllm-cache"
mkdir -p "$DG_JIT_CACHE_DIR" "$TRITON_CACHE_DIR" "$VLLM_CACHE_ROOT"

# Unpack the stack to node-local NVMe and repair the shebangs, which record the
# absolute path of the job that built the venv.
tar -xf "$PD/llm-stack/vllm.tar" -C "$TMPDIR"
for f in "$TMPDIR"/venv/bin/*; do
    [ -f "$f" ] && [ "$(head -c2 "$f" 2>/dev/null)" = '#!' ] && sed -i "1s|.*|#!$TMPDIR/venv/bin/python|" "$f"
done
PY="$TMPDIR/venv/bin/python"

cat > "$TMPDIR/run.py" <<'PYEOF'
import os, time
from vllm import LLM, SamplingParams
t0 = time.time()
llm = LLM(model=os.environ["MODEL"], tensor_parallel_size=4,
          gpu_memory_utilization=0.90, max_model_len=8192,
          trust_remote_code=True,
          # DeepSeek V4 refuses anything else; harmless elsewhere to omit.
          kv_cache_dtype="fp8")
print(f"loaded in {time.time()-t0:.0f}s", flush=True)
out = llm.generate(["Your prompt here"], SamplingParams(max_tokens=128, temperature=0))
print(out[0].outputs[0].text)
PYEOF

# Log to a file. Piping a running engine through `head` closes the pipe and
# kills it mid-load, which is indistinguishable from a hang.
MODEL="$MODEL" "$PY" "$TMPDIR/run.py" > "$PD/runs/$SLURM_JOB_ID.log" 2>&1
echo "exit $?  log: $PD/runs/$SLURM_JOB_ID.log"
