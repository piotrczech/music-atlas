#!/bin/bash
# Serve a model too large for one node, across several Lem GPU nodes.
#
# Verify every new shape: send a prompt with a known answer and read the reply.
# SGLANG_SHARED_EXPERT_TP1=1 makes a server start and emit token salad; it is
# avoided here by choosing TP 24, which divides the shared expert exactly.
#
# Read the "Sizing, and the arithmetic that picks the shape" section of
# references/large-models.md BEFORE editing this. The parallel shape is not a
# tuning knob here: on a 1.6 TB checkpoint, expert parallelism delivered 2,250
# tokens/s aggregate where pipeline parallelism delivered 45 (LLM-12). Work the
# shape out from the model's config.json first; it decides the node count.

#SBATCH --job-name=EXP-001-serve
#SBATCH --account=YOUR_SERVICE_ACCOUNT
#SBATCH --partition=lem-gpu-short            # 3 d; lem-gpu-normal is 7 d
#SBATCH --nodes=6                            # world size is nodes x 4
#SBATCH --ntasks-per-node=1                  # one launcher per node, not one per GPU
#SBATCH --gres=gpu:hopper:4,storage:local:100g
#SBATCH --cpus-per-task=60                   # <= 16 per GPU, so <= 64 on a 4-GPU node
#SBATCH --mem=950gb                          # <= 250 GB per GPU. Asking 250 on a 4-GPU
                                             # node gets the loader OOM-killed at exit 137
#SBATCH --time=2:00:00
#SBATCH --output=%x_%j.out
#SBATCH --error=%x_%j.err

set -uo pipefail
source /usr/local/sbin/modules.sh
module load Python/3.12.3-GCCcore-13.3.0     # the venv links against this module's libpython
module load CUDA/13.0.0                      # or deep_gemm dies on import: no CUDA_HOME, no nvcc (LLM-21)

PD="/lustre/pd03/${SLURM_JOB_ACCOUNT}"
MODEL="$PD/models/YOUR_MODEL"
LOG="$PD/serve-$SLURM_JOB_ID"; mkdir -p "$LOG"

# Every cache defaults into $HOME, which has a 50 GB quota. One export per line:
# a second assignment in the same export cannot read the first.
export HF_HOME="$TMPDIR/hf"
export HF_HUB_OFFLINE=1 PYTHONNOUSERSITE=1 OMP_NUM_THREADS=8
export DG_JIT_CACHE_DIR="$PD/llm-stack/dg-cache"
export DG_CACHE_DIR="$PD/llm-stack/dg-cache"
export TRITON_CACHE_DIR="$PD/llm-stack/triton-cache"
export SGLANG_CACHE_DIR="$PD/llm-stack/sgl-cache"

# NOTE: no SGLANG_SHARED_EXPERT_TP1 here, deliberately. TP 24 divides the shared
# expert (3072/24 = 128) so the escape hatch is unnecessary, and using it
# produces token salad (LLM-14).

# Nothing here resolves to the right network by hostname: the hostname gives
# management ethernet, which carries no inter-node traffic. There is no ib0.
mapfile -t NODES < <(scontrol show hostnames "$SLURM_JOB_NODELIST")
HEAD_IP=$(srun --overlap --nodes=1 --ntasks=1 -w "${NODES[0]}" bash -c \
  "ip -o -4 addr show | awk '\$4 ~ /^10[.]43[.]/ {print \$4}' | cut -d/ -f1 | head -1")
echo "nodes: ${NODES[*]}   head: $HEAD_IP   world: $((SLURM_NNODES * 4))"

# Unpack the prebuilt stack onto each node's NVMe and repair the shebangs,
# which record the absolute path of the job that built the venv (LLM-09).
srun --ntasks-per-node=1 --ntasks="$SLURM_NNODES" bash -c '
  tar -C "$TMPDIR" -xf '"$PD"'/llm-stack/sglang-py312.tar
  for f in "$TMPDIR"/venv/bin/*; do
      [ -f "$f" ] && [ "$(head -c2 "$f")" = "#!" ] &&
        sed -i "1s|.*|#!$TMPDIR/venv/bin/python|" "$f"
  done'

# --ep with data-parallel attention and head: whole experts per rank, so nothing
# is cut along the 128-wide axis, no pipeline stage, and CUDA graphs stay on.
# Per-rank logs go to PD so a failure leaves evidence after the job ends.
srun --ntasks-per-node=1 --ntasks="$SLURM_NNODES" --export=ALL bash -c \
 'export PATH="$TMPDIR/venv/bin:$PATH"
  IB=$(ip -o -4 addr show | awk "\$4 ~ /^10[.]43[.]/ {print \$2; exit}")
  export NCCL_SOCKET_IFNAME="$IB" GLOO_SOCKET_IFNAME="$IB"
  "$TMPDIR"/venv/bin/python -m sglang.launch_server \
     --model-path '"$MODEL"' --trust-remote-code \
     --tp 24 --ep 24 --data-parallel-size 24 \
     --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1 \
     --chunked-prefill-size 12288 --page-size 64 --max-running-requests 48 \
     --mem-fraction-static 0.97 --context-length 4096 \
     --nnodes '"$SLURM_NNODES"' --node-rank $SLURM_NODEID \
     --dist-init-addr '"$HEAD_IP"':5000 --host 0.0.0.0 --port 30000 \
     > '"$LOG"'/node_$SLURM_NODEID.log 2>&1' &
SRUN_PID=$!

# Poll health rather than sleeping: a 1.6 TB checkpoint took 735 s to first reply.
for i in $(seq 1 170); do
    curl -sf -m 5 "http://$HEAD_IP:30000/health" >/dev/null 2>&1 && { echo "up after $((i*15))s"; break; }
    kill -0 "$SRUN_PID" 2>/dev/null || { echo "server died; see $LOG/node_0.log"; exit 1; }
    sleep 15
done

# ---- your batch of work goes here, against http://$HEAD_IP:30000/v1/completions
# Serve this class of model in batch. One request gets ~17 tokens/s out of 32
# H100s; 256 concurrent requests share ~2,250 (LLM-15).

kill "$SRUN_PID" 2>/dev/null
