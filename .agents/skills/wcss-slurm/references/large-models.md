# Serving large language models

**TL;DR**
- A Lem GPU node is 4 H100s, 95,830 MiB each, about 374 GiB.
- Work the parallel shape out from `config.json` before requesting nodes. It
  decides the node count and is worth more than any other tuning (LLM-12).
- H100 is Hopper: FP8 is native, FP4 is not. An `expert_dtype: fp4` or MXFP4
  config targets Blackwell and is the wrong download (LLM-07).
- The site's vLLM container is v0.9.0.1 and knows none of today's
  architectures. `pip install vllm` from a compute node does (LLM-01).
- **Nothing here resolves to the right network by hostname.** NCCL finds
  InfiniBand itself; gloo and Ray must be told (LLM-04, LLM-05).
- **Read the model's output before believing any throughput number.** A broken
  layout serves nonsense at full speed with normal latencies (LLM-18).

## The green path

### One node, weights under about 300 GiB

Tensor parallelism over NVLink, CUDA graphs on, nothing else. This is the fast
tier: interactive speed, and where you want to be if the model fits.

```bash
#SBATCH --nodes=1 --gres=gpu:hopper:4,storage:local:400g
#SBATCH --cpus-per-task=60 --mem=950gb      # 4 GPUs allow 1000 GB; 250 GB gets you OOM-killed
module load Python/3.12.3-GCCcore-13.3.0    # the venv links against this module's libpython
module load CUDA/13.0.0                     # nothing else sets CUDA_HOME
python -m sglang.launch_server --model-path "$MODEL" --tp 4 --trust-remote-code
```

### Several nodes, beyond that

For a Mixture-of-Experts model, expert parallelism. Not pipeline parallelism,
which forbids CUDA graphs and collapses to a tenth of the throughput. Six nodes
at TP 24 with EP 24, verified correct end to end:

```bash
#SBATCH --nodes=6 --ntasks-per-node=1 --gres=gpu:hopper:4,storage:local:100g
#SBATCH --cpus-per-task=60 --mem=950gb
IB=$(ip -o -4 addr show | awk '$4 ~ /^10\.43\./ {print $2; exit}')
export NCCL_SOCKET_IFNAME="$IB" GLOO_SOCKET_IFNAME="$IB"
python -m sglang.launch_server --model-path "$MODEL" --trust-remote-code \
  --tp 24 --ep 24 --data-parallel-size 24 \
  --enable-dp-attention --enable-dp-lm-head --moe-dense-tp-size 1 \
  --chunked-prefill-size 12288 --page-size 64 --max-running-requests 48 \
  --mem-fraction-static 0.97 --context-length 4096 \
  --nnodes 6 --node-rank "$SLURM_NODEID" --dist-init-addr "$HEAD_IP":5000 \
  --host 0.0.0.0 --port 30000
```

Budget about 12 minutes from job start to the first healthy reply for a 1.6 TB
checkpoint, and **poll `/health` rather than sleeping a fixed time**. Pipeline
parallelism took 18 minutes for the same weights.

**Do not add `SGLANG_SHARED_EXPERT_TP1=1`**, whatever a search result tells you.
It makes an otherwise-refusing server start, and the output is then token salad
(LLM-14, LLM-18). TP 24 is chosen precisely so the flag is unnecessary.

Either runtime serves one node; the template
[job-vllm.sbatch](../templates/job-vllm.sbatch) uses vLLM and carries the cache
and shebang handling already. Across nodes the shape above was measured in
SGLang: use [job-sglang-multinode.sbatch](../templates/job-sglang-multinode.sbatch).

### Five rules that decide whether it is efficient

1. **Take the memory the GPUs entitle you to**: `--mem=950gb` on a 4-GPU node (GPU-04).
2. **Move every cache to PD.** Hugging Face, Triton, DeepGEMM and the runtime all
   default into `$HOME`, which has a 50 GB quota. One `export` per line.
3. **Build the environment once**, keep it as one tar on PD, unpack per job (LLM-03).
4. **Batch, never serve interactively.** One request gets a fraction of what the
   same GPUs deliver under load.
5. **Short wall limits while debugging**, and per-rank logs on PD so a failure
   leaves evidence after the job ends.

## Sizing, and the arithmetic that picks the shape

| | measured |
|---|---|
| HBM per GPU / per node | 95,830 MiB / about 374 GiB |
| Cores and RAM | at most 16 cores and 250 GB **per GPU** |
| Node-local NVMe | up to 7 TB with `--gres=storage:local:Ng` |
| Interconnect, all-reduce bf16 1 GiB | 337 GB/s NVLink, 152 GB/s InfiniBand (LLM-02) |
| Wall limit | 3 d on `lem-gpu-short`, 7 d on `lem-gpu-normal` |

A Mixture-of-Experts model holds every expert resident, so the weights must fit
however few are active per token.

| Checkpoint | Size | Nodes |
|---|---|---|
| DeepSeek-V4-Pro FP8 (`sgl-project`) | 1.61 TB | 6 |
| DeepSeek-V4-Pro official | 0.89 TB, FP4 experts, Blackwell only | not on Lem |
| DeepSeek-V4-Flash | 0.17 TB | 1 |

`[M:LLM-13]` **An FP8 checkpoint with `weight_block_size [128, 128]` stores one
scale per 128x128 tile, so every sharded dimension must stay a multiple of 128
after division.** Ten minutes with `config.json` saves a wasted allocation,
because otherwise the failure arrives as `output_size of gate's and up's weight
= 96 is not divisible by weight quantization block_n = 128` only after the queue
and the load. For DeepSeek V4 Pro:

| Quantity | Value | Constrains TP to |
|---|---|---|
| `moe_intermediate_size` | 3,072 = 24 x 128 | a divisor of 24 |
| `hidden_size` | 7,168 = 56 x 128 | a divisor of 56 |
| `vocab_size` | 129,280 = 2^8 x 5 x 101 | in practice a power of two |
| `n_routed_experts` | 384 = 2^7 x 3 | EP must divide 384 |

Those intersect at 8, and 8 GPUs hold 752 GB against 1.61 TB, so **pure tensor
parallelism cannot serve this model at any node count**. Expert parallelism is
the way out: whole experts go to whole ranks, so no expert tensor is cut along a
128-wide axis, and data-parallel attention plus a data-parallel head replicate
what is left. That allows 24 ranks with no pipeline stage and CUDA graphs intact.

**To choose a shape for a different model:**

1. Read `config.json` for the block size, `moe_intermediate_size`,
   `hidden_size`, `vocab_size` and the expert count.
2. Divide the checkpoint by 374 GiB for a node floor, then add a third for cache.
3. Pick the largest TP that satisfies every divisibility rule above. For a
   shared expert of width `moe_intermediate_size`, TP must divide that over 128.
4. If no valid TP is large enough and the model is MoE, add expert parallelism.
   Pipeline parallelism is the last resort, for dense models only.

## Building the stack and fetching weights

Build once in a CPU job, keep one archive on PD, unpack per node.

```bash
module load Python/3.12.3-GCCcore-13.3.0
export PIP_CACHE_DIR="$TMPDIR/pipcache"          # never $HOME
python -m venv "$TMPDIR/venv"
"$TMPDIR/venv/bin/pip" install vllm "ray[default]"
tar -C "$TMPDIR" -cf "$PD/llm-stack/vllm.tar" venv
```

`[M:LLM-03]` The archive is 8.1 GB and 82,000 files and unpacks from PD to
node-local NVMe in 10 to 13 seconds. After unpacking, repair the shebangs, which
record the absolute path of the job that built the venv (LLM-09):

```bash
for f in "$TMPDIR"/venv/bin/*; do
    [ -f "$f" ] && [ "$(head -c2 "$f")" = '#!' ] && sed -i "1s|.*|#!$TMPDIR/venv/bin/python|" "$f"
done
```

Download weights to PD, never `$HOME`, in a CPU job so no GPU hours go on the network:

```bash
export HF_HOME="$TMPDIR/hf" HF_HUB_DISABLE_XET=1
hf download <repo> --local-dir "$PD/models/<name>" --max-workers 16
```

`[M:LLM-06]` **Authenticate**, or transfers run at about 50 MB/s instead of
1.1 GB/s, nine hours instead of half an hour for 1.6 TB. Keep the token in a
mode-600 file, read it inside the job, never in the script or the log.
**Disable Xet**, whose signed URLs expire mid-download: the client aborts with
`File reconstruction error: CAS Client Error` and still exits zero having
fetched one shard of sixty-four. Loop until the shard count is right.

## When it breaks

Most of these look like a hang or an unrelated error. Find the message first.

| Symptom | Cause | Fix |
|---|---|---|
| Silent hang across nodes, no error, no timeout | vLLM's symmetric-memory rendezvous. One node only warns, which is why single-node hides it (LLM-10) | `export VLLM_ALLREDUCE_USE_SYMM_MEM=0` |
| `Unable to find address`, or gloo hangs | Hostname resolves to management ethernet, which carries no inter-node traffic. There is no `ib0` (LLM-04) | Set `NCCL_SOCKET_IFNAME` and `GLOO_SOCKET_IFNAME` from the 10.43.x interface |
| `Failed to connect to GCS ... Timed out` | `ray start` daemonised and SLURM reaped the step (LLM-05) | Background each `srun ... ray start --block &`, and pass `--node-ip-address` on InfiniBand |
| `FileNotFoundError: 'ninja'` though ninja is installed | The venv's `bin` is not on `PATH` (LLM-11) | `export PATH="$TMPDIR/venv/bin:$PATH"` in the launcher's own shell |
| `bad interpreter` running `pip`/`vllm`/`ray` | Console-script shebangs name the building job's `$TMPDIR` (LLM-09) | Rewrite shebangs after unpacking |
| `libpython3.12.so.1.0: cannot open shared object file` | The venv's python is a symlink into the module tree (PY-06) | `module load Python/3.12.3-GCCcore-13.3.0` |
| `assert cuda_home is not None` in `deep_gemm`, server dies ~30 s in | GPU nodes set neither `CUDA_HOME` nor `nvcc` (LLM-21) | `module load CUDA/13.0.0` |
| `must be divisible by page_size`, or `max_running_request is zero`, naming numbers you never typed | DP attention divides `chunked_prefill_size` and the in-flight limit by `dp_size` before validating (LLM-20) | Use multiples of `dp_size x page_size`: 12,288 and 48 at 24 ranks |
| `output_size ... = 96 is not divisible by weight quantization block_n = 128` | TP does not divide `moe_intermediate_size / 128` | Change TP. **Not** `SGLANG_SHARED_EXPERT_TP1=1`, which serves salad (LLM-14) |
| Exit 137 partway through loading a large checkpoint | Host RAM, not GPU. `--mem=250gb` on a 4-GPU node | `--mem=950gb` |
| `DeepseekV4 fp8_ds_mla layout only supports fp8 kv-cache, got auto` | KV dtype unset | `kv_cache_dtype="fp8"` |
| `Total number of attention heads (14) must be divisible by tensor parallel size (4)` | Wrong TP for the head count | Read `num_attention_heads` before choosing the shape |
| First load takes tens of minutes | About 2,000 DeepGEMM kernels JIT-compiling, cache defaulting into `$HOME` | Point `DG_JIT_CACHE_DIR`, `DG_CACHE_DIR`, `TRITON_CACHE_DIR` at PD, one export per line |
| Engine dies mid-load with no message | `... | head -20` closed the pipe and SIGPIPEd it | Redirect to a file, grep the file afterwards |
| `A: unbound variable` at job start | `export A="$PD/x" B="$A"` expands `$A` before defining it | One export per line |

**When a distributed job hangs silently, photograph the stack rather than
permuting environment variables.** `py-spy dump --pid $PID --nonblocking` on the
launcher and each child found the symmetric-memory rendezvous in four minutes,
after hours of guessing had not.

## Correctness is not throughput

`[M:LLM-18]` **A speed number says nothing about whether the output is right.**
This is not a caution, it is what happened here. An expert-parallel server
loaded all 64 shards, passed health checks, honoured concurrency limits, streamed
at 4,270 tokens/s with normal latencies, and produced this for every prompt in
every language:

```
ation,unt ... ontype >>.ughaunt  y.  aft  "):叶v"  "algorithm ()
```

Six of nine test generations never stopped and hit the token limit. Benchmarking
with `ignore_eos` hides the failure perfectly, because output is forced past the
natural stop either way. That is exactly how it got published here.

`[M:LLM-19]` **Schema-constrained JSON hides it completely.** Asked for JSON
against a schema, the same broken server returned
`{"city": "", "country": "", "population": 1, "landmarks": []}`: every required
key, correct types, passes a structural validator. Grammar-constrained decoding
masks the logits so only tokens continuing a legal document survive, which means
it manufactures well-formed output from noise. The same server could not emit
parseable JSON at all once the constraint was removed. **Never accept schema
conformance as evidence that a deployment works.**

`[M:LLM-14]` Bisected to `SGLANG_SHARED_EXPERT_TP1=1`. At TP 32 the shared
expert is 3,072 over 32 = 96 and the server refuses to start; that flag
replicates the module instead of slicing it, which starts the server and
corrupts the output. Expert parallelism, DP attention, the DP head and
`--moe-dense-tp-size 1` are all innocent: with the flag off and those three on,
the same six nodes answer Canberra, 391, and red/blue/yellow, each stopping by
itself. The flag also costs memory, pushing weights from 95.7% of HBM to 99.7%.

**The check to run before trusting any new shape:** one prompt with a verifiable
answer, `ignore_eos` off, and read the reply. Confirm it stops on its own. One
request, and it is the difference between a result and a number.

## Measured throughput

`[M:LLM-17]` One node, DeepSeek-V4-Flash, TP 4:

| | |
|---|---|
| Weights in HBM | 37.8 GiB per GPU |
| Model load | 66 to 73 s from node-local NVMe |
| KV cache | 39.65 GiB per GPU, 103,449 tokens (LLM-08) |
| One request, CUDA graphs on | 144.5 tokens/s |
| Batch of 16 | 1,345.8 tokens/s |
| Eager, batch of 2 | 13 tokens/s |

The eager figure is a floor, not a benchmark. **Turning CUDA graphs on is worth
more than a factor of ten**, and this tier is the only one that reaches an
interactive speed.

`[M:LLM-23]` Six nodes, V4 Pro, TP 24 with EP 24, the shape that answers
correctly. 512-token prompts, 256-token completions:

| Concurrency | Aggregate | Per request | Errors |
|---|---|---|---|
| 1 | 23.9 tokens/s | 23.9 | 0 |
| 8 | 186.9 tokens/s | 23.4 | 0 |
| 32 | 591.4 tokens/s | 18.5 | 0 |
| 64 | 1,234.6 tokens/s | 19.3 | 0 |
| **128** | **2,077.3 tokens/s** | 16.3 | 0 |
| 256 | 1,586.5 tokens/s | 13.5 | 64 dropped |
| 512 | - | - | server dead |

`[M:LLM-12]` For contrast, on the same checkpoint: pipeline parallelism over 6
nodes reached 5.8 tokens/s for one request and 45 aggregate, with CUDA graphs
forced off and stages idling. The broken expert-parallel shape reached 4,270
(LLM-15, LLM-16), roughly twice the correct one, because it left about 12 GiB of
cache per rank where the correct shape leaves about 1.

`[M:LLM-22]` **At TP 24 the binding limit is prompt length, not throughput.**
DP attention replicates the dense half on every rank, so weights alone need
95.6% of each GPU and the server says so exactly as
`minimum viable = 1 - available/pre = 0.9563`. What is left is the cache.

| `--mem-fraction-static` | context | result |
|---|---|---|
| 0.98 | 49,152 | dies in DeepGEMM warmup: wants 2.1 GB free, has 1.6 |
| 0.97 | 32,768 | serves short batches, then CUDA OOM |
| 0.967 | 24,576 | completes; 107 of 150 documents fit, 43 exceed the context |

This shape screens abstracts comfortably and handles documents to about 21k
tokens. More nodes do not help: TP must stay in {8, 12, 24}, so per-rank weights
are fixed. Budget context from the cache that is left, then cap prompt length.

`[M:LLM-24]` **Past about 128 concurrent it does not queue, it dies.** Triton
kernels device-load lazily after serving starts, by which point free memory is
0.11 to 0.91 GiB, and the log warns:

```
Triton kernel 'fused_moe_kernel' device-loaded after serving started (free device mem: 0.60 GiB).
Pre-load it during engine init to avoid CUDA OOM.
```

The next lazy kernel OOMs a scheduler and the server goes with it. Cap client
concurrency at the peak. Do not probe the cliff with work you care about.

## Cost

A GPU node is 4 GPU-hours per wall-clock hour, so six nodes is 24 per hour and a
3,000-hour grant buys about 125 hours. Bring-up is where the hours go: budget
several short failed jobs, keep wall limits short while debugging so a hang
costs minutes, and always write per-rank logs to PD.
