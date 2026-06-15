# AGENTS.md

Guidelines for AI coding agents (Claude Code, Cursor, Copilot, etc.) working in this repository.

## Project Context

LLMServingSim 2.0 is a cycle-level LLM serving simulator. It combines a Python frontend
(`serving/`, run as `python -m serving`) with ASTRA-Sim (C++ analytical network simulator)
as the backend. The profiling pipeline (`profiler/`) generates per-hardware latency data
that drives the simulation, and the bench module (`bench/`) runs vLLM end-to-end to
validate the simulator against ground truth.

### Repository structure

```
LLMServingSim/
├── serving/                    # Simulator (`python -m serving`)
│   ├── __main__.py             # Simulation entry point + main loop
│   ├── core/                   # Internals
│   │   ├── scheduler.py        # vLLM-style continuous batching scheduler
│   │   ├── trace_generator.py  # Builds execution traces from profiled latencies
│   │   ├── memory_model.py     # Memory tracking, KV cache, tensor sizes
│   │   ├── graph_generator.py  # Chakra protobuf graph generation
│   │   ├── controller.py       # IPC with ASTRA-Sim subprocess
│   │   ├── router.py           # Request routing across instances
│   │   ├── gate_function.py    # MoE expert token routing
│   │   ├── config_builder.py   # Cluster config → ASTRA-Sim input files
│   │   ├── power_model.py      # Power/energy estimation
│   │   ├── pim_model.py        # PIM device model
│   │   ├── request.py          # Request/Batch data classes
│   │   ├── radix_tree.py       # Prefix cache radix tree (from SGLang)
│   │   ├── logger.py           # Rich-based logger + stdio capture
│   │   └── utils.py            # Model config loading, formatting
│   └── run.sh                  # Example invocations across cluster configs
├── configs/
│   ├── cluster/                # Cluster topology configs (hardware, memory, instances)
│   ├── model/                  # Model architecture configs (subset of HF config.json)
│   └── pim/                    # PIM device configs (DRAMSim3 INI format)
├── workloads/                   # Request trace datasets (.jsonl)
│   └── generators/             # ShareGPT/etc → JSONL workload generators
├── profiler/                   # vLLM-based layerwise profiler (`python -m profiler`)
│   ├── __main__.py             # CLI dispatch (profile / slice)
│   ├── core/                   # internals
│   │   ├── runner.py           # Orchestration (spin_up → categories → spin_down)
│   │   ├── config.py           # Architecture / ProfileArgs / engine defaults
│   │   ├── engine.py           # vLLM lifecycle (tmpdir-based local config load)
│   │   ├── categories.py       # Dense / PerSequence / Attention / Expert
│   │   ├── skew.py             # Heterogeneous-decode skew sweep
│   │   ├── fit_alpha.py        # 5-axis weighted-LS alpha fit
│   │   ├── writer.py           # CSV + meta.yaml writer, TP-stable replication
│   │   ├── logger.py           # Rich-based logger + stdio capture
│   │   └── hooks/              # vLLM-internal-API touchpoints (worker ext, MoE patch, etc.)
│   ├── models/                 # Architecture yamls, one per HF `model_type`
│   ├── power/                  # nvidia-smi / IPMI power-logging helpers
│   ├── perf/                   # Output: perf/<hw>/<model>/<variant>/tp<N>/{dense,per_sequence,attention,moe,skew,skew_fit}.csv
│   ├── v0/                     # Legacy (pre-rewrite) profiler, kept for reference
│   ├── profile.sh              # Editable user template (MODEL / HARDWARE / TP_DEGREES / …)
│   └── profile-all.sh          # Helper: sweeps several MODELs × TP degrees
├── bench/                      # vLLM end-to-end benchmark + sim validation (`python -m bench`)
│   ├── __main__.py             # CLI dispatch (run / validate)
│   ├── core/                   # internals
│   │   ├── runner.py           # AsyncLLM driver, captures RequestStateStats
│   │   ├── recorder.py         # writes meta.json / requests.jsonl / timeseries.csv
│   │   ├── stat_logger.py      # custom vLLM StatLoggerBase that fills timeseries
│   │   ├── validate.py         # bench-vs-sim comparison entry point
│   │   ├── plots.py            # throughput / running-waiting / latency-CDF plot helpers
│   │   └── logger.py           # Rich-based logger + stdio capture
│   ├── results/                # output: bench/results/<run_id>/
│   ├── bench.sh                # host-side wrapper for `python -m bench run`
│   └── validate.sh             # host-side wrapper for `python -m bench validate`
├── scripts/                    # Shared shell entry points (env / build, not module-specific)
│   ├── docker-vllm.sh          # vLLM container (profiler + bench)
│   ├── docker-sim.sh           # simulator container
│   ├── install-vllm.sh         # bare-metal vLLM install (uv venv)
│   └── compile.sh              # ASTRA-Sim + Chakra build
└── astra-sim/                  # ASTRA-Sim C++ backend (submodule)
    ├── inputs/                 # Generated configs (network, memory, system)
    └── extern/graph_frontend/chakra/  # Chakra trace converter
```

Per-paper artifact evaluation scripts (the previous `evaluation/`
directory) live on dedicated branches (`ispass26-artifact`, etc.) and
are not part of the main branch's tree.

### Simulation flow

1. `serving/__main__.py` parses CLI args and cluster config
2. `config_builder.py` generates ASTRA-Sim input files (network.yml, system.json, memory_expansion.json)
3. ASTRA-Sim subprocess is launched
4. Per iteration:
   - `scheduler.py` forms a batch under memory and token budget constraints
   - `trace_generator.py` looks up profiled latencies and emits a text trace
   - `graph_generator.py` converts the trace to a Chakra protobuf graph
   - `controller.py` feeds the graph path to ASTRA-Sim, reads back cycle count
   - `scheduler.py` updates request state, marks completions
5. Results are printed and optionally saved to CSV

### Key data flow

```
profile.csv (profiled latencies)
    ↓ _load_perf_db() + _lookup_latency_ns()
trace_generator.py → text trace file
    ↓ Chakra converter
graph_generator.py → .et protobuf file
    ↓ stdin/stdout IPC
ASTRA-Sim (C++) → cycle count
    ↓
scheduler.py → next iteration
```

## Code Style & Formatting

- **Python**: 4-space indentation, snake_case for functions/variables, PascalCase for classes
- **No enforced formatter** — match surrounding code style in the file you're editing
- **CLI flags**: use hyphens (`--cluster-config`, `--max-num-seqs`)
- **Internal Python**: use underscores (`max_num_seqs`, `enable_chunked_prefill`)
- **JSON config filenames**: descriptive snake_case (`single_node_pim_instance.json`)
- **Imports**: keep minimal and consistent; `serving/` modules use relative imports
- **Comments**: use English only — no Korean or other non-English text in comments, docstrings, or log messages

## Architecture Patterns

### Profiler (`profiler/`)
The profiler uses vLLM's built-in `layerwise_profile()` via a worker extension class to
capture per-layer CUDA kernel timings from real vLLM execution paths. Architecture is
dispatched by the HF config's `model_type` field against YAML catalogs under
`profiler/models/<model_type>.yaml`, which bind canonical layer names (dense /
per-sequence / attention / moe) to vLLM class names.

Every TP degree is profiled on a **single GPU**: the engine is always booted with
`tensor_parallel_size=1`, and per-rank shapes are emulated by dividing `SHARD_FIELDS`
(e.g. `hidden_size`, `num_attention_heads`) by TP via `hf_overrides`. Collective
timings are left to ASTRA-Sim. The model's full `config.json` (read from
`configs/model/<org>/<name>.json`, or auto-fetched from the HF Hub on first run)
is written to a tmpdir at spin-up so vLLM never needs Hub access.

Attribution: the base layerwise-profile methodology (worker-extension hook into
vLLM's `layerwise_profile()`, single-GPU TP emulation via `hf_overrides`) is
adapted from [@waneon](https://github.com/waneon). The unified 4D attention
sweep, the heterogeneous-decode skew sweep in `profiler/core/skew.py`, and
the 5-axis weighted-LS alpha fit in `profiler/core/fit_alpha.py` are
developed in this repo.

Each run produces a per-category CSV bundle:

```
perf/<hw>/<model>/<variant>/
  meta.yaml                              profiler/vLLM version, effective engine kwargs, GPU,
                                         timestamps, compact sweep specs, skew_fit summary
  tp<N>/
    dense.csv                            layer, tokens, time_us
    per_sequence.csv                     layer, sequences, time_us
    attention.csv                        prefill_chunk, kv_prefill, n_decode, kv_decode, time_us
    moe.csv                              tokens, activated_experts, time_us   (MoE only)
    skew.csv                             raw heterogeneous-decode shots        (skew enabled)
    skew_fit.csv                         fitted per-bucket alpha table         (skew enabled)
```

`<variant>` is auto-derived from weight + KV dtype (e.g. `bf16`, `bf16-kvfp8`,
`fp8-kvfp8`) unless `--variant` is set. Times are in **microseconds**. Layers marked
`tp_stable: true` in the yaml (layernorms, sampler) are profiled once at TP=1 and
replicated into other `tp<N>/` folders by the writer.

The profiler Docker uses **vLLM v0.19.0** (`vllm/vllm-openai:v0.19.0` or
`v0.19.0-cu130` for CUDA 13.x). The MoE hook patches `FusedMoE.forward_native` for
forced expert routing — method name is version-specific.

### Skew profiling & alpha fit
FlashAttention's varlen kernel pays tile-padding + SM-imbalance costs when a
decode batch has non-uniform kv lengths. The uniform attention grid can't see
that (every shot uses a single kv_decode value), so `skew.py` runs a second
sweep on bimodal batches and measures three latencies per case — `t_mean`
(all decodes at the batch mean), `t_max` (all at the max), and `t_skew` (the
actual bimodal mix). The normalised alpha ∈ [0, 1],
`alpha = (t_skew − t_mean) / (t_max − t_mean)`, tells the simulator how far
along the mean→max line a skewed batch lands.

- **Sweep structure**: Tier 1 is a factorial over `(n, ratio, pc, kp, kvs)`
  at `_SKEW_REP = 4.0`; Tier 2 adds a skew-axis sweep at a handful of anchor
  pivots (`skew ∈ {1.5, 2, 4, 8, 16}`). Any CLI `SKEW_<axis>_FACTOR`
  (default 2.0) coarsens that axis geometrically — higher = faster, lower
  = denser. Factors and grid specs land in `meta.yaml::skew_profile`.
- **Fit**: `fit_alpha.py` groups rows by the 5-axis key
  `pc | n_label | skew_rate_label | kv_big_label | kp_label` and runs a
  weighted least-squares fit per cell. Axis ablation on the widened
  ~13k-sample dataset picked the 5-axis scheme (test p50/p90 ≈ 2.7% / 14.8%
  on TP=1 vs 3.5% / 16.4% for the previous 3-axis fit).
- **Data-driven bucket axes**: `n` and `kp` get one bucket per unique
  profiled value (`kp=0` sentinel + overflow), `kv_big` uses log-4x bins
  extended to the observed max, `skew_rate` is a fixed normalised [0, 1]
  scheme, and `pc` is keyed raw. Derived axes are written to
  `meta.yaml::skew_fit.bucket_axes`; the simulator reads them from there
  so widening the profile sweep lights up finer resolution without any
  simulator code change.
- **Storage**: the full (bucket → alpha) mapping spills to
  `tp<N>/skew_fit.csv` with columns `pc, n_label, skew_rate_label,
  kv_big_label, kp_label, alpha, n_samples`. `meta.yaml::skew_fit.per_tp[tp]`
  keeps only a summary (`method`, `n_samples`, `alpha_default`,
  `rel_err_p50/p90/p99`, `signed_mean`, `bucket_table` pointer). This
  turns meta.yaml from ~3100 lines into ~100 lines per variant. The
  simulator hydrates the CSV back into memory on `_load_perf_db()`.
- **Disable**: `SKIP_SKEW=1` skips the sweep entirely (simulator falls
  back to a pooled constant alpha). `ONLY_SKEW=1` skips every other
  category and refreshes just `skew.csv` + `skew_fit.csv`.

### Feasibility bounds shared by attention and skew
Both the uniform attention sweep and the skew sweep cap `n_reqs > max_num_seqs`
(strict `>`, not `>=`) so that `n = MSQ` **pure** cases (no prefill chunk) fit.
This uses vLLM V1's `input_batch` buffer exactly up to `MSQ`. Mixed cases at
`n = MSQ` need `MSQ + 1` requests and are still filtered. If a runtime workload
needs mixed-regime data at `n = X`, profile with `MAX_NUM_SEQS ≥ X + 1`.

### Canonical layer names (simulator ↔ profiler, unified)
The simulator consumes the profiler's per-category CSVs directly. Canonical
layer names match vLLM's own attribute names. `trace_generator` walks the
`sequence:` section of `profiler/models/<model_type>.yaml`; the table below
lists where each layer appears in the profiler CSVs and how the simulator keys
the lookup.

| Layer | Category (CSV) | Key semantics |
|-------|----------------|---------------|
| `embedding` | dense | `tokens = total_len` |
| `layernorm` | dense (tp_stable) | `tokens = total_len` |
| `qkv_proj` | dense | `tokens = total_len` |
| `qk_norm` | dense (tp_stable; Qwen3 only) | `tokens = total_len` |
| `rotary_emb` | dense | `tokens = total_len` |
| `attention` | attention | `(prefill_chunk, kv_prefill, n_decode, kv_decode)` |
| `o_proj` | dense + ALLREDUCE after (TP>1) | `tokens = total_len` |
| `gate_up_proj` | dense | `tokens = total_len` |
| `act_fn` | dense | `tokens = total_len` |
| `down_proj` | dense + ALLREDUCE after (TP>1) | `tokens = total_len` |
| `final_layernorm` | dense (tp_stable) | `tokens = total_len` |
| `lm_head` | per_sequence | `sequences = num_requests` |
| `sampler` | per_sequence (tp_stable) | `sequences = num_requests` |
| `moe` | moe (always profiled at tp=1; wrapped in EP ALLTOALL) | `(local_tokens, activated_experts)` |

### Trace generator structure
`trace_generator.py` walks the architecture yaml's `sequence:` section to emit
each iteration. Composable helpers:
- `resolve_variant()` / `_load_perf_db()` / `_load_architecture()` — resolve
  the variant folder, load meta.yaml, load per-category CSVs, and attach the
  architecture catalog + sequence.
- `_lookup_dense()` / `_lookup_per_sequence()` / `_lookup_attention()` /
  `_lookup_moe()` — category-specific lookups. Attention uses 4D lookup
  (nearest-neighbour on `prefill_chunk, n_decode`, bilinear on
  `kv_prefill, kv_decode`).
- `_emit_sequence()` — walks a list of canonical names from the yaml, attaches
  TP ALLREDUCE to `o_proj`/`down_proj`, swaps in PIM attention before the
  NPU attention kernel when offloading is enabled, and one-shot-warns when a
  sequence layer is missing from the profile CSVs.
- `_emit_prologue()` / `_emit_pre_attn_layers()` / `_emit_post_attn_layers()` /
  `_emit_final_layers()` — thin wrappers over `_emit_sequence`.
- `_synthesize_interleaved_trace()` — alternates two `BatchCtx` objects for
  sub-batch interleaving.
- `_emit_final_layers()` — final_layernorm → lm_head → sampler (sampler output goes to REMOTE)

### Trace file format
Each trace is a tab-separated text file consumed by the Chakra converter:

```
COLOCATED		model_parallel_NPU_group: {npu_group}
{num_layers}
Layername    comp_time    input_loc    input_size    weight_loc    weight_size    output_loc    output_size    comm_type    comm_size    misc
embedding_0  5621         REMOTE:0     40            LOCAL         1050673152     LOCAL         81920          NONE         0            NONE
...
sampler_291  25933        LOCAL        2565120       LOCAL         0              REMOTE:0      40             NONE         0            NONE
```

- `comp_time`: latency in nanoseconds (from profile.csv, converted at load time)
- `input_loc`/`weight_loc`/`output_loc`: `LOCAL` (NPU), `REMOTE:{node_id}` (CPU), `CXL:{id}`
- `comm_type`: `NONE`, `ALLREDUCE`, `ALLTOALL`, or with dimension scoping `ALLREDUCE:1,0`, `ALLTOALL:0,1`
  (the `:dim0,dim1` suffix maps to ASTRA-Sim's `involved_dim` BoolList for multi-dimensional topologies)
- `misc`: `NONE` or batch tag for sub-batch interleaving (`BATCH_1`, `BATCH_2`)
- First layer (embedding) input comes from `REMOTE` (CPU → NPU), last layer (sampler) output goes to `REMOTE` (NPU → CPU)
- MoE uses `EXPERT {i}` / `EXPERT END` markers (comm_type on EXPERT line can include dimension scoping)
- PIM uses `PIM {channel}` / `PIM END` markers

### Performance DB and latency lookup
The simulator loads per-category CSVs via `_load_perf_db()` and dispatches
lookups by catalog category: `_lookup_dense` (1D linear over tokens),
`_lookup_per_sequence` (1D linear over sequences), `_lookup_attention` (4D:
nearest-neighbour on `(prefill_chunk, n_decode)` + bilinear on `(kv_prefill,
kv_decode)`), and `_lookup_moe` (2D over `(tokens, activated_experts)`,
profiled at tp=1). All lookups extrapolate (time_us is linearly
extended) rather than clamping. Latencies are stored as microseconds in the
CSVs and converted to nanoseconds at load time. No calibration scaling —
profiled latencies are used directly.

Attention with skew correction: `_lookup_attention_with_skew` does two 4D
lookups (at `kv_decode_mean` and `kv_decode_max`) and blends them using
`alpha` resolved from `meta.yaml::skew_fit` by `_skew_alpha`. The bucket key
is `pc={pc}|{n_label}|{sr_label}|{kvb_label}|{kp_label}`, built against
`skew_fit.bucket_axes` from the meta (falling back to module defaults for
older profiles). `_hydrate_skew_fit_tables()` reads each TP's `skew_fit.csv`
into the in-memory `alpha_by_bucket` map on first load.

Profile CSV path: `profiler/perf/<hardware>/<model>/<variant>/tp<N>/{dense,
per_sequence,attention,moe,skew,skew_fit}.csv` (resolved as
`../profiler/perf/...` from the `astra-sim/` working directory).

Variant resolution: `trace_generator.resolve_variant(dtype, kv_cache_dtype,
model_config)` mirrors the profiler's `effective_variant` — weight dtype is
the CLI value or `torch_dtype` from the model config (default `bfloat16`),
KV dtype appends a `-kv<short>` suffix when not `auto`. Runtime lookups verify
the resulting folder exists; a mismatch raises a clear `FileNotFoundError`
pointing at the missing variant.

FP8 KV cache (`--kv-cache-dtype fp8`) resolves to a `<dtype>-kvfp8` variant
folder (e.g. `bf16-kvfp8`). The `kv_cache_dtype` parameter is threaded through
`generate_trace` → `resolve_variant` → `_load_perf_db`. In `memory_model.py`,
`kv_fp` is 1 byte for fp8 (vs `fp` for others), halving KV cache memory usage.

Runtime vs. profiled warnings: on first load of a `(hardware, model, variant)`,
the simulator compares the CLI's `--max-num-batched-tokens` and `--max-num-seqs`
against `meta.yaml`'s `engine_effective` values and logs a one-shot warning
when the runtime exceeds the profiler's sweep bounds (lookups will extrapolate).

### Agentic session support (dependency chains)
The simulator supports closed-loop agentic workloads (SWE-bench, tool-calling agents)
where LLM calls within a session form a dependency chain interleaved with tool calls.

**Dataset format:** Each JSONL line is a session with `sub_requests[]`. Each sub-request
has `input_toks`, `output_toks`, `tool_duration_ns` (wait time after this LLM call before
the next can start). Flat requests (no `sub_requests` key) are also supported for backward
compatibility. Both formats can coexist in the same file.

**Router dependency tracking** (`router.py`):
- `load_requests()` auto-detects flat vs agentic format. For agentic sessions, only the
  first sub-request is queued; the rest are stored in `_deferred_sessions`
- `notify_request_completed(request_id, completion_time_ns)` releases the next sub-request
  at `completion_time + tool_duration_ns` and inserts it sorted into `_pending_requests`
- `has_deferred_sessions()` prevents premature simulation exit while sessions are in-flight
- `scheduler.add_request()` uses `bisect.insort` (not `append`) to maintain arrival-time
  sort order when dynamically released sub-requests enter the queue

**Time advancement:** When all instances are idle but deferred sub-requests have future
arrival times (tool calls still running), `serving/__main__.py` advances `current` to the next pending
arrival time to avoid busy-looping.

### Scheduler and memory model
- `scheduler.py` implements vLLM-style continuous batching with chunked prefill (default on)
- Token budget controlled by `--max-num-batched-tokens` (default 2048) and `--max-num-seqs` (default 128)
- `--long-prefill-token-threshold` caps per-request tokens per step for chunked prefill
- KV cache is managed in blocks of `--block-size` tokens (default 16)
- Prefix caching via RadixAttention is enabled by default (`--enable-prefix-caching`)
- Memory tracking in `memory_model.py` covers NPU, CPU, and CXL tiers
- `calculate_sizes(parallel=)` computes per-layer tensor sizes — `parallel` is TP for dense
  layers and EP for MoE experts. Uses `head_dim`, `q_dim`, `kv_dim`
- MoE expert weights are sharded by `ep_size` (not `tp_size`)
- Prompt throughput (`prompt_t` in `add_done()`) includes prefix cache hit tokens,
  not just actually computed prefill tokens. This matches vLLM's reported prompt
  throughput which counts all input tokens including cached ones

### CLI argument conventions
CLI flags follow vLLM naming where applicable:
- `--dtype` (`float16`, `bfloat16`, `float32`, `int8`) — model weight precision
- `--skip-prefill` — skip the prefill phase (decode only)
- `--request-routing-policy` (`LOAD`, `RR`, `RAND`, `CUSTOM`) — request routing across instances
- `--expert-routing-policy` (`BALANCED`, `RR`, `RAND`, `CUSTOM`) — expert token routing for MoE
  (block-copy optimization is controlled separately via `--enable-block-copy`, default on)
- Boolean flags use `argparse.BooleanOptionalAction` (e.g., `--enable-prefix-caching` /
  `--no-enable-prefix-caching`)

### Head dimension
Some models (e.g., Qwen3) have `head_dim != hidden_size // num_attention_heads`. Always use:
```python
head_dim = config.get('head_dim', n_embd // n_head)
q_dim = n_head * head_dim        # NOT n_embd
kv_dim = kv_head * head_dim      # NOT n_embd // group
```

### Model configs
Model architecture configs live in `configs/model/{org}/{model}.json`. These are subsets
of HuggingFace `config.json` containing fields the simulator needs (`hidden_size`,
`num_attention_heads`, `num_hidden_layers`, `num_key_value_heads`, `intermediate_size`,
`vocab_size`, `head_dim`, `num_local_experts`, `num_experts_per_tok`).

The simulator loads these via `get_config(model_name)` in `utils.py`.

### Cluster configs
Cluster configs in `configs/cluster/` define hardware topology. Key instance fields:
- `hardware`: must match a directory name in `profiler/perf/<hardware>/`
- `model_name`: must match a config in `configs/model/{model_name}.json`
- `num_npus`: total GPUs for the instance (optional, inferred from `tp_size * pp_size`)
- `tp_size`: tensor parallel degree (required or inferred)
- `pp_size`: pipeline parallel degree (optional, default 1)
- `ep_size`: expert parallel degree (optional, default `tp_size` for MoE, 1 for dense)
- `dp_group`: DP group ID string (optional, instances with same string share experts)
- `npu_mem.mem_bw`: NPU memory bandwidth (also set as `local-mem-bw` in system.json)
- `cpu_mem.mem_bw`: CPU memory bandwidth (set as remote memory in memory_expansion.json)
- `link_bw`: inter-node bandwidth in GB/s (set in network.yml)
- `link_latency`: inter-node link latency in ns

Parallelism inference: users may provide partial info (e.g., `num_npus=4, tp_size=2`)
and `config_builder.py` infers the rest (`pp_size=2`). Validation ensures
`num_npus = tp_size * pp_size` and `ep_size` divides `num_local_experts`.

TP and EP share the same GPUs: non-MoE layers use TP (ALLREDUCE), MoE layers use EP
(ALLTOALL). DP is achieved via multiple instances with the same `dp_group`.

`config_builder.py` reads the cluster config and generates three ASTRA-Sim input files:
- `astra-sim/inputs/network/network.yml` — topology and bandwidth
- `astra-sim/inputs/system/system.json` — scheduling policy and memory bandwidth
- `astra-sim/inputs/memory/memory_expansion.json` — remote (CPU) memory config

### Working directory
`serving/__main__.py` changes cwd to `astra-sim/` early in execution. All relative paths in the simulator
resolve from `astra-sim/`, not the repo root. Paths to `configs/`, `workloads/`, `profiler/`
are relative to the repo root and prefixed with `../` in code.

### Communication sizes for ASTRA-Sim
ASTRA-Sim expects the **total** data size for collectives (not per-NPU). It divides by N
internally (`msg_size = data_size / nodes_in_ring`).
- ALLREDUCE on `o_proj` and `down_proj`: pass full output tensor size
- ALLTOALL for MoE: pass full activation tensor size

### Multi-dimensional topology and `involved_dim`
For DP+EP configurations, the network topology is 2D: `npus_count: [tp_size, dp_group_size]`.
Collectives are scoped to specific dimensions via the `involved_dim` BoolList attribute
on COMM_COLL_NODE protobuf nodes:
- ALLREDUCE (TP): `involved_dim=[True, False]` — dim 0 only
- ALLTOALL (EP): `involved_dim=[False, True]` — dim 1 only (or `[True, True]` if EP spans TP+DP)

The `involved_dim` is encoded in the trace `comm_type` field as `ALLTOALL:0,1` (parsed by
the Chakra converter's `_parse_comm_type`). ASTRA-Sim's `Workload::issue_comm()` reads this
and passes it to `generate_all_to_all()`, which skips dimensions where `involved_dim` is false.

The `system.json` collective implementations must have one entry per topology dimension
(e.g., `"all-to-all-implementation": ["ring", "ring"]` for 2D). `config_builder.py`
generates this automatically based on whether DP groups are present.

### MoE expert blocks
Expert blocks use `EXPERT {i}` / `EXPERT END` markers for ASTRA-Sim. Each EP rank
gets a per-rank latency from profiled data based on its local token count and activated
experts (`key_0=local_tokens, key_1=activated_experts`, profiled at tp=1). Ranks execute
in parallel and sync at the ALLTOALL barrier. Expert-to-rank assignment uses even
partitioning: `expert_id * ep_size // num_experts`.

### DP+EP wave synchronization
For DP groups (instances with the same `dp_group`), wave synchronization is achieved
through two mechanisms:
1. **Python-side dp_pending barrier**: trace generation is deferred until all DP group
   members have scheduled their batches. The ALLTOALL `comm_size` is synchronized to
   `max(total_len) * hidden_size * fp` across the group.
2. **ASTRA-Sim ALLTOALL barrier**: all DP group instances' `.et` files are placed in a
   shared workload folder. The ALLTOALL collectives in both files have matching stream
   IDs, causing ASTRA-Sim to block until both NPUs reach the collective.

When one DP instance is idle (no requests), a dummy batch (1 decode token) is created
so it can participate in the ALLTOALL sync. When one instance finishes all requests,
it continues generating dummy batches until all DP group members are done.

### Chakra graph converter
The Chakra converter (`astra-sim/extern/graph_frontend/chakra/src/converter/llm_converter.py`)
transforms text traces into protobuf `.et` files. It creates:
- `MEM_LOAD_NODE` for the first layer's input (from REMOTE/CPU memory)
- `COMP_NODE` for each computation layer
- `MEM_STORE_NODE` for the last layer's output (to REMOTE/CPU memory)
- `COMM_COLL_NODE` for ALLREDUCE/ALLTOALL (with optional `involved_dim` BoolList attribute)

The converter parses `comm_type` strings like `ALLTOALL:0,1` via `_parse_comm_type()`,
splitting into `comm_type="ALLTOALL"` and `involved_dim=[False, True]`.

The MEM_STORE node uses the **last layer's** `output_memory_loc`. This is why the sampler
(not lm_head) must have `output_loc=REMOTE:{node_id}`.

Memory location types: `LOCAL` (NPU) = 1, `REMOTE` (CPU) = 2, `CXL` = 3, `STORAGE` = 4.
These must match the C++ enum in `astra-sim/astra-sim/system/AstraMemoryAPI.hh`.

### Docker environments
- **vLLM container** (used by `python -m profiler`, `python -m bench`, and
  `python -m workloads.generators`): `vllm/vllm-openai:v0.19.0` (or
  `v0.19.0-cu130` for CUDA 13.x)
  - Launched via `scripts/docker-vllm.sh`
  - Mounts the **LLMServingSim repo root** as `/workspace`; container cwd
    is `/workspace`, so `python -m profiler …` etc. work directly
  - Pre-installs `datasets` and `matplotlib` on first start (extra deps
    used by the workload generator and bench plots; vLLM brings the rest)
  - Set `HF_TOKEN` in `scripts/docker-vllm.sh` for gated-config auto-download
- **Simulator container**: `astrasim/tutorial-micro2024` + Python deps
  - Launched via `scripts/docker-sim.sh`
  - Mounts the repo root at `/app/LLMServingSim`; ASTRA-Sim + Chakra are
    built inside via `scripts/compile.sh` on first use

## Physical Design Constants (Glass-Photonic FB) — 확정값

### Waveguide bandwidth & N_WG convention
- 1 WG = 32 λ × 32 Gb/s = 1.024 Tb/s = **128 GB/s (unidirectional)**. WG는 단방향
  매체 → 양방향은 TX WG + RX WG 별도.
- **N_WG = 방향당(per-direction) WG 수** (canonical 정의). ASTRA 입력은 단방향:
  `intra_opt_bw = N_WG × 128 GB/s`.
- bundle 매핑: x{K} bundle = K WG total = K/2 per direction.
  예: 6×6-4c cap **x6 = 3 TX + 3 RX = N_WG=3** (단방향 384 GB/s); 4×4 cap **x10 = N_WG=5**
  (640 GB/s). **cap은 격자마다 다르다** (아래 표) — N_WG=3은 6×6 cap이지 보편값 아님.

### Micro-bump 제약 → 격자별 N_WG 상한 (CRITICAL)
WG 수 상한은 micro-bump 예산과 FlattenedButterfly degree로 결정된다 (격자별로 다름).

기본 상수:
- 1 WG = 32 λ → 32 data micro-bumps (single-ended)
- 타일 1600 mm² (2 × 800 mm² dies), bump 밀도 1.083 bumps/mm² (PanelScale 실측: 832/768mm²)
- 예산 = 1600 × 1.083 ≈ 1733 bumps → **54 WG/GPU** (양방향 총합)

계산: 54 WG/GPU ÷ degree[=(rows-1)+(cols-1)] = WG/pair(총) ÷ 2 = 방향당 N_WG 상한.

| 격자 | degree | WG/pair(총) | N_WG/dir | cap(floor) | 그래프선(ceil) | bundle |
|------|--------|-------------|----------|------------|----------------|--------|
| 4×4  | 6      | 9.0         | 4.5      | 4          | 5              | x9~10  |
| 6×6 / 6×6-4c | 10 | 5.4      | 2.7      | 2          | 3              | x6     |
| 8×8  | 14     | 3.9         | 1.9      | 1          | 2              | x4     |

- degree↑(큰 격자) → pair 많아 가늘어짐 → 상한↓; degree↓(작은 격자) → 굵어짐 → 상한↑.
- **feasible(N_WG ≤ cap)** vs **aggressive(N_WG > cap)**: aggressive는 타일 면적 증대 /
  bump 밀도 상향으로만 정당화 (예: 6×6 N_WG=3 = 60 WG/GPU = 1733 대비 11% 초과;
  우리 타일이 PanelScale XPU 768mm²의 2배라 밀도 1.08→~1.2로 올리면 가능).
- trade-off: **4×4** = pair당 BW 높음(cap 5)·제조 쉬움 / 패널당 16 GPU → inter-panel 일찍.
  **6×6-4c** = 패널당 32 GPU(큰 EP 단일 패널) / pair당 BW 제한(cap 3).
- 코드: `compute_nwg_cap(rows, cols)` 가 degree·상한 자동 계산. `plot_panel_dse.py`가
  격자별 cap선(수직 점선, ceil) + aggressive 음영을 그린다.
- **논문 결론 구조: breakeven N_WG < feasible cap → 실현가능 + NVL72 능가 동시 증명**
  (단, 아래 'Collective topology 주의'의 inter-panel 모델 수정 후 재확인 대상).

### Power
- energy efficiency = 1.15 pJ/bit (PanelScale Table 1)
- interconnect_power = total_WG × 1.024e12 bit/s × 1.15e-12 J/bit
- x5 (50 WG/GPU) → 59 W (5.9% TDP), x6 (60 WG/GPU) → 71 W (7% of 1 kW TDP)
- vs NVL72 NVSwitch **540 W/rack** (36 × 15 W); 우리는 passive WG → switch 전력 0
- per-GPU 집계 BW: 글래스 7.68 TB/s vs NVL72 1.8 TB/s = **4.3×**

### Optical link latency (TeraPHY 기반)
- Ref: Wade et al., "TeraPHY: A Chiplet Technology for Low-Power, High-Bandwidth
  In-Package Optical I/O," Hot Chips 2019 / IEEE Micro 2020
- 분해: E/O+O/E (CPO) ~10 ns + glass propagation ~2 ns (dist-5 324mm, n=1.5)
  + WDM mux/demux (AWG) ~수 ns
- **intra-panel optical = 100 ns (보수적 기본값)** = CPO 10 + WDM + IOX coupling + 마진.
  ablation 대안 30 ns (실측 기반).
- **inter-panel fiber = 500 ns** (co-located: edge transceiver ~50-200ns + 짧은
  fiber 1-2m ~5-10ns = intra의 5배). 5000ns는 ~1km fiber라 비현실적이라 폐기.
  ablation: 500↔5000 ns가 TPOT <1% (BW-bound) → 값 자체는 결과에 robust.

### Latency 비교 (시뮬레이션 입력)
| 연결 | latency | 근거 |
|------|---------|------|
| NVL72 intra-rack | 1000 ns | NVSwitch hop |
| 우리 intra-panel | 100 ns | TeraPHY CPO (switch 없음, 10× 낮음) |
| 우리 inter-panel | 500 ns | edge transceiver + 짧은 fiber (intra의 5배; <1% 영향) |

### Sweep & NVL72 baseline
- **N_WG sweep = [1, 2, 3, 4, 5, 6, 8]** (각 격자 cap선 포함: 4×4→5, 6×6→3)
- **epscale/headline은 각 패널을 자기 cap의 N_WG로 실행**: 4×4 → `--fixed-wg 5`,
  6×6-4c → `--fixed-wg 3`. ⚠️ 단일 N_WG를 모든 패널에 쓰지 말 것 — 4×4를 wg3로 돌리면
  intra optical 384 GB/s(제대로면 640)로 glass 과소provisioning.
- intra latency 고정 = 100 ns
- NVL72 baseline: **BW 900 GB/s unidirectional** (1800 bidir ÷2), **latency 1000 ns**,
  inter-rack IB 50 GB/s @EP>64

### Convention 주의 (uni/bi)
- ASTRA BW 입력은 **unidirectional**. NVL72·글래스 모두 단방향으로 통일 적용.
- 양쪽 동일 convention이면 절대값과 무관하게 비율(4.3×)·breakeven 유지.

### MoE collective 모델 (해결 — reach 헤드라인의 핵심)
- **근본 원인**: MoE dispatch/combine을 vLLM 기본 **AllGather/ReduceScatter**로 모델하면
  multi-dim에서 계층 분해돼 느린 inter-domain(IB / inter-panel)에 데이터가 조금만 실려
  → NVL72 IB cliff 과소 → crossover 미발현.
- **수정**: 大EP 실제 통신은 **router-directed true all-to-all (DeepEP)** — 토큰이 흩어진
  expert rank로 직접 가 locality가 없으니 cross-domain 전량이 느린 링크를 탄다.
  `MOE_ALLTOALL=1` env로 dispatch/combine을 ALLTOALL로 전환 (`trace_generator._emit_moe_block`),
  default off (기존 동작 보존).
- **검증 (mini rack=4, EP 4→32, batch128, glass wg5)**: NVL72/glass TPOT 비 =
  1.05 → 1.47 → 3.23 → **6.39×**. NVL72 exposed 4%→95%(IB 폭발), glass 3%→69%(평탄). crossover 확정.
- **부수 결론**: glass inter-panel은 **Ring 유지로 충분** — cross-domain 대역차(optical 512
  vs IB 50 = 10×)가 Ring 홉수차를 압도해 glass가 Ring인데도 이김. crossbar로 바꿀 필요 없음.
- TODO: `MOE_ALLTOALL` env를 정식 config/CLI 플래그로 승격; 大EP=all2all 근거(DeepEP) 본문 명시.

## README and docs split

The repo has two documentation surfaces with deliberate scope:

- **`README.md`** — minimal front door. About / Getting Started / Publications /
  Citation only. Logo + link bar (Website / Documentation / Contribute /
  Contact / Changelog) point everything else out to the website. **Do not
  re-add detailed content (CLI flag tables, dataset schema, profiler
  walkthroughs, validation plots, etc.) to the README** — it lives on the
  website now.
- **`docs/`** — the public docs site (Docusaurus 3, deployed at
  `https://llmservingsim.ai`). All long-form content lives here. See
  `docs/AGENTS.md` for site-specific conventions.

When you add a new feature with user-visible behavior, document it on the
website (not the README).

## Commit & Pull Request Guidelines

- Short imperative commit messages: `Fix incorrect evict_size accumulation`,
  `Add Qwen3 model support`
- Keep commits focused — one logical change per commit
- Include the exact command used for validation and note any output CSV path in PRs
- Describe which simulation mode is affected and the config/dataset used

## Testing & Validation

No dedicated unit-test suite. Validate by:
1. Running the smallest relevant `python -m serving …` scenario and inspecting
   the per-request CSV.
2. For end-to-end accuracy checks against real vLLM, use `python -m bench run`
   followed by `python -m bench validate` (see `bench/README.md`).
3. For profiler changes: edit `MODEL` / `HARDWARE` in `profiler/profile.sh`
   and run `./profiler/profile.sh` from the repo root inside the vLLM container.

## Common Pitfalls

- **Don't edit `astra-sim/`** unless the change targets simulator integration
  (e.g., `llm_converter.py`, `Workload.cc`, input configs)
- **Don't commit large files**: generated traces, output CSVs, `.et` files are gitignored
- **Don't use machine-specific absolute paths** in configs or code — use relative paths
  rooted at the repo
- **Don't add `getattr` fallbacks** for Request attributes — initialize all attributes
  in `Request.__init__` and access directly
- **Don't assume `hidden_size == num_heads * head_dim`** — use explicit `head_dim` from config
- **Use canonical vLLM layer names** (`qkv_proj`, `o_proj`, `gate_up_proj`,
  `act_fn`, `down_proj`, `rotary_emb`, `qk_norm`, `attention`, `layernorm`,
  `final_layernorm`, `embedding`, `lm_head`, `sampler`, `moe`). Every name the
  simulator emits must also appear in the architecture yaml's catalog.
- **Profiler CSVs store microseconds** (`time_us` column) — the simulator
  multiplies by 1000 and rounds to nanoseconds at load time
- **First and last trace layers must use REMOTE** — the Chakra converter creates a MEM_LOAD
  node from the first layer's input_loc and a MEM_STORE node from the last layer's output_loc;
  if either is LOCAL without local_mem configured, ASTRA-Sim crashes
- **memory_expansion.json only has remote_mem by default** — local_mem is not configured unless
  `--enable-local-offloading` is used; weight loads from LOCAL go through compute time, not memory
- **`config_builder.py` regenerates ASTRA-Sim inputs on every run** — don't manually edit
  `astra-sim/inputs/` files expecting them to persist

## 실행 환경 (Local Mac + Docker)

Claude Code는 Mac 파일시스템에서 파일을 직접 수정하고,
명령어 실행은 아래 형식으로 Docker 컨테이너 안에서 수행한다.

```bash
docker exec servingsim_docker bash -c "cd /app/LLMServingSim && <명령어>"
```

### 테스트 커맨드

EP=2 smoke test (항상 먼저 실행, 통과해야 함):
```bash
docker exec servingsim_docker bash -c "cd /app/LLMServingSim && python -m serving --cluster-config configs/cluster/single_node_moe_dp_ep_instance.json --dtype bfloat16 --block-size 16 --dataset workloads/example_trace.jsonl --output outputs/test_ep2.csv --num-req 10 --log-level INFO"
```

EP=4 FC test (수정 후 검증 목표):
```bash
docker exec servingsim_docker bash -c "cd /app/LLMServingSim && python -m serving --cluster-config configs/cluster/nvl72_ep4.json --dtype bfloat16 --block-size 16 --dataset workloads/example_trace.jsonl --output outputs/test_ep4.csv --num-req 10 --log-level INFO"
```

## 현재 진행 중인 작업: EP=4 Deadlock 수정

### 문제
EP=4에서 ASTRA-Sim이 "Waiting" 메시지를 반환하지 않아 hang 발생.

### 원인
serving/__main__.py의 dp_group 동기화 로직에서
각 인스턴스가 chakra converter를 --num-npus 1 --npu-offset N 으로 따로 호출함.
4개 et 파일이 따로 생성되고 ASTRA-Sim에 동시에 전달되는 구조가 불안정함.

### 제안된 수정 방향
serving/core/graph_generator.py 수정:
- 현재: 인스턴스마다 chakra --num-npus 1 --npu-offset N 호출
- 변경: EP 전체를 한 번에 chakra --num-npus EP --npu-offset 0 으로 호출
- 효과: dp_group 동기화 로직 제거 가능, ASTRA-Sim 호출 단순화

### 수정 파일 우선순위
1. serving/core/graph_generator.py  ← 핵심
2. serving/__main__.py              ← dp_group sync 단순화
3. serving/core/trace_generator.py  ← 필요 시 AllToAll 크기 재계산
