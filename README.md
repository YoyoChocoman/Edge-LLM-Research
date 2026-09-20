# Edge-LLM-Research
> **Research Status:** Bottleneck Characterization / Mechanism Investigation
>
> The project has progressed from aggregate baseline profiling to micro-level latency instrumentation. We have identified and characterized a deterministic, context-dependent latency anomaly during autoregressive decoding. Current efforts are focused on low-level root-cause analysis before designing systems-level interventions (e.g., routing, caching).

An empirical systems research project investigating the inference characteristics and bottlenecks of Large Language Models (LLMs) deployed in resource-constrained edge environments.

## 1. Project Paradigm & Methodology
This repository originated as a Minimum Viable Product (MVP) for a local-inference vocabulary tutor. It has since evolved into an **experimental research platform**.

Instead of treating the LLM as a generic black-box API, this project utilizes the vocabulary tutoring application (which involves constrained decoding, Retrieval-Augmented Generation, and structured outputs) as a **stable, realistic application workload**. By constraining the model's output variability (achieving a 100% JSON parsing success rate in the evaluated workload via multi-turn Few-Shot CoT), we establish a highly controlled and reproducible workload for evaluating underlying system performance—specifically focusing on Time-to-First-Token (TTFT), Time-Per-Output-Token (TPOT), and Inter-Chunk Latency dynamics.

## 2. Repository Architecture
The repository strictly separates the application workload from experimental instrumentation and analytical scripts:

```text
.
├── src/                    # The Application Workload (FastAPI, SQLite, Llama.cpp)
│   ├── api/                # HTTP layer and concurrency control (Mutex locks)
│   ├── db/                 # Vector retrieval and persistent storage
│   └── llm/                # Structured output generation and CoT prompting
├── experiments/            # Core Systems Research & Profiling
│   ├── exp1_output_length/         # Output-length scaling analysis
│   ├── exp2_chunk_dynamics/        # Micro-level inter-chunk latency profiling
│   ├── exp3_prompt_variance/       # Context-position dependency isolation
│   ├── exp4_fa_intervention/       # Flash Attention intervention
│   └── exp5_spike_intervention/    # Runtime parameter ablation and long-context characterization
├── methods/                # (Planned) Future systems-level optimization and intervention methods
├── docs/                   # Documentation and detailed experiment reports
│   ├── BASELINE.md         # Baseline system metrics summary
│   └── MVP_EVALUATION.md   # Record of prompt tuning used to stabilize the workload output
├── tests/                  # Legacy MVP evaluation and testing scripts
└── requirements.txt
```

## 3. Experimental Progression & Findings
Our research is conducted iteratively, transitioning from aggregate observations to micro-level bottleneck isolation:

### Exp 1: Output Length Scaling (`experiments/exp1_output_length`)
Established that aggregate decode time exhibits a strong linear relationship with actual output length. However, we observed that Time-Per-Output-Token (TPOT) is not a strict constant, displaying a slight positive correlation with generation length, prompting deeper investigation into per-token dynamics.

### Exp 2: Per-Chunk Latency Dynamics (`experiments/exp2_chunk_dynamics`)
Transitioned to micro-level instrumentation measuring application-observed inter-chunk latency ($\Delta t_i$). Revealed that decoding speed degradation is not smooth. Instead, we identified a **deterministic latency spike** (up to ~40ms) followed by a **persistent step-up** in base latency during continuous generation.

### Exp 3: Context-Dependent Anomalies (`experiments/exp3_prompt_variance`)
Utilized orthogonal prompts to isolate the trigger for the latency spike. Findings demonstrated that, under the evaluated prompts and configuration, the spike is not explained by specific text content or generation position. It is consistently anchored to a **Total Context Length boundary** (Input Tokens + Output Tokens), occurring consistently around ~256 and ~512 total tokens.

### Exp 4: Flash Attention Intervention (`experiments/exp4_fa_intervention`)
Introduced Flash Attention as an intervention variable. Results indicated that Flash Attention substantially **mitigated the post-spike step-up**, consistent with the hypothesis that this component of the latency is related to attention memory I/O. However, Flash Attention **failed to eliminate** the spike itself. The spike's location and presence remained invariant.

### Exp 5: Parameter Ablation & Long-Context (`experiments/exp5_spike_intervention`)
Ablated high-level runtime parameters (`n_ctx` and `n_batch`) and observed no shift in spike locations. Extended the generation window to >2000 tokens, revealing a **strict 256-token periodic recurrence** of the spike after the initial observed event. Furthermore, observed a gradual, long-term degradation of TPOT (from ~7.5ms to ~8.9ms) over 2000 tokens, which Flash Attention could not completely suppress.

*Conclusion to date: The 256-token periodicity is not affected by the tested high-level runtime parameters (`n_ctx` and `n_batch`). This suggests that the observed periodicity is governed by a lower-level mechanism within the inference runtime. KV-cache memory management or fixed-granularity allocation remains a plausible hypothesis, but has not yet been directly verified.*

## 4. Environment Setup & Reproducibility
**Prerequisites:**
- Python 3.10+
- `uv` package manager

**Installation:**
```bash
git clone https://github.com/YoyoChocoman/Edge-LLM-Research.git
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# Note: For NVIDIA GPU acceleration, compile llama-cpp-python with CUDA:
# CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python
```

**Model Acquisition:**
Current experiments are standardized on Llama-3-8B (Q4_K_M).
```bash
mkdir models
hf download lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir ./models
```

## 5. Running Experiments
Each experiment resides in its respective directory under `experiments/`. **To ensure relative paths and configurations resolve correctly, you must change your working directory into the specific experiment folder before executing any scripts.**

A typical execution flow involves running a benchmark script to generate raw JSON data, followed by an analysis/plotting script.

Example (Exp 5 - n_batch ablation):
```bash
cd experiments/exp5_spike_intervention/

# Run an intervention benchmark
python scripts/nbatch_interv.py

# Analyze the resulting latency data
python scripts/nbatch_analyze.py results/nbatch_switch.json

# Plot the comparison
python scripts/plot_nbatch_comp.py results/nbatch_switch.json
```