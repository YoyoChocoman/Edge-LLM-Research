# Experiment 1: Output-Length Scaling Characterization

## 1. Research Question
How does the actual output sequence length ($N_{output}$) affect aggregate decoding latency and the Time Per Output Token (TPOT) under a fixed edge-LLM workload?

This experiment aims to characterize the baseline scaling behavior of the inference engine before introducing optimizations like speculative decoding or dynamic routing.

## 2. Methodology
To isolate the effect of generation length, we utilized a One-Factor-At-A-Time (OFAT) approach, bypassing the application layer (Pydantic/FastAPI) to directly instrument the `llama-cpp-python` engine.

- **Fixed Variables:**
  - Hardware: NVIDIA GeForce RTX 5070 Ti (16GB VRAM)
  - Model: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
  - Prompt: Static evaluation prompt (~61 tokens)
  - Sampling: Greedy decoding (`temperature = 0.0`) to ensure deterministic output paths
  - Concurrency: $C = 1$ (Sequential execution)
- **Independent Variable:** `max_tokens` (Sweeping from 25 to 600)
- **Analysis Variable:** `Actual Output Tokens` ($N_{output}$)
- **Dependent Variables:** Aggregate Decode Time ($T_{decode}$), Actual Output Tokens ($N_{output}$), Average TPOT.
- **Sample Size:** 20 independent runs per setting.

### Methodological Note: Token Counting Discrepancy
During data collection, we observed that the measured `actual_tokens` consistently exceeded the configured `max_tokens` limit (e.g., `max_tokens=25` yielded `actual_tokens=29`). This discrepancy is likely related to tokenization differences between the re-tokenized output string and the generation loop's internal token counter.

We do not investigate this tokenization mismatch further, as it does not affect the experimental findings. The `max_tokens` parameter successfully served its structural purpose: functioning as an independent throttle to create distinct output length buckets. Since all downstream latency regressions and TPOT calculations strictly depend on the empirically measured `actual_tokens` ($N_{output}$), the underlying scaling analysis remains valid.

## 3. Findings

### Finding 1: Aggregate Decode Latency Exhibits Strong Linear Scaling
We modeled the decoding latency as a linear function of actual output tokens:
$$T_{decode} = \beta_1 N_{output} + \beta_0$$

The empirical data yielded a highly robust linear fit:
- **$R^2$ = 0.9916**
- **Slope ($\beta_1$) = 7.76 ms/token**
- **Intercept ($\beta_0$) = -120.47 ms**

**Conclusion 1:** Under the current hardware and workload configuration, the aggregate decoding latency scales highly linearly with the actual number of generated tokens.

### Finding 2: The Breakdown of the Simple TPOT Model
A naive constant-marginal-cost model assumes that $T_{decode} = T_{fixed} + c \cdot N$. This implies that TPOT should follow:
$$TPOT = c + \frac{T_{fixed}}{N}$$
If $T_{fixed}$ represents a physical system overhead, it must be positive, which means TPOT should monotonically *decrease* and asymptote to $c$ as $N$ grows.

However, our observation contradicts this model:
- The correlation between actual tokens and TPOT is strongly positive (**Pearson $r = 0.8725$**, **$R^2 = 0.7612$**).
- TPOT steadily *increases* from ~6.0 ms at $N=25$ to ~7.5 ms at $N=350$.
- The negative intercept does not represent a physically meaningful fixed latency; rather, it indicates that the fitted linear model should not be extrapolated to $N_{output}=0$

**Conclusion 2:** A simple fixed-overhead + constant marginal decoding-cost model does not adequately explain the observed TPOT behavior in our edge-LLM deployment.

## 4. Discussion & Open Questions

The observed TPOT behavior is inconsistent with the simple constant-marginal-cost model and motivates direct measurement of per-token decoding dynamics.

We hypothesize that the per-token decoding cost changes dynamically during the autoregressive generation process. Possible mechanisms to investigate include:
1. Context-dependent KV-cache access cost
2. Memory bandwidth pressure
3. Other context-length-dependent effects

### Next Step: Per-Token Decoding Dynamics
To move beyond aggregate averages and isolate the root cause, the next experiment will transition from macroscopic to microscopic instrumentation.

**Next Experiment Objective:** How does the per-token decoding latency ($t_i$) evolve at each specific generation position $i$ during autoregressive decoding?