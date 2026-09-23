# Experiment 6-A: Direct Python-Binding Decode Measurement

## 1. Research Question

Experiments 2–5 identified reproducible, context-dependent latency spikes through application-observed streaming measurements. Do these spikes persist when inference bypasses high-level chat formatting and streaming iteration?

This experiment moves the measurement boundary to direct Python-binding calls, separating token evaluation from sampling before investigating the native runtime.

## 2. Methodology

- **Hardware & Model:** NVIDIA GeForce RTX 5070 Ti, `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` via `llama-cpp-python`.
- **Configuration:** Full GPU offload, `n_ctx=2048`, Flash Attention disabled, greedy sampling (`temperature=0.0`).
- **Workload:** A fixed 123-token prompt, a 5-step warm-up, and 20 sequential runs with a maximum of 500 decode steps.
- **Measurement:** After prefill and an initial sample, alternate `llm.eval([token])` and `llm.sample()`. Record both durations, evaluated/sampled token IDs, and the binding's token count after each evaluation using `time.perf_counter_ns()`.

The position before evaluation is reconstructed as `n_past_after_eval - 1`. This measures the binding's token count, not the runtime's padded KV length. The prompt differs from the earlier high-level workload, so generation-step numbers are not directly interchangeable.

For spike classification, each run uses the median of the available neighboring steps within ±3 positions, excluding the target. A spike must exceed this reference by **both 20% and 0.8 ms**. The component analysis separately pools neighboring observations across runs to describe local latency levels.

## 3. Observations

### 3.1 Spikes Persist in the Direct Evaluation Path

| Decode Step | Tokens Before Eval | Eval P50 | Pooled Local Reference | P50 − Reference | Spike Rate |
| :--- | :--- | :--- | :--- | :--- | :--- |
| 1 | 123 | 2.727 ms | 0.497 ms | +2.230 ms | 20/20 |
| **134** | **256** | **2.890 ms** | **0.476 ms** | **+2.414 ms** | **20/20** |
| **390** | **512** | **2.872 ms** | **0.444 ms** | **+2.428 ms** | **20/20** |

> **Note:** Local references and excess values in this table use the pooled component analysis; spike rates use the per-run rule defined above.

The two periodic events occur while evaluating the token after the binding's context count reaches 256 and 512. Their separation is exactly 256 decode steps. Step 1 is treated separately as a first-decode event and is not evidence of periodic recurrence.

**Observation:** Removing high-level formatting and streaming iteration does not eliminate the context-aligned spikes.

### 3.2 Evaluation and Sampling Have Different Timing Contributions

| Tokens Before Eval | Eval P50 | Sample P50 | Eval + Sample P50 | Local Eval + Sample Reference |
| :--- | :--- | :--- | :--- | :--- |
| 256 | 2.890 ms | 7.307 ms | 10.235 ms | 7.504 ms |
| 512 | 2.872 ms | 7.083 ms | 9.976 ms | 7.372 ms |

Local evaluation latency is approximately 0.4–0.5 ms, while sampling occupies roughly 6.7–6.9 ms around these boundaries. The combined local latency remains around 7–8 ms, consistent in scale with earlier high-level measurements.

At the first periodic boundary, the pooled-reference excess is **+2.414 ms for eval**, **+0.418 ms for sample**, and **+2.731 ms for their combined interval**. The anomaly is therefore primarily visible in the evaluation path.

These P50 differences are not additive: the median of `eval + sample` need not equal the sum of the separate medians. Sampling may also include pending GPU work or synchronization, so its measured duration cannot be interpreted as sampling computation alone. The combined interval excludes token-to-text processing and other high-level overhead.

![Evaluation and sampling latency decomposition](figures/component.png)

## 4. Conclusion

The periodic spikes persist in direct Python-binding evaluation at token counts 256 and 512, with **20/20 reproducibility** at both positions. High-level chat formatting and streaming iteration are therefore insufficient as the sole explanation.

The experiment does not separate Python-binding overhead from native llama.cpp or CUDA behavior. This motivates Exp6-B: reproducing the same token trajectory through native C++ decode calls.