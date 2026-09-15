# Experiment 2: Per-Chunk Streaming Latency Dynamics

## 1. Research Question
While Experiment 1 demonstrated an aggregate linear relationship between output length and total decode time, does the marginal cost of generation (time per chunk) remain constant throughout a single continuous generation process?

This experiment investigates the micro-level latency dynamics—specifically, the inter-chunk arrival latency ($\Delta t_i$)—to determine whether decoding speed remains stationary, degrades smoothly, or exhibits discrete/non-linear transitions over the course of generation.

## 2. Methodology
- **Metric Definition:** We measured Application-Observed Inter-Chunk Latency ($\Delta t_i = t_i - t_{i-1}$), utilizing high-resolution monotonic timestamps from Python's `time.perf_counter()`. The Time-To-First-Token (TTFT) was strictly isolated from this dataset to prevent initialization overhead from skewing the decode metrics.
- **Hardware & Model:** NVIDIA GeForce RTX 5070 Ti, `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` running via `llama-cpp-python`.
- **Workload:** $N = 20$ independent, sequential runs using a fixed deterministic prompt (`temperature = 0.0`) and a large output limit (`max_tokens = 500`).
- **Tooling:** We developed custom Python profiling scripts for timestamp collection, targeted anomaly detection (`analyze_anomaly.py`), and step-up statistical verification (`verify_step_up.py`).

## 3. Observations
Plotting the inter-chunk latency over the generation position revealed a highly non-uniform latency distribution. We identified five distinct phenomena that warrant further investigation:

*   **a. Initial Warm-up Overhead:** Even with TTFT isolated, Chunks 1~5 exhibited elevated latencies (~9.0 ms) before stabilizing, likely due to initial generator instantiation or C++/Python boundary overheads.
*   **b. Minor Latency Elevation (~Chunk 70):** A slight but visible increase in the P50 latency occurred around chunk 70 (from ~7.4 ms to ~7.5 ms) without a subsequent recovery.
*   **c. Massive Deterministic Spike (~Chunk 108):** A singular, massive latency spike occurred consistently at the exact same generation position across all runs.
*   **d. Post-Spike Step-Up:** Following the spike, the P50 base latency did not return to pre-spike levels; it permanently shifted upwards.
*   **e. Increased Tail Latency Spread:** The gap between P50 and P95 noticeably widened after the spike, suggesting a heavier tail in the observed latency distribution.

## 4. Deep Dive: The Chunk 108 Anomaly & Step-Up
In this phase, we focused our quantitative analysis exclusively on observations (c) and (d).

#### 4.1 Anomaly Localization
Using our anomaly detection script (`analyze_anomaly.py`) targeting the window of chunks 100-120, we identified that the spike occurred precisely at **Chunk 108** across all 20 runs. The generated content at this exact moment was consistently the string `' user'` (context: `". In the user's sentence,"`). The latency at this specific chunk spiked to >10.0 ms (with outliers up to 40.45 ms).

#### 4.2 Post-Spike Step-Up
To verify the post-spike step-up (observation d), we conducted a statistical comparison between a clean pre-spike window (Chunks 53–103) and a clean post-spike window (Chunks 113–163). (`spike_analysis.py`)

**Aggregate Statistical Analysis:**
*   Global P50 Before Spike: **7.3425 ms**
*   Global P50 After Spike: **7.6867 ms**
*   Step-up Delta: **+0.3442 ms (+4.69%)**
*   Per-Run Consistency: **20 / 20 runs (100.0%)** demonstrated this latency step-up.

#### 4.3 Token Composition Sanity Check
Because streaming chunks do not necessarily correspond one-to-one with generated tokens, we performed an additional tokenization-based sanity check. For each run, the cumulative streamed output was re-tokenized using the same Llama tokenizer (`analyze_token_emission.py`), and the incremental token count associated with each chunk was computed. Around the identified anomaly, Chunk 108 increased the cumulative output token count by 1 token, despite exhibiting substantially elevated latency. In comparison, Chunk 109 increased the cumulative count by 2 tokens while showing a lower latency. This observation suggests that the Chunk 108 spike cannot be readily explained simply by a larger number of output tokens contained in that chunk.

> *This analysis reconstructs token counts from the cumulative streamed text rather than directly instrumenting internal decoder token emissions; therefore, the result is treated as a sanity check rather than a definitive measurement of per-token decoder iterations.*

## 5. Discussion & Hypotheses
The observed behavior is highly reproducible under the tested configuration. The token composition sanity check further suggests that the Chunk 108 spike
cannot be readily attributed to an unusually large number of output tokens within that chunk.

Therefore, the current evidence is more consistent with a state-dependent latency transition than with a simple per-chunk output-size effect. However, the underlying mechanism remains unresolved. We currently hypothesize two potential mechanisms, though establishing causation requires further lower-level (C++/CUDA) profiling:
1.  **Memory Allocation Boundaries:** The underlying engine (`llama.cpp`) may allocate KV cache memory in predefined blocks. Crossing Chunk 108 may trigger a block reallocation or cause the active memory footprint to exceed a specific cache hierarchy boundary (e.g., L2 cache limit), resulting in a permanent latency penalty for subsequent reads.
2.  **Content-dependent execution:** The spike coincides with a specific generated token (' user'), so a content-dependent execution effect cannot yet be completely ruled out. However, the persistent post-spike shift is difficult to explain solely through the observed token content.

**Conclusion:** The results show that the aggregate linear relationship observed in Experiment 1 does not imply a smooth, linear increase in per-chunk decoding latency. Under the tested hardware, model, runtime, and workload, decoding latency exhibits non-stationary behavior, including a reproducible spike at chunk 108 followed by a persistent elevation in P50 latency. The mechanism responsible for this regime transition remains an open question for future investigation.