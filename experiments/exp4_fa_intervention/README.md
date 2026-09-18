# Experiment 4: Flash Attention Intervention on Context-Dependent Latency

## 1. Research Question
In Experiment 3, we observed two distinct latency phenomena linked to specific total context boundaries (roughly every 256 tokens):
1. A transient but massive latency spike.
2. A persistent step-up in base inter-chunk latency following the first spike.

This experiment investigates whether these two phenomena are related to the memory-access overhead of the standard attention execution path. We intervene by enabling Flash Attention (FA), a tiled attention algorithm designed to reduce intermediate memory traffic and improve memory efficiency. By comparing FA-ON and FA-OFF under otherwise identical conditions, we examine whether Flash Attention changes:
- the location and magnitude of the transient spike
- the persistent post-boundary step-up in baseline decoding latency.

The key objective is to distinguish whether the spike and the persistent step-up respond differently to the same attention-level intervention.

## 2. Methodology

### 2.1 Intervention Variable
The flash_attn flag in llama-cpp-python was toggled between:
*   FA OFF: standard attention execution path
*   FA ON: Flash Attention execution path

### 2.2 Controlled Variables
The experiment was replicated across three orthogonal prompts from Experiment 3:
*   A_Control, B_Longer_Context, C_Different_Content
*   Hardware: NVIDIA GeForce RTX 5070 Ti
*   Model: `Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` via `llama.cpp`
*   Decoding Parameters: `temperature = 0.0` (Greedy decoding)
*   Runs: 20 repeated iterations per prompt

The same hardware, model, runtime configuration, and generation settings were maintained across FA-ON and FA-OFF conditions. This design allows us to determine whether the observed effects persist across differences in input length and generated content.

### 2.3 Metrics
We evaluate the following metrics:
1. **Spike location:** Total context position at which the transient latency spike occurs.
2. **Spike magnitude:** P50 latency measured within the spike window.
3. **Baseline latency:** P50 latency measured in stable windows immediately before and after the relevant boundary.
4. **Step-up delta:** Difference between post-boundary and pre-boundary baseline P50 latency.

The primary comparison is the change in step-up delta between FA-OFF and FA-ON conditions.

## 3. Observations

### 3.1 Spike Location Remains Invariant
Enabling Flash Attention did not shift the generation positions at which the major latency spikes occurred.
| Prompt Type | Total Ctx at Spike 1 | Total Ctx at Spike 2 | FA OFF Spike P50 | FA ON Spike P50 |
| :--- | :--- | :--- | :--- | :--- |
| **A_Control** | 241 | N/A | 10.07 ms | 9.09 ms |
| **B_Longer_Context** | 241 | 497 | 10.58 ms | 9.31 ms |
| **C_Diff_Content** | 241 | 497 | 9.69 ms | 9.68 ms |

**Observation:** The major spikes remained aligned with total-context positions 241 and 497 under both FA-OFF and FA-ON conditions.

Flash Attention therefore did **not appear to change the trigger location** of the observed spikes. The absolute spike magnitude was somewhat lower under FA in Prompts A and B, while remaining almost unchanged in Prompt C. Thus, the intervention affects spike magnitude to some extent, but does not eliminate the event or shift its observed location. This distinction is important: **Flash Attention does not eliminate the transient spike, even though it may modestly reduce its magnitude.**

### 3.2 Mitigation of the Post-Spike Step-Up
In contrast to the persistent spike itself, Flash Attention substantially reduced the persistent latency increase following the first context boundary.

**First Boundary (~256 Total Context) P50 Latency Shifts:**

| Prompt | FA State | Before P50 | After P50 | Step-Up Delta | Mitigation Effect |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A | OFF | 7.1166 ms | 7.4284 ms | **+0.3118 ms** | - |
| A | ON  | 6.8503 ms | 6.8683 ms | **+0.0180 ms** | **94.2% reduction** |
| B | OFF | 7.0560 ms | 7.3837 ms | **+0.3277 ms** | - |
| B | ON  | 6.8791 ms | 6.9935 ms | **+0.1144 ms** | **65.1% reduction** |
| C | OFF | 7.0641 ms | 7.3768 ms | **+0.3127 ms** | - |
| C | ON  | 6.7856 ms | 6.8581 ms | **+0.0725 ms** | **76.8% reduction** |

**Observation**: Flash Attention reduced the initial decoding baseline latency (from ~7.1ms to ~6.8ms) and heavily suppressed the subsequent step-up penalty in all three prompts, although its magnitude varied.

These results provide strong evidence that the persistent step-up is sensitive to the attention execution strategy and is likely associated with memory-access or memory-bandwidth costs that can be reduced by Flash Attention.

### 3.3 Behavior at the Second Boundary (~512 Total Context)
Prompts B and C also contained a second major spike near total context 497.
The behavior after this second boundary differed from that observed at the first boundary:

| Prompt | FA State | Before P50 (Spike 2) | After P50 (Spike 2) | Step-Up Delta |
| :--- | :--- | :--- | :--- | :--- |
| B | OFF | 7.5378 ms | 7.3441 ms | **-0.1937 ms** (Step-down) |
| B | ON  | 7.1619 ms | 7.2344 ms | **+0.0725 ms** (Slight step-up)|
| C | OFF | 7.5319 ms | 7.3400 ms | **-0.1919 ms** (Step-down) |
| C | ON  | 6.9903 ms | 7.0769 ms | **+0.0866 ms** (Slight step-up)|

**Observation**: Without Flash Attention, crossing the second boundary was associated with a small decrease in the measured baseline latency. With Flash Attention, this became a small increase.

We document this divergence but do not use it as evidence for the primary Exp4 conclusion. The second-boundary behavior appears qualitatively different from the first-boundary persistent step-up and requires separate investigation. Therefore, the primary intervention result of Exp4 is based on the first boundary, where the FA-OFF/FA-ON difference is consistent across all three prompts.

## 4. Conclusion
Experiment 4 reveals an important separation between the two latency phenomena observed in Experiment 3.

### 4.1 Flash Attention mitigates marginal cost degradation
Across all three prompts, enabling Flash Attention substantially reduced the persistent increase in baseline inter-chunk decoding latency following the first context boundary.The measured step-up reduction ranged from approximately 65% to 94%. This provides strong evidence that the persistent step-up is associated with computational or memory-access costs in the attention execution path that are sensitive to Flash Attention's memory-efficient execution strategy.

However, the experiment does not by itself identify the exact low-level mechanism responsible for the step-up. Kernel-level profiling or source-level analysis would be required to establish the precise causal mechanism.

### 4.2 The transient spike behaves differently from the persistent step-up
The transient spikes remained aligned with total-context positions 241 and 497 under both FA-OFF and FA-ON conditions. Although FA modestly reduced spike magnitude in some prompts, it **did not** eliminate the spike; shift its observed trigger location; or produce the same strong mitigation observed for the persistent step-up.

Therefore, the results indicate that the transient spike and the persistent post-spike step-up are likely governed by at least partially distinct mechanisms.

In particular, the persistence of the context-aligned spike under Flash Attention suggests that it is not simply a consequence of the standard attention implementation's memory-bandwidth overhead. This result motivates a separate investigation of the factors controlling the spike itself.