# 實驗三：隨上下文位置變化的延遲轉換分析 (Context-Position-Dependent Latency Transitions)

## 1. 研究問題
在實驗二中，我們觀察到了在自迴歸解碼 (autoregressive decoding) 過程中，出現了決定性的延遲突刺 (deterministic latency spike) 與區塊間延遲的永久性階梯式上升 (step-up)。本實驗旨在釐清，在受評估的配置下，究竟是哪個可觀測變數 (observable variable) 最強烈地控制了此異常現象。具體而言，我們調查延遲突刺是否由以下因素觸發：
1. **輸出位置 (Output Position)：** 綁定於特定的生成步驟（例如：在生成第 100 個 chunk 時發生）。
2. **內容 (Content)：** 由特定的字串模式或分詞邊界觸發（例如：特定的 JSON 語法或換行符號）。
3. **總上下文長度 (Total Context Length)：** 綁定於輸入 Tokens 與已生成 Chunks 的累加總和。

## 2. 實驗方法
我們設計了一組正交測試腳本 (`prompt_variance.py`)，使用三個不同的 Prompt 來分離「輸入長度」與「內容」這兩個變因：
*   **Prompt A (Control/對照組)：** 基準評估提示詞（預估輸入：133 tokens）。
*   **Prompt B (Longer Context/較長上下文)：** 與 A 任務相同，但在系統指令中加入了填充文字（預估輸入：161 tokens）。
*   **Prompt C (Different Content/不同內容)：** 評估另一個不同的單字，以改變生成的輸出字串內容（預估輸入：144 tokens）。

**控制變因：**
*   硬體：NVIDIA GeForce RTX 5070 Ti
*   模型：`Meta-Llama-3-8B-Instruct-Q4_K_M.gguf` 透過 `llama.cpp` 執行
*   解碼參數：`temperature = 0.0` (貪婪解碼 Greedy decoding)
*   執行次數：每個 Prompt 進行 20 次重複迭代

我們將區塊間延遲 ($\Delta t_i$) 的視覺化 X 軸，從單純的生成步驟，改為以 **總上下文長度 (Total Context Length)**（Input Tokens + Output Chunk Index）進行對齊。

## 3. 觀察結果

### 3.1 突刺對齊分析 (`analyze_spike.py`)
當以「總上下文長度」為基準繪製圖表時，第一次持續性的延遲轉換，在三個不同的 Prompt 中，皆完美對齊在相同的「名義總上下文位置 (nominal total-context position)」。

| Prompt 類型         | Input Tokens | Spike 1 Chunk Index | Spike 1 總上下文 | Spike 2 Chunk Index | Spike 2 總上下文 |
|---------------------|--------------|---------------------|------------------|---------------------|------------------|
| A_Control           | 133          | 108                 | **241**          | N/A (執行結束)      | N/A              |
| B_Longer_Context    | 161          | 80                  | **241**          | 336                 | **497**          |
| C_Different_Content | 144          | 97                  | **241**          | 353                 | **497**          |

**結論：** 這些結果與「固定輸出位置」的解釋不符，也無法用所評估提示詞中生成的 Token 具體內容來解釋。數據支持「名義總上下文位置」是本實驗中觀測到的主要控制變因。

### 3.2 週期性與 256-Token 間隔
兩次觀測到的突刺間距（497 - 241）恰好為 **256**。
雖然第一次突刺發生在觀測到的總上下文長度 241 處，但我們推測缺失的 ~15 個 tokens 是來自 LLM 聊天模板的隱藏開銷（例如，被注入的角色標記如 `<|start_header_id|>user<|end_header_id|>`）。兩次觀測到的事件位置在名義上下文位置上精確相差了 256。此週期性與執行期狀態（runtime-state）的邊界特徵一致。一個可能的解釋是，提示詞格式或特殊 Token 引入了固定的偏移量，將名義位置 241 與 497 對應到內部記憶體位置的 256 與 512 附近。

因此，KV-cache 或記憶體管理邊界成為一個極具可能的機制假說，但本實驗並未直接觀測底層的執行期 Token 計算、KV-cache 分配或 GPU 記憶體事件。要確立該物理機制，需要進行執行期層級（runtime-level）的檢測埋點。

### 3.3 突刺後的延遲偏移 (`step_up_multi.py`)
我們針對每個突刺前後的乾淨視窗（clean window）進行了 P50 中位數延遲的階梯分析，以觀察推論速度是否有持續性的偏移。下表總結了各個 Prompt 在跨越邊界時的原始測量值與變化百分比：

| Prompt | 突刺類型 (上下文邊界) | 突刺前 P50 (ms) | 突刺後 P50 (ms) | Delta (ms) | 變化幅度 (%) |
| :--- | :--- | :--- | :--- | :--- | :--- |
| A_Control | Spike 1 (~256 tokens) | 7.3230 | 7.6289 | +0.3059 | **+4.18%** |
| B_Longer_Context | Spike 1 (~256 tokens) | 7.3296 | 7.5814 | +0.2519 | **+3.44%** |
| C_Different_Content| Spike 1 (~256 tokens) | 7.3901 | 7.6783 | +0.2881 | **+3.90%** |
| B_Longer_Context | Spike 2 (~512 tokens) | 7.7283 | 7.5467 | -0.1816 | **-2.35%** |
| C_Different_Content| Spike 2 (~512 tokens) | 7.7627 | 7.6254 | -0.1373 | **-1.77%** |

**觀察一：第一次的階梯式上升 (~256 邊界)**

在第一次突刺（跨越約 256 總上下文邊界）之後，我們觀察到所有執行（runs）皆出現高度一致的延遲**增加**（階梯式上升）。P50 延遲大約衰退了 +3.4% 至 +4.2%。這暗示了一種持續性的額外開銷，可能是由於在連續的注意力運算中，需要遍歷一個額外的記憶體區塊所致。

**觀察二：第二次的階梯式下降 (~512 邊界)**

出乎意料地，當跨越第二次突刺（Prompt B 與 C 中的 ~512 總上下文邊界）時，中位數延遲相較於其突刺前的穩定區間，出現了微幅的**減少**（階梯式下降），降幅約為 -1.7% 至 -2.4%（儘管仍高於最初小於 256 上下文時的基準線）。

## 4. 待解問題與未來研究方向
所觀測到的 256 位置週期性，為 KV-cache 或底層記憶體管理的假說提供了動機，但目前的實驗僅確立了相關性（correlation）而非機制（mechanism）。

潛在的影響因子可能涉及 L1/L2 快取的時間與空間局部性（locality behaviors）、硬體預取（prefetching）閾值，或是其他底層作業系統的記憶體分頁（paging）機制。我們將此特定「階梯式下降」現象的根本原因（root-cause analysis）留待未來的研究深入探討。