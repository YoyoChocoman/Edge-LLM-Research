# 實驗一：輸出長度擴展特性分析 (Output-Length Scaling Characterization)

## 1. 研究問題
在固定的邊緣大型語言模型 (Edge-LLM) 工作負載下，實際輸出序列長度 ($N_{output}$) 如何影響總解碼延遲 (Aggregate Decoding Latency) 與每輸出 token 耗時 (TPOT)？

本實驗旨在引入投機解碼 (speculative decoding) 或動態路由 (dynamic routing) 等最佳化技術前，先描繪出推論引擎的基準縮放行為 (baseline scaling behavior)。

## 2. 研究方法
為隔離生成長度所帶來的影響，我們採用單一變數法 (One-Factor-At-A-Time, OFAT)，繞過應用層 (Pydantic/FastAPI)，直接對 `llama-cpp-python` 引擎進行檢測埋點 (instrumentation)。

- **控制變因 (Fixed Variables)：**
  - 硬體：NVIDIA GeForce RTX 5070 Ti (16GB VRAM)
  - 模型：`Meta-Llama-3-8B-Instruct-Q4_K_M.gguf`
  - 提示詞：靜態評估提示詞 (約 61 tokens)
  - 採樣方式：貪婪解碼 (`temperature = 0.0`)，確保輸出路徑具備決定性
  - 併發數：$C = 1$ (循序執行)
- **自變數 (Independent Variable)：** `max_tokens` (從 25 掃描至 600)
- **分析變數 (Analysis Variable)：** 實際輸出 tokens 數 (`Actual Output Tokens`, $N_{output}$)
- **應變數 (Dependent Variables)：** 總解碼時間 ($T_{decode}$)、實際輸出 tokens 數 ($N_{output}$)、平均 TPOT
- **樣本數 (Sample Size)：** 每個設定進行 20 次獨立測試。

### 研究方法附註：Token 計數差異 (Token Counting Discrepancy)
在資料收集過程中，我們觀察到測量所得的 `actual_tokens` 持續超出設定的 `max_tokens` 上限（例如：設定 `max_tokens=25` 卻產出 `actual_tokens=29`）。此差異很可能與「對輸出字串重新進行 tokenization 的結果」與「生成迴圈內部 token 計數器」兩者之間的計算差異有關。

我們不對此 tokenization 差異做進一步探究，因其不影響實驗發現。`max_tokens` 參數已成功達成其結構性目的：作為一個獨立的節流閥 (throttle) 來建立不同的輸出長度區間。由於所有下游的延遲迴歸分析與 TPOT 計算，皆嚴格依賴於經驗測量所得的 `actual_tokens` ($N_{output}$)，因此底層的縮放分析依然有效。

## 3. 實驗發現

### 發現一：總解碼延遲呈現強烈的線性縮放關係
我們將解碼延遲建模為實際輸出 tokens 數的線性函數：
$$T_{decode} = \beta_1 N_{output} + \beta_0$$

經驗數據顯示出極具強健性的線性擬合結果：
- **$R^2$ = 0.9916**
- **斜率 ($\beta_1$) = 7.76 ms/token**
- **截距 ($\beta_0$) = -120.47 ms**

**結論一：** 在當前的硬體與工作負載配置下，總解碼延遲與實際生成的 token 數量呈現高度的線性縮放關係。

### 發現二：簡易 TPOT 模型的失效
一個單純的固定邊際成本模型 (constant-marginal-cost model) 假設 $T_{decode} = T_{fixed} + c \cdot N$。這暗示 TPOT 應遵循以下公式：
$$TPOT = c + \frac{T_{fixed}}{N}$$
若 $T_{fixed}$ 代表物理系統的額外開銷 (overhead)，它必須為正值，這意味著隨著 $N$ 增長，TPOT 應該呈現單調*遞減*，並漸近於 $c$。

然而，我們的觀察結果與此模型相悖：
- 實際 token 數與 TPOT 之間呈現強烈的正相關（**Pearson $r = 0.8725$**, **$R^2 = 0.7612$**）。
- TPOT 從 $N=25$ 時的約 6.0 ms，穩定*上升*至 $N=350$ 時的約 7.5 ms。
- 負的截距並不代表具備物理意義的固定延遲；相反地，這指出不應將此線性擬合模型外推 (extrapolate) 至 $N_{output}=0$ 的情況。

**結論二：** 「簡單的固定開銷加上恆定邊際解碼成本」之模型，不足以解釋在我們的邊緣 LLM 部署中所觀察到的 TPOT 行為。

## 4. 討論與未解問題

觀察到的 TPOT 行為與簡單的固定邊際成本模型不一致，這構成了直接測量「單一 token 解碼動態 (per-token decoding dynamics)」的動機。

我們假設在自迴歸 (autoregressive) 生成過程中，單一 token 的解碼成本會動態改變。可能需要調查的機制包含：
1. 與上下文長度相關的 KV-cache 存取成本
2. 記憶體頻寬壓力
3. 其他依賴於上下文長度的效應

### 下一步：單一 Token 解碼動態分析 (Per-Token Decoding Dynamics)
為了超越總體平均值並找出根本原因，下一個實驗將從巨觀 (macroscopic) 檢測轉向微觀 (microscopic) 檢測。

**下個實驗目標：** 在自迴歸解碼過程中，單一 token 的解碼延遲 ($t_i$) 在每一個特定的生成位置 $i$ 上是如何演變的？