# Edge-LLM-Research
> **研究狀態：** 瓶頸特徵分析 / 機制調查
>
> 本專案已從總體基準線效能側寫（aggregate baseline profiling），推進至微觀級別的延遲檢測埋點（micro-level latency instrumentation）。我們在自迴歸解碼（autoregressive decoding）過程中，發現並定性了一個具備決定性、且高度依賴上下文的延遲異常現象。目前的研究重點在於進行底層的根因分析，隨後才會設計系統層級的介入方案（例如：路由 routing、快取 caching）。

這是一個實證系統研究專案，旨在探討部署於資源受限之邊緣環境（edge environments）中的大型語言模型（LLMs），其推論特徵與系統瓶頸。

## 1. 專案典範與研究方法
本儲存庫最初是作為一個本地推論單字導師的最小可行性產品（MVP）而建立，現已演進為一個**實驗研究平台**。

本專案並不將 LLM 視為通用的黑箱 API，而是利用單字導師應用程式（涵蓋限制解碼、檢索增強生成 RAG 與結構化輸出）作為一個**穩定且寫實的應用負載（workload）**。透過限制模型輸出的變異性（藉由多輪 Few-Shot CoT，在評估負載中達成 100% 的 JSON 解析成功率），我們建立了一個高度可控且可重現的負載，以評估底層系統效能——特別聚焦於首字延遲（TTFT）、單字元輸出時間（TPOT）以及 Chunk 間延遲的動態變化（Inter-Chunk Latency dynamics）。

## 2. 儲存庫架構
本儲存庫嚴格區分了應用負載、實驗檢測腳本與分析腳本：

```text
.
├── src/                    # 應用負載 (FastAPI, SQLite, Llama.cpp)
│   ├── api/                # HTTP 層與併發控制 (Mutex 鎖)
│   ├── db/                 # 向量檢索與持久化儲存
│   └── llm/                # 結構化輸出生成與 CoT 提示詞
├── experiments/            # 核心系統研究與效能側寫
│   ├── exp1_output_length/         # 輸出長度擴展性分析
│   ├── exp2_chunk_dynamics/        # 微觀層級的 Chunk 間延遲側寫
│   ├── exp3_prompt_variance/       # 上下文位置依賴性隔離
│   ├── exp4_fa_intervention/       # Flash Attention 介入實驗
│   └── exp5_spike_intervention/    # 執行期參數消融與長上下文特徵分析
├── methods/                # (計畫中) 未來系統級優化與介入方法
├── docs/                   # 文件與詳細實驗報告
│   ├── BASELINE.md         # 基準系統指標總結
│   └── MVP_EVALUATION.md   # 穩定負載輸出的 Prompt 微調紀錄
├── tests/                  # 舊版 MVP 評估與測試腳本
└── requirements.txt
```

## 3. 實驗進程與發現
我們的研究採迭代方式進行，從總體觀察逐漸過渡到微觀級別的瓶頸隔離：

### Exp 1: 輸出長度擴展性分析 (`experiments/exp1_output_length`)
確立了總體解碼時間（aggregate decode time）與實際輸出長度呈現強烈的線性關係。然而，我們觀察到單字元輸出時間（TPOT）並非嚴格的常數，而是與生成長度呈現輕微的正相關，這促使我們進一步深入研究每個 Token 的動態變化。

### Exp 2: 每個 Chunk 的延遲動態 (`experiments/exp2_chunk_dynamics`)
轉換為微觀層級的檢測埋點，測量應用程式觀察到的 Chunk 間延遲（$\Delta t_i$）。揭露了解碼速度的退化並非平滑的。相反地，我們發現了一個**具備決定性的延遲突刺（deterministic latency spike）**（高達 ~40ms），隨後在連續生成過程中出現了**持續性的基礎延遲階梯式墊高（persistent step-up）**現象。

### Exp 3: 依賴上下文的異常現象 (`experiments/exp3_prompt_variance`)
利用正交提示詞（orthogonal prompts）來隔離延遲突刺的觸發條件。結果顯示，在受評估的提示詞與設定下，突刺無法由特定的文本內容或生成位置來解釋。它一致地錨定於一個**總上下文長度邊界（Total Context Length boundary）**（Input Tokens + Output Tokens），並穩定地在約 ~256 與 ~512 總 tokens 的位置發生。

### Exp 4: Flash Attention 介入實驗 (`experiments/exp4_fa_intervention`)
引入 Flash Attention 作為介入變數。結果指出 Flash Attention 大幅**減輕了突刺後的階梯式墊高**，這與該延遲構成要素與 Attention 記憶體 I/O 相關的假說相符。然而，Flash Attention **未能消除**突刺本身。突刺的位置與存在依然保持不變。

### Exp 5: 參數消融與長上下文 (`experiments/exp5_spike_intervention`)
對高階執行期參數（`n_ctx` 與 `n_batch`）進行消融實驗（Ablation），觀察到突刺位置並未發生偏移。將生成視窗延伸至 >2000 個 tokens，揭露了在初始觀測事件之後，突刺會以**嚴格的 256-token 週期性重現**。此外，觀察到 TPOT 在 2000 個 tokens 的過程中出現了漸進的長期退化（從 ~7.5ms 升至 ~8.9ms），這是 Flash Attention 也無法完全抑制的。

*目前的結論：256-token 的週期性不受所測試的高階執行期參數（`n_ctx` 與 `n_batch`）影響。這表明觀察到的週期性是由推論執行期（inference runtime）內部較低階的機制所控制。KV-cache 記憶體管理或固定粒度的記憶體配置仍然是合理的假說，但尚未被直接驗證。*

## 4. 環境設定與可重現性
**系統需求 (Prerequisites):**
- Python 3.10+
- `uv` 套件管理員

**安裝:**
```bash
git clone https://github.com/YoyoChocoman/Edge-LLM-Research.git
uv venv
source .venv/bin/activate
uv pip install -r requirements.txt

# 備註: 若需啟用 NVIDIA GPU 加速，請以 CUDA 編譯 llama-cpp-python:
# CMAKE_ARGS="-DGGML_CUDA=on" uv pip install llama-cpp-python
```

**取得模型:**
目前的實驗統一使用 Llama-3-8B (Q4_K_M)。
```bash
mkdir models
hf download lmstudio-community/Meta-Llama-3-8B-Instruct-GGUF Meta-Llama-3-8B-Instruct-Q4_K_M.gguf --local-dir ./models
```

## 5. 執行實驗
每個實驗皆放置於 `experiments/` 底下各自的目錄中。**為確保相對路徑與設定檔能正確解析，在執行任何腳本前，必須先將工作目錄切換至特定的實驗資料夾內。**

典型的執行流程包含：先執行 benchmark 腳本以生成原始的 JSON 數據，接著執行分析/繪圖腳本。

範例 (Exp 5 - n_batch 參數消融):
```bash
cd experiments/exp5_spike_intervention/

# 執行介入測試腳本
python scripts/nbatch_interv.py

# 分析生成的延遲數據
python scripts/nbatch_analyze.py results/nbatch_switch.json

# 繪製對比圖表
python scripts/plot_nbatch_comp.py results/nbatch_switch.json
```