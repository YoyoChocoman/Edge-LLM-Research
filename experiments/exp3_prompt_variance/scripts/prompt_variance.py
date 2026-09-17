import sys
import os
import time
import json
from datetime import datetime

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from llama_cpp import Llama

MODEL_PATH = "../../models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
MAX_TOKENS = 500
RUNS_PER_PROMPT = 20
RESULTS_FILE = f"results/exp3_raw_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

PROMPTS = {
    "A_Control": (
        "Evaluate if the user's sentence uses the target word correctly.\n\n"
        "<target_word>\nmitigate\n</target_word>\n\n"
        "<target_definition>\nMake a situation less severe\n</target_definition>\n\n"
        "<user_sentence>\nThe government implemented new flood defenses to mitigate the damage caused by heavy rains.\n</user_sentence>\n\n"
        "Execution Steps for 'reasoning':\n"
        "1. Check the Part of Speech.\n"
        "2. Check the Semantics.\n"
        "3. Make a final judgment.\n\n"
        "Provide an extremely detailed, step-by-step reasoning."
    ),
    "B_Longer_Context": (
        "Evaluate if the user's sentence uses the target word correctly.\n\n"
        "<target_word>\nmitigate\n</target_word>\n\n"
        "<target_definition>\nMake a situation less severe\n</target_definition>\n\n"
        "<user_sentence>\nThe government implemented new flood defenses to mitigate the damage caused by heavy rains.\n</user_sentence>\n\n"
        "Execution Steps for 'reasoning':\n"
        "1. Check the Part of Speech.\n"
        "2. Check the Semantics.\n"
        "3. Make a final judgment.\n\n"
        "Provide an extremely detailed, step-by-step reasoning. "
        "Furthermore, please ensure your analysis is highly academic and strictly objective. "          # padding
        "This evaluation will be used for grading advanced level students. Accuracy is paramount."      # padding
    ),
    "C_Different_Content": (
        "Evaluate if the user's sentence uses the target word correctly.\n\n"
        "<target_word>\nephemeral\n</target_word>\n\n"
        "<target_definition>\nLasting for a very short time\n</target_definition>\n\n"
        "<user_sentence>\nThe fragile beauty of the spring cherry blossoms is known to ephemeral almost as soon as they reach full bloom.\n</user_sentence>\n\n"
        "Execution Steps for 'reasoning':\n"
        "1. Check the Part of Speech.\n"
        "2. Check the Semantics.\n"
        "3. Make a final judgment.\n\n"
        "Provide an extremely detailed, step-by-step reasoning."
    )
}

def run_chunk_profiling(llm, prompt, run_id):
    messages = [
        {"role": "system", "content": "You are a highly analytical GRE linguistic judge."},
        {"role": "user", "content": prompt}
    ]

    response_stream = llm.create_chat_completion(
        messages=messages, temperature=0.0, max_tokens=MAX_TOKENS, stream=True
    )

    chunks_data = []
    chunk_idx = 0
    t_first_content = None

    for chunk in response_stream:
        choices = chunk.get("choices", [])
        if not choices: continue

        content = choices[0].get("delta", {}).get("content", "")
        if content:
            t_current = time.perf_counter()
            if t_first_content is None:
                t_first_content = t_current
                chunks_data.append({"chunk_idx": chunk_idx, "content": content, "timestamp_abs": t_current, "is_first": True})
            else:
                chunks_data.append({"chunk_idx": chunk_idx, "content": content, "timestamp_abs": t_current, "is_first": False})
            chunk_idx += 1

    return {"run_id": run_id, "chunks": chunks_data}

def main():
    os.makedirs(os.path.dirname(RESULTS_FILE), exist_ok=True)
    print(f"Loading Model: {MODEL_PATH}")
    llm = Llama(model_path=MODEL_PATH, n_gpu_layers=-1, n_ctx=2048, verbose=False)

    print("Warming up...")
    _ = run_chunk_profiling(llm, PROMPTS["A_Control"], 0)

    experiment_data = {"metadata": {"experiment": "Prompt Variance (Stage 1)"}, "results": {}}

    for p_name, p_content in PROMPTS.items():
        print(f"\nRunning {p_name} ({RUNS_PER_PROMPT} runs)...")
        sys_msg = "You are a highly analytical GRE linguistic judge."
        full_prompt_bytes = f"{sys_msg}\n{p_content}".encode('utf-8')
        input_tokens = len(llm.tokenize(full_prompt_bytes, add_bos=True))

        runs = []
        for run in range(1, RUNS_PER_PROMPT + 1):
            runs.append(run_chunk_profiling(llm, p_content, run))
            print(".", end="", flush=True)

        experiment_data["results"][p_name] = {"input_tokens": input_tokens, "runs": runs}

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)

    print(f"\nData generation complete. Saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()