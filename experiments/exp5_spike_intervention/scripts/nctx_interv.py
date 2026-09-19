import time
import json
import gc
from datetime import datetime
from llama_cpp import Llama

MODEL_PATH = "../../models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
MAX_TOKENS = 500
RUNS_PER_SETTING = 20
N_CTX_VALUES = [256, 512, 1024, 2048]
RESULTS_FILE = f"results/nctx_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

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

def run_chunk_profiling(llm, prompt_text, run_id):
    messages = [
        {"role": "system", "content": "You are a highly analytical GRE linguistic judge."},
        {"role": "user", "content": prompt_text}
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
    experiment_data = {"metadata": {"experiment": "n_ctx Intervention"}, "results": {}}

    print("Starting n_ctx Intervention Benchmark (Multi-Prompt)...")
    print("-" * 60)

    for n_ctx_val in N_CTX_VALUES:
        gc.collect()

        llm = Llama(
            model_path=MODEL_PATH,
            n_gpu_layers=-1,
            n_ctx=n_ctx_val,
            n_batch=512,
            flash_attn=True,
            verbose=False
        )

        sys_msg = "You are a highly analytical GRE linguistic judge."

        for prompt_name, prompt_text in PROMPTS.items():
            print(f"Running n_ctx={n_ctx_val} | {prompt_name}...")
            input_tokens = len(llm.tokenize(f"{sys_msg}\n{prompt_text}".encode('utf-8'), add_bos=True))

            # Warm up
            _ = run_chunk_profiling(llm, prompt_text, 0)

            runs = []
            for run in range(1, RUNS_PER_SETTING + 1):
                runs.append(run_chunk_profiling(llm, prompt_text, run))
                print(".", end="", flush=True)
            print()

            key_name = f"n_ctx_{n_ctx_val}_{prompt_name}"
            experiment_data["results"][key_name] = {
                "n_ctx": n_ctx_val,
                "prompt_name": prompt_name,
                "input_tokens": input_tokens,
                "runs": runs
            }

        del llm

    with open(RESULTS_FILE, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)

    print("-" * 60)
    print(f"Data successfully saved to {RESULTS_FILE}")

if __name__ == "__main__":
    main()