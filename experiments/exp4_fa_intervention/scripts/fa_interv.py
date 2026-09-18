import time
import json
import argparse
from datetime import datetime
from llama_cpp import Llama

MODEL_PATH = "../../models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
MAX_TOKENS = 500
RUNS = 20

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

def run_inference(llm, prompt_text, run_id):
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
    parser = argparse.ArgumentParser(description="Multi-Prompt FA Intervention Profiling")
    parser.add_argument("--use-fa", action="store_true", help="Enable Flash Attention")
    args = parser.parse_args()

    fa_status = "ON" if args.use_fa else "OFF"
    result_path = f"results/multi_prompt_fa_{fa_status.lower()}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

    print(f"Initializing Model (Flash Attention: {fa_status})...")
    llm = Llama(
        model_path=MODEL_PATH,
        n_gpu_layers=-1,
        n_ctx=2048,
        flash_attn=args.use_fa,
        verbose=False
    )

    sys_msg = "You are a highly analytical GRE linguistic judge."

    experiment_data = {
        "metadata": {
            "experiment": "Multi-Prompt Flash Attention Intervention",
            "flash_attention": args.use_fa
        },
        "results": {}
    }

    for prompt_name, prompt_text in PROMPTS.items():
        print(f"\nProcessing {prompt_name}...")

        input_tokens = len(llm.tokenize(f"{sys_msg}\n{prompt_text}".encode('utf-8'), add_bos=True))

        prompt_results = {
            "input_tokens": input_tokens,
            "runs": []
        }

        print("Warming up...")
        _ = run_inference(llm, prompt_text, 0)

        print(f"Running Benchmark (N={RUNS})...")
        for i in range(1, RUNS + 1):
            res = run_inference(llm, prompt_text, i)
            prompt_results["runs"].append(res)
            print(".", end="", flush=True)
        print()

        experiment_data["results"][prompt_name] = prompt_results

    with open(result_path, 'w', encoding='utf-8') as f:
        json.dump(experiment_data, f, indent=2, ensure_ascii=False)

    print(f"\nAll data successfully saved to {result_path}")

if __name__ == "__main__":
    main()