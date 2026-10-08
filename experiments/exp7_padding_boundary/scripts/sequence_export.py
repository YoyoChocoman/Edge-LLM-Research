import argparse
from pathlib import Path
from llama_cpp import Llama

MODEL = "../../models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
DECODE_STEPS = 1024
OUTPUT = "results/sequence.txt"
PROMPT = (
    "Evaluate if the user's sentence uses the target word correctly.\n\n"
    "<target_word>\nmitigate\n</target_word>\n\n"
    "<target_definition>\nMake a situation less severe\n</target_definition>\n\n"
    "<user_sentence>\nThe government implemented new flood defenses to "
    "mitigate the damage caused by heavy rains.\n</user_sentence>\n\n"
    "Execution Steps for 'reasoning':\n"
    "1. Check the Part of Speech.\n"
    "2. Check the Semantics.\n"
    "3. Make a final judgment.\n\n"
    "Provide an extremely detailed, step-by-step reasoning."
)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", default=MODEL)
    parser.add_argument("--steps", type=int, default=DECODE_STEPS)
    parser.add_argument("--output", type=Path, default=OUTPUT)
    args = parser.parse_args()

    llm = Llama(model_path=args.model, n_gpu_layers=-1, n_ctx=2048, n_batch=512, flash_attn=False, seed=42, verbose=False)
    prompt = llm.tokenize(PROMPT.encode(), add_bos=True)

    if not 0 < len(prompt) <= 512 or args.steps < 1 or len(prompt) + args.steps > 2048:
        parser.error("Prompt/replay exceeds the native batch or context limits")
    llm.eval(prompt)
    token = llm.sample(temp=0.0, repeat_penalty=1.0)

    # Intentionally continue past EOS: this is a fixed-length replay workload.
    rows = []
    for _ in range(args.steps):
        llm.eval([token])
        sampled = llm.sample(temp=0.0, repeat_penalty=1.0)
        rows.append((token, sampled))
        token = sampled

    args.output.parent.mkdir(parents=True, exist_ok=True)

    with args.output.open("w") as out:
        out.write(f"{len(prompt)} {len(rows)}\n")
        out.write(" ".join(map(str, prompt)) + "\n")
        for token, sampled in rows:
            out.write(f"{token} {sampled}\n")

    llm.close()
    print(f"Sequence: {args.output}; prompt={len(prompt)}, decode={len(rows)}")


if __name__ == "__main__":
    main()
