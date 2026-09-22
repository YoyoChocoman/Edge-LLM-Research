import argparse
import json
from pathlib import Path
from llama_cpp import Llama

SOURCE = "../a_python_binding_decode/results/low_level.json"
MODEL = "../../../models/Meta-Llama-3-8B-Instruct-Q4_K_M.gguf"
RESULT_FILE = "results/sequence.txt"
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

def export(source, output, model):
    data = json.loads(source.read_text())
    llm = Llama(model_path=str(model), vocab_only=True, verbose=False)
    prompt_tokens = llm.tokenize(PROMPT.encode(), add_bos=True)

    _, prompt_data = next(iter(data["results"].items()))
    run = prompt_data["runs"][0]
    steps = run["decode_steps"]

    evaluated = [row["evaluated_token_id"] for row in steps]
    sampled = [row["sampled_next_token_id"] for row in steps]

    with output.open("w") as f:
        f.write(f"{len(prompt_tokens)} {len(evaluated)}\n")
        f.write(" ".join(map(str, prompt_tokens)) + "\n")
        for token, expected in zip(evaluated, sampled):
            f.write(f"{token} {expected}\n")

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source", type=Path, default=SOURCE)
    p.add_argument("--output", type=Path, default=RESULT_FILE)
    p.add_argument("--model", type=Path, default=MODEL)
    args = p.parse_args()
    export(args.source, args.output, args.model)

if __name__ == "__main__":
    main()