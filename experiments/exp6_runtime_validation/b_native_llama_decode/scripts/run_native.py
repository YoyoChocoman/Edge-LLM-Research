import argparse
import importlib.metadata
import os
import subprocess
import llama_cpp
from datetime import datetime, timezone
from pathlib import Path

def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--mode", choices=["async", "sync"], default="async")
    args = p.parse_args()

    # Locate installed llama-cpp-python / libllama
    lib = Path(llama_cpp.__file__).resolve().parent / "lib"
    version = importlib.metadata.version("llama-cpp-python")
    source = next((Path.home() / ".cache/uv/sdists-v9/pypi/llama-cpp-python" / version).glob("*/src/vendor/llama.cpp"), None).resolve()

    # Experiment output directory
    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    dest = Path(f"results/{tag}_{args.mode}")
    dest.mkdir(parents=True, exist_ok=False)

    # Runtime environment
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(lib) + (":" + env["LD_LIBRARY_PATH"] if env.get("LD_LIBRARY_PATH") else "")

    # Build native Exp6-B
    command = [
        "g++",
        "-std=c++17",
        "-O2",
        "-I" + str(source / "include"),
        "-I" + str(source / "ggml/include"),
        "scripts/native.cpp",
        "-L" + str(lib),
        "-Wl,-rpath," + str(lib),
        "-lllama",
        "-o",
        "build/exp6b_native"
    ]

    print("Building Exp6-B...", flush=True)
    subprocess.run(command, check=True, env=env)

    # Run native.cpp
    native_command = ["build/exp6b_native", args.mode, str(dest / "native.json")]

    # Execute native measurement
    print(f"Running native Exp6-B ({args.mode})...", flush=True)
    subprocess.run(native_command, check=True, env=env)
    print(f"Exp6-B complete: {dest}", flush=True)

if __name__ == "__main__":
    main()