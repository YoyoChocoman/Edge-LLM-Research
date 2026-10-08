"""Run one async batch. Alternate padding groups manually between batches."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess

RUNS = 10

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--padding", type=int, choices=[256, 512], required=True)
    parser.add_argument("--runs", type=int, default=RUNS)
    args = parser.parse_args()

    home = (Path(".runtime") / f"pad{args.padding}").resolve()
    source = home / "source"
    lib = home / "build/bin"
    if not (lib / "libggml-cuda.so").exists():
        raise RuntimeError("Build this padding group's CUDA runtime first")

    binary = home / "exp7_native"
    subprocess.run([
        "g++", "-std=c++17", "-O2", "-DEXP7_TRACE", f"-DEXP7_PADDING_SIZE={args.padding}",
        "-I", str(source / "include"), "-I", str(source / "ggml/include"),
        "scripts/native.cpp", "-L", str(lib), f"-Wl,-rpath,{lib}",
        "-lllama", "-lggml-base", "-o", str(binary),
    ], check=True)

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")

    output = Path("results") / f"{tag}_pad{args.padding}"
    output.mkdir(parents=True)

    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")

    subprocess.run([str(binary), "async", str(output / "native.json"), str(output / "events.json"), str(args.runs)], env=env, check=True)

    print(f"Results: {output}")


if __name__ == "__main__":
    main()
