"""Compile the replay tool and run it against one private runtime."""
import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import subprocess


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "trace"], default="trace")
    parser.add_argument("--mode", choices=["async", "sync"], default="async")
    args = parser.parse_args()

    home = (Path(".runtime") / args.variant).resolve()
    source = home / "source"
    lib = home / "build/bin"

    if not (lib / "libggml-cuda.so").exists():
        raise RuntimeError("Build this variant's CUDA runtime with build_runtime.py first")

    binary = home / "exp6c_native"
    command = [
        "g++", "-std=c++17", "-O2", "-I", str(source / "include"),
        "-I", str(source / "ggml/include"), "scripts/native.cpp",
        "-L", str(lib), f"-Wl,-rpath,{lib}", "-lllama", "-lggml-base", "-o", str(binary),
    ]

    if args.variant == "trace":
        command.append("-DEXP6_TRACE")
    subprocess.run(command, check=True)

    tag = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S_%fZ")
    output = Path("results") / f"{tag}_{args.variant}_{args.mode}"
    output.mkdir(parents=True, exist_ok=True)
    env = dict(os.environ)
    env["LD_LIBRARY_PATH"] = str(lib) + os.pathsep + env.get("LD_LIBRARY_PATH", "")
    subprocess.run([str(binary), args.mode, str(output / "native.json"), str(output / "events.json")], env=env, check=True)
    print(f"Results: {output}")


if __name__ == "__main__":
    main()
