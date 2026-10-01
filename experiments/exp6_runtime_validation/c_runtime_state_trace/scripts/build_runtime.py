"""Build a private baseline or traced runtime; never modify the installed package."""
import argparse
from importlib.metadata import version
from pathlib import Path
import shutil
import subprocess

from patch_runtime import patch_runtime

RUNTIME = Path(".runtime")
CUDA_ARCH = "120a-real"
JOBS = 4


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--variant", choices=["baseline", "trace"], default="trace")
    parser.add_argument("--source", type=Path, help="llama.cpp source directory; default: installed package's uv source cache")
    parser.add_argument("--cuda-compiler", help="Full path to nvcc, e.g. /usr/local/cuda-12.8/bin/nvcc")
    parser.add_argument("--cuda-arch", default=CUDA_ARCH)
    parser.add_argument("--jobs", type=int, default=JOBS)
    args = parser.parse_args()

    # Both variants start from the same local snapshot.
    snapshot = RUNTIME / "source"
    if not snapshot.exists():
        source = args.source
        if source is None:
            cache = Path.home() / ".cache/uv/sdists-v9/pypi/llama-cpp-python" / version("llama-cpp-python")
            sources = list(cache.glob("*/src/vendor/llama.cpp"))
            if len(sources) != 1:
                raise RuntimeError("Specify --source: expected one cached llama.cpp source directory")
            source = sources[0]
        shutil.copytree(source, snapshot, ignore=shutil.ignore_patterns(".git", "build", "__pycache__"))
        print(f"Source snapshot: {source.resolve()} -> {snapshot}", flush=True)
    elif args.source:
        raise RuntimeError(".runtime/source already exists; --source only applies to the first build")

    home = RUNTIME / args.variant
    source = home / "source"
    if not source.exists():
        shutil.copytree(snapshot, source)
    if args.variant == "trace" and not (source / "ggml/include/exp6-trace.h").exists():
        patch_runtime(source)

    build = home / "build"
    command = [
        "cmake", "-S", str(source), "-B", str(build),
        "-DCMAKE_BUILD_TYPE=Release", "-DBUILD_SHARED_LIBS=ON",
        "-DGGML_CUDA=ON", "-DGGML_CUDA_GRAPHS=ON", "-DGGML_BACKEND_DL=OFF",
        f"-DCMAKE_CUDA_ARCHITECTURES={args.cuda_arch}",
        "-DLLAMA_BUILD_TESTS=OFF", "-DLLAMA_BUILD_EXAMPLES=OFF",
        "-DLLAMA_BUILD_TOOLS=OFF", "-DLLAMA_BUILD_SERVER=OFF", "-DLLAMA_CURL=OFF",
    ]
    if args.cuda_compiler:
        command.append(f"-DCMAKE_CUDA_COMPILER={args.cuda_compiler}")
    subprocess.run(command, check=True)
    subprocess.run(["cmake", "--build", str(build), "--target", "llama", "--parallel", str(args.jobs)], check=True)
    print(f"Ready: {build / 'bin'}")


if __name__ == "__main__":
    main()
