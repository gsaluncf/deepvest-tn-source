#!/usr/bin/env bash
# Build the TN-enabled server and keep its shared libraries beside it.
set -euo pipefail
root="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")/.." && pwd)"
backend="${1:-cpu}"
jobs="${TN_BUILD_JOBS:-4}"
case "$backend" in cpu) cuda=OFF ;; cuda) cuda=ON ;; *) echo 'Usage: build_llama.sh [cpu|cuda]' >&2; exit 2 ;; esac
for command in git cargo cmake ninja python3 c++; do
    command -v "$command" >/dev/null || { echo "Missing build dependency: $command" >&2; exit 1; }
done
if [[ "$cuda" == ON ]]; then
    command -v nvcc >/dev/null || { echo 'CUDA builds require nvcc on PATH.' >&2; exit 1; }
fi
cuda_options=()
if [[ "$cuda" == ON ]]; then
    cuda_options+=("-DCMAKE_CUDA_ARCHITECTURES=${TN_CUDA_ARCHITECTURES:-native}")
fi
source_root="$root/native/tn-llama"
work="$root/.runtime/llama-build"
upstream="$work/upstream"
build="$work/$backend"
revision=3f545beccee69d9975f466ec7e45fd9aacd8ba90
mkdir -p "$work"
if [[ ! -d "$upstream" ]]; then
    git init "$upstream"
    git -C "$upstream" remote add origin https://github.com/ggml-org/llama.cpp.git
    git -C "$upstream" fetch --depth 1 origin "$revision"
    git -C "$upstream" checkout --detach FETCH_HEAD
fi
[[ "$(git -C "$upstream" rev-parse HEAD)" == "$revision" ]] || { echo 'Unexpected llama.cpp revision.' >&2; exit 1; }
patch="$source_root/patches/llama.cpp/tn-governed-loop.patch"
if ! git -C "$upstream" apply --reverse --check "$patch" 2>/dev/null; then
    [[ -z "$(git -C "$upstream" status --porcelain)" ]] || { echo 'Build checkout has unrelated changes.' >&2; exit 1; }
    git -C "$upstream" apply --check "$patch"
    git -C "$upstream" apply "$patch"
fi
export CARGO_TARGET_DIR="$work/cargo-target"
if ! native_output=$(cargo rustc --manifest-path "$source_root/Cargo.toml" --locked --release -p tn-runtime --lib -j "$jobs" -- --print native-static-libs 2>&1); then
    printf '%s\n' "$native_output" >&2
    exit 1
fi
printf '%s\n' "$native_output"
native_libraries=$(printf '%s\n' "$native_output" | sed -n 's/.*native-static-libs: //p' | tail -1 | xargs | tr ' ' ';')
[[ -n "$native_libraries" ]] || { echo 'Rust did not report native link libraries.' >&2; exit 1; }
cmake -S "$upstream" -B "$build" -G Ninja \
    -DCMAKE_BUILD_TYPE=Release -DCMAKE_BUILD_RPATH_USE_ORIGIN=ON \
    -DLLAMA_BUILD_SERVER=ON -DLLAMA_BUILD_TESTS=OFF -DLLAMA_BUILD_EXAMPLES=OFF \
    -DLLAMA_BUILD_TOOLS=ON -DLLAMA_BUILD_APP=OFF -DLLAMA_BUILD_UI=OFF \
    -DGGML_NATIVE=OFF -DGGML_CUDA="$cuda" "${cuda_options[@]}" \
    -DLLAMA_TN_GOVERNED=ON -DLLAMA_TN_PROJECT_ROOT="$source_root" \
    -DLLAMA_TN_RUNTIME_LIBRARY="$CARGO_TARGET_DIR/release/libtn_runtime.a" \
    -DLLAMA_TN_RUNTIME_NATIVE_LIBS="$native_libraries"
cmake --build "$build" --target llama-server --parallel "$jobs"
python3 "$root/scripts/package_llama.py" "$build/bin" "$root/.runtime/dependencies/tn-llama" "$backend"
