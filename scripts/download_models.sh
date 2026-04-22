#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$ROOT"

mkdir -p models vendor

Q8_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q8_0.gguf"
Q4_URL="https://huggingface.co/TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF/resolve/main/tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

Q8_OUT="models/tinyllama-1.1b-chat.Q8_0.gguf"
Q4_OUT="models/tinyllama-1.1b-chat.Q4_K_M.gguf"

fetch() {
  local url="$1" out="$2"
  if [ -f "$out" ]; then
    echo "exists: $out"
    return
  fi
  echo "downloading $out"
  curl -L --fail -o "$out" "$url"
}

fetch "$Q8_URL" "$Q8_OUT"
fetch "$Q4_URL" "$Q4_OUT"

if [ ! -x "vendor/llama.cpp/build/bin/llama-server" ]; then
  echo "building llama.cpp with metal"
  if [ ! -d vendor/llama.cpp ]; then
    git clone --depth 1 https://github.com/ggerganov/llama.cpp vendor/llama.cpp
  fi
  cmake -S vendor/llama.cpp -B vendor/llama.cpp/build -DGGML_METAL=ON -DLLAMA_CURL=OFF
  cmake --build vendor/llama.cpp/build --target llama-server -j
else
  echo "llama-server already built"
fi

echo "done. models in ./models, binary at vendor/llama.cpp/build/bin/llama-server"
