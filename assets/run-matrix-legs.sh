#!/bin/bash
# Empirical matrix runner: ik_llama.cpp and llama.cpp HIP on Huihui-30B and Qwen3.8-27B
set -euo pipefail

BENCH=~/v620/bench
REPO_ASSETS=~/source/repos/v620-q4-three-way/assets
IK_SRV=~/ik_llama.cpp/build-hip/bin/llama-server
HIP_SRV=~/llama.cpp/build-hip-new/bin/llama-server
HUIHUI_GGUF=/mnt/hdd500/models/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-Q4_K_M.gguf
AB27_GGUF=/mnt/hdd500/models/Qwen3.8-27B-Heretic-Abliterated-Q4_K_M.gguf
JINJA=~/v620/Qwen3.5-4B-anysystem.jinja

export HSA_ENABLE_SDMA=0

run_server_and_bench () {
  local label=$1 srv=$2 model=$3 alias=$4 ctx=$5 fa=$6 extra=$7
  echo "=================================================="
  echo "$(date +%T) Starting $label ($srv)"
  echo "=================================================="
  pkill -x llama-server || true
  sleep 4

  local fa_arg="-fa on -ctk q8_0 -ctv q8_0"
  if [ "$fa" = "off" ]; then
    fa_arg="-fa off"
  fi

  nohup "$srv" -m "$model" -ngl 99 -c "$ctx" $fa_arg \
    --host 127.0.0.1 --port 8082 --alias "$alias" --cache-ram 0 \
    --reasoning-budget 0 --reasoning-format deepseek $extra \
    > "$BENCH/srv-$label.log" 2>&1 &

  local ok=0
  for i in $(seq 1 120); do
    if curl -s -m 3 http://127.0.0.1:8082/health 2>/dev/null | grep -q '"status":"ok"'; then
      ok=1
      break
    fi
    sleep 3
  done

  if [ "$ok" -ne 1 ]; then
    echo "$(date +%T) FAILED to start $label"
    tail -n 25 "$BENCH/srv-$label.log"
    return 1
  fi

  local loaded=$(curl -s http://127.0.0.1:8082/props | python3 -c "import json,sys; print(json.load(sys.stdin).get('model_path','?').split('/')[-1])" 2>/dev/null)
  echo "$(date +%T) $label loaded: $loaded"

  cd "$BENCH" && python3 "$REPO_ASSETS/llm-bench.py" "$label" http://127.0.0.1:8082 > "$BENCH/$label.run.log" 2>&1
  echo "$(date +%T) $label battery complete"

  pkill -x llama-server || true
  sleep 4

  if [ -f "$BENCH/$label.json" ]; then
    cp -v "$BENCH/$label.json" "$REPO_ASSETS/$label.json"
  fi
}

echo "Runner ready."
