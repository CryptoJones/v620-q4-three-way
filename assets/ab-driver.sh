#!/bin/bash
# Three-leg Q4 A/B under pinned 3D_FULL_SCREEN profile.
BENCH=~/v620/bench
SRV=~/llama.cpp/build-vulkan-new/bin/llama-server
JINJA=~/v620/Qwen3.5-4B-anysystem.jinja

run_leg () {
  local label=$1 model=$2 alias=$3 extra=$4
  pkill -x llama-server; sleep 3
  nohup $SRV -m "$model" -ngl 99 -c 131072 -fa on -ctk q8_0 -ctv q8_0 \
    --host 127.0.0.1 --port 8082 --alias "$alias" --cache-ram 0 \
    --reasoning-budget 0 --reasoning-format deepseek $extra \
    > $BENCH/srv-$label.log 2>&1 &
  until curl -s -m 3 http://127.0.0.1:8082/health 2>/dev/null | grep -q '"status":"ok"'; do sleep 3; done
  local loaded=$(curl -s http://127.0.0.1:8082/props | python3 -c "import json,sys; print(json.load(sys.stdin).get('model_path','?').split('/')[-1])" 2>/dev/null)
  echo "$(date +%T) $label loaded: $loaded"
  cd $BENCH && python3 llm-bench.py "$label" http://127.0.0.1:8082 > $label.run.log 2>&1
  echo "$(date +%T) $label battery done"
}

run_leg ab27-hi  ~/models/Qwen3.8-27B-Heretic-Abliterated-Q4_K_M.gguf qwen3.8-27b-q4 "--chat-template-file $JINJA"
run_leg flint-hi ~/models/Flint-40B-A2.5B-abliterated-Q4_K_M.gguf flint-40b-q4 ""
run_leg huihui-hi ~/models/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-Q4_K_M.gguf huihui-coder-30b-q4 ""
pkill -x llama-server
echo ALL-DONE > $BENCH/_ab_driver.done
