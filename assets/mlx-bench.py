#!/usr/bin/env python3
"""Speed benchmark for an MLX-LM model on ROCm (gfx1030).

Sends exact-length token prompts with the prompt cache cleared between runs,
generates a fixed 256 tokens, and records MLX's prefill / generation timings
plus V620 VRAM in use. Output matches llm-bench.py JSON schema.

  mlx-bench.py <label> <model_path> [sizes]
"""
import glob
import json
import os
import sys
import time

os.environ.setdefault("ROCM_PATH", "/usr")
os.environ.setdefault("ROCM_HOME", "/usr")

import mlx.core as mx
from mlx_lm import load
from mlx_lm.generate import generate_step
from mlx_lm.sample_utils import make_sampler

LABEL = sys.argv[1]
MODEL_PATH = os.path.expanduser(sys.argv[2])
SIZES = [(512, 3), (4096, 3), (16384, 2), (32768, 1), (65536, 1)]
if len(sys.argv) > 3:
    SIZES = [(int(x), 1) for x in sys.argv[3].split(",")]
GEN = 256
OUT = os.path.expanduser(f"~/v620/bench/{LABEL}.json")
OUT_REPO = os.path.expanduser(f"~/source/repos/v620-q4-three-way/assets/{LABEL}.json")
os.makedirs(os.path.dirname(OUT), exist_ok=True)
os.makedirs(os.path.dirname(OUT_REPO), exist_ok=True)


def vram_mib():
    for f in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):
        try:
            total = int(open(f.replace("used", "total")).read())
            if total > 24 * 2**30:  # the 32 GB V620
                return int(open(f).read()) // 2**20
        except Exception:
            continue
    return None


print(f"Loading model from {MODEL_PATH}...", flush=True)
model, tokenizer = load(MODEL_PATH)
vram_idle = vram_mib()
print(f"Model loaded. VRAM idle: {vram_idle} MiB", flush=True)

# Fixed corpus: llama.cpp source text, tokenized by the model under test.
text = ""
need_chars = max(s for s, _ in SIZES) * 5
for p in sorted(glob.glob(os.path.expanduser("~/llama.cpp/src/*.cpp"))) + \
         sorted(glob.glob(os.path.expanduser("~/llama.cpp/ggml/src/**/*.c*"), recursive=True)):
    text += open(p, errors="ignore").read()
    if len(text) > need_chars:
        break

tokens = tokenizer.encode(text)
assert len(tokens) >= max(s for s, _ in SIZES), f"Need {max(s for s, _ in SIZES)} tokens, got {len(tokens)}"

build_str = f"mlx-{mx.__version__}"

results = {
    "label": LABEL,
    "model": os.path.basename(MODEL_PATH),
    "build": build_str,
    "date": time.strftime("%F %T"),
    "vram_idle_mib": vram_idle,
    "runs": [],
}

if os.path.exists(OUT):
    try:
        existing = json.load(open(OUT))
        results["runs"] = existing.get("runs", [])
        print(f"Resuming {LABEL}: found {len(results['runs'])} completed runs in {OUT}", flush=True)
    except Exception:
        pass

sampler = make_sampler(temp=0.0)

for size, reps in SIZES:
    done_reps = sum(1 for r in results["runs"] if r["prompt_tokens"] == size)
    if done_reps >= reps:
        print(f"Skipping size {size}: already completed {done_reps}/{reps} reps", flush=True)
        continue
    for rep in range(done_reps, reps):
        prompt_slice = tokens[rep * 7 : rep * 7 + size]
        prompt_arr = mx.array(prompt_slice)

        # Clear any caches between runs
        mx.clear_cache()

        t0 = time.perf_counter()
        # Save partial results so runs survive crashes
        for path in (OUT, OUT_REPO):
            with open(path, "w") as f:
                json.dump(results, f, indent=1)

        # Generate exactly GEN tokens (max_tokens=GEN+1: token 0 finishes prefill, 1..GEN are decode)
        token_gen = generate_step(
            prompt_arr,
            model,
            max_tokens=GEN + 1,
            sampler=sampler,
            prefill_step_size=512,
        )

        t_prefill_start = time.perf_counter()
        prefill_s = None
        gen_tokens_count = 0
        t_gen_start = None

        for n, (tok, logprobs) in enumerate(token_gen):
            if n == 0:
                mx.eval(tok)
                t_gen_start = time.perf_counter()
                prefill_s = t_gen_start - t_prefill_start
            else:
                gen_tokens_count += 1
                if gen_tokens_count == GEN:
                    mx.eval(tok)
                    break

        mx.synchronize()
        t_end = time.perf_counter()
        wall_s = t_end - t0
        gen_s = t_end - t_gen_start if t_gen_start else 0.0

        prefill_tps = round(size / prefill_s, 1) if prefill_s and prefill_s > 0 else 0.0
        gen_tps = round(gen_tokens_count / gen_s, 2) if gen_s > 0 else 0.0

        run = {
            "prompt_tokens": size,
            "prefill_tps": prefill_tps,
            "prefill_s": round(prefill_s, 2) if prefill_s else 0.0,
            "gen_tokens": gen_tokens_count,
            "gen_tps": gen_tps,
            "wall_s": round(wall_s, 2),
            "vram_mib": vram_mib(),
        }
        results["runs"].append(run)
        print(json.dumps(run), flush=True)

for path in (OUT, OUT_REPO):
    with open(path, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", path)
