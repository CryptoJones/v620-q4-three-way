#!/usr/bin/env python3
"""Speed benchmark for a running llama-server (same API claude-local uses).

Sends exact-length token prompts with the prompt cache off, generates a fixed
256 tokens, and records the server's own prefill / generation timings plus
V620 VRAM in use. Run it unchanged against each model to compare them.

  llm-bench.py <label> [base_url]   -> writes ~/v620/bench/<label>.json
"""
import glob, json, os, sys, time, urllib.request

LABEL = sys.argv[1]
BASE = sys.argv[2] if len(sys.argv) > 2 else "http://127.0.0.1:8081"
SIZES = [(512, 3), (4096, 3), (16384, 2), (32768, 1), (65536, 1)]
# Optional third arg: comma-separated prompt sizes, one run each (e.g. 131072,261888).
if len(sys.argv) > 3:
    SIZES = [(int(x), 1) for x in sys.argv[3].split(",")]
GEN = 256
OUT = os.path.expanduser(f"~/v620/bench/{LABEL}.json")


def post(path, body):
    req = urllib.request.Request(BASE + path, json.dumps(body).encode(),
                                 {"Content-Type": "application/json"})
    with urllib.request.urlopen(req, timeout=4 * 3600) as r:
        return json.load(r)


def vram_mib():
    for f in glob.glob("/sys/class/drm/card*/device/mem_info_vram_used"):
        total = int(open(f.replace("used", "total")).read())
        if total > 24 * 2**30:  # the 32 GB V620
            return int(open(f).read()) // 2**20
    return None


# Fixed corpus: llama.cpp source text, tokenized by the model under test.
text = ""
need_chars = max(s for s, _ in SIZES) * 5
for p in sorted(glob.glob(os.path.expanduser("~/llama.cpp/src/*.cpp"))) + \
         sorted(glob.glob(os.path.expanduser("~/llama.cpp/ggml/src/**/*.c*"), recursive=True)):
    text += open(p, errors="ignore").read()
    if len(text) > need_chars:
        break
tokens = post("/tokenize", {"content": text})["tokens"]
assert len(tokens) >= max(s for s, _ in SIZES), len(tokens)

props = json.load(urllib.request.urlopen(BASE + "/props"))
# Clamp prompts so prompt + GEN fits the server's context (a 128k prompt in a 128k ctx is a 400).
n_ctx = props.get("default_generation_settings", {}).get("n_ctx") or 0
if n_ctx:
    SIZES = [(min(s, n_ctx - GEN - 64), r) for s, r in SIZES]
results = {"label": LABEL, "model": props.get("model_alias") or props.get("model_path"),
           "build": props.get("build_info"), "date": time.strftime("%F %T"),
           "vram_idle_mib": vram_mib(), "runs": []}

for size, reps in SIZES:
    for rep in range(reps):
        # Offset the prompt per rep so nothing can be reused between runs.
        prompt = tokens[rep * 7: rep * 7 + size]
        t0 = time.time()
        json.dump(results, open(OUT, "w"), indent=1)  # partial results survive a crash
        r = post("/completion", {"prompt": prompt, "n_predict": GEN, "temperature": 0,
                                 "ignore_eos": True, "cache_prompt": False})
        t = r["timings"]
        run = {"prompt_tokens": t["prompt_n"], "prefill_tps": round(t["prompt_per_second"], 1),
               "prefill_s": round(t["prompt_ms"] / 1000, 2), "gen_tokens": t["predicted_n"],
               "gen_tps": round(t["predicted_per_second"], 2), "wall_s": round(time.time() - t0, 2),
               "vram_mib": vram_mib()}
        results["runs"].append(run)
        print(json.dumps(run), flush=True)

json.dump(results, open(OUT, "w"), indent=1)
print("wrote", OUT)
