<p align="center"><em>Proudly Made in Nebraska. Go Big Red! 🌽 <a href="https://xkcd.com/2347/">https://xkcd.com/2347/</a></em></p>

# V620 Q4 Three-Way: Dense 27B vs Two Sparse Coders on a Thunderbolt Radeon

[![Hardware](https://img.shields.io/badge/GPU-Radeon%20Pro%20V620%2032GB%20(gfx1030)-red)](https://www.amd.com/en/products/pro/graphics/radeon-pro-v620.html)
[![Link](https://img.shields.io/badge/Link-Thunderbolt%204%20(docked)-blue)](https://en.wikipedia.org/wiki/Thunderbolt_(interface))
[![Engine](https://img.shields.io/badge/llama.cpp%20(Vulkan)-b388--d854d98-orange)](https://github.com/ggml-org/llama.cpp)
[![Quant](https://img.shields.io/badge/Quant-Q4_K_M%20(all%20three)-purple)](#the-matchup)
[![Date](https://img.shields.io/badge/Measured-2026--11--10%20green)](#method)

**Three Q4_K_M models, one 32 GB Radeon Pro V620 in a Thunderbolt dock, one identical speed battery each: does the dense model's per-token brains or the sparse models' per-token speed win on the card we actually own?**

---

## The Matchup

All three are abliterated (refusal-damped), all three quantized identically to Q4_K_M, all three loaded one-at-a-time on the same card under the same power profile with the same harness.

| Model | Architecture | Total / active params | Weights | Peak VRAM (131k ctx) |
|---|---|---|---:|---:|
| **Qwen3.8-27B Heretic** | Dense | 27B / 27B | 16.5 GB | 16 991 MiB |
| **Flint 40B-A2.5B** (Recursive) | MoE, coding-agent tuned | 39.8B / ~2.5B | 24.2 GB | 20 745 MiB |
| **Huihui-Qwen3-Coder-30B** | MoE, coding specialist | 30.5B / 3.3B | 18.6 GB | 19 847 MiB |

## Results — speed battery (steady-state t/s, prefill / gen)

| Prompt | 27B dense | Flint 40B | Coder-30B | Fastest |
|---:|---|---|---|---|
| 512 | 337 / 11.0 | 696 / 26.5 | **911 / 40.7** | Coder-30B (3.7× / 3.7×) |
| 4k | 478 / 10.1 | 1104 / 24.3 | **1272 / 33.3** | Coder-30B |
| 16k | 451 / 9.5 | **1055 / 22.9** | 769 / 22.6 | Flint (prefill), tie (gen) |
| 32k | 372 / 8.9 | **811 / 21.7** | 486 / 19.4 | Flint |
| 65k | 318 / 9.2 | **694 / 15.4** | 311 / 14.4 | Flint |

**The shape of the result:** the Qwen3-Coder wins short-context work outright — at chat/repo-question sizes it is nearly 4× the dense 27B. Flint wins long-context work — its expert routing keeps generation and prefill from collapsing as the KV cache grows, where the Coder-30B's prefill falls off a cliff (1272 → 311 t/s from 4k to 65k). The dense 27B is the slowest everywhere, as physics demands, but its curve is the flattest: it pays its 27B-per-token tax uniformly and degrades the least with depth.

**Nobody sweeps.** Pick by workload: short-context agentic coding → Coder-30B; long-context repo work → Flint; "one model for everything and I'll wait" → the dense 27B.

## Method

- **Host:** ronin28 (Lenovo, Ryzen AI laptop-class host) with Radeon Pro V620 32 GB in a Thunderbolt 4 dock (Navi 21, gfx1030, PCIe Gen4 x16 negotiated at 16 GT/s).
- **Engine:** llama.cpp `build-vulkan-new` (b388-d854d98), `-ngl 99 -c 131072 -fa on -ctk q8_0 -ctv q8_0 --cache-ram 0`.
- **Harness:** `assets/llm-bench.py` — fixed llama.cpp-source corpus tokenized per model, prompt-cache off, 3 repeats at 512/4k/16k, 2 at 32k (fewer at depth for time), steady-state run reported. Driver: `assets/ab-driver.sh`.
- **Isolation:** one model on the card at a time; the driver verifies the loaded model path from `/props` before benching (see gotcha #3).
- Raw data: [`ab27-hi.json`](assets/ab27-hi.json) · [`flint-hi.json`](assets/flint-hi.json) · [`huihui-hi.json`](assets/huihui-hi.json)

## Gotchas — the four things that changed our numbers

1. **The power profile is a benchmark variable.** Out of the box the V620 runs the `BOOTUP_DEFAULT` desktop profile: boost to ~1600 MHz, nap between tokens. LLM decode is memory-bound with modest compute, so the card treats generation like UI bursts. Pinning `3D_FULL_SCREEN` (`echo 1 > /sys/class/drm/card0/device/pp_power_profile_mode`) lifted sustained clocks to ~2310 MHz and prefill by up to +21% at short context. Long-context generation is KV/link-bound and barely moves. **Every number in this repo is pinned-profile.** Unpinned numbers are a different benchmark.
2. **KV precision is a speed dial, not a free knob.** A 2026-10-26 baseline run of Flint at q4_0 KV measured 33.6 t/s gen at 512; tonight's q8_0 KV run measures 26.5. Doubled KV bytes per token cross the Thunderbolt link on every token. We chose q8_0 for quality; the ~20% generation cost is the price, stated here so nobody reads it as a regression.
3. **llama-server answers ANY requested model name with its single loaded model.** A client can request "flint" while a Qwen answers every token, silently, with no error possible. This exact phantom-label failure killed a previous trial. The driver now asserts the loaded path from `/props` before trusting a single number.
4. **VRAM cap is 30 704 MiB, not 32 768, and overflow is fatal, not slow.** Past the cap, spill lands in the GTT (host RAM mapped into GPU page tables — every access crosses Thunderbolt), past ~700 MiB of GTT the first request triggers a PCIe AER storm, and the card drops off the bus until reboot. All three models here peak ≤ 20 745 MiB at 131k ctx with wide margin.

## Not measured (yet)

- **Quality.** This repo measures speed honestly and says nothing about which model writes better code — the fixed coding-prompt battery (win/tie/loss, human-judged) is the planned companion. Speed alone always favors the sparse model; that's the trap this section exists to admit.
- **HIP/ROCm backend.** A `build-hip-new` (gfx1030, `GGML_HIP_NO_VMM=ON`) exists on the host; a Vulkan-vs-HIP A/B is a 15-minute experiment.
- **MTP speculative decoding** for the MoEs (proven 2.4× on a different model on this card class).

## Provenance

- Models: [mradermacher/Flint-40B-A2.5B-abliterated-GGUF](https://huggingface.co/mradermacher/Flint-40B-A2.5B-abliterated-GGUF) · [mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-GGUF](https://huggingface.co/mradermacher/Huihui-Qwen3-Coder-30B-A3B-Instruct-abliterated-GGUF) · Qwen3.8-27B-Heretic-Abliterated (RVN Q4_K_M)
- Staged on the fleet NAS (telesto), copied to the test host over LAN — fleet convention.
- Related: [CryptoJones/strata-heterogeneous-v100-3060](https://github.com/CryptoJones/strata-heterogeneous-v100-3060)
