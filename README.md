# MiniMax H3 Local Video Agent & Wan2GP Generative Studio 🎬⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.10 / CUDA 13](https://img.shields.io/badge/PyTorch-2.10%20%7C%20CUDA%2013-red.svg)](https://pytorch.org/)
[![GPU: Universal Support](https://img.shields.io/badge/GPU-RTX%203060%20to%205090%20%2F%20RTX%206000%20Ada%20%2F%20A100%20%2F%20H100-brightgreen.svg)](https://nvidia.com)
[![Engine: Wan2GP v12.6](https://img.shields.io/badge/Engine-Wan2GP%20v12.6-purple.svg)](https://github.com/deepbeepmeep/Wan2GP)

A universal, high-performance local AI generative studio for running **MiniMax H3** and the entire **Wan2GP (WanGP)** generative ecosystem across all NVIDIA GPU tiers (from 8GB/12GB consumer cards like the RTX 3060/4070 up to 16GB/24GB/32GB/48GB+ cards like the RTX 4080/5080/4090/5090, RTX 6000 Ada, A100, and H100).

Generate everything from **synchronized 720p/1080p cinematic videos with native 32kHz stereo audio and character lip-sync** in 2–4 minutes, to **5-minute complete songs with lyrics**, **character motion transfers**, **interactive video inpainting**, and **4K infographics**.

---

## 📋 Table of Contents
1. [🚀 1-Click Quickstart (Windows & Linux)](#-1-click-quickstart)
2. [🖥️ Universal Hardware & VRAM Guide (8GB to 80GB+)](#️-universal-hardware--vram-guide)
3. [🧠 MiniMax H3 Deep-Dive: Architectures & Use Cases](#-minimax-h3-deep-dive)
4. [⚙️ Comprehensive Settings, Modes & What They Do](#️-comprehensive-settings--modes-guide)
   - [Quantization & Text Encoders](#1-quantization--text-encoders)
   - [Acceleration & Step Skipping (Turbo LoRA, First Block Cache, SageAttn)](#2-acceleration--step-skipping)
   - [Generation Phases & Latent Upscaling (1-Phase, 2-Phase, Tiling)](#3-generation-phases--latent-upscaling)
   - [Longer Videos, Overlap Frames & Storyboarding](#4-longer-videos-sliding-windows--storyboarding)
   - [Audio Conditioning & Lip-Sync Dialogue Syntax](#5-audio-conditioning--lip-sync-dialogue-syntax)
5. [🎨 The Full Wan2GP Studio: What Models Are Included](#-the-full-wan2gp-studio)
6. [🧰 Built-In Tools & Post-Processing Suite](#-built-in-tools--post-processing-suite)
7. [⚠️ Known Limitations, Constraints & Trade-Offs](#️-known-limitations--trade-offs)
8. [📦 Model Catalog & Download Options](#-model-catalog--download-options)
9. [🛠️ Troubleshooting & Memory FAQs](#️-troubleshooting--memory-faqs)
10. [🙏 Credits & License](#-credits--license)

---

## 🚀 1-Click Quickstart

### Windows (1-Click)
1. **Clone this repository:**
   ```cmd
   git clone https://github.com/PatelVrund-GameDev/minimax-h3-local-runner.git
   cd minimax-h3-local-runner
   ```
2. **Run the Installer:** Double-click **`setup_windows.bat`** *(Creates Python 3.11 `.venv`, installs PyTorch with CUDA, Triton Windows, SageAttention 2.2.0, and dependencies)*.
3. **Download Models:** Double-click **`download_models.bat`** *(Select your hardware tier to automatically fetch checkpoints with resume support)*.
4. **Start the Web UI:** Double-click **`run_windows.bat`** and open `http://127.0.0.1:7860` in your browser.

---

### Linux / WSL (1-Click)
```bash
git clone https://github.com/PatelVrund-GameDev/minimax-h3-local-runner.git
cd minimax-h3-local-runner
chmod +x *.sh

./setup_linux.sh          # 1. Automatic installation
./download_models.sh      # 2. Interactive model downloader
./run_linux.sh            # 3. Launch Gradio Web UI
```

---

## 📂 Checkpoints & Expected File Locations

If you use `download_models.bat` (or manually place files), here is the exact folder structure and expected file paths on disk:

```text
local-video-agent/
 └── wan2gp_core/
      ├── ckpts/
      │    ├── MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors   <-- Main Model (Pruned 20B INT8)
      │    ├── MiniMax-H3-FL2VA_int8_convrot.safetensors               <-- Main Model (Full 33B INT8)
      │    ├── MiniMax-H3-video_vae_fp16.safetensors                   <-- Video VAE
      │    ├── MiniMax-H3-audio_vae_fp32.safetensors                   <-- Audio VAE
      │    ├── minimax_h3/
      │    │    └── minimax_h3_latent_upscaler_3d_bf16.safetensors     <-- 3D Latent Upscaler
      │    └── Qwen3-VL-32B-Instruct/
      │         ├── Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors <-- Text Encoder (Quanto INT8)
      │         ├── config.json
      │         ├── tokenizer.json
      │         ├── tokenizer_config.json
      │         ├── preprocessor_config.json
      │         └── vocab.json
      └── loras/
           └── minimax_h3/
                ├── minimax_h3_lightx2v_fl2v_turbo_4step_alpha128_v1.0_768p_bf16.safetensors <-- 4-Step Turbo LoRA
                └── minimax_h3_lightx2v_fl2v_turbo_8step_alpha8_v1.0_bf16.safetensors        <-- 8-Step Turbo LoRA
```

### 🔍 Understanding the Web UI "URLs" Tab:
Inside the Web UI under the **URLs** tab:
* **Main Checkpoints**: Displays the supported Hugging Face URLs for the selected model. When generating, Wan2GP automatically matches these filenames to your local files inside `wan2gp_core/ckpts/` and loads them directly from disk.
* **Text Encoder Checkpoints**: Left **blank by default**. Leaving this empty tells Wan2GP to load the official built-in text encoder (`wan2gp_core/ckpts/Qwen3-VL-32B-Instruct/Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors`). You can also manually paste custom paths or browse via the yellow folder icon.
* **Video & Audio VAE File**: Left **blank by default** to use `MiniMax-H3-video_vae_fp16.safetensors` and `MiniMax-H3-audio_vae_fp32.safetensors`.

---

## 🖥️ Universal Hardware & VRAM Guide

MiniMax H3 and Wan2GP are built to scale dynamically from budget laptops to massive server clusters:

| Hardware Tier | Typical GPUs | VRAM | Recommended Model & Text Encoder | Max Target Resolution | Workflow & Expected Speed |
|---|---|---|---|---|---|
| **Tier 1: Entry / Budget** | RTX 3060 (12G), RTX 4060 Ti (16G), RTX 2080 Ti (11G) | **8 – 12 GB** | **FL2VA Pruned 20B (INT8)** + GGUF Q4 Text Enc + FP8 VAE | **480p – 720p** | Turbo LoRA (4–6 steps) + Two-Phase Tiling (~1–3 min) |
| **Tier 2: Mid / High Consumer** | RTX 3080, 4070, 4070 Ti, 4080, **RTX 5080** | **12 – 16 GB** | **FL2VA Pruned 20B (INT8)** + Quanto INT8 Text Enc | **720p Native / 1080p 2-Phase** | Turbo LoRA (6–8 steps) + First Block Cache (0.08) (~2–4 min) |
| **Tier 3: Top Enthusiast** | RTX 3090, 4090, **RTX 5090** | **24 – 32 GB** | **FL2VA 33B (INT8 or BF16)** / Pruned 20B | **1080p Native** | Single-Phase direct 1080p, 8–12 steps Turbo or 20 steps standard (~1.5–3 min) |
| **Tier 4: Enterprise Workstation** | RTX 6000 Ada, A100 (80G), H100, B200 | **48 – 80 GB+** | **FL2VA 33B Full BF16** + Full BF16 Text Enc | **1080p / 4K** | Full unpruned precision, Ralston 2S sampler, continuous multi-window movie rendering (~30s–2 min) |

*System RAM: 32 GB or more is recommended when streaming quantized weights.*

---

## 🧠 MiniMax H3 Deep-Dive

MiniMax H3 is a multi-modal foundation model capable of producing high-definition video along with native 32kHz stereo audio and speech lip-synchronization in a single diffusion process.

```mermaid
graph TD
    Prompt[Text / Dialogue Prompt] --> H3[MiniMax H3 Multi-Modal Diffusion]
    StartImg[Optional Start Image] --> H3
    EndImg[Optional End Image] --> H3
    RefMedia[Optional Multi-References] --> H3
    H3 --> VideoStream[High Definition Video 24 FPS]
    H3 --> AudioStream[Synchronized 32kHz Stereo Audio & Lip-Sync]
```

### 1. FL2VA (First/Last Frame to Video & Audio)
* **What it does**: Generates video + audio from text alone, a start image, an end image, or both start and end images.
* **Pruned 20B (`minimax_h3_fl2va_pruned`)**: Rank-8 pruned architecture that retains ~95% of visual quality while reducing VRAM usage by 40% and running significantly faster. **Best for 8GB–16GB VRAM.**
* **Full 33B (`minimax_h3_fl2va`)**: The complete unpruned model. Delivers the absolute highest fidelity for intricate background physics, lighting, and facial micro-expressions. **Best for 24GB+ VRAM.**

### 2. Ref2VA (Reference to Video & Audio)
* **What it does**: Ingests up to **9 reference images**, **2 reference video clips**, and **2 audio clips** to maintain consistent character faces, costumes, environments, and voices across separate shots without fixing the camera to a start frame.
* **Best used for**: Multi-shot storytelling, recurring cast members, and style transfer.

### 3. PDD (Parallel Denoising Diffusion) Models
* **What it does**: Merges 4 learned denoising-interval outputs simultaneously, covering 32 intervals in **just 8 model evaluations**.
* **Best used for**: Fixed ultra-fast 8-step generation with the Euler scheduler.

---

## ⚙️ Comprehensive Settings & Modes Guide

### 1. Quantization & Text Encoders
* **Quanto INT8 (`Qwen3-VL-32B...quanto_bf16_int8`)**: Takes ~6 GB VRAM. Recommended default for prompt comprehension without high memory overhead.
* **GGUF Q4_K_M (`qwen3vl...Q4_K_M.gguf`)**: Quantized format taking ~14 GB System RAM / 7 GB VRAM. Ideal for budget PCs with 16GB–24GB System RAM.
* **Full BF16 (`Qwen3-VL-32B...bf16`)**: Full 16-bit precision text encoder (~64 GB file size). Recommended for 24GB–48GB+ workstation GPUs.

---

### 2. Acceleration & Step Skipping
* **LightX2V Turbo LoRA**:
  * Cuts sampling steps from 20–30 down to **4 to 8 steps**, reducing generation time by **10x** (from 30+ minutes down to 2–4 minutes).
  * *Multiplier (0.5 – 0.6)*: Balanced motion stability and prompt fidelity.
  * *Multiplier (0.8 – 1.0)*: Maximum convergence speed.
* **First Block Cache (TeaCache-style)**:
  * Measures the output change in transformer block 0. If negligible, it skips re-evaluating blocks 1–49 for that step.
  * `0.06 (Low)`: Best visual fidelity.
  * `0.08 (Balanced)`: **Recommended default** (~25% speedup with imperceptible difference).
  * `0.12+ (High)`: Ultra-fast draft mode.
* **Attention Kernels**:
  * `SageAttention 2.2.0`: ~2x faster attention evaluation for RTX 30/40/50 series GPUs.
  * `Sol-Attn (Sparse Attention)`: 10–30% extra speedup on Blackwell (RTX 50-series) and Ada Lovelace (RTX 40-series) when using Triton 3.6+.
* **Samplers (Euler vs Ralston 2S)**:
  * *Euler*: Standard fast solver used with Turbo LoRAs.
  * *Ralston 2S*: Second-order Runge-Kutta solver with `1/4, 3/4` weighting. Produces cleaner motion and detail retention, but evaluates the transformer twice per step (**2x slower**).

---

### 3. Generation Phases & Latent Upscaling
Under *Advanced Mode -> General -> Phases*:
1. **One Phase (Direct)**: Direct generation at target resolution. Best whole-frame coherence; highest peak VRAM.
2. **Two Phases (Latent Upscaler)**: Generates base motion at half resolution (360p), upscales the 3D latent with `minimax_h3_latent_upscaler_3d_bf16`, and performs a 3-step high-res refinement.
3. **Two Phases with Tiling (Low VRAM 1080p/4K)**: Splits the high-resolution refinement into 4 spatial quadrants. **Allows 1080p/4K generation on 12GB–16GB VRAM cards without crashing!**
   * *Tip*: Lower *Phase 2 Noise Level Start* (default `0.05` -> `0.03`) to eliminate quadrant seams.

---

### 4. Longer Videos, Sliding Windows & Storyboarding
MiniMax H3 generates 4 to 15 seconds per window natively at 24 FPS. To produce clips of **30s, 60s, or full minutes**:
* **Sliding Window Continuation**:
  * Set overlap frames to H3 standard increments: `18`, `35`, or `52` frames. Wan2GP automatically carries the previous segment's end motion and audio harmonics across the join for smooth continuity.
* **Hard Scene Cuts**:
  * Use `[/new_shot]` in your prompt to direct a camera cut between sliding windows without visual bleeding.
* **Frames Injection**:
  * Insert reference images at exact frame numbers (e.g. Frame 1, Frame 72, or `L` for end frame) to anchor key visual events.

---

### 5. Audio Conditioning & Lip-Sync Dialogue Syntax
MiniMax H3 accepts structured multimodal prompts for dialogue, Foley sound effects, and musical score:

```text
integrated_multimodal_description: [Shot 1] A cinematic medium shot of a detective standing in the rain under neon lights. He adjusts his coat, looks directly at the camera, and says clearly (S1) <d>[English] We only have five minutes before the system locks us out.</d> While speaking, he activates a glowing data chip in his hand.
overall_soundscape: Rain falling on asphalt, puddle splashes, distant siren hum, the scientist's synchronized vocal delivery, and an electronic pulse chime.
non_diegetic_music: A dark atmospheric synthwave bassline rising slowly in the background.
```

* `(S1) <d>[Language] Text here.</d>`: Marks spoken dialogue for Speaker 1 with lip synchronization.
* `overall_soundscape: ...`: Describes diegetic Foley sound effects and ambient noise.
* `non_diegetic_music: ...`: Describes the musical score and background instruments.

---

## 🎨 The Full Wan2GP Studio

Wan2GP is a complete multi-modal studio. You can switch models on the fly without restarting:

```
Wan2GP Studio
 ├── Video Models:
 │    ├── MiniMax H3 (FL2VA, Ref2VA, PDD)
 │    ├── LTX-2.3 / 2.5 (22B native audio-video)
 │    ├── Wan 2.1 & 2.2 (T2V, I2V, TI2V 5B)
 │    ├── VACE (Inpainting & Outpainting)
 │    ├── Bernini-R (Video-to-Video & Multi-ref)
 │    ├── SCAIL-2 & Wan Animate (Motion transfer)
 │    └── MultiTalk / InfiniteTalk / LongCat (Talking heads)
 ├── Image Models:
 │    ├── Krea 2 (RAW, Turbo, Identity Edit)
 │    ├── Qwen Image Edit Plus (20B multi-subject composition)
 │    ├── Ideogram 4 (Typography, posters, graphic design)
 │    ├── Flux 1/2 (Schnell, Dev, Chroma, Klein)
 │    ├── Z-Image Turbo (Efficient 6B distilled model)
 │    └── SenseNova-U1.5 (Native 4K infographics)
 └── Audio & TTS Models:
      ├── MiniMax Music 3 (5-minute full songs with lyrics)
      ├── Qwen3 TTS Base (Zero-shot voice cloning)
      ├── IndexTTS 2 & 2.5 (Emotional expressive speech)
      ├── ACE-Step 1.5 XL / HeartMuLa (Full songs)
      └── Stable Audio 3 (Sound effects & ambience)
```

---

## 🧰 Built-In Tools & Post-Processing Suite

* **Matanyone**: Interactive video segmentation. Click on a person/object in frame 1, and it automatically tracks and creates a video mask across all frames.
* **SeedVC**: Voice-swapping tool. Replace the actor's voice in an existing video with any cloned voice sample.
* **MMAudio & PrismAudio**: Analyzes silent video clips and synthesizes matching synchronized sound effects.
* **RIFE 2x / 4x**: Temporal frame interpolation (turns 24 FPS video into fluid 60 FPS / 120 FPS).
* **FlashVSR & LTX Spatial Upscalers**: Super-resolution models that upscale 480p/720p videos up to 1080p/4K.
* **Deepy (Creative AI Agent)**:
  * *Deepy Zero*: Fast assistant for immediate prompts and setting overrides.
  * *Deepy Prime (Qwen3.8 27B / Claude / Codex)*: Autonomous agent that plans multi-scene video productions, manages project files, and integrates with **Model Context Protocol (MCP)** servers.

---

## ⚠️ Known Limitations & Trade-Offs

| Area | Limitation / Constraint | Solution & Workaround |
|---|---|---|
| **Single GPU Only** | Wan2GP runs on **one GPU at a time**; dual cards (e.g. 2x T4 or 2x 3090) do **not** pool VRAM together. | Select the fastest single GPU. Use MMGP offloading to leverage System RAM. |
| **System RAM Requirement** | When offloading weights to fit low VRAM, models require **substantial System RAM** (32 GB RAM recommended). | Use **GGUF Q4 text encoders** and **FP8 mixed VAEs** to minimize RAM overhead. |
| **Two-Phase Tiling Seams** | Tiling the 2nd phase into 4 quadrants can occasionally cause visible seams if motion is erratic. | Lower the *Phase 2 Noise Level Start* slider (default `0.05` -> `0.03`) for smoother blending. |
| **Turbo LoRA Detail Trade-off** | 4-step Turbo LoRA is ~10x faster (2 min vs 30 min) but can slightly soften fine background textures compared to 20-step unaccelerated generation. | Use 8 steps instead of 4 steps, or set the Turbo LoRA multiplier to `0.5` – `0.6`. |
| **Deepy Prime VRAM** | Running local Deepy Prime with Qwen3.8 27B requires 16GB–24GB VRAM on its own. | Use **Remote LLMs** (Claude Code / OpenAI Codex / OpenCode) in Configuration so Deepy uses 0 VRAM! |
| **Video Duration Limits** | MiniMax H3's native training window is **4–15 seconds at 24 FPS**. | Use **Sliding Windows** with `18`, `35`, or `52` overlap frames to stitch continuous 30s, 60s, or multi-minute sequences. |

---

## 📦 Model Catalog & Download Options

Run the interactive downloader anytime:
```cmd
python download_models.py
```

### Download Presets:
* **`1. entry` (~38.0 GB)**: Pruned INT8 + GGUF Q4 Text Encoder + FP8 VAE + Turbo LoRA *(For 8GB–12GB VRAM: RTX 3060/4060)*.
* **`2. balanced` (~39.5 GB)**: Pruned INT8 + Quanto INT8 Text Encoder + FP16 VAE + Latent Upscaler + Turbo LoRA *(For 12GB–16GB VRAM: RTX 5080/4080/4070/3080)*.
* **`3. enthusiast` (~51.0 GB)**: Full 33B INT8 ConvRot + Quanto INT8 + FP16 VAE + Latent Upscaler + Turbo LoRA *(For 24GB–32GB VRAM: RTX 4090/3090/5090)*.
* **`4. workstation` (~132.0 GB)**: Full 33B BF16 + Full BF16 Text Encoder + VAEs + All LoRAs *(For 48GB–80GB+ VRAM: RTX 6000 Ada, A100, H100)*.
* **`5. turbo_only` (~0.4 GB)**: Just the Latent 3D Upscaler and LightX2V Turbo LoRAs.

---

## 🛠️ Troubleshooting & Memory FAQs

### 1. `CUDA out of memory` (OOM)
* Switch from `One Phase` to `Two Phases with Tiling` under *Advanced Mode*.
* Use the **Pruned INT8 ConvRot** model rather than unpruned BF16.
* Ensure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set (automatically handled by `run_windows.bat` and `run_linux.sh`).

### 2. Downloads filling up the `C:` drive
* The launcher scripts automatically route `HF_HOME`, `TORCH_HOME`, and `TMPDIR` to the project folder on your large storage drive.

### 3. Audio and video out of sync
* Ensure the output frame rate is set to **24 FPS** (MiniMax H3's native training rate).

---

## 🙏 Credits & License

* **[MiniMax AI](https://huggingface.co/MiniMaxAI)**: MiniMax H3 foundation model.
* **[DeepBeepMeep (Wan2GP)](https://github.com/deepbeepmeep/Wan2GP)**: Wan2GP engine, INT8 ConvRot quantization, and Web UI.
* **[LightX2V](https://github.com/LightX2V)**: Turbo LoRA acceleration models.
* **[Qwen Team](https://github.com/QwenLM/Qwen)**: Qwen3-VL multimodal vision-language models.

This repository is open-sourced under the [MIT License](LICENSE).
