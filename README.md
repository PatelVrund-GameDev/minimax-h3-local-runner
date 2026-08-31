# MiniMax H3 Local Video Agent 🎬⚡

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](LICENSE)
[![Python 3.11](https://img.shields.io/badge/Python-3.11-blue.svg)](https://www.python.org/)
[![PyTorch 2.10 / CUDA 13](https://img.shields.io/badge/PyTorch-2.10%20%7C%20CUDA%2013-red.svg)](https://pytorch.org/)
[![GPU: RTX 5080 / 4080 / 3080](https://img.shields.io/badge/GPU-RTX%205080%20%7C%2012GB--16GB%20VRAM-green.svg)](https://nvidia.com)
[![Engine: Wan2GP](https://img.shields.io/badge/Engine-Wan2GP%20v12.6-purple.svg)](https://github.com/deepbeepmeep/Wan2GP)

Run **MiniMax H3** (FL2VA & Ref2VA) locally on your PC with consumer NVIDIA GPUs (e.g. **RTX 5080, RTX 4080, RTX 4070, RTX 3080, RTX 4090**). Generate high quality **720p 15-second videos with native synchronized stereo audio** using only **~12GB VRAM**!

Powered by the high-performance **Wan2GP** engine with **SageAttention 2.2.0**, **First Block Caching**, and **LightX2V Turbo LoRA (4–8 step inference)**.

---

## 🌟 Key Features

- **Local & Private**: No cloud subscriptions, no Kaggle/Colab session timeouts, and full offline generation.
- **Low VRAM Friendly (12GB–16GB)**: Optimized **INT8 ConvRot** quantization and **Quanto/GGUF text encoders** allow 720p 15-second video generation on an RTX 5080 / 4080 / 4070 in ~12GB VRAM.
- **Synchronized Video + 32kHz Stereo Audio**: MiniMax H3 generates video along with synchronized character speech, lip-syncing, sound effects, and background music in a single pass.
- **Ultra-Fast Generation (Turbo LoRA)**: Generates 5–15 second clips in **2 to 4 minutes** using 4–8 steps (compared to 30+ minutes standard sampling).
- **First Block Caching (0.08)**: Reuses transformer block evaluations for a ~20–30% boost in speed with zero quality degradation.
- **1-Click Launchers**: Automated setup and run scripts for Windows and Linux.
- **Smart Drive Storage**: Keeps large model weights and caches on your high-capacity drive (avoiding `C:` drive overflow).

---

## 🚀 Quick Start

### Windows (1-Click Setup)

1. **Clone the repository:**
   ```cmd
   git clone https://github.com/YOUR_USERNAME/local-video-agent.git
   cd local-video-agent
   ```

2. **Run the 1-Click Installer:**
   Double-click **`setup_windows.bat`** (or run via PowerShell/CMD):
   ```cmd
   .\setup_windows.bat
   ```
   *This automatically sets up a Python 3.11 virtual environment, installs PyTorch with CUDA acceleration, Triton Windows, and SageAttention 2.2.0.*

3. **Download Model Weights:**
   Double-click **`download_models.bat`** (or run `python download_models.py --preset recommended`).
   *Downloads the quantized 20B MiniMax H3 model, Qwen3-VL INT8 text encoder, VAEs, and Turbo LoRA (~38 GB total).*

4. **Launch the Web Interface:**
   Double-click **`run_windows.bat`**.
   *The Gradio Web UI will start and automatically be accessible at `http://127.0.0.1:7860`.*

---

### Linux / WSL (1-Click Setup)

```bash
git clone https://github.com/YOUR_USERNAME/local-video-agent.git
cd local-video-agent
chmod +x *.sh

# 1. Install dependencies
./setup_linux.sh

# 2. Download models
./download_models.sh --preset recommended

# 3. Launch Web UI
./run_linux.sh
```

---

## 💻 Hardware Requirements & Performance

| GPU Model | VRAM | Recommended Model | Resolution | Steps (Turbo LoRA) | Est. Time (15s Video) |
|---|---|---|---|---|---|
| **RTX 5080 (Blackwell)** | **16 GB** | **FL2VA Pruned INT8** | **720p (1280x720)** | **6 – 8 steps** | **~2.5 – 4 mins** |
| **RTX 4080 / 4080 Super** | **16 GB** | **FL2VA Pruned INT8** | **720p (1280x720)** | **6 – 8 steps** | **~3 – 5 mins** |
| **RTX 4070 Ti / 3080 (12G)** | **12 GB** | **FL2VA Pruned INT8** | **480p – 720p** | **4 – 6 steps** | **~4 – 6 mins** |
| **RTX 4090 / 3090** | **24 GB** | **FL2VA Pruned / BF16** | **720p – 1080p** | **6 – 8 steps** | **~2 – 3 mins** |

*Note: System RAM recommended: 32 GB or more. Disk space required: ~40 GB free on target drive.*

---

## 📦 Model Files & Download Catalog

All weights are sourced from [`DeepBeepMeep/MiniMax-H3`](https://huggingface.co/DeepBeepMeep/MiniMax-H3):

| Component | File Name | Size | Description |
|---|---|---|---|
| **Main Checkpoint (Recommended)** | `MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors` | **~21 GB** | 20B Pruned INT8 ConvRot for 12G-16G GPUs |
| **Main Checkpoint (Full BF16)** | `MiniMax-H3-FL2VA_bf16.safetensors` | **~66 GB** | 33B Full precision for 24GB+ GPUs |
| **Text Encoder** | `Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors` | **~16.5 GB** | Quantized 50-layer Qwen3-VL text encoder |
| **Video VAE** | `MiniMax-H3-video_vae_fp16.safetensors` | **~1.4 GB** | MiniMax Video VAE (FP16) |
| **Audio VAE** | `MiniMax-H3-audio_vae_fp32.safetensors` | **~0.3 GB** | MiniMax Audio VAE (32kHz Stereo) |
| **Latent Upscaler** | `minimax_h3_latent_upscaler_3d_bf16.safetensors` | **~0.2 GB** | Latent 3D Upscaler for two-phase generation |
| **Turbo LoRA** | `minimax_h3_lightx2v_fl2v_turbo_4step_alpha16_v0.1.safetensors` | **~0.1 GB** | 4-step acceleration Turbo LoRA |

### Manual Download CLI Options:
```bash
# Recommended for RTX 5080 (16GB VRAM) / RTX 4080 / 4070
python download_models.py --preset recommended

# Low RAM systems (with GGUF text encoder & FP8 VAE)
python download_models.py --preset recommended_gguf

# Full precision for 24GB+ GPUs
python download_models.py --preset full

# List available catalog items
python download_models.py --list
```

---

## ⚙️ Recommended Settings for RTX 5080 (16GB VRAM)

Inside the Wan2GP Web Interface (`http://127.0.0.1:7860`):

1. **Model Selection**: Select `MiniMax H3 FL2VA Pruned 20B`.
2. **LoRA Settings**:
   - Check `Turbo LoRA` (LightX2V 4-Step).
   - Set **LoRA Multiplier** to `0.5` – `0.6`.
3. **Sampling Settings**:
   - **Inference Steps**: `6` to `8` steps (Euler sampler).
   - **Guidance Scale**: `4.0` – `5.0`.
4. **Step Acceleration**:
   - Under *Advanced Mode / Steps Skipping*, enable **First Block Cache** (`Balanced (0.08)`).
5. **Resolution & Duration**:
   - Resolution: `1280x720` (720p) or `854x480` (480p).
   - Duration: `5` to `15` seconds (24 FPS).

---

## 📝 Prompting Guide for Synchronized Dialogue & Audio

MiniMax H3 understands structured multimodal prompts for character dialogue, sound effects, and background audio:

```text
integrated_multimodal_description: [Shot 1] A cinematic close-up of an astronaut inside a dimly lit spacecraft looking out at a nebula. The camera slowly dollies forward. He touches his radio headset and says clearly (S1) <d>[English] Mission control, we have confirmed contact.</d> A subtle green light pulses across his helmet visor as he speaks.
overall_soundscape: Low cabin hum, air ventilation hiss, faint radio static, crystal-clear vocal delivery, and an electronic confirmation chime.
non_diegetic_music: A deep atmospheric ambient synth swell rising slowly in the background.
```

- `(S1) <d>[English] Your text here.</d>`: Marks spoken dialogue for speaker 1 with automated lip synchronization.
- `overall_soundscape: ...`: Describes diegetic Foley sound effects and ambient noise.
- `non_diegetic_music: ...`: Describes musical score and instruments.

---

## 🛠️ Troubleshooting & FAQ

### 1. `CUDA out of memory` during generation
- Make sure you are using the **Pruned INT8 ConvRot** model (`MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors`).
- If generating at 720p or 1080p, enable **Two Phases with Tiling** in *Advanced Mode / Phases*.
- Ensure `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True` is set (automatically handled by `run_windows.bat`).

### 2. Downloads filling up the `C:` drive
- `run_windows.bat` and `setup_windows.bat` redirect `HF_HOME`, `TORCH_HOME`, and `TMPDIR` to the project directory on your large storage drive.

### 3. SageAttention / Triton not loading
- Run `setup_windows.bat` to ensure `triton-windows` and `sageattention 2.2.0` are installed into the active `.venv`.

---

## 🙏 Credits & Acknowledgments

- **[MiniMax AI](https://huggingface.co/MiniMaxAI)** for creating and open-sourcing the remarkable MiniMax H3 multimodal foundation model.
- **[DeepBeepMeep (Wan2GP)](https://github.com/deepbeepmeep/Wan2GP)** for the optimized GPU-poor inference engine, INT8 ConvRot quantization, and Web UI.
- **[LightX2V](https://github.com/LightX2V)** for the MiniMax H3 4-step Turbo LoRA weights.
- **[Qwen Team (Alibaba Cloud)](https://github.com/QwenLM/Qwen)** for Qwen3-VL multimodal text encoder.

---

## 📄 License

This repository is licensed under the [MIT License](LICENSE). Checkpoint weights are subject to the upstream MiniMax and Wan2GP model licenses.
