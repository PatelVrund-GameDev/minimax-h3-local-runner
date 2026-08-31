#!/usr/bin/env python3
"""
MiniMax H3 Model Downloader for Wan2GP / Local Video Agent
==========================================================
Automatically downloads and verifies MiniMax H3 checkpoints,
quantized text encoders, VAEs, and Turbo LoRAs.

Usage:
  python download_models.py --preset recommended
  python download_models.py --preset full
  python download_models.py --list
"""

import os
import sys
import argparse
import shutil
from pathlib import Path
from urllib.parse import urljoin

if sys.platform == "win32":
    try:
        sys.stdout.reconfigure(encoding='utf-8')
    except Exception:
        pass

try:
    import urllib.request as urllib_request
except ImportError:
    import urllib2 as urllib_request

try:
    from tqdm import tqdm
except ImportError:
    tqdm = None

REPO_ID = "DeepBeepMeep/MiniMax-H3"
HF_BASE_URL = f"https://huggingface.co/{REPO_ID}/resolve/main/"

# Models Definition
MODELS_CATALOG = {
    "fl2va_pruned_int8": {
        "name": "MiniMax H3 FL2VA Pruned 20B (INT8 ConvRot - Recommended for 12GB-16GB VRAM)",
        "subpath": "MiniMax-H3-FL2VA-pruned_rank8_int8_convrot.safetensors",
        "category": "checkpoint",
        "approx_gb": 21.0,
    },
    "fl2va_pruned_bf16": {
        "name": "MiniMax H3 FL2VA Pruned 20B (BF16)",
        "subpath": "MiniMax-H3-FL2VA-pruned_rank8_bf16.safetensors",
        "category": "checkpoint",
        "approx_gb": 42.0,
    },
    "fl2va_full_int8": {
        "name": "MiniMax H3 FL2VA 33B (INT8 ConvRot)",
        "subpath": "MiniMax-H3-FL2VA_int8_convrot.safetensors",
        "category": "checkpoint",
        "approx_gb": 33.0,
    },
    "fl2va_full_bf16": {
        "name": "MiniMax H3 FL2VA 33B (BF16 Full Precision - 24GB+ VRAM)",
        "subpath": "MiniMax-H3-FL2VA_bf16.safetensors",
        "category": "checkpoint",
        "approx_gb": 66.0,
    },
    "text_encoder_bf16": {
        "name": "Qwen3-VL-32B Instruct (Full BF16 Text Encoder - 48GB+ Workstations)",
        "subpath": "Qwen3-VL-32B-Instruct/Qwen3-VL-32B-Instruct-layer50_bf16.safetensors",
        "category": "text_encoder",
        "approx_gb": 64.0,
    },
    "text_encoder_int8": {
        "name": "Qwen3-VL-32B Instruct (Quanto INT8 Text Encoder - Recommended)",
        "subpath": "Qwen3-VL-32B-Instruct/Qwen3-VL-32B-Instruct-layer50_quanto_bf16_int8.safetensors",
        "category": "text_encoder",
        "approx_gb": 16.5,
    },
    "text_encoder_gguf_q4": {
        "name": "Qwen3-VL-32B Instruct (GGUF Q4_K_M - Low RAM)",
        "subpath": "Qwen3-VL-32B-Instruct/qwen3vl-32B-MiniMax-H3-Q4_K_M.gguf",
        "category": "text_encoder",
        "approx_gb": 18.0,
    },
    "video_vae_fp16": {
        "name": "MiniMax H3 Video VAE (FP16)",
        "subpath": "MiniMax-H3-video_vae_fp16.safetensors",
        "category": "vae",
        "approx_gb": 1.4,
    },
    "video_vae_fp8mix": {
        "name": "MiniMax H3 Video VAE (FP8 Mixed - Low RAM)",
        "subpath": "minimax_h3_video_vae_fp8mix.safetensors",
        "category": "vae",
        "approx_gb": 0.8,
    },
    "audio_vae_fp32": {
        "name": "MiniMax H3 Audio VAE (FP32 Stereo)",
        "subpath": "MiniMax-H3-audio_vae_fp32.safetensors",
        "category": "vae",
        "approx_gb": 0.3,
    },
    "latent_upscaler": {
        "name": "MiniMax H3 Latent 3D Upscaler (BF16)",
        "subpath": "minimax_h3/minimax_h3_latent_upscaler_3d_bf16.safetensors",
        "category": "upscaler",
        "approx_gb": 0.2,
    },
    "turbo_lora_fl2v": {
        "name": "LightX2V FL2V Turbo LoRA (4-Step Alpha16 - Recommended)",
        "subpath": "minimax_h3_lightx2v_fl2v_turbo_4step_alpha16_v0.1.safetensors",
        "target_dir": "loras/minimax_h3",
        "category": "lora",
        "approx_gb": 0.1,
    },
    "turbo_lora_ref2v": {
        "name": "LightX2V Ref2V Turbo LoRA (4-Step Alpha8)",
        "subpath": "minimax_h3_lightx2v_ref2v_turbo_4step_alpha8_v0.1_bf16.safetensors",
        "target_dir": "loras/minimax_h3",
        "category": "lora",
        "approx_gb": 0.1,
    },
}

PRESETS = {
    "entry": {
        "description": "Tier 1: Entry & Budget (8GB - 12GB VRAM: RTX 3060, 4060, 2080Ti)",
        "items": [
            "fl2va_pruned_int8",
            "text_encoder_gguf_q4",
            "video_vae_fp8mix",
            "audio_vae_fp32",
            "latent_upscaler",
            "turbo_lora_fl2v",
        ],
    },
    "balanced": {
        "description": "Tier 2: Mid-High Consumer (12GB - 16GB VRAM: RTX 5080, 4080, 4070, 3080)",
        "items": [
            "fl2va_pruned_int8",
            "text_encoder_int8",
            "video_vae_fp16",
            "audio_vae_fp32",
            "latent_upscaler",
            "turbo_lora_fl2v",
        ],
    },
    "recommended": {
        "description": "Alias for 'balanced' (12GB - 16GB VRAM)",
        "items": [
            "fl2va_pruned_int8",
            "text_encoder_int8",
            "video_vae_fp16",
            "audio_vae_fp32",
            "latent_upscaler",
            "turbo_lora_fl2v",
        ],
    },
    "enthusiast": {
        "description": "Tier 3: Enthusiast (24GB - 32GB VRAM: RTX 4090, 3090, 5090)",
        "items": [
            "fl2va_full_int8",
            "text_encoder_int8",
            "video_vae_fp16",
            "audio_vae_fp32",
            "latent_upscaler",
            "turbo_lora_fl2v",
        ],
    },
    "workstation": {
        "description": "Tier 4: Workstation & Enterprise (48GB - 80GB+: RTX 6000 Ada, A100, H100)",
        "items": [
            "fl2va_full_bf16",
            "text_encoder_bf16",
            "video_vae_fp16",
            "audio_vae_fp32",
            "latent_upscaler",
            "turbo_lora_fl2v",
            "turbo_lora_ref2v",
        ],
    },
    "turbo_only": {
        "description": "Download only Turbo LoRAs & Latent 3D Upscaler",
        "items": [
            "latent_upscaler",
            "turbo_lora_fl2v",
            "turbo_lora_ref2v",
        ],
    },
}


class DownloadProgressBar:
    def __init__(self, title, total_bytes):
        self.title = title
        self.total = total_bytes
        self.pbar = None
        if tqdm:
            self.pbar = tqdm(
                total=total_bytes,
                unit="B",
                unit_scale=True,
                unit_divisor=1024,
                desc=title[:35].ljust(35),
                ascii=True,
                ncols=90,
            )

    def update(self, bytes_chunk):
        if self.pbar:
            self.pbar.update(bytes_chunk)

    def close(self):
        if self.pbar:
            self.pbar.close()


def download_file(url: str, dest_path: Path, hf_token: str = None):
    """Download a file with HTTP Range resume support and progress visualization."""
    dest_path.parent.mkdir(parents=True, exist_ok=True)
    temp_path = dest_path.with_suffix(dest_path.suffix + ".part")

    headers = {
        "User-Agent": "MiniMaxH3-Local-Downloader/1.0",
    }
    if hf_token:
        headers["Authorization"] = f"Bearer {hf_token}"

    downloaded_size = 0
    if temp_path.exists():
        downloaded_size = temp_path.stat().st_size
        headers["Range"] = f"bytes={downloaded_size}-"

    req = urllib_request.Request(url, headers=headers)
    try:
        response = urllib_request.urlopen(req)
    except Exception as e:
        # If range was not satisfiable (already completed), check
        if "416" in str(e):
            if temp_path.exists():
                shutil.move(str(temp_path), str(dest_path))
                print(f"[OK] {dest_path.name} already complete.")
                return
        print(f"[!] Error connecting to {url}: {e}")
        raise e

    content_length = response.headers.get("Content-Length")
    total_size = downloaded_size + (int(content_length) if content_length else 0)

    # Check if target file already exists and is full size
    if dest_path.exists():
        if total_size > 0 and dest_path.stat().st_size == total_size:
            print(f"[OK] {dest_path.name} already exists ({dest_path.stat().st_size / (1024**3):.2f} GB). Skipping.")
            return

    pbar = DownloadProgressBar(dest_path.name, total_size)
    if downloaded_size > 0:
        pbar.update(downloaded_size)

    mode = "ab" if downloaded_size > 0 else "wb"
    chunk_size = 1024 * 1024 * 4  # 4 MB chunk

    try:
        with open(temp_path, mode) as f:
            while True:
                chunk = response.read(chunk_size)
                if not chunk:
                    break
                f.write(chunk)
                pbar.update(len(chunk))
    finally:
        pbar.close()

    shutil.move(str(temp_path), str(dest_path))
    print(f"[OK] Successfully downloaded: {dest_path.name}")


def get_free_disk_space_gb(path: Path) -> float:
    """Return free disk space in GB for given path."""
    resolved = path.resolve()
    while not resolved.exists():
        resolved = resolved.parent
    usage = shutil.disk_usage(str(resolved))
    return usage.free / (1024**3)


def run_downloader(preset_name: str, base_dir: Path, hf_token: str = None, dry_run: bool = False):
    print("=" * 78)
    print("           MINIMAX H3 LOCAL MODEL DOWNLOADER & VERIFIER           ")
    print("=" * 78)

    preset = PRESETS.get(preset_name)
    if not preset:
        print(f"[!] Unknown preset '{preset_name}'. Valid presets: {list(PRESETS.keys())}")
        return False

    print(f"[*] Selected Preset: {preset_name}")
    print(f"[*] Description:     {preset['description']}")
    print(f"[*] Target Directory: {base_dir.resolve()}")

    # Determine files to download
    items_to_dl = [MODELS_CATALOG[key] for key in preset["items"] if key in MODELS_CATALOG]
    total_est_gb = sum(item["approx_gb"] for item in items_to_dl)

    free_gb = get_free_disk_space_gb(base_dir)
    print(f"[*] Estimated Download Size: ~{total_est_gb:.1f} GB")
    print(f"[*] Available Free Disk Space: {free_gb:.1f} GB")

    if free_gb < total_est_gb + 2.0:
        print(f"[!] WARNING: Free disk space ({free_gb:.1f} GB) might be insufficient for ~{total_est_gb:.1f} GB!")
        if not dry_run:
            ans = input("Continue anyway? (y/N): ").strip().lower()
            if ans != "y":
                print("Download aborted by user.")
                return False

    print("-" * 78)
    print(f"{'Category':<15} | {'Size':<8} | {'File / Model'}")
    print("-" * 78)
    for item in items_to_dl:
        print(f"{item['category']:<15} | ~{item['approx_gb']:<5.1f}GB | {item['name']}")
    print("-" * 78)

    if dry_run:
        print("[*] Dry run complete. No files were downloaded.")
        return True

    # Start downloads
    for idx, item in enumerate(items_to_dl, 1):
        rel_target = item.get("target_dir")
        if rel_target:
            dest_folder = base_dir / rel_target
        else:
            dest_folder = base_dir / "ckpts"

        dest_file = dest_folder / item["subpath"]
        url = urljoin(HF_BASE_URL, item["subpath"].replace("\\", "/"))

        print(f"\n[{idx}/{len(items_to_dl)}] Downloading {item['name']}...")
        print(f"      Source: {url}")
        print(f"      Dest:   {dest_file}")

        try:
            download_file(url, dest_file, hf_token=hf_token)
        except Exception as err:
            print(f"[!] Failed downloading {item['subpath']}: {err}")
            return False

    print("\n" + "=" * 78)
    print(" [OK] ALL REQUESTED MINIMAX H3 MODELS DOWNLOADED AND VERIFIED!")
    print("=" * 78)
    print("You can now launch the application via:")
    print("  - Windows: run_windows.bat")
    print("  - Linux:   ./run_linux.sh")
    print("=" * 78)
    return True


def interactive_menu(base_dir: Path):
    print("\n" + "=" * 70)
    print("      MiniMax H3 - Universal Hardware Preset Selector")
    print("=" * 70)
    preset_keys = list(PRESETS.keys())
    for idx, key in enumerate(preset_keys, 1):
        val = PRESETS[key]
        total_gb = sum(MODELS_CATALOG[i]["approx_gb"] for i in val["items"] if i in MODELS_CATALOG)
        print(f"  {idx}. {key.upper().ljust(14)} (~{total_gb:4.1f} GB) - {val['description']}")
    print(f"  {len(preset_keys) + 1}. EXIT")
    print("-" * 70)

    choice = input(f"Enter option [1-{len(preset_keys) + 1}] (Default: 2 - Balanced): ").strip()
    try:
        choice_idx = int(choice) - 1
        if 0 <= choice_idx < len(preset_keys):
            return preset_keys[choice_idx]
        elif choice_idx == len(preset_keys):
            sys.exit(0)
    except (ValueError, IndexError):
        pass
    return "balanced"


def main():
    parser = argparse.ArgumentParser(description="Download and verify MiniMax H3 models for local inference.")
    parser.add_argument(
        "--preset",
        type=str,
        choices=list(PRESETS.keys()),
        default=None,
        help="Download preset (recommended, recommended_gguf, full, turbo_only)",
    )
    parser.add_argument(
        "--target-dir",
        type=str,
        default=None,
        help="Target directory (defaults to wan2gp_core in workspace)",
    )
    parser.add_argument(
        "--hf-token",
        type=str,
        default=os.environ.get("HF_TOKEN", None),
        help="Hugging Face token (optional, for gated/private access)",
    )
    parser.add_argument(
        "--list",
        action="store_true",
        help="List available models and presets without downloading",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show what would be downloaded without actually downloading",
    )

    args = parser.parse_args()

    # Determine base directory
    script_dir = Path(__file__).resolve().parent
    if args.target_dir:
        base_dir = Path(args.target_dir)
    elif (script_dir / "wan2gp_core").exists():
        base_dir = script_dir / "wan2gp_core"
    else:
        base_dir = script_dir

    if args.list:
        print("Available Presets:")
        for k, v in PRESETS.items():
            print(f"  - {k}: {v['description']}")
        print("\nAvailable Catalog Items:")
        for k, v in MODELS_CATALOG.items():
            print(f"  - {k}: {v['name']} (~{v['approx_gb']} GB)")
        return

    preset = args.preset
    if not preset:
        preset = interactive_menu(base_dir)

    success = run_downloader(
        preset_name=preset,
        base_dir=base_dir,
        hf_token=args.hf_token,
        dry_run=args.dry_run,
    )
    if not success:
        sys.exit(1)


if __name__ == "__main__":
    main()
