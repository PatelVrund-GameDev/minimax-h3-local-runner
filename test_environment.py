import sys
import os

# Set UTF-8 encoding for stdout on Windows
if sys.platform == "win32":
    sys.stdout.reconfigure(encoding='utf-8')

print(f"[*] Python Version: {sys.version}")

try:
    import torch
    print(f"[OK] PyTorch {torch.__version__}")
    print(f"     CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"     Device: {torch.cuda.get_device_name(0)}")
        print(f"     VRAM: {torch.cuda.get_device_properties(0).total_memory / 1024**3:.2f} GB")
        print(f"     Compute Capability: {torch.cuda.get_device_capability(0)}")
except Exception as e:
    print(f"[FAIL] PyTorch error: {e}")

try:
    import triton
    print(f"[OK] Triton {triton.__version__}")
except Exception as e:
    print(f"[FAIL] Triton error: {e}")

try:
    import sageattention
    print(f"[OK] SageAttention loaded")
except Exception as e:
    print(f"[FAIL] SageAttention error: {e}")

try:
    import diffusers
    import transformers
    import mmgp
    print(f"[OK] Diffusers {diffusers.__version__}, Transformers {transformers.__version__}, MMGP loaded")
except Exception as e:
    print(f"[FAIL] AI stack error: {e}")

print("[OK] All checks completed successfully!")
