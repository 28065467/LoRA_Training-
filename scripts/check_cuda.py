"""Standalone CUDA-availability check, run via subprocess so the calling notebook's
ipykernel process never imports torch itself (importing torch reserves a CUDA context /
some VRAM in whatever process does it, for the life of that process — running this as a
short-lived subprocess means that memory is released the moment it exits).

Usage: python check_cuda.py
Exit code 0 + prints device info if CUDA is available, exit code 1 otherwise.
"""
import sys

import torch

if not torch.cuda.is_available():
    print("CUDA is not available.")
    sys.exit(1)

print(f"CUDA available: {torch.cuda.get_device_name(0)}")
sys.exit(0)
