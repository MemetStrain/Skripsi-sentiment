# Save this as check_cuda.py in Idxchannel folder
import torch
import sys

print("=" * 70)
print("CUDA DIAGNOSTIC REPORT")
print("=" * 70)

print(f"\nPython version: {sys.version}")
print(f"PyTorch version: {torch.__version__}")

# Check CUDA availability
print(f"\nCUDA available: {torch.cuda.is_available()}")

if torch.cuda.is_available():
    print(f"CUDA version: {torch.version.cuda}")
    print(f"cuDNN version: {torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else 'N/A'}")
    print(f"Number of GPUs: {torch.cuda.device_count()}")
    
    for i in range(torch.cuda.device_count()):
        print(f"\nGPU {i}:")
        print(f"  Name: {torch.cuda.get_device_name(i)}")
        props = torch.cuda.get_device_properties(i)
        print(f"  Total Memory: {props.total_memory / (1024**3):.2f} GB")
        print(f"  Compute Capability: {props.major}.{props.minor}")
    
    # Test CUDA tensor creation
    try:
        x = torch.randn(3, 3).cuda()
        print(f"\n✓ CUDA tensor test: SUCCESS")
        print(f"  Device: {x.device}")
        del x
        torch.cuda.empty_cache()
    except Exception as e:
        print(f"\n✗ CUDA tensor test: FAILED")
        print(f"  Error: {e}")
else:
    print("\nCUDA is NOT available.")
    print("\nPossible reasons:")
    print("  1. PyTorch was installed as CPU-only version")
    print("  2. No NVIDIA GPU detected")
    print("  3. CUDA drivers not installed")
    print("\nTo install PyTorch with CUDA support:")
    print("  Visit: https://pytorch.org/get-started/locally/")
    print("  Or use: pip install torch --index-url https://download.pytorch.org/whl/cu118")

print("\n" + "=" * 70)

# Test device selection
print("\nTesting device selection logic:")
DEVICE = 'cuda' if torch.cuda.is_available() else 'cpu'
print(f"Selected device: {DEVICE}")

if DEVICE != 'cpu':
    device_id = 0
    print(f"Using device: cuda:{device_id}")
    torch.cuda.set_device(device_id)
    print("✓ Device set successfully")
else:
    print("Using CPU (CUDA not available)")