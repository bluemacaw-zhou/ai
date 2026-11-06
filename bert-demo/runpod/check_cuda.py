"""Check the GPU environment without project dependencies or downloaded assets."""
import subprocess
import sys


def main():
    print("=== NVIDIA driver ===", flush=True)
    subprocess.run(["nvidia-smi"], check=True)

    import torch

    print("Python:", sys.version.split()[0], flush=True)
    print("PyTorch:", torch.__version__, flush=True)
    print("CUDA runtime:", torch.version.cuda, flush=True)
    print("CUDA available:", torch.cuda.is_available(), flush=True)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable; GPU environment verification failed")
    print("GPU:", torch.cuda.get_device_name(0), flush=True)
    print("VRAM: %.1f GiB" % (
        torch.cuda.get_device_properties(0).total_memory / 1024**3
    ), flush=True)

    x = torch.randn(512, 512, device="cuda", requires_grad=True)
    loss = (x @ x.T).square().mean()
    loss.backward()
    torch.cuda.synchronize()
    if not torch.isfinite(loss).item():
        raise RuntimeError("Non-finite loss")
    if x.grad is None or not torch.isfinite(x.grad).all().item():
        raise RuntimeError("Missing or non-finite gradients")
    print("Loss:", loss.item(), flush=True)
    print("PASS: CUDA computation and backward propagation", flush=True)


if __name__ == "__main__":
    main()
