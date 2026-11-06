"""Real offline forward/backward check; CUDA is required unless CPU is explicit."""
import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import torch
from post_processing_model import PostProcessingModel
from review_dataset import ReviewDataset
from tool import TrainingTool


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--device", choices=["cpu", "cuda"], default="cuda")
    args = parser.parse_args()
    if args.device == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("CUDA unavailable: check GPU assignment, driver and CUDA wheel")
    device = torch.device(args.device)
    print(f"torch={torch.__version__}, runtime CUDA={torch.version.cuda}, device={device}")
    if device.type == "cuda":
        print(f"GPU={torch.cuda.get_device_name(0)}")
    tokenizer = PostProcessingModel.load_tokenizer(ROOT / "model/bert-base-chinese")
    for split in ("train", "validation", "test"):
        dataset = ReviewDataset(split)
        assert len(dataset) > 0, f"Empty split: {split}"
        print(f"{split}: {len(dataset)} samples")
    loader = ReviewDataset.create_data_loader("train", tokenizer, 2, 512)
    inputs, labels = TrainingTool.move_batch_to_device(next(iter(loader)), device)
    model = PostProcessingModel.create_model(ROOT / "model/bert-base-chinese", device)
    model.train()
    optimizer = TrainingTool.create_optimizer(model, 1e-3)
    before = model.classifier.weight.detach().clone()
    optimizer.zero_grad(set_to_none=True)
    logits = model(**inputs)
    assert logits.shape == (2, 2)
    loss = torch.nn.functional.cross_entropy(logits, labels)
    assert torch.isfinite(loss)
    loss.backward()
    assert model.classifier.weight.grad is not None
    assert all(p.grad is None for p in model.pretrained_model.parameters())
    optimizer.step()
    assert not torch.equal(before, model.classifier.weight.detach())
    if device.type == "cuda":
        torch.cuda.synchronize()
        print(f"peak allocated VRAM={torch.cuda.max_memory_allocated() / 2**20:.1f} MiB")
    print(f"PASS: offline model/data loading + forward/backward/update, loss={loss.item():.6f}")


if __name__ == "__main__":
    main()
