import json
from pathlib import Path


ROOT = Path(
    r"D:\Bachelor of Technology\IIIT Dharwad\Academics\Rachith Bharadwaj T N 24BDS062\2nd Year\4th SEM\Deep Learning\DL-Project\CvT vs ConvNeXT"
)
OUT = ROOT / "all_model_training_testing_optimization.ipynb"


def md_cell(text):
    return {
        "cell_type": "markdown",
        "metadata": {},
        "source": text.splitlines(keepends=True),
    }


def code_cell(text):
    return {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": text.splitlines(keepends=True),
    }


cells = [
    md_cell(
        """# Combined Training, Testing, and Optimization Notebook

This notebook restores one detailed code file for the full Deep Learning project. It keeps **both datasets in one notebook** and covers the full workflow for:

- CvT-13 on Food-101
- ConvNeXtV2-Tiny on Food-101
- CvT-13 on iNaturalist / iNatLoc500
- ConvNeXtV2-Tiny on iNaturalist / iNatLoc500

The notebook is organized so that the common training utilities are defined once and each experiment is run from a separate configuration cell.
"""
    ),
    md_cell(
        """## 1. Path Setup

Update the dataset paths before running. This notebook is intended to sit beside the final report in the parent `DL-Project` folder, so it resolves the cleaned repository as `DL-Project/CvT vs ConvNeXT`.
"""
    ),
    code_cell(
        """from pathlib import Path

PROJECT_ROOT = Path.cwd()
REPO_ROOT = PROJECT_ROOT / "CvT vs ConvNeXT"
ARTIFACT_ROOT = REPO_ROOT / "log files"
FIGURE_ROOT = REPO_ROOT / "figures"
OUTPUT_ROOT = PROJECT_ROOT / "final_report_runs"
OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

# Update these before training.
FOOD101_DATA_ROOT = Path(r"D:\\path\\to\\food101")
INATLOC500_DATA_ROOT = Path(r"D:\\path\\to\\inatloc500_flat")

print("Project root:", PROJECT_ROOT)
print("Repository root:", REPO_ROOT)
print("Food-101 path exists:", FOOD101_DATA_ROOT.exists())
print("iNaturalist path exists:", INATLOC500_DATA_ROOT.exists())
"""
    ),
    md_cell(
        """## 2. Dependencies

The main runs use `torch`, `torchvision`, `timm`, and `transformers`. Optional accelerators are wrapped safely so the notebook still works when FlashAttention 2 or ScheduleFree is not installed.
"""
    ),
    code_cell(
        """import inspect
import json
import math
import random
import time
from dataclasses import asdict, dataclass

import timm
import torch
import torch.nn.functional as F
from torch import nn
from torch.optim import AdamW
from torch.optim.lr_scheduler import LambdaLR
from torch.utils.data import DataLoader, Subset
from torchvision import datasets, transforms
from transformers import CvtForImageClassification

try:
    from flash_attn import flash_attn_func
except ImportError:
    flash_attn_func = None

try:
    import schedulefree
except ImportError:
    schedulefree = None

IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", device)
"""
    ),
    md_cell(
        """## 3. Common Utilities

This cell defines the reusable training, validation, testing, checkpointing, and metric-saving functions used by all four experiment blocks.
"""
    ),
    code_cell(
        """@dataclass
class ExperimentConfig:
    tag: str
    dataset_name: str
    model_family: str
    model_name: str
    num_classes: int
    batch_size: int
    accum_steps: int
    epochs: int
    lr: float
    weight_decay: float
    image_size: int
    num_workers: int = 4
    use_amp: bool = True
    use_channels_last: bool = True
    use_tf32: bool = True
    use_fused_adamw: bool = True
    use_torch_compile: bool = False
    compile_backend: str = "inductor"
    compile_mode: str = "default"
    use_schedulefree: bool = False
    warmup_epochs: int = 1
    label_smoothing: float = 0.1
    val_fraction: float = 0.1
    seed: int = 42


def set_seed(seed: int = 42):
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def save_json(path: Path, payload: dict):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def accuracy_topk(logits, targets, topk=(1, 5)):
    with torch.no_grad():
        maxk = min(max(topk), logits.shape[1])
        _, pred = logits.topk(maxk, dim=1, largest=True, sorted=True)
        pred = pred.t()
        correct = pred.eq(targets.view(1, -1).expand_as(pred))
        scores = []
        for k in topk:
            k = min(k, logits.shape[1])
            score = correct[:k].reshape(-1).float().sum(0) / targets.size(0)
            scores.append(score.item())
        return scores


def get_logits(outputs):
    return outputs.logits if hasattr(outputs, "logits") else outputs


def split_indices(total_size, val_fraction=0.1, seed=42):
    indices = list(range(total_size))
    rng = random.Random(seed)
    rng.shuffle(indices)
    val_size = max(1, int(total_size * val_fraction))
    val_indices = indices[:val_size]
    train_indices = indices[val_size:]
    return train_indices, val_indices


def build_transforms(image_size):
    train_tfms = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.RandomResizedCrop(image_size),
        transforms.RandomHorizontalFlip(),
        transforms.RandAugment(),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        transforms.RandomErasing(p=0.25),
    ])
    eval_tfms = transforms.Compose([
        transforms.Resize((image_size + 32, image_size + 32)),
        transforms.CenterCrop(image_size),
        transforms.ToTensor(),
        transforms.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])
    return train_tfms, eval_tfms


def build_food101_splits(data_root: Path, image_size: int, val_fraction=0.1, seed=42):
    train_tfms, eval_tfms = build_transforms(image_size)
    train_dir = data_root / "train"
    test_dir = data_root / "test"
    if train_dir.exists() and test_dir.exists():
        raw_train = datasets.ImageFolder(train_dir)
        train_idx, val_idx = split_indices(len(raw_train), val_fraction, seed)
        train_ds = Subset(datasets.ImageFolder(train_dir, transform=train_tfms), train_idx)
        val_ds = Subset(datasets.ImageFolder(train_dir, transform=eval_tfms), val_idx)
        test_ds = datasets.ImageFolder(test_dir, transform=eval_tfms)
        num_classes = len(raw_train.classes)
        return train_ds, val_ds, test_ds, num_classes

    raw_train = datasets.Food101(root=str(data_root), split="train", download=False)
    train_idx, val_idx = split_indices(len(raw_train), val_fraction, seed)
    train_ds = Subset(datasets.Food101(root=str(data_root), split="train", download=False, transform=train_tfms), train_idx)
    val_ds = Subset(datasets.Food101(root=str(data_root), split="train", download=False, transform=eval_tfms), val_idx)
    test_ds = datasets.Food101(root=str(data_root), split="test", download=False, transform=eval_tfms)
    return train_ds, val_ds, test_ds, 101


def build_inatloc500_splits(data_root: Path, image_size: int):
    train_tfms, eval_tfms = build_transforms(image_size)
    train_ds = datasets.ImageFolder(data_root / "train", transform=train_tfms)
    val_ds = datasets.ImageFolder(data_root / "val", transform=eval_tfms)
    test_ds = datasets.ImageFolder(data_root / "test", transform=eval_tfms)
    return train_ds, val_ds, test_ds, len(train_ds.classes)


def create_loaders(train_ds, val_ds, test_ds, batch_size, num_workers):
    train_loader = DataLoader(train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers, pin_memory=True)
    return train_loader, val_loader, test_loader


def maybe_set_tf32(enabled: bool):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = enabled
        torch.backends.cudnn.allow_tf32 = enabled


def build_model(cfg: ExperimentConfig):
    if cfg.model_family == "cvt":
        model = CvtForImageClassification.from_pretrained(
            cfg.model_name,
            num_labels=cfg.num_classes,
            ignore_mismatched_sizes=True,
        )
    elif cfg.model_family == "convnext":
        model = timm.create_model(cfg.model_name, pretrained=True, num_classes=cfg.num_classes)
    else:
        raise ValueError(f"Unsupported model_family: {cfg.model_family}")
    return model


def maybe_compile(model, cfg: ExperimentConfig):
    if cfg.use_torch_compile and hasattr(torch, "compile"):
        model = torch.compile(model, backend=cfg.compile_backend, mode=cfg.compile_mode)
    return model


def build_optimizer(params, cfg: ExperimentConfig):
    if cfg.use_schedulefree:
        if schedulefree is None:
            raise ImportError("schedulefree is not installed")
        return schedulefree.AdamWScheduleFree(
            params,
            lr=cfg.lr,
            weight_decay=cfg.weight_decay,
        )

    kwargs = {"lr": cfg.lr, "weight_decay": cfg.weight_decay}
    if cfg.use_fused_adamw and "fused" in inspect.signature(AdamW).parameters and torch.cuda.is_available():
        kwargs["fused"] = True
    return AdamW(params, **kwargs)


def build_scheduler(optimizer, steps_per_epoch, cfg: ExperimentConfig):
    if cfg.use_schedulefree:
        return None
    total_steps = max(1, steps_per_epoch * cfg.epochs)
    warmup_steps = max(1, steps_per_epoch * cfg.warmup_epochs)

    def lr_lambda(step):
        if step < warmup_steps:
            return float(step + 1) / float(warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    return LambdaLR(optimizer, lr_lambda)


def run_epoch(model, loader, optimizer, criterion, scaler, scheduler, cfg: ExperimentConfig, train=True):
    model.train(train)
    if train and hasattr(optimizer, "train"):
        optimizer.train()
    if (not train) and hasattr(optimizer, "eval"):
        optimizer.eval()

    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_seen = 0

    if train:
        optimizer.zero_grad(set_to_none=True)

    for step, (images, targets) in enumerate(loader):
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if cfg.use_channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        autocast_enabled = cfg.use_amp and device.type == "cuda"
        context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)

        with torch.set_grad_enabled(train):
            with context:
                logits = get_logits(model(images))
                loss = criterion(logits, targets)
                loss_for_step = loss / cfg.accum_steps if train else loss

            if train:
                scaler.scale(loss_for_step).backward()
                should_step = (step + 1) % cfg.accum_steps == 0 or (step + 1) == len(loader)
                if should_step:
                    scaler.step(optimizer)
                    scaler.update()
                    optimizer.zero_grad(set_to_none=True)
                    if scheduler is not None:
                        scheduler.step()

        top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_seen += batch_size

    return {
        "loss": total_loss / max(1, total_seen),
        "top1": total_top1 / max(1, total_seen),
        "top5": total_top5 / max(1, total_seen),
    }


@torch.no_grad()
def evaluate(model, loader, criterion, cfg: ExperimentConfig):
    model.eval()
    total_loss = 0.0
    total_top1 = 0.0
    total_top5 = 0.0
    total_seen = 0

    for images, targets in loader:
        images = images.to(device, non_blocking=True)
        targets = targets.to(device, non_blocking=True)
        if cfg.use_channels_last:
            images = images.contiguous(memory_format=torch.channels_last)

        autocast_enabled = cfg.use_amp and device.type == "cuda"
        context = torch.autocast(device_type="cuda", dtype=torch.float16, enabled=autocast_enabled) if device.type == "cuda" else torch.autocast(device_type="cpu", enabled=False)
        with context:
            logits = get_logits(model(images))
            loss = criterion(logits, targets)

        top1, top5 = accuracy_topk(logits, targets, topk=(1, 5))
        batch_size = targets.size(0)
        total_loss += loss.item() * batch_size
        total_top1 += top1 * batch_size
        total_top5 += top5 * batch_size
        total_seen += batch_size

    return {
        "loss": total_loss / max(1, total_seen),
        "top1": total_top1 / max(1, total_seen),
        "top5": total_top5 / max(1, total_seen),
    }


def load_datasets(cfg: ExperimentConfig, dataset_root: Path):
    if cfg.dataset_name == "food101":
        return build_food101_splits(dataset_root, cfg.image_size, cfg.val_fraction, cfg.seed)
    if cfg.dataset_name == "inatloc500":
        return build_inatloc500_splits(dataset_root, cfg.image_size)
    raise ValueError(f"Unsupported dataset_name: {cfg.dataset_name}")


def run_experiment(cfg: ExperimentConfig, dataset_root: Path):
    set_seed(cfg.seed)
    maybe_set_tf32(cfg.use_tf32)

    train_ds, val_ds, test_ds, num_classes = load_datasets(cfg, dataset_root)
    cfg.num_classes = num_classes
    train_loader, val_loader, test_loader = create_loaders(
        train_ds, val_ds, test_ds, cfg.batch_size, cfg.num_workers
    )

    model = build_model(cfg).to(device)
    if cfg.use_channels_last:
        model = model.to(memory_format=torch.channels_last)
    model = maybe_compile(model, cfg)

    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    optimizer = build_optimizer(model.parameters(), cfg)
    scheduler = build_scheduler(optimizer, len(train_loader), cfg)
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == "cuda"))

    output_dir = OUTPUT_ROOT / cfg.tag
    output_dir.mkdir(parents=True, exist_ok=True)

    history = []
    best_state = None
    best_val_top1 = -1.0
    start_time = time.time()

    for epoch in range(1, cfg.epochs + 1):
        train_metrics = run_epoch(model, train_loader, optimizer, criterion, scaler, scheduler, cfg, train=True)
        val_metrics = evaluate(model, val_loader, criterion, cfg)

        row = {
            "epoch": epoch,
            "train_loss": train_metrics["loss"],
            "train_top1": train_metrics["top1"],
            "train_top5": train_metrics["top5"],
            "val_loss": val_metrics["loss"],
            "val_top1": val_metrics["top1"],
            "val_top5": val_metrics["top5"],
            "lr": optimizer.param_groups[0]["lr"],
        }
        history.append(row)
        print(f"{cfg.tag} | epoch {epoch:02d}/{cfg.epochs} | train_top1={row['train_top1']:.4f} | val_top1={row['val_top1']:.4f}")

        if row["val_top1"] > best_val_top1:
            best_val_top1 = row["val_top1"]
            best_state = {k: v.detach().cpu() for k, v in model.state_dict().items()}
            torch.save(best_state, output_dir / "best_checkpoint.pth")

    total_minutes = (time.time() - start_time) / 60.0
    if best_state is not None:
        model.load_state_dict(best_state)

    test_metrics = evaluate(model, test_loader, criterion, cfg)
    torch.save(model.state_dict(), output_dir / "last_checkpoint.pth")

    summary = {
        "config": asdict(cfg),
        "best_val_top1": best_val_top1,
        "test_top1": test_metrics["top1"],
        "test_top5": test_metrics["top5"],
        "test_loss": test_metrics["loss"],
        "training_minutes": total_minutes,
        "output_dir": str(output_dir),
        "history": history,
    }
    save_json(output_dir / "summary.json", summary)
    save_json(output_dir / "history.json", {"history": history})
    return summary
"""
    ),
    md_cell(
        """## 4. Optimization Snippets Used in the Project

These are the same implementation themes used in the report: Torch.compile, Flash Attention 2, SDPA-backed attention, ScheduleFree, and fused AdamW.
"""
    ),
    code_cell(
        """# Torch.compile
example_model = nn.Linear(128, 64)
if hasattr(torch, "compile"):
    compiled_example = torch.compile(example_model, backend="inductor", mode="default")

# Flash Attention 2 direct call
if flash_attn_func is not None and torch.cuda.is_available():
    q = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 128, 8, 64, device="cuda", dtype=torch.float16)
    flash_out = flash_attn_func(q, k, v, dropout_p=0.0, causal=False)

# Flash Attention through SDPA
if torch.cuda.is_available():
    from torch.nn.attention import SDPBackend, sdpa_kernel

    q = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)
    k = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)
    v = torch.randn(2, 8, 128, 64, device="cuda", dtype=torch.float16)
    with sdpa_kernel(backends=[SDPBackend.FLASH_ATTENTION]):
        sdpa_out = F.scaled_dot_product_attention(q, k, v, dropout_p=0.0)

# AdamWScheduleFree
if schedulefree is not None:
    schedulefree_optimizer = schedulefree.AdamWScheduleFree(
        example_model.parameters(),
        lr=2.5e-4,
        weight_decay=1e-4,
    )

# Fused AdamW
fused_kwargs = {"lr": 3e-4, "weight_decay": 1e-4}
if "fused" in inspect.signature(AdamW).parameters and torch.cuda.is_available():
    fused_kwargs["fused"] = True
fused_optimizer = AdamW(example_model.parameters(), **fused_kwargs)
"""
    ),
    md_cell(
        """## 5. Food-101: CvT-13 Training, Validation, and Testing

This block recreates the Food-101 CvT training path with the same family of optimizations mentioned in the saved logs: AMP, channels-last layout, TF32, AdamW, and optional compile.
"""
    ),
    code_cell(
        """food101_cvt_cfg = ExperimentConfig(
    tag="food101_cvt13",
    dataset_name="food101",
    model_family="cvt",
    model_name="microsoft/cvt-13",
    num_classes=101,
    batch_size=32,
    accum_steps=1,
    epochs=10,
    lr=3e-4,
    weight_decay=1e-4,
    image_size=224,
    use_amp=True,
    use_channels_last=True,
    use_tf32=True,
    use_fused_adamw=True,
    use_torch_compile=True,
    compile_backend="aot_eager",
    compile_mode="default",
    label_smoothing=0.1,
)

# Uncomment to run:
# food101_cvt_summary = run_experiment(food101_cvt_cfg, FOOD101_DATA_ROOT)
"""
    ),
    md_cell(
        """## 6. Food-101: ConvNeXtV2-Tiny Training, Validation, and Testing

This block reconstructs the Food-101 ConvNeXtV2-Tiny path with fused AdamW, compile support, cosine learning-rate decay, and full held-out testing code.
"""
    ),
    code_cell(
        """food101_convnext_cfg = ExperimentConfig(
    tag="food101_convnextv2_tiny",
    dataset_name="food101",
    model_family="convnext",
    model_name="convnextv2_tiny",
    num_classes=101,
    batch_size=16,
    accum_steps=2,
    epochs=10,
    lr=1e-4,
    weight_decay=1e-4,
    image_size=224,
    use_amp=True,
    use_channels_last=True,
    use_tf32=True,
    use_fused_adamw=True,
    use_torch_compile=True,
    compile_backend="inductor",
    compile_mode="reduce-overhead",
    label_smoothing=0.1,
)

# Uncomment to run:
# food101_convnext_summary = run_experiment(food101_convnext_cfg, FOOD101_DATA_ROOT)
"""
    ),
    md_cell(
        """## 7. iNaturalist: CvT-13 Training, Validation, Optimization, and Testing

This block follows the optimized iNaturalist CvT configuration from the retained logs: batch size 32, fused AdamW, AMP, channels-last, TF32, and SDPA-backed attention without compile.
"""
    ),
    code_cell(
        """inat_cvt_cfg = ExperimentConfig(
    tag="inatloc500_cvt13",
    dataset_name="inatloc500",
    model_family="cvt",
    model_name="microsoft/cvt-13",
    num_classes=500,
    batch_size=32,
    accum_steps=1,
    epochs=10,
    lr=3e-4,
    weight_decay=1e-4,
    image_size=224,
    use_amp=True,
    use_channels_last=True,
    use_tf32=True,
    use_fused_adamw=True,
    use_torch_compile=False,
    compile_backend="aot_eager",
    compile_mode="default",
    label_smoothing=0.1,
)

# Uncomment to run:
# inat_cvt_summary = run_experiment(inat_cvt_cfg, INATLOC500_DATA_ROOT)
"""
    ),
    md_cell(
        """## 8. iNaturalist: ConvNeXtV2-Tiny Training, Validation, Optimization, and Testing

This block mirrors the strongest optimized iNaturalist ConvNeXtV2-Tiny run from the saved log trail, including the `convnextv2_tiny.fcmae_ft_in22k_in1k` backbone and effective batch size 32 through accumulation.
"""
    ),
    code_cell(
        """inat_convnext_cfg = ExperimentConfig(
    tag="inatloc500_convnextv2_tiny",
    dataset_name="inatloc500",
    model_family="convnext",
    model_name="convnextv2_tiny.fcmae_ft_in22k_in1k",
    num_classes=500,
    batch_size=16,
    accum_steps=2,
    epochs=10,
    lr=3e-4,
    weight_decay=1e-4,
    image_size=224,
    use_amp=True,
    use_channels_last=True,
    use_tf32=True,
    use_fused_adamw=True,
    use_torch_compile=False,
    compile_backend="disabled",
    compile_mode="default",
    label_smoothing=0.1,
)

# Uncomment to run:
# inat_convnext_summary = run_experiment(inat_convnext_cfg, INATLOC500_DATA_ROOT)
"""
    ),
    md_cell(
        """## 9. Optional Evaluation-Only Helpers for Retained Checkpoints

If the cleaned repository already contains a checkpoint, this helper lets you load it and run testing without repeating training.
"""
    ),
    code_cell(
        """def evaluate_existing_checkpoint(cfg: ExperimentConfig, dataset_root: Path, checkpoint_path: Path):
    _, _, test_ds, num_classes = load_datasets(cfg, dataset_root)
    cfg.num_classes = num_classes
    _, _, test_loader = create_loaders(test_ds, test_ds, test_ds, cfg.batch_size, cfg.num_workers)
    model = build_model(cfg).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    model.load_state_dict(state, strict=False)
    criterion = nn.CrossEntropyLoss(label_smoothing=cfg.label_smoothing)
    return evaluate(model, test_loader, criterion, cfg)


# Example:
# food101_cvt_eval = evaluate_existing_checkpoint(
#     food101_cvt_cfg,
#     FOOD101_DATA_ROOT,
#     REPO_ROOT / "cvt" / "checkpoints" / "cvt13_compile_sdpa_math_bs64_best.pth",
# )
"""
    ),
    md_cell(
        """## 10. Retained Artifact Summary

This final cell helps cross-check the recreated notebook against the saved logs and report evidence.
"""
    ),
    code_cell(
        """artifact_files = [
    REPO_ROOT / "log files" / "food101_cvt13.log",
    REPO_ROOT / "log files" / "food101_convnextv2_tiny.log",
    REPO_ROOT / "log files" / "inaturalist_cvt13.log",
    REPO_ROOT / "log files" / "inaturalist_convnextv2_tiny.log",
    REPO_ROOT / "cvt" / "checkpoints" / "cvt13_compile_metrics.json",
    REPO_ROOT / "cvt" / "checkpoints" / "cvt13_compile_sdpa_math_bs64_test_metrics.json",
    REPO_ROOT / "convnext" / "convnextv2_tiny_food101_10ep_bs32" / "history.csv",
]

for path in artifact_files:
    print(path.name, "->", "FOUND" if path.exists() else "MISSING")
"""
    ),
]


notebook = {
    "cells": cells,
    "metadata": {
        "kernelspec": {
            "display_name": "Python 3",
            "language": "python",
            "name": "python3",
        },
        "language_info": {
            "name": "python",
            "version": "3.11",
        },
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}


OUT.write_text(json.dumps(notebook, indent=2), encoding="utf-8")
print(OUT)
