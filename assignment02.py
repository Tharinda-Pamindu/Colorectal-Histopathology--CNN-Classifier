"""
CAME Assignment 02 - Colorectal Histopathology CNN Classifier (PyTorch).

Custom CNN trained FROM SCRATCH (random init, no pretrained weights) to classify
64x64 H&E colorectal tissue patches into 9 classes:
    ADI, BACK, DEB, LYM, MUC, MUS, NORM, STR, TUM

Colab workflow
--------------
1. Runtime -> Change runtime type -> T4 GPU.
2. Put this file in /content (upload it or use %%writefile assignment02.py).
3. !python assignment02.py
   (optional env vars: CAME_EPOCHS, CAME_BATCH_SIZE, CAME_PER_CLASS, CAME_HF_MODE)
4. main() trains, restores the best validation checkpoint, encodes it, verifies
   load_trained_model(cpu) and writes `assignment02_submission.py`: a copy of
   this file with TRAINED_MODEL_B64 already filled in. Download that file, rename
   it to assignment02.py and upload it to CMS.

Importing this module does NOT train and does NOT download data.
"""

import base64
import copy
import io
import os
import re
import time
import zlib
from pathlib import Path

import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
from torchvision import transforms
from torchvision.datasets import ImageFolder

# ---------------------------------------------------------------------------
# Base64 string of the Colab-trained best_model.pt (filled in after training).
# ---------------------------------------------------------------------------
TRAINED_MODEL_B64 = """
"""

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
NUM_CLASSES = 9
IMAGE_SIZE = 64
NORM_MEAN = (0.5, 0.5, 0.5)
NORM_STD = (0.5, 0.5, 0.5)
SEED = 42
VAL_FRACTION = 0.15

HF_DATASET_ID = "1aurent/NCT-CRC-HE"
HF_SPLIT = "NCT_CRC_HE_100K"

# Optional per-epoch hook: callable(epoch, record, improved). Set only by main()
# (MLflow tracking on Colab); it stays None on CMS, so grading is unaffected.
_EPOCH_CALLBACK = None


# ===========================================================================
# Task 1 - Data preprocessing & loaders
# ===========================================================================
def official_classes() -> list[str]:
    return ["ADI", "BACK", "DEB", "LYM", "MUC", "MUS", "NORM", "STR", "TUM"]


def make_transforms(train: bool):
    if train:
        return transforms.Compose([
            transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(NORM_MEAN, NORM_STD),
        ])
    return transforms.Compose([
        transforms.Resize((IMAGE_SIZE, IMAGE_SIZE)),
        transforms.ToTensor(),
        transforms.Normalize(NORM_MEAN, NORM_STD),
    ])


class NCTPatchDataset(Dataset):
    """In-memory dataset of pre-resized 64x64 RGB patches (Colab / Hugging Face path).

    images: uint8 tensor of shape (N, 64, 64, 3); labels: long tensor of shape (N,).
    Each item is (tensor_3x64x64, int_label_0_to_8).
    """

    def __init__(self, images: torch.Tensor, labels: torch.Tensor, transform, extra_aug: bool = False):
        self.images = images
        self.labels = labels
        self.transform = transform
        self.extra_aug = extra_aug
        # Mild stain/colour variation to help generalise to unseen patients.
        self.color_jitter = transforms.ColorJitter(
            brightness=0.10, contrast=0.10, saturation=0.10, hue=0.02
        )

    def __len__(self) -> int:
        return int(self.labels.numel())

    def __getitem__(self, idx):
        from PIL import Image

        img = Image.fromarray(self.images[idx].numpy())  # (64, 64, 3) uint8 -> RGB
        if self.extra_aug:
            # Tissue has no canonical orientation -> random 90-degree rotations.
            k = int(torch.randint(0, 4, (1,)).item())
            if k:
                img = img.rotate(90 * k)
            img = self.color_jitter(img)
        x = self.transform(img)
        return x, int(self.labels[idx])


def _label_mapper(features):
    """Map Hugging Face labels (ClassLabel int or str) to official indices 0..8."""
    classes = official_classes()
    remap = None
    try:
        names = list(getattr(features["label"], "names", None) or [])
        if names:
            remap = {i: classes.index(n.strip().upper()) for i, n in enumerate(names)
                     if n.strip().upper() in classes}
    except Exception:
        remap = None

    def to_idx(value) -> int:
        if isinstance(value, str):
            return classes.index(value.strip().upper())
        value = int(value)
        if remap is not None:
            return remap.get(value, -1)
        return value

    return to_idx


def _patch_to_uint8(image_field) -> torch.Tensor:
    """Decode one HF image field (PIL or {'bytes', 'path'}) to a (64, 64, 3) uint8 tensor."""
    import numpy as np
    from PIL import Image

    if isinstance(image_field, dict):
        if image_field.get("bytes") is not None:
            img = Image.open(io.BytesIO(image_field["bytes"]))
        else:
            img = Image.open(image_field["path"])
    else:
        img = image_field
    img = img.convert("RGB")
    # Same resize as make_transforms(False) so train / test pixels match.
    img = transforms.Resize((IMAGE_SIZE, IMAGE_SIZE))(img)
    return torch.from_numpy(np.array(img, dtype=np.uint8).copy())


def _list_hf_shards() -> list[str]:
    """Return the parquet shard paths of the NCT_CRC_HE_100K split (relative to the repo)."""
    try:
        from huggingface_hub import HfFileSystem

        fs = HfFileSystem()
        files = sorted(fs.glob(f"datasets/{HF_DATASET_ID}/data/{HF_SPLIT}-*.parquet"))
        shards = ["data/" + f.rsplit("/", 1)[-1] for f in files]
        if shards:
            return shards
    except Exception as exc:  # pragma: no cover - network dependent
        print(f"[data] could not list shards via HfFileSystem ({exc}); using default pattern")
    return [f"data/{HF_SPLIT}-{i:05d}-of-00031-*.parquet" for i in range(31)]


def _collect_streaming(per_class: int):
    """Stream the NCT_CRC_HE_100K split ONE parquet shard at a time until every class
    has `per_class` patches. Avoids downloading the full ~15 GB split and keeps RAM
    bounded: only a single shard stream is open at any moment and it is freed before
    the next one is opened.

    Round r reads rows [r*R, (r+1)*R) from each of the 31 shards in turn. The shards
    are grouped by class, so round 0 already samples every class from several shards
    (= several slides); later rounds run only if a class is still short. The read
    order is fixed, so the subset is deterministic.
    """
    import gc

    from datasets import Image as HFImage
    from datasets import load_dataset

    shards = _list_hf_shards()
    default_rows = max(64, (per_class * NUM_CLASSES * 13) // (10 * len(shards)))
    rows_per_round = int(os.environ.get("CAME_ROWS_PER_SHARD", default_rows))
    max_rounds = int(os.environ.get("CAME_MAX_ROUNDS", 12))
    print(f"[data] streaming {HF_DATASET_ID}:{HF_SPLIT}: {len(shards)} shards, "
          f"{rows_per_round} rows per shard per round (one shard open at a time)")

    buckets = {c: [] for c in range(NUM_CLASSES)}
    exhausted = set()
    to_idx = None
    scanned = 0
    t0 = time.time()

    def all_full() -> bool:
        return all(len(buckets[c]) >= per_class for c in range(NUM_CLASSES))

    for rnd in range(max_rounds):
        if all_full() or len(exhausted) == len(shards):
            break
        for si, shard in enumerate(shards):
            if all_full():
                break
            if si in exhausted:
                continue
            ds = load_dataset(HF_DATASET_ID, data_files={HF_SPLIT: shard},
                              split=HF_SPLIT, streaming=True)
            try:
                ds = ds.cast_column("image", HFImage(decode=False))  # decode only what we keep
            except Exception:
                pass
            if to_idx is None:
                to_idx = _label_mapper(ds.features)
            if rnd:
                ds = ds.skip(rnd * rows_per_round)
            n_read = 0
            for example in ds.take(rows_per_round):
                n_read += 1
                y = to_idx(example["label"])
                if 0 <= y < NUM_CLASSES and len(buckets[y]) < per_class:
                    buckets[y].append(_patch_to_uint8(example["image"]))
            scanned += n_read
            if n_read < rows_per_round:
                exhausted.add(si)
            del ds
            gc.collect()
            counts = [len(buckets[c]) for c in range(NUM_CLASSES)]
            print(f"[data] round {rnd + 1} shard {si + 1}/{len(shards)} | scanned={scanned} "
                  f"per-class={counts} ({time.time() - t0:.0f}s)")
    return buckets


def _collect_full(per_class: int):
    """Download the full NCT_CRC_HE_100K split and draw a seeded balanced subset."""
    from datasets import Image as HFImage
    from datasets import load_dataset

    ds = load_dataset(HF_DATASET_ID, split=HF_SPLIT)
    to_idx = _label_mapper(ds.features)
    labels = torch.tensor([to_idx(v) for v in ds["label"]], dtype=torch.long)
    g = torch.Generator().manual_seed(SEED)
    chosen = []
    for c in range(NUM_CLASSES):
        idx = (labels == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.numel(), generator=g)][:per_class]
        chosen.extend(idx.tolist())
    chosen.sort()
    sub = ds.select(chosen)
    try:
        sub = sub.cast_column("image", HFImage(decode=False))
    except Exception:
        pass
    buckets = {c: [] for c in range(NUM_CLASSES)}
    for example in sub:
        y = to_idx(example["label"])
        buckets[y].append(_patch_to_uint8(example["image"]))
    return buckets


def _load_hf_subset():
    """Return (images uint8 (N,64,64,3), labels long (N,)) for a balanced subset, cached on disk."""
    per_class = int(os.environ.get("CAME_PER_CLASS", 1000))
    mode = os.environ.get("CAME_HF_MODE", "stream").lower()
    cache = Path(os.environ.get("CAME_CACHE", f"nct_crc_he_100k_subset_{per_class}.pt"))

    if cache.exists():
        print(f"[data] loading cached subset from {cache}")
        blob = torch.load(cache, map_location="cpu", weights_only=True)
        return blob["images"], blob["labels"]

    if mode == "full":
        buckets = _collect_full(per_class)
    else:
        buckets = _collect_streaming(per_class)

    images, labels = [], []
    for c in range(NUM_CLASSES):
        if not buckets[c]:
            raise RuntimeError(f"No patches collected for class {official_classes()[c]}")
        images.extend(buckets[c])
        labels.extend([c] * len(buckets[c]))
    images = torch.stack(images).to(torch.uint8)
    labels = torch.tensor(labels, dtype=torch.long)
    try:
        torch.save({"images": images, "labels": labels}, cache)
        print(f"[data] cached subset to {cache}")
    except Exception as exc:
        print(f"[data] could not cache subset ({exc})")
    return images, labels


def _stratified_split(labels: torch.Tensor, val_fraction: float = VAL_FRACTION, seed: int = SEED):
    """Deterministic per-class 85 / 15 split. Returns (train_idx, val_idx) long tensors."""
    g = torch.Generator().manual_seed(seed)
    train_idx, val_idx = [], []
    for c in range(NUM_CLASSES):
        idx = (labels == c).nonzero(as_tuple=False).view(-1)
        idx = idx[torch.randperm(idx.numel(), generator=g)]
        n_val = int(round(idx.numel() * val_fraction))
        val_idx.append(idx[:n_val])
        train_idx.append(idx[n_val:])
    return torch.cat(train_idx), torch.cat(val_idx)


def build_datasets():
    data_dir = os.environ.get("CAME_DATA_DIR")
    if data_dir:
        # CMS path: local ImageFolder only, no `datasets` import, no internet.
        train_dataset = ImageFolder(os.path.join(data_dir, "train"), transform=make_transforms(True))
        val_dataset = ImageFolder(os.path.join(data_dir, "validation"), transform=make_transforms(False))
        return train_dataset, val_dataset

    # Colab path: Hugging Face 1aurent/NCT-CRC-HE, split NCT_CRC_HE_100K.
    images, labels = _load_hf_subset()
    train_idx, val_idx = _stratified_split(labels)
    extra_aug = os.environ.get("CAME_EXTRA_AUG", "1") != "0"
    train_dataset = NCTPatchDataset(images[train_idx], labels[train_idx],
                                    make_transforms(True), extra_aug=extra_aug)
    val_dataset = NCTPatchDataset(images[val_idx], labels[val_idx],
                                  make_transforms(False), extra_aug=False)
    print(f"[data] train={len(train_dataset)} val={len(val_dataset)} "
          f"(per-class train counts={torch.bincount(labels[train_idx], minlength=NUM_CLASSES).tolist()})")
    return train_dataset, val_dataset


def make_loaders(train_dataset, val_dataset, batch_size: int):
    num_workers = int(os.environ.get("CAME_WORKERS", 0))
    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True,
                              num_workers=num_workers)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers)
    return train_loader, val_loader


# ===========================================================================
# Task 2 - Custom CNN (from scratch)
# ===========================================================================
def _conv_bn_relu(in_ch: int, out_ch: int) -> list:
    return [
        nn.Conv2d(in_ch, out_ch, kernel_size=3, padding=1, bias=False),
        nn.BatchNorm2d(out_ch),
        nn.ReLU(inplace=True),
    ]


def build_model(num_classes: int = 9) -> torch.nn.Module:
    """Funnel CNN 3 -> 32 -> 64 -> 128 -> 256 channels, 64 -> 4 spatial.

    Uses only allowed layers (Sequential, Conv2d, BatchNorm2d, ReLU, MaxPool2d,
    AdaptiveAvgPool2d, Flatten, Dropout, Linear). Returns raw logits (N, num_classes).
    Trainable parameters (num_classes=9): 437,481.
    """
    features = nn.Sequential(
        # Block 1: 3 -> 32, 64x64 -> 32x32
        nn.Sequential(*_conv_bn_relu(3, 32), *_conv_bn_relu(32, 32), nn.MaxPool2d(2, 2)),
        # Block 2: 32 -> 64, 32x32 -> 16x16
        nn.Sequential(*_conv_bn_relu(32, 64), *_conv_bn_relu(64, 64), nn.MaxPool2d(2, 2)),
        # Block 3: 64 -> 128, 16x16 -> 8x8
        nn.Sequential(*_conv_bn_relu(64, 128), nn.MaxPool2d(2, 2)),
        # Block 4: 128 -> 256, 8x8 -> 4x4
        nn.Sequential(*_conv_bn_relu(128, 256), nn.MaxPool2d(2, 2)),
    )
    classifier = nn.Sequential(
        nn.AdaptiveAvgPool2d(1),
        nn.Flatten(),
        nn.Dropout(p=0.3),
        nn.Linear(256, num_classes),
    )
    return nn.Sequential(features, classifier)


def count_parameters(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# ===========================================================================
# Task 3 - Training loop, validation & best checkpoint restoration
# ===========================================================================
def train_model(train_loader, val_loader, epochs: int, device: torch.device):
    torch.manual_seed(42)
    model = build_model(9).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)

    best_state = copy.deepcopy(model.state_dict())
    best_val_acc = -1.0
    best_val_loss = float("inf")
    history = []

    for epoch in range(epochs):
        t0 = time.time()

        # ---- training phase ----
        model.train()
        loss_sum, correct, seen = 0.0, 0, 0
        for images, labels in train_loader:
            images = images.to(device)
            labels = labels.to(device).long()
            optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, labels)
            loss.backward()
            optimizer.step()
            bs = labels.size(0)
            loss_sum += loss.item() * bs
            correct += (logits.argmax(dim=1) == labels).sum().item()
            seen += bs
        train_loss = loss_sum / max(seen, 1)
        train_acc = correct / max(seen, 1)

        # ---- validation phase ----
        model.eval()
        loss_sum, correct, seen = 0.0, 0, 0
        with torch.no_grad():
            for images, labels in val_loader:
                images = images.to(device)
                labels = labels.to(device).long()
                logits = model(images)
                loss = criterion(logits, labels)
                bs = labels.size(0)
                loss_sum += loss.item() * bs
                correct += (logits.argmax(dim=1) == labels).sum().item()
                seen += bs
        val_loss = loss_sum / max(seen, 1)
        val_acc = correct / max(seen, 1)

        # ---- checkpoint tracking ----
        if abs(val_acc - best_val_acc) <= 1e-12:
            improved = val_loss < best_val_loss
        else:
            improved = val_acc > best_val_acc
        if improved:
            best_val_acc = val_acc
            best_val_loss = val_loss
            best_state = copy.deepcopy(model.state_dict())

        history.append({
            "train_loss": float(train_loss),
            "train_acc": float(train_acc),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        })
        print(f"Epoch {epoch + 1:02d}/{epochs} | train_loss={train_loss:.4f} "
              f"train_acc={train_acc:.4f} | val_loss={val_loss:.4f} val_acc={val_acc:.4f}"
              f"{' *best*' if improved else ''} | {time.time() - t0:.1f}s")
        if _EPOCH_CALLBACK is not None:
            try:
                _EPOCH_CALLBACK(epoch, history[-1], improved)
            except Exception as exc:  # tracking must never break training
                print(f"[tracking] epoch callback failed: {exc}")

    model.load_state_dict(best_state)
    return model, history


# ===========================================================================
# Task 4 - Inference & metrics
# ===========================================================================
def predict(model: torch.nn.Module, loader, device: torch.device) -> torch.Tensor:
    model.eval()
    preds = []
    with torch.no_grad():
        for batch in loader:
            images = batch[0] if isinstance(batch, (list, tuple)) else batch
            images = images.to(device)
            logits = model(images)
            preds.append(logits.argmax(dim=1).detach().cpu().long())
    if not preds:
        return torch.empty(0, dtype=torch.long)
    return torch.cat(preds).long()


def classification_report(y_true: torch.Tensor, y_pred: torch.Tensor) -> dict:
    n = NUM_CLASSES
    y_true = torch.as_tensor(y_true).detach().cpu().long().view(-1)
    y_pred = torch.as_tensor(y_pred).detach().cpu().long().view(-1)
    if y_true.numel() != y_pred.numel():
        raise ValueError(f"y_true has {y_true.numel()} items but y_pred has {y_pred.numel()}")

    confusion = torch.zeros((n, n), dtype=torch.long)
    if y_true.numel() > 0:
        valid = (y_true >= 0) & (y_true < n) & (y_pred >= 0) & (y_pred < n)
        flat = y_true[valid] * n + y_pred[valid]
        confusion = torch.bincount(flat, minlength=n * n).reshape(n, n).long()
        accuracy = (y_true == y_pred).double().mean().item()
    else:
        accuracy = 0.0

    support = confusion.sum(dim=1)
    present = support > 0
    if present.any():
        recalls = confusion.diag()[present].double() / support[present].double()
        macro_recall = recalls.mean().item()
    else:
        macro_recall = 0.0

    return {
        "accuracy": float(accuracy),
        "macro_recall": float(macro_recall),
        "confusion": confusion,
    }


# ===========================================================================
# Task 5 - Embedding the Colab-trained model
# ===========================================================================
def encode_model_to_b64(model: torch.nn.Module) -> str:
    """Run this on Colab after training and paste the output into TRAINED_MODEL_B64."""
    state = {k: v.detach().cpu().half() for k, v in model.state_dict().items()}
    buf = io.BytesIO()
    torch.save(state, buf)
    compressed = zlib.compress(buf.getvalue(), level=9)
    return base64.b64encode(compressed).decode("ascii")


def load_trained_model(device: torch.device) -> torch.nn.Module:
    """Rebuilds build_model(9) and loads TRAINED_MODEL_B64 on CMS."""
    if not TRAINED_MODEL_B64.strip():
        raise RuntimeError("TRAINED_MODEL_B64 is empty - train on Colab and paste the encoded weights.")
    model = build_model(9)
    raw = zlib.decompress(base64.b64decode(TRAINED_MODEL_B64.strip()))
    state = torch.load(io.BytesIO(raw), map_location="cpu", weights_only=True)
    state = {k: v.float() if v.is_floating_point() else v for k, v in state.items()}
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def _write_submission_file(b64: str, out_path: str = "assignment02_submission.py"):
    """Copy this source file with TRAINED_MODEL_B64 filled in (wrapped at 100 chars;
    base64.b64decode ignores the newlines). Returns the written path or None."""
    try:
        source = Path(__file__).read_text(encoding="utf-8")
    except Exception as exc:
        print(f"[export] could not read source file ({exc}); paste trained_model_b64.txt manually")
        return None
    wrapped = "\n".join(b64[i:i + 100] for i in range(0, len(b64), 100))
    new_source, n_subs = re.subn(
        r'^TRAINED_MODEL_B64 = """.*?"""',
        lambda _m: 'TRAINED_MODEL_B64 = """\n' + wrapped + '\n"""',
        source, count=1, flags=re.DOTALL | re.MULTILINE,
    )
    if n_subs != 1:
        print("[export] TRAINED_MODEL_B64 placeholder not found; paste trained_model_b64.txt manually")
        return None
    Path(out_path).write_text(new_source, encoding="utf-8")
    print(f"[export] wrote {out_path} ({len(new_source) / 1e6:.2f} MB)")
    return out_path


# ===========================================================================
# MLflow experiment tracking (Colab only - never imported on CMS)
# ===========================================================================
MLFLOW_EXPERIMENT = os.environ.get("CAME_MLFLOW_EXPERIMENT", "came-a02-nct-crc-cnn")
MLFLOW_RANK_METRIC = "best_val_macro_recall"


def _mlflow_start(params: dict):
    """Start an MLflow run and log params. Returns the mlflow module, or None if
    tracking is disabled (CAME_MLFLOW=0) or mlflow is not installed."""
    if os.environ.get("CAME_MLFLOW", "1") == "0":
        return None
    try:
        import mlflow
    except ImportError:
        print("[mlflow] not installed (pip install mlflow) - continuing without tracking")
        return None
    mlflow.set_tracking_uri(os.environ.get("MLFLOW_TRACKING_URI", "sqlite:///mlflow.db"))
    mlflow.set_experiment(MLFLOW_EXPERIMENT)
    run = mlflow.start_run(run_name=os.environ.get("CAME_RUN_NAME"))
    mlflow.log_params(params)
    mlflow.set_tags({"task": "NCT-CRC-HE 9-class", "framework": "pytorch", "pretrained": "no"})
    print(f"[mlflow] run {run.info.run_id} in experiment '{MLFLOW_EXPERIMENT}' "
          f"({mlflow.get_tracking_uri()})")
    return mlflow


def _mlflow_epoch_callback(mlflow):
    """Live per-epoch logging + running best-so-far metrics."""
    def callback(epoch: int, record: dict, improved: bool) -> None:
        mlflow.log_metrics(record, step=epoch + 1)
        if improved:
            mlflow.log_metrics({
                "best_val_acc": record["val_acc"],
                "best_val_loss": record["val_loss"],
                "best_epoch": epoch + 1,
            }, step=epoch + 1)
    return callback


def _mlflow_log_results(mlflow, report: dict, cpu_report: dict, history: list,
                        train_seconds: float, artifacts: list) -> None:
    classes = official_classes()
    conf = report["confusion"]
    support = conf.sum(dim=1).clamp(min=1).double()
    per_class = (conf.diag().double() / support).tolist()

    metrics = {
        "best_val_accuracy": report["accuracy"],
        MLFLOW_RANK_METRIC: report["macro_recall"],
        "cpu_fp16_val_accuracy": cpu_report["accuracy"],
        "cpu_fp16_val_macro_recall": cpu_report["macro_recall"],
        "train_seconds": train_seconds,
    }
    metrics.update({f"recall_{name}": r for name, r in zip(classes, per_class)})
    mlflow.log_metrics(metrics)
    mlflow.set_tag("cms_target_met", str(cpu_report["macro_recall"] >= 0.80))

    mlflow.log_dict(history, "history.json")
    mlflow.log_dict({
        "classes": classes,
        "confusion": conf.tolist(),
        "per_class_recall": dict(zip(classes, per_class)),
    }, "confusion_matrix.json")
    for path in artifacts:
        if path and Path(path).exists():
            mlflow.log_artifact(path)


def _mlflow_print_leaderboard(mlflow, top_k: int = 5) -> None:
    """Print the best runs so far, ranked by validation macro recall."""
    try:
        runs = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT],
            order_by=[f"metrics.{MLFLOW_RANK_METRIC} DESC"],
            max_results=top_k,
        )
    except Exception as exc:
        print(f"[mlflow] could not query runs: {exc}")
        return
    if runs is None or len(runs) == 0:
        return
    wanted = ["run_id", "tags.mlflow.runName", f"metrics.{MLFLOW_RANK_METRIC}",
              "metrics.best_val_accuracy", "metrics.best_epoch",
              "params.epochs", "params.per_class", "params.batch_size", "params.extra_aug"]
    cols = [c for c in wanted if c in runs.columns]
    print(f"\n[mlflow] Top {len(runs)} runs by {MLFLOW_RANK_METRIC}:")
    print(runs[cols].to_string(index=False))
    best = runs.iloc[0]
    print(f"[mlflow] best run: {best['run_id']} -> download its "
          f"assignment02_submission.py artifact to submit that model\n")


# ===========================================================================
# Colab entry point
# ===========================================================================
def main():
    global TRAINED_MODEL_B64, _EPOCH_CALLBACK

    torch.manual_seed(SEED)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print("CUDA available:", torch.cuda.is_available(), "| Device:",
          torch.cuda.get_device_name(0) if torch.cuda.is_available() else "CPU")

    batch_size = int(os.environ.get("CAME_BATCH_SIZE", 64 if torch.cuda.is_available() else 32))
    epochs = int(os.environ.get("CAME_EPOCHS", 12))

    train_dataset, val_dataset = build_datasets()
    train_loader, val_loader = make_loaders(train_dataset, val_dataset, batch_size)

    mlflow = _mlflow_start({
        "epochs": epochs,
        "batch_size": batch_size,
        "per_class": int(os.environ.get("CAME_PER_CLASS", 1000)),
        "hf_mode": os.environ.get("CAME_HF_MODE", "stream"),
        "extra_aug": os.environ.get("CAME_EXTRA_AUG", "1") != "0",
        "train_size": len(train_dataset),
        "val_size": len(val_dataset),
        "optimizer": "AdamW",
        "lr": 1e-3,
        "weight_decay": 1e-4,
        "seed": SEED,
        "image_size": IMAGE_SIZE,
        "num_params": count_parameters(build_model(9)),
        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else "cpu",
    })
    if mlflow is not None:
        _EPOCH_CALLBACK = _mlflow_epoch_callback(mlflow)

    try:
        print(f"Model parameters: {count_parameters(build_model(9)):,}")
        print(f"Training for {epochs} epochs, batch size {batch_size}")
        t0 = time.time()
        model, history = train_model(train_loader, val_loader, epochs, device)
        train_seconds = time.time() - t0
        print(f"Training finished in {train_seconds:.1f}s")

        y_true = torch.cat([labels.long() for _, labels in val_loader])
        y_pred = predict(model, val_loader, device)
        report = classification_report(y_true, y_pred)
        print(f"Best checkpoint (GPU) | val accuracy={report['accuracy']:.4f} "
              f"macro_recall={report['macro_recall']:.4f}")
        print("Confusion matrix (rows=true, cols=pred):", official_classes())
        print(report["confusion"])

        torch.save(model.state_dict(), "best_model.pt")
        b64 = encode_model_to_b64(model)
        Path("trained_model_b64.txt").write_text(b64, encoding="ascii")
        print(f"Saved best_model.pt and trained_model_b64.txt ({len(b64) / 1e6:.2f} MB of base64)")

        # Verify the exact CMS loading path on CPU (fp16 round trip).
        TRAINED_MODEL_B64 = b64
        cpu = torch.device("cpu")
        cpu_model = load_trained_model(cpu)
        cpu_report = classification_report(y_true, predict(cpu_model, val_loader, cpu))
        print(f"load_trained_model(cpu) OK | val accuracy={cpu_report['accuracy']:.4f} "
              f"macro_recall={cpu_report['macro_recall']:.4f}")

        submission = _write_submission_file(b64)

        if mlflow is not None:
            _mlflow_log_results(mlflow, report, cpu_report, history, train_seconds,
                                ["best_model.pt", submission])
            mlflow.end_run()
            _mlflow_print_leaderboard(mlflow)

        # Returned for notebook use (plots / checks); ignored when run as a script.
        return {
            "model": model,
            "history": history,
            "report": report,
            "cpu_report": cpu_report,
            "val_loader": val_loader,
            "submission": submission,
        }
    except BaseException:
        if mlflow is not None:
            mlflow.end_run(status="FAILED")
        raise
    finally:
        _EPOCH_CALLBACK = None


def _run_on_cms_if_needed() -> bool:
    zpath = Path("cms_eval_data.zip")
    if not zpath.exists():
        return False
    import sys, runpy, zipfile
    sys.modules["a02_cnn"] = sys.modules["__main__"]
    with zipfile.ZipFile(zpath) as archive:
        archive.extract("cms_grader.py", ".")
    runpy.run_path("cms_grader.py", run_name="__main__")
    return True


if __name__ == "__main__":
    if not _run_on_cms_if_needed():
        main()
