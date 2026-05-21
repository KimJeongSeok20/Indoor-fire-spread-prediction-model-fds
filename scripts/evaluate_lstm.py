from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import joblib
import numpy as np
import torch
from sklearn.metrics import mean_squared_error, r2_score
from torch.utils.data import DataLoader
from tqdm import tqdm

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.data.dataset import myDataset
from src.models.lstm import TemperatureLSTM, device


DEFAULT_DATASET = PROJECT_ROOT / "data" / "train_data"
DEFAULT_CHECKPOINT = PROJECT_ROOT / "checkpoints" / "lstm2_100.pth"
DEFAULT_SCALER_DIR = PROJECT_ROOT / "scalers"


def resolve_path(path: str | Path) -> Path:
    path = Path(path)
    return path if path.is_absolute() else PROJECT_ROOT / path


def load_checkpoint(model: torch.nn.Module, checkpoint_path: Path) -> tuple[int, int]:
    try:
        checkpoint = torch.load(checkpoint_path, map_location=device, weights_only=True)
    except TypeError:
        checkpoint = torch.load(checkpoint_path, map_location=device)

    state_dict = checkpoint.get("model_state_dict", checkpoint) if isinstance(checkpoint, dict) else checkpoint
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    return len(missing), len(unexpected)


def load_scalers(scaler_dir: Path) -> dict[str, object]:
    return {
        "temp": joblib.load(scaler_dir / "temp_scaler.pkl"),
        "co": joblib.load(scaler_dir / "co_scaler.pkl"),
        "soot": joblib.load(scaler_dir / "soot_scaler.pkl"),
    }


def inverse_values(scaler: object, values: torch.Tensor) -> np.ndarray:
    flat = values.detach().cpu().numpy().reshape(-1, 1)
    return scaler.inverse_transform(flat).flatten()


def evaluate(
    model: torch.nn.Module,
    dataloader: DataLoader,
    scaler_dir: Path,
    max_batches: int | None = None,
) -> dict[str, dict[str, float]]:
    model.eval()
    scalers = load_scalers(scaler_dir)

    preds: dict[str, list[torch.Tensor]] = {"temp": [], "co": [], "soot": []}
    targets: dict[str, list[torch.Tensor]] = {"temp": [], "co": [], "soot": []}
    elapsed = 0.0
    sample_count = 0

    with torch.no_grad():
        for batch_index, (data, label) in enumerate(tqdm(dataloader, desc="Evaluating")):
            if max_batches is not None and batch_index >= max_batches:
                break

            data = data.to(device)
            label = label.to(device).float()

            start = time.perf_counter()
            temp, co, soot = model(data)
            elapsed += time.perf_counter() - start
            sample_count += data.size(0)

            preds["temp"].append(temp.cpu())
            preds["co"].append(co.cpu())
            preds["soot"].append(soot.cpu())
            targets["temp"].append(label[:, 0, :, :].cpu())
            targets["co"].append(label[:, 1, :, :].cpu())
            targets["soot"].append(label[:, 2, :, :].cpu())

    metrics: dict[str, dict[str, float]] = {}
    for key in ("temp", "co", "soot"):
        pred = inverse_values(scalers[key], torch.cat(preds[key], dim=0))
        target = inverse_values(scalers[key], torch.cat(targets[key], dim=0))
        metrics[key] = {
            "r2": float(r2_score(target, pred)),
            "rmse": float(np.sqrt(mean_squared_error(target, pred))),
        }

    metrics["average"] = {
        "r2": float(np.mean([metrics[key]["r2"] for key in ("temp", "co", "soot")])),
        "rmse": float(np.mean([metrics[key]["rmse"] for key in ("temp", "co", "soot")])),
        "seconds_per_sample": float(elapsed / sample_count) if sample_count else 0.0,
        "samples": float(sample_count),
    }
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser(description="Evaluate the LSTM model.")
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET), help="Dataset directory.")
    parser.add_argument("--checkpoint", default=str(DEFAULT_CHECKPOINT), help="LSTM checkpoint path.")
    parser.add_argument("--scalers", default=str(DEFAULT_SCALER_DIR), help="Scaler directory.")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--num-workers", type=int, default=0)
    parser.add_argument("--max-batches", type=int, default=None, help="Optional quick-test batch limit.")
    args = parser.parse_args()

    dataset_path = resolve_path(args.dataset)
    checkpoint_path = resolve_path(args.checkpoint)
    scaler_dir = resolve_path(args.scalers)

    dataset = myDataset(str(dataset_path), scaler_dir=scaler_dir)
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=torch.cuda.is_available(),
    )

    model = TemperatureLSTM().to(device)
    missing_count, unexpected_count = load_checkpoint(model, checkpoint_path)

    metrics = evaluate(model, dataloader, scaler_dir, max_batches=args.max_batches)
    print(f"Device: {device}")
    print(f"Checkpoint: {checkpoint_path}")
    print(f"missing_keys={missing_count}, unexpected_keys={unexpected_count}")
    for key, values in metrics.items():
        if key == "average":
            continue
        print(f"{key}: r2={values['r2']:.4f}, rmse={values['rmse']:.4f}")
    print(
        "average: "
        f"r2={metrics['average']['r2']:.4f}, "
        f"rmse={metrics['average']['rmse']:.4f}, "
        f"samples={metrics['average']['samples']:.0f}, "
        f"seconds_per_sample={metrics['average']['seconds_per_sample']:.6f}"
    )


if __name__ == "__main__":
    main()
