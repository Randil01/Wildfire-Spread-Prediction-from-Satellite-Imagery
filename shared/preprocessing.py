"""
shared/preprocessing.py

Shared preprocessing pipeline for the project

ALL FOUR MODELS MUST IMPORT FROM THIS MODULE. This guarantees identical
train/val/test splits, normalization statistics, missing-value handling and
class-imbalance strategy across all four architectures, so the final
comparison in Section 7 of the report is fair (same preprocessing = only the
architecture differs).

Usage:
    from shared.preprocessing import get_dataloaders

    train_loader, val_loader, test_loader, pos_weight = get_dataloaders(
        data_dir="data", batch_size=32
    )

    for x, y, mask in train_loader:
        # x:    (B, 12, 64, 64) float32, normalized input channels
        # y:    (B, 1, 64, 64)  float32, target fire mask (0/1, or -1 = uncertain)
        # mask: (B, 1, 64, 64)  float32, 1.0 where y is valid, 0.0 where y == -1
        ...
"""

import glob
import json
import os
from dataclasses import dataclass, asdict
from typing import Dict, List, Tuple

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import Dataset, DataLoader

# ---------------------------------------------------------------------------
# 0. Config — fixed for every member, do not change without team agreement
# ---------------------------------------------------------------------------

SEED = 42
PATCH_SIZE = 64

INPUT_FEATURES: List[str] = [
    "elevation", "th", "vs", "tmmn", "tmmx", "sph", "pr",
    "pdsi", "NDVI", "population", "erc", "PrevFireMask",
]
TARGET_FEATURE = "FireMask"
ALL_FEATURES = INPUT_FEATURES + [TARGET_FEATURE]
NUM_CHANNELS = len(INPUT_FEATURES)  # 12

# Sentinel value used in the raw dataset for uncertain / unlabeled pixels
UNCERTAIN_LABEL_VALUE = -1.0

# Index of the wind-direction channel — needed for augmentation (see below)
WIND_DIRECTION_CHANNEL = INPUT_FEATURES.index("th")

STATS_PATH = os.path.join(os.path.dirname(__file__), "normalization_stats.json")

np.random.seed(SEED)
torch.manual_seed(SEED)


# ---------------------------------------------------------------------------
# 1. Raw TFRecord parsing
# ---------------------------------------------------------------------------

def _feature_description() -> Dict[str, "tf.io.FixedLenFeature"]:
    return {
        name: tf.io.FixedLenFeature([PATCH_SIZE, PATCH_SIZE], tf.float32)
        for name in ALL_FEATURES
    }


def _parse_example(example_proto):
    parsed = tf.io.parse_single_example(example_proto, _feature_description())
    inputs = tf.stack([parsed[name] for name in INPUT_FEATURES], axis=0)  # (C, H, W)
    target = parsed[TARGET_FEATURE][tf.newaxis, ...]                      # (1, H, W)
    return inputs, target


def load_split_arrays(tfrecord_paths: List[str]) -> Tuple[np.ndarray, np.ndarray]:
    """Reads a list of TFRecord files fully into memory as numpy arrays.

    Returns:
        inputs:  (N, 12, 64, 64) float32
        targets: (N, 1, 64, 64)  float32, values in {-1, 0, 1}
    """
    if not tfrecord_paths:
        raise FileNotFoundError(
            "No TFRecord files provided. Did you download the dataset into "
            "data/? See the root README.md for download instructions."
        )

    raw_dataset = tf.data.TFRecordDataset(tfrecord_paths)
    parsed_dataset = raw_dataset.map(_parse_example)

    all_inputs, all_targets = [], []
    for inputs, target in parsed_dataset:
        all_inputs.append(inputs.numpy())
        all_targets.append(target.numpy())

    if not all_inputs:
        raise ValueError(f"TFRecord files were found but contained 0 examples: {tfrecord_paths}")

    return np.stack(all_inputs), np.stack(all_targets)


# ---------------------------------------------------------------------------
# 2. Fixed train / val / test split
# ---------------------------------------------------------------------------

def get_split_files(data_dir: str) -> Dict[str, List[str]]:
    """Uses the dataset's OWN predefined train/eval/test TFRecord files.

    The Kaggle release ships pre-split files (filenames containing
    'train' / 'eval' / 'test'). We deliberately use the authors' own split
    rather than re-shuffling everything ourselves, for two reasons:
      1. It avoids leaking spatially/temporally correlated patches (e.g. the
         same fire on adjacent days) across train and test.
      2. It guarantees all 4 members get an IDENTICAL split by construction,
         with no risk of someone using a different random seed by mistake.
    """
    splits = {
        "train": sorted(glob.glob(os.path.join(data_dir, "*train*.tfrecord*"))),
        "val": sorted(glob.glob(os.path.join(data_dir, "*eval*.tfrecord*"))),
        "test": sorted(glob.glob(os.path.join(data_dir, "*test*.tfrecord*"))),
    }
    for split_name, files in splits.items():
        if not files:
            raise FileNotFoundError(
                f"No files found for split '{split_name}' in '{data_dir}'. "
                "Check that the downloaded filenames contain 'train' / 'eval' / "
                "'test', or adjust the glob pattern in get_split_files()."
            )
    return splits


# ---------------------------------------------------------------------------
# 3. Missing-value / uncertain-label handling
# ---------------------------------------------------------------------------

def build_valid_mask(targets: np.ndarray) -> np.ndarray:
    """Boolean/float mask, True (1.0) where the label is a real 0/1, False (0.0)
    where it's the -1 'uncertain' sentinel.

    Every member's loss function MUST multiply by this mask (or gather only
    valid pixels) so uncertain/unlabeled pixels never contribute to training
    loss or to reported evaluation metrics. This is applied identically for
    all 4 models — it is NOT something each member decides independently.
    """
    return (targets != UNCERTAIN_LABEL_VALUE).astype(np.float32)


def _fill_nans_with_medians(inputs: np.ndarray, channel_medians: np.ndarray) -> np.ndarray:
    """Replaces NaNs in input channels with that channel's TRAINING-set median.

    channel_medians must come from compute_normalization_stats() on the
    training set only, and be reused unchanged for val/test — never
    recomputed per split, or you leak information from val/test into
    preprocessing (an explicit requirement of the assignment brief).
    """
    inputs = inputs.copy()
    for c in range(inputs.shape[1]):
        channel = inputs[:, c, :, :]
        nan_mask = np.isnan(channel)
        if nan_mask.any():
            channel[nan_mask] = channel_medians[c]
            inputs[:, c, :, :] = channel
    return inputs


# ---------------------------------------------------------------------------
# 4. Normalization (train-set statistics only, saved + shared)
# ---------------------------------------------------------------------------

@dataclass
class NormalizationStats:
    means: List[float]
    stds: List[float]
    medians: List[float]  # used for NaN-filling

    def save(self, path: str = STATS_PATH) -> None:
        with open(path, "w") as f:
            json.dump(asdict(self), f, indent=2)

    @classmethod
    def load(cls, path: str = STATS_PATH) -> "NormalizationStats":
        with open(path) as f:
            return cls(**json.load(f))


def compute_normalization_stats(train_inputs: np.ndarray) -> NormalizationStats:
    """Computes per-channel mean/std/median from the TRAINING SET ONLY.

    Saved to shared/normalization_stats.json and committed to the repo so
    every member's model applies the EXACT same normalization. This
    satisfies the assignment requirement that "all preprocessing ... decisions
    must be made using only the training and validation data."
    """
    means, stds, medians = [], [], []
    for c in range(train_inputs.shape[1]):
        channel = train_inputs[:, c, :, :]
        median = float(np.nanmedian(channel))
        medians.append(median)
        clean_channel = np.nan_to_num(channel, nan=median)
        means.append(float(clean_channel.mean()))
        stds.append(float(clean_channel.std() + 1e-8))  # epsilon avoids div-by-zero
    return NormalizationStats(means=means, stds=stds, medians=medians)


def normalize_inputs(inputs: np.ndarray, stats: NormalizationStats) -> np.ndarray:
    inputs = _fill_nans_with_medians(inputs, np.array(stats.medians))
    means = np.array(stats.means).reshape(1, -1, 1, 1)
    stds = np.array(stats.stds).reshape(1, -1, 1, 1)
    return (inputs - means) / stds


# ---------------------------------------------------------------------------
# 5. Class-imbalance strategy
# ---------------------------------------------------------------------------

def compute_pos_weight(train_targets: np.ndarray) -> float:
    """Positive-class weight for weighted BCE, computed on valid (non -1)
    pixels of the TRAINING set only.

    Fire pixels are heavily outnumbered by no-fire pixels. Every member uses
    this SAME weight (e.g. torch.nn.BCEWithLogitsLoss(pos_weight=...)) so the
    class-imbalance strategy is a controlled variable across the 4-model
    comparison, not a hidden confound that makes results incomparable.
    """
    valid_mask = build_valid_mask(train_targets).astype(bool)
    valid_labels = train_targets[valid_mask]
    num_pos = (valid_labels == 1).sum()
    num_neg = (valid_labels == 0).sum()
    if num_pos == 0:
        raise ValueError("No positive (fire) pixels found in the training targets — check parsing.")
    return float(num_neg / num_pos)


# ---------------------------------------------------------------------------
# 6. PyTorch Dataset
# ---------------------------------------------------------------------------

class WildfireDataset(Dataset):
    """One sample = (input_tensor, target_tensor, valid_mask_tensor).

    input_tensor:  (12, 64, 64) float32, normalized
    target_tensor: (1, 64, 64)  float32, values in {0, 1}; uncertain pixels
                   are LEFT as -1 here — you must use valid_mask to exclude
                   them, not filter them out beforehand (that would change
                   the spatial shape).
    valid_mask:    (1, 64, 64)  float32, 1.0 = valid label, 0.0 = uncertain (-1)
    """

    def __init__(self, inputs: np.ndarray, targets: np.ndarray, augment: bool = False):
        self.inputs = inputs
        self.targets = targets
        self.valid_mask = build_valid_mask(targets)
        self.augment = augment

    def __len__(self) -> int:
        return len(self.inputs)

    def __getitem__(self, idx: int):
        x = self.inputs[idx]
        y = self.targets[idx]
        m = self.valid_mask[idx]

        if self.augment:
            x, y, m = self._augment(x, y, m)

        return (
            torch.from_numpy(x.copy()).float(),
            torch.from_numpy(y.copy()).float(),
            torch.from_numpy(m.copy()).float(),
        )

    @staticmethod
    def _augment(x: np.ndarray, y: np.ndarray, m: np.ndarray):
        """Random flips + 90-degree rotation, applied identically to input,
        target and mask so they stay spatially aligned.

        IMPORTANT: wind direction ('th') is a DIRECTIONAL quantity, measured
        in degrees. Flipping/rotating the grid without also transforming the
        wind-direction VALUES would create physically inconsistent samples
        (e.g. a wind arrow pointing the wrong way relative to the fire
        shape). This is handled explicitly below — do not skip it if you
        modify this augmentation function.
        """
        if np.random.rand() < 0.5:  # horizontal flip
            x = x[:, :, ::-1]
            y = y[:, :, ::-1]
            m = m[:, :, ::-1]
            x[WIND_DIRECTION_CHANNEL] = (180.0 - x[WIND_DIRECTION_CHANNEL]) % 360.0

        if np.random.rand() < 0.5:  # vertical flip
            x = x[:, ::-1, :]
            y = y[:, ::-1, :]
            m = m[:, ::-1, :]
            x[WIND_DIRECTION_CHANNEL] = (360.0 - x[WIND_DIRECTION_CHANNEL]) % 360.0

        k = np.random.randint(0, 4)  # 0 / 90 / 180 / 270 degree rotation
        if k:
            x = np.rot90(x, k=k, axes=(1, 2))
            y = np.rot90(y, k=k, axes=(1, 2))
            m = np.rot90(m, k=k, axes=(1, 2))
            x[WIND_DIRECTION_CHANNEL] = (x[WIND_DIRECTION_CHANNEL] - 90.0 * k) % 360.0

        return x, y, m


# ---------------------------------------------------------------------------
# 7. Top-level entry point — this is the ONLY function other code should call
# ---------------------------------------------------------------------------

def get_dataloaders(
    data_dir: str = "data",
    batch_size: int = 32,
    augment_train: bool = True,
    num_workers: int = 2,
) -> Tuple[DataLoader, DataLoader, DataLoader, float]:
    """Builds train/val/test DataLoaders with the shared, fixed preprocessing.

    Every member calls ONLY this function to get their data. Do not
    re-implement loading, splitting, or normalization inside your own model
    folder — importing from here is what makes the 4-way comparison fair.

    Returns:
        train_loader, val_loader, test_loader, pos_weight
    """
    split_files = get_split_files(data_dir)

    train_inputs, train_targets = load_split_arrays(split_files["train"])
    val_inputs, val_targets = load_split_arrays(split_files["val"])
    test_inputs, test_targets = load_split_arrays(split_files["test"])

    # Normalization stats: computed ONCE on train, cached to disk, reused by everyone
    if os.path.exists(STATS_PATH):
        stats = NormalizationStats.load()
    else:
        stats = compute_normalization_stats(train_inputs)
        stats.save()

    train_inputs = normalize_inputs(train_inputs, stats)
    val_inputs = normalize_inputs(val_inputs, stats)
    test_inputs = normalize_inputs(test_inputs, stats)

    pos_weight = compute_pos_weight(train_targets)

    train_ds = WildfireDataset(train_inputs, train_targets, augment=augment_train)
    val_ds = WildfireDataset(val_inputs, val_targets, augment=False)
    test_ds = WildfireDataset(test_inputs, test_targets, augment=False)

    g = torch.Generator()
    g.manual_seed(SEED)

    train_loader = DataLoader(
        train_ds, batch_size=batch_size, shuffle=True, num_workers=num_workers, generator=g
    )
    val_loader = DataLoader(val_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)
    test_loader = DataLoader(test_ds, batch_size=batch_size, shuffle=False, num_workers=num_workers)

    return train_loader, val_loader, test_loader, pos_weight


if __name__ == "__main__":
    train_loader, val_loader, test_loader, pos_weight = get_dataloaders()

    print(f"Train batches: {len(train_loader)}")
    print(f"Val batches:   {len(val_loader)}")
    print(f"Test batches:  {len(test_loader)}")
    print(f"pos_weight for BCEWithLogitsLoss: {pos_weight:.3f}")

    x, y, m = next(iter(train_loader))
    print(f"Input batch shape:  {tuple(x.shape)}")   # (B, 12, 64, 64)
    print(f"Target batch shape: {tuple(y.shape)}")   # (B, 1, 64, 64)
    print(f"Mask batch shape:   {tuple(m.shape)}")   # (B, 1, 64, 64)
