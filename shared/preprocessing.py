import glob
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, List, Tuple, Iterator

import numpy as np
import tensorflow as tf
import torch
from torch.utils.data import IterableDataset, DataLoader


# ============================================================
# 1. FIXED CONFIGURATION
# ============================================================

SEED = 42
PATCH_SIZE = 64

INPUT_FEATURES: List[str] = [
    "elevation",
    "th",
    "vs",
    "tmmn",
    "tmmx",
    "sph",
    "pr",
    "pdsi",
    "NDVI",
    "population",
    "erc",
    "PrevFireMask",
]

TARGET_FEATURE = "FireMask"

ALL_FEATURES = INPUT_FEATURES + [TARGET_FEATURE]

NUM_CHANNELS = len(INPUT_FEATURES)

UNCERTAIN_LABEL_VALUE = -1.0

WIND_DIRECTION_CHANNEL = INPUT_FEATURES.index("th")


# ============================================================
# PHYSICAL VALID RANGES
# ============================================================

FEATURE_VALID_RANGES = {
    "elevation": (-500.0, 9000.0),
    "th": (0.0, 360.0),
    "vs": (0.0, 30.0),
    "tmmn": (200.0, 330.0),
    "tmmx": (200.0, 350.0),
    "sph": (0.0, 0.03),
    "pr": (0.0, 50.0),
    "pdsi": (-10.0, 10.0),
    "NDVI": (-10000.0, 10000.0),
    "population": (0.0, 100000.0),
    "erc": (0.0, 120.0),
    "PrevFireMask": (-1.0, 1.0),
}


# ============================================================
# GOOGLE DRIVE PROJECT PATH
# ============================================================

PROJECT_ROOT = Path("/content/drive/MyDrive/wildfire_project")

DATA_DIR = PROJECT_ROOT / "data"

STATS_PATH = str(
    PROJECT_ROOT / "shared" / "normalization_stats.json"
)

DEFAULT_DATA_DIR = str(DATA_DIR)


# Reproducibility
np.random.seed(SEED)
torch.manual_seed(SEED)
# ============================================================
# 2. RAW TFRECORD PARSING
# ============================================================

def _feature_description() -> Dict[str, "tf.io.FixedLenFeature"]:
    """
    Define the structure of one TFRecord example.
    """

    return {
        name: tf.io.FixedLenFeature(
            [PATCH_SIZE, PATCH_SIZE],
            tf.float32
        )
        for name in ALL_FEATURES
    }


def _parse_example(example_proto):
    """
    Parse one TFRecord example.

    Returns:
        inputs: (12, 64, 64)
        target: (1, 64, 64)
    """

    parsed = tf.io.parse_single_example(
        example_proto,
        _feature_description()
    )

    inputs = tf.stack(
        [parsed[name] for name in INPUT_FEATURES],
        axis=0
    )

    target = parsed[TARGET_FEATURE][tf.newaxis, ...]

    return inputs, target
    # ============================================================
# 3. TRAIN / VALIDATION / TEST FILES
# ============================================================

def get_split_files(
    data_dir: str = DEFAULT_DATA_DIR
) -> Dict[str, List[str]]:
    """
    Find the predefined train/eval/test TFRecord files.

    Only actual .tfrecord files are included.
    .zip files are ignored.
    """

    splits = {
        "train": sorted(
            glob.glob(
                os.path.join(
                    data_dir,
                    "*train*.tfrecord"
                )
            )
        ),

        "val": sorted(
            glob.glob(
                os.path.join(
                    data_dir,
                    "*eval*.tfrecord"
                )
            )
        ),

        "test": sorted(
            glob.glob(
                os.path.join(
                    data_dir,
                    "*test*.tfrecord"
                )
            )
        ),
    }

    for split_name, files in splits.items():

        if not files:
            raise FileNotFoundError(
                f"No TFRecord files found for split "
                f"'{split_name}' in '{data_dir}'."
            )

    return splits
    # ============================================================
# 4. VALID MASK
# ============================================================

def build_valid_mask(
    targets: np.ndarray
) -> np.ndarray:
    """
    1.0 where target is a valid 0/1 label.
    0.0 where target is -1 (uncertain).
    """

    return (
        targets != UNCERTAIN_LABEL_VALUE
    ).astype(np.float32)
    # ============================================================
# 5. INVALID INPUT HANDLING
# ============================================================

def _feature_valid_mask(
    values: np.ndarray,
    feature_name: str
) -> np.ndarray:

    lower, upper = FEATURE_VALID_RANGES[feature_name]

    return (
        np.isfinite(values)
        & (values >= lower)
        & (values <= upper)
    )


def _clean_single_input_with_medians(
    inputs: np.ndarray,
    channel_medians: np.ndarray
) -> np.ndarray:
    """
    Clean one input sample.

    inputs shape:
        (12, 64, 64)

    This version works on ONE sample rather than
    the entire dataset.
    """

    cleaned = inputs.copy()

    for channel_index, feature_name in enumerate(
        INPUT_FEATURES
    ):

        values = cleaned[channel_index]

        valid = _feature_valid_mask(
            values,
            feature_name
        )

        values[~valid] = channel_medians[channel_index]

        cleaned[channel_index] = values

    return cleaned
    # ============================================================
# 6. NORMALIZATION STATISTICS
# ============================================================

@dataclass
class NormalizationStats:

    means: List[float]

    stds: List[float]

    medians: List[float]

    version: int = 2


    def save(
        self,
        path: str = STATS_PATH
    ) -> None:

        with open(path, "w") as file:

            json.dump(
                asdict(self),
                file,
                indent=2
            )


    @classmethod
    def load(
        cls,
        path: str = STATS_PATH
    ) -> "NormalizationStats":

        with open(path) as file:

            payload = json.load(file)

        payload.setdefault(
            "version",
            1
        )

        return cls(**payload)
        # ============================================================
# 7. NORMALIZE ONE SAMPLE
# ============================================================

def normalize_single_input(
    inputs: np.ndarray,
    stats: NormalizationStats
) -> np.ndarray:
    """
    Clean and normalize one sample.

    Input:
        (12, 64, 64)

    Output:
        (12, 64, 64)
    """

    medians = np.asarray(
        stats.medians,
        dtype=np.float32
    )

    inputs = _clean_single_input_with_medians(
        inputs.astype(np.float32, copy=False),
        medians
    )

    means = np.asarray(
        stats.means,
        dtype=np.float32
    ).reshape(-1, 1, 1)

    stds = np.asarray(
        stats.stds,
        dtype=np.float32
    ).reshape(-1, 1, 1)

    normalized = (
        inputs - means
    ) / stds

    return normalized.astype(
        np.float32,
        copy=False
    )
    # ============================================================
# 8. LOAD SHARED NORMALIZATION STATISTICS
# ============================================================

def load_normalization_stats() -> NormalizationStats:

    if not os.path.exists(STATS_PATH):

        raise FileNotFoundError(
            "\nShared normalization statistics were not found.\n"
            f"Expected file:\n{STATS_PATH}\n\n"
            "Please make sure normalization_stats.json exists "
            "inside wildfire_project/shared/."
        )

    stats = NormalizationStats.load(
        STATS_PATH
    )

    if stats.version != 2:

        raise ValueError(
            f"Expected normalization statistics "
            f"version 2, but found version {stats.version}."
        )

    return stats
    # ============================================================
# 9. STREAMING WILDFIRE DATASET
# ============================================================

class StreamingWildfireDataset(
    IterableDataset
):
    """
    Memory-efficient dataset.

    TFRecord examples are read one at a time.

    The complete dataset is NOT loaded into RAM.
    """

    def __init__(
        self,
        tfrecord_paths: List[str],
        stats: NormalizationStats,
        augment: bool = False,
        shuffle: bool = False,
        shuffle_buffer: int = 2048,
        wind_direction_mean: float = 0.0,
        wind_direction_std: float = 1.0,
    ):

        super().__init__()

        self.tfrecord_paths = tfrecord_paths

        self.stats = stats

        self.augment = augment

        self.shuffle = shuffle

        self.shuffle_buffer = shuffle_buffer

        self.wind_direction_mean = (
            wind_direction_mean
        )

        self.wind_direction_std = (
            wind_direction_std
        )


    def _augment(
        self,
        x: np.ndarray,
        y: np.ndarray,
        m: np.ndarray
    ):

        # Convert normalized wind direction
        # back to degrees.

        x[WIND_DIRECTION_CHANNEL] = (
            x[WIND_DIRECTION_CHANNEL]
            * self.wind_direction_std
            + self.wind_direction_mean
        )


        # Horizontal flip

        if np.random.rand() < 0.5:

            x = x[:, :, ::-1]

            y = y[:, :, ::-1]

            m = m[:, :, ::-1]

            x[WIND_DIRECTION_CHANNEL] = (
                180.0
                - x[WIND_DIRECTION_CHANNEL]
            ) % 360.0


        # Vertical flip

        if np.random.rand() < 0.5:

            x = x[:, ::-1, :]

            y = y[:, ::-1, :]

            m = m[:, ::-1, :]

            x[WIND_DIRECTION_CHANNEL] = (
                360.0
                - x[WIND_DIRECTION_CHANNEL]
            ) % 360.0


        # Rotation

        k = np.random.randint(
            0,
            4
        )

        if k:

            x = np.rot90(
                x,
                k=k,
                axes=(1, 2)
            )

            y = np.rot90(
                y,
                k=k,
                axes=(1, 2)
            )

            m = np.rot90(
                m,
                k=k,
                axes=(1, 2)
            )

            x[WIND_DIRECTION_CHANNEL] = (
                x[WIND_DIRECTION_CHANNEL]
                - 90.0 * k
            ) % 360.0


        # Convert wind direction back
        # to normalized form.

        x[WIND_DIRECTION_CHANNEL] = (
            x[WIND_DIRECTION_CHANNEL]
            - self.wind_direction_mean
        ) / self.wind_direction_std


        return x, y, m


    def __iter__(self) -> Iterator:

        dataset = tf.data.TFRecordDataset(
            self.tfrecord_paths
        )

        dataset = dataset.map(
            _parse_example,
            num_parallel_calls=tf.data.AUTOTUNE
        )


        if self.shuffle:

            dataset = dataset.shuffle(
                buffer_size=self.shuffle_buffer,
                seed=SEED,
                reshuffle_each_iteration=True
            )


        for inputs, target in dataset:

            # TensorFlow tensor → NumPy

            x = inputs.numpy()

            y = target.numpy()


            # Valid mask

            m = build_valid_mask(y)


            # Normalize

            x = normalize_single_input(
                x,
                self.stats
            )


            # Augmentation only for training

            if self.augment:

                x, y, m = self._augment(
                    x,
                    y,
                    m
                )


            # Make arrays contiguous

            x = np.ascontiguousarray(
                x,
                dtype=np.float32
            )

            y = np.ascontiguousarray(
                y,
                dtype=np.float32
            )

            m = np.ascontiguousarray(
                m,
                dtype=np.float32
            )


            # NumPy → PyTorch

            yield (
                torch.from_numpy(x).float(),
                torch.from_numpy(y).float(),
                torch.from_numpy(m).float(),
            )
            # ============================================================
# 10. MEMORY-SAFE POSITIVE CLASS WEIGHT
# ============================================================

def compute_pos_weight_streaming(
    train_tfrecord_paths: List[str],
    max_weight: float = 20.0
) -> float:
    """
    Calculate positive-class weight without loading
    the entire training target dataset into RAM.
    """

    dataset = tf.data.TFRecordDataset(
        train_tfrecord_paths
    )

    dataset = dataset.map(
        _parse_example,
        num_parallel_calls=tf.data.AUTOTUNE
    )


    num_pos = 0

    num_neg = 0


    for _, target in dataset:

        target_np = target.numpy()

        valid = (
            target_np
            != UNCERTAIN_LABEL_VALUE
        )

        valid_labels = target_np[valid]

        num_pos += int(
            np.sum(valid_labels == 1)
        )

        num_neg += int(
            np.sum(valid_labels == 0)
        )


    if num_pos == 0:

        raise ValueError(
            "No positive fire pixels were found "
            "in the training targets."
        )


    raw_weight = (
        float(num_neg)
        / float(num_pos)
    )


    pos_weight = float(
        np.clip(
            raw_weight,
            1.0,
            max_weight
        )
    )


    print(
        f"Training positive pixels: {num_pos:,}"
    )

    print(
        f"Training negative pixels: {num_neg:,}"
    )

    print(
        f"Raw pos_weight: {raw_weight:.3f}"
    )

    print(
        f"Clipped pos_weight: {pos_weight:.3f}"
    )


    return pos_weight
    # ============================================================
# 11. MAIN DATALOADER FUNCTION
# ============================================================

def get_dataloaders(
    data_dir: str = DEFAULT_DATA_DIR,
    batch_size: int = 32,
    augment_train: bool = True,
    num_workers: int = 0,
) -> Tuple[
    DataLoader,
    DataLoader,
    DataLoader,
    float
]:
    """
    Create shared train, validation and test DataLoaders.

    IMPORTANT:
    The entire dataset is NOT loaded into RAM.
    """

    # --------------------------------------------------------
    # Find split files
    # --------------------------------------------------------

    split_files = get_split_files(
        data_dir
    )


    print(
        "TFRecord files found:"
    )

    print(
        f"  Train: {len(split_files['train'])}"
    )

    print(
        f"  Val:   {len(split_files['val'])}"
    )

    print(
        f"  Test:  {len(split_files['test'])}"
    )


    # --------------------------------------------------------
    # Load shared normalization statistics
    # --------------------------------------------------------

    stats = load_normalization_stats()


    # --------------------------------------------------------
    # Wind direction statistics
    # --------------------------------------------------------

    wind_mean = stats.means[
        WIND_DIRECTION_CHANNEL
    ]

    wind_std = stats.stds[
        WIND_DIRECTION_CHANNEL
    ]


    # Prevent division by zero

    if wind_std == 0:

        wind_std = 1.0


    # --------------------------------------------------------
    # Calculate pos_weight without loading all data
    # --------------------------------------------------------

    pos_weight = compute_pos_weight_streaming(
        split_files["train"]
    )


    # --------------------------------------------------------
    # Create streaming datasets
    # --------------------------------------------------------

    train_ds = StreamingWildfireDataset(
        tfrecord_paths=split_files["train"],
        stats=stats,
        augment=augment_train,
        shuffle=True,
        shuffle_buffer=2048,
        wind_direction_mean=wind_mean,
        wind_direction_std=wind_std,
    )


    val_ds = StreamingWildfireDataset(
        tfrecord_paths=split_files["val"],
        stats=stats,
        augment=False,
        shuffle=False,
        wind_direction_mean=wind_mean,
        wind_direction_std=wind_std,
    )


    test_ds = StreamingWildfireDataset(
        tfrecord_paths=split_files["test"],
        stats=stats,
        augment=False,
        shuffle=False,
        wind_direction_mean=wind_mean,
        wind_direction_std=wind_std,
    )


    # --------------------------------------------------------
    # Create PyTorch DataLoaders
    # --------------------------------------------------------

    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


    val_loader = DataLoader(
        val_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


    test_loader = DataLoader(
        test_ds,
        batch_size=batch_size,
        num_workers=num_workers,
        pin_memory=torch.cuda.is_available(),
    )


    return (
        train_loader,
        val_loader,
        test_loader,
        pos_weight
    )
    # ============================================================
# 12. PREPROCESSING SMOKE TEST
# ============================================================

def run_preprocessing_smoke_test():
    """
    Load only one batch from each split.

    This verifies:
      - TFRecord parsing
      - normalization
      - valid mask
      - tensor shapes
      - finite values
      - shared pos_weight
    """

    print("=" * 60)
    print("RUNNING PREPROCESSING SMOKE TEST")
    print("=" * 60)


    # Use a small batch for the smoke test

    train_loader, val_loader, test_loader, pos_weight = (
        get_dataloaders(
            batch_size=8,
            augment_train=False,
            num_workers=0,
        )
    )


    print()
    print(
        f"pos_weight for BCEWithLogitsLoss: "
        f"{pos_weight:.3f}"
    )


    # --------------------------------------------------------
    # Train batch
    # --------------------------------------------------------

    print()
    print("Loading one training batch...")

    x_train, y_train, m_train = next(
        iter(train_loader)
    )


    print(
        f"Train input shape:  {tuple(x_train.shape)}"
    )

    print(
        f"Train target shape: {tuple(y_train.shape)}"
    )

    print(
        f"Train mask shape:   {tuple(m_train.shape)}"
    )


    # --------------------------------------------------------
    # Validation batch
    # --------------------------------------------------------

    print()
    print("Loading one validation batch...")

    x_val, y_val, m_val = next(
        iter(val_loader)
    )


    print(
        f"Val input shape:  {tuple(x_val.shape)}"
    )

    print(
        f"Val target shape: {tuple(y_val.shape)}"
    )

    print(
        f"Val mask shape:   {tuple(m_val.shape)}"
    )


    # --------------------------------------------------------
    # Test batch
    # --------------------------------------------------------

    print()
    print("Loading one test batch...")

    x_test, y_test, m_test = next(
        iter(test_loader)
    )


    print(
        f"Test input shape:  {tuple(x_test.shape)}"
    )

    print(
        f"Test target shape: {tuple(y_test.shape)}"
    )

    print(
        f"Test mask shape:   {tuple(m_test.shape)}"
    )


    # --------------------------------------------------------
    # Shape checks
    # --------------------------------------------------------

    assert x_train.shape[1:] == (
        NUM_CHANNELS,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    assert y_train.shape[1:] == (
        1,
        PATCH_SIZE,
        PATCH_SIZE,
    )

    assert m_train.shape == y_train.shape


    # --------------------------------------------------------
    # Numerical checks
    # --------------------------------------------------------

    assert torch.isfinite(
        x_train
    ).all()

    assert torch.isfinite(
        y_train
    ).all()

    assert torch.isfinite(
        m_train
    ).all()


    # --------------------------------------------------------
    # Mask check
    # --------------------------------------------------------

    unique_mask_values = torch.unique(
        m_train
    )

    print()
    print(
        "Unique mask values:",
        unique_mask_values.tolist()
    )


    # Mask should contain only 0 and 1

    assert all(
        value.item() in [0.0, 1.0]
        for value in unique_mask_values
    )


    print()
    print("=" * 60)
    print("PREPROCESSING SMOKE TEST PASSED")
    print("=" * 60)
