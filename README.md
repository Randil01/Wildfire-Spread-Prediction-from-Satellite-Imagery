# Next-Day Wildfire Spread Prediction

Predicting next-day wildfire spread as a pixel-wise binary segmentation task, using four deep-learning architectures trained and evaluated under identical, fair experimental conditions.

## Problem statement

Given a stack of environmental and remote-sensing feature layers for a region on day *t*, predict the binary fire mask for day *t + 1*. Every model in this repo solves the exact same task with the exact same input/output shapes, so results are directly comparable.

## Assignment-aligned model plan

This is a supervised deep-learning segmentation project. The four required architectures are:

1. CNN Baseline — implemented in `models/cnn_baseline.ipynb`
2. U-Net — implemented in `models/unet.ipynb`
3. ResNet-UNet — implemented in `models/resnet_unet.ipynb`
4. Attention U-Net — implemented in `models/attention_unet.ipynb`

All four notebooks must use [shared/preprocessing.ipynb](shared/preprocessing.ipynb) and [shared/evaluate.ipynb](shared/evaluate.ipynb). Architecture code, training configuration, and model-specific experiments belong in the respective model notebook. Dataset parsing, splitting, normalization, uncertain-label masking, shared metrics, and cross-model comparison must not be reimplemented separately.

## Dataset

**Name:** Next Day Wildfire Spread
**Source:** Kaggle — https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread
**Citation:** F. Huot, R. L. Hu, N. Goyal, T. Sankar, M. Ihme, and Y.-F. Chen, "Next Day Wildfire Spread: A Machine Learning Data Set to Predict Wildfire Spreading from Remote-Sensing Data," *IEEE Transactions on Geoscience and Remote Sensing*, vol. 60, pp. 1–13, 2022.

### What it contains

- **~15,000 samples**, split roughly 8:1:1 into train / eval / test
- Each sample is a **64×64 pixel patch**, where each pixel represents roughly a 1 km × 1 km area of the contiguous United States (2012–2020)
- **12 input channels** per sample, all aligned to the same 64×64 spatial grid:

| Channel | Description |
|---|---|
| `elevation` | Terrain height above sea level |
| `th` | Wind direction |
| `vs` | Wind speed |
| `tmmn` | Minimum daily temperature |
| `tmmx` | Maximum daily temperature |
| `sph` | Specific humidity |
| `pr` | Precipitation |
| `pdsi` | Palmer Drought Severity Index (drought index) |
| `NDVI` | Normalized Difference Vegetation Index (vegetation health) |
| `population` | Population density |
| `erc` | Energy Release Component (fuel/fire-intensity potential) |
| `PrevFireMask` | Fire mask on day *t* (today) |

- **Target/label:** `FireMask` — the binary fire mask on day *t + 1* (tomorrow), same 64×64 shape. Pixels can be `1` (fire), `0` (no fire), or `-1` (uncertain/unlabeled — decide during preprocessing whether to mask these out of the loss)
- **File format:** TFRecord (`.tfrecord` files, split into train/eval/test)
- Data originally aggregated via Google Earth Engine from VIIRS active-fire detections, GRIDMET weather data, and other public geospatial sources

### How to download

**Option A — Kaggle CLI (recommended):**

```bash
pip install kaggle --break-system-packages
# Place your kaggle.json API token in ~/.kaggle/kaggle.json first
# (Generate one at https://www.kaggle.com/settings -> API -> Create New Token)

kaggle datasets download -d fantineh/next-day-wildfire-spread -p data
cd data && unzip next-day-wildfire-spread.zip
```

**Option B — Manual:**
1. Go to https://www.kaggle.com/datasets/fantineh/next-day-wildfire-spread
2. Click "Download" (requires a free Kaggle account)
3. Unzip into `data/`

Do **not** commit the raw dataset to Git — it's large. `data/` is in `.gitignore`; only the loading/preprocessing code is tracked.

### Reading the data

The dataset comes as TFRecords. Parse with TensorFlow:

```python
import tensorflow as tf

feature_names = [
    'elevation', 'th', 'vs', 'tmmn', 'tmmx', 'sph', 'pr',
    'pdsi', 'NDVI', 'population', 'erc', 'PrevFireMask', 'FireMask'
]

def parse_fn(example_proto):
    feature_description = {
        name: tf.io.FixedLenFeature([64, 64], tf.float32) for name in feature_names
    }
    return tf.io.parse_single_example(example_proto, feature_description)

raw_dataset = tf.data.TFRecordDataset('data/next_day_wildfire_spread_train_00.tfrecord')
parsed_dataset = raw_dataset.map(parse_fn)
```

The shared `shared/preprocessing.ipynb` notebook wraps this into ready-to-use PyTorch `DataLoader` objects with the fixed train/val/test split, normalization, NaN handling, augmentation, and class-imbalance weight already applied. Every model notebook must run this shared notebook rather than writing its own loader. See [models/README.md](models/README.md) for the model implementation contract.

The shared `shared/evaluate.ipynb` notebook provides common metrics, visualizations, saved result format, and final comparison. Every member must use it for model evaluation so the comparison is fair.

## Repo structure

```
wildfire-prediction/
├── data                       # downloaded TFRecords (gitignored)
├── shared/
│   ├── preprocessing.ipynb     # shared Dataset/DataLoader and preprocessing
│   ├── evaluate.ipynb           # shared metrics, plots, and model comparison
│   └── normalization_stats.json # shared training-set statistics
├── models/
│   ├── README.md              # model implementation instructions
│   ├── cnn_baseline.ipynb      # Member 1
│   ├── unet.ipynb              # Member 2
│   ├── resnet_unet.ipynb       # Member 3
│   └── attention_unet.ipynb    # Member 4
├── requirment.txt             # Python dependencies
└── README.md
```

## Setup

```bash
git clone <repo-url>
cd wildfire-prediction
python -m venv .venv
\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirment.txt
```

In VS Code, select the Jupyter kernel **Wildfire Spread (.venv)** before running the notebooks. The dataset files must be in `data/`, and the model notebooks should be run from top to bottom.