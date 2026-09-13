# Model Implementation Instructions

Each model notebook must use the shared preprocessing notebook. This keeps the train, validation, and test data identical across the CNN, U-Net, ResNet-UNet, and Attention U-Net experiments.

The four required model notebooks are:

- `cnn_baseline.ipynb` — CNN baseline
- `unet.ipynb` — U-Net
- `resnet_unet.ipynb` — ResNet-UNet
- `attention_unet.ipynb` — Attention U-Net

## Required first cells

Place these cells at the beginning of every model notebook:

```python
from pathlib import Path
from IPython import get_ipython

preprocessing_notebook = Path("shared/preprocessing.ipynb")
if not preprocessing_notebook.exists():
    preprocessing_notebook = Path("../shared/preprocessing.ipynb")

get_ipython().run_line_magic("run", str(preprocessing_notebook))
```

Run the model notebook from the repository root or from the `models/` folder. The shared notebook creates `train_loader`, `val_loader`, `test_loader`, and `pos_weight` in the model notebook's namespace. Do not parse TFRecords, create a second dataset class, split the data, or calculate normalization statistics inside a model notebook.

After training, every model notebook must run the shared evaluator:

```python
%run ../shared/evaluate.ipynb

results = evaluate_model(
    model,
    test_loader,
    device,
    model_name="cnn_baseline",  # change for your model
    training_time_sec=elapsed_time,
    num_params=sum(parameter.numel() for parameter in model.parameters()),
)
save_results(results, "../results/cnn_baseline_metrics.json")
```

Use `val_loader` for checkpoint and hyperparameter decisions. Use `test_loader` only once for the final report evaluation.

## Build the loaders

```python
import torch

criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight, dtype=torch.float32)
)
```

`num_workers=0` is recommended on Windows while developing. It can be increased after the data pipeline works reliably.

## Batch contract

Each loader yields `(inputs, targets, valid_mask)`:

- `inputs`: float32 tensor with shape `(B, 12, 64, 64)`
- `targets`: float32 tensor with shape `(B, 1, 64, 64)`; values are `0`, `1`, or `-1`
- `valid_mask`: float32 tensor with shape `(B, 1, 64, 64)`; `1` means a valid target and `0` means uncertain (`-1`)

The model must return logits with shape `(B, 1, 64, 64)`.

## Masked training loss

Uncertain target pixels must not affect training. Use the shared `pos_weight` and apply `valid_mask` to the unreduced loss:

```python
criterion = torch.nn.BCEWithLogitsLoss(
    pos_weight=torch.tensor(pos_weight, device=device),
    reduction="none",
)

for inputs, targets, valid_mask in train_loader:
    inputs = inputs.to(device)
    targets = targets.to(device)
    valid_mask = valid_mask.to(device)

    logits = model(inputs)
    pixel_loss = criterion(logits, targets.clamp(0, 1))
    loss = (pixel_loss * valid_mask).sum() / valid_mask.sum().clamp_min(1.0)

    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
```

Do not use the `-1` target value directly in BCE. Clamp it only after retaining `valid_mask`, as shown above.

## Validation and testing

Use `val_loader` for model selection and `test_loader` only for the final report. Apply `valid_mask` to every pixel-level loss or metric in both phases. Do not augment validation or test data:

```python
_, val_targets, val_mask = next(iter(val_loader))
_, test_targets, test_mask = next(iter(test_loader))
```

All four models must use the same batch size, preprocessing notebook, split files, normalization statistics, mask handling, and evaluation rules. Only the architecture and its explicitly reported hyperparameters should differ.

## Dataset location

Place the downloaded TFRecord files directly in `data/`. The filenames must contain `train`, `eval`, and `test` so `get_dataloaders()` can find each split.

The notebook computes normalization statistics from the training split and stores them in `shared/normalization_stats.json`. Do not delete or replace this file between model runs unless the dataset changes.

## Minimal smoke test

Before training, verify the loader and model output:

```python
x, y, mask = next(iter(train_loader))
assert x.shape[1:] == (12, 64, 64)
assert y.shape[1:] == (1, 64, 64)
assert mask.shape == y.shape
assert torch.isfinite(x).all()
```
