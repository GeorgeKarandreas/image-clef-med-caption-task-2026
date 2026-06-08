# AUEB/DMST Implementation for the ImageCLEFmedical 2026 Caption Task

This repository contains the training, inference, and evaluation code used for our work in the ImageCLEFmedical 2026 Concept Detection and Caption Prediction Tasks.

The ImageCLEF datasets and trained model checkpoints are not included in this repository. Datasets should be placed under `data/`, and locally trained checkpoints are written under `models/`.

## Setup

This project uses Python 3.12 and `uv` for dependency management.

Set up the environment:

```bash
uv sync
```

Run scripts through `uv run`:

```bash
uv run path/to/script.py
```

Note: This project was developed and tested on a CUDA GPU environment using PyTorch CUDA 12.8 wheels.

## Expected Data Layout

Place the ImageCLEF data under `data/` using the paths expected by the scripts:

```text
data/
  dev_concept/
    concepts.csv
    images/
  dev_caption/
    captions.csv
    images/
  dev_concept_synth/
    concepts.csv
    images/
  dev_caption_synth/
    captions.csv
    images/
  test/
    images/
  test_synth/
    images/
```

## Running Scripts

### Concept Training

Train a concept model:

```bash
uv run scripts/concept/train_concept_model.py convnext 224 42
```

Arguments:

- `model_name`: Model backbone. Available options are `convnext`, `convnext-tiny`, `resnet50`, `tresnet`, `dinov2`, `efficientnetv2`, or `swinsmall`

- `image_size`: Square image size for transforms

- `seed`: Run seed

Optional flags:

```bash
uv run scripts/concept/train_concept_model.py convnext 224 42 --prediction-threshold 0.6 --no-deterministic
```

Outputs:

- Model checkpoints under `models/<model_name>/<run_id>/model.pt`

- Validation submissions and run logs under `results/concept/`

### Concept Test Ensemble

Generate a concept test submission from the hard-coded checkpoint ensemble:

```bash
uv run scripts/concept/test_mixed_model_ensemble.py
```

Before running, edit `MODEL_PATHS` and `MODEL_WEIGHTS` in `scripts/concept/test_mixed_model_ensemble.py` so they point to checkpoints that exist locally.

### Caption Training

Train the caption model:

```bash
uv run scripts/caption/train_caption_model.py
```

Outputs:

- Model checkpoint under `models/caption/<run_id>`

- Run logs under `results/caption/training_runs.csv`

### Caption Test Inference

Generate a caption test submission:

```bash
uv run scripts/caption/test_caption_model.py
```

Optional flags:

```bash
uv run scripts/caption/test_caption_model.py --checkpoint models/caption/<run_id> --test-dir data/test/images --output results/caption/caption_submission.csv --batch-size 32 --num-beams 5
```

If `--checkpoint` is omitted, the script uses the newest checkpoint in `models/caption/`.

### Synthetical Variants

Synthetic concept and caption scripts mirror the main scripts but read/write the `_synth` folders:

```bash
uv run scripts/concept_synth/train_concept_synth_model.py convnext 224 42
uv run scripts/concept_synth/test_synth_mixed_model_ensemble.py
uv run scripts/caption_synth/train_caption_synth_model.py
uv run scripts/caption_synth/test_caption_synth_model.py
```

## Project Structure

```text
notebooks/
  captions_task_notebook.ipynb       Caption task experiments
  concepts_task_notebook.ipynb       Concept task experiments

scripts/
  concept/                           Concept training and test ensemble scripts
  caption/                           Caption training and test inference scripts
  concept_synth/                     Synthetic concept data scripts
  caption_synth/                     Synthetic caption data scripts

utils/
  shared/                            Shared project utilities
  concept/                           Concept Task utilities
  caption/                           Caption Task utilities

results/
  concept/                           Concept submissions and training run logs
  caption/                           Caption submissions and training run logs
  concept_synth/                     Synthetic concept outputs
  caption_synth/                     Synthetic caption outputs

data/                                Local datasets
models/                              Local model checkpoints
```

## Notes

- Run commands from the repository root.

- `data/` and `models/` are ignored by git.

- Training/inference can download pretrained model weights from Hugging Face, OpenCLIP, `timm`, or PyTorch sources, so network access may be needed on first run.

- Most hyperparameters are constants near the top of each script.
