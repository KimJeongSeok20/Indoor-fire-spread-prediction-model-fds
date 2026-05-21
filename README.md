# Fire Field Prediction Models

Sensor-sequence models for predicting 2D fire-field outputs:

- `TEMPERATURE`
- `CO_FRACTION`
- `SOOT_VISIBILITY`

The repository contains TCN, LSTM, and ConvLSTM model code, pretrained checkpoints, and fitted scalers.

## Repository Layout

```text
src/
  data/
    dataset.py          # Dataset loader and scaler handling
  models/
    tcn.py              # TCN model and training entrypoint
    lstm.py             # LSTM model and training entrypoint
    convlstm.py         # ConvLSTM model and training entrypoint
scripts/
  train_tcn.py
  train_lstm.py
  train_convlstm.py
  evaluate_tcn.py
  evaluate_convlstm.py
checkpoints/
  TCN_base.pth
  lstm1_100.pth
  lstm2_100.pth
  convlstm_base.pth
scalers/
  temp_scaler.pkl
  co_scaler.pkl
  soot_scaler.pkl
  devc_scaler.pkl
data/
  train_data/           # Full training CSV dataset
  test_data/            # Publishable model-ready test cases
  psm/                  # Centralized PyroSim PSM source files
  raw/                  # Local ZIP/PSM/source files; ignored by git
outputs/                # Generated checkpoints and plots
```

Large raw/generated simulation folders are intentionally excluded by `.gitignore`.

## Install

```bash
pip install -r requirements.txt
```

Install the PyTorch build that matches your CUDA environment if GPU training is required.

## Data Format

The training dataset has been normalized into 50 model-ready CSV cases:

```text
data/train_data/
  a_5MW/ ... t_5MW/
  a_7MW/ ... t_7MW/
  a_9MW/ ... j_9MW/
```

Each model-ready dataset case should be a folder containing:

- `{case_name}_devc.csv`
- `slice_0_1.csv`, `slice_1_2.csv`, ...

Example:

```text
data/test_data/
  test_3MW/
    test_3MW_devc.csv
    slice_0_1.csv
    slice_1_2.csv
```

`*_devc.csv` provides the 70 temperature sensor input columns used by the models. `slice_*.csv` files provide the three output maps: `TEMPERATURE`, `CO_FRACTION`, and `SOOT_VISIBILITY`.

The publishable fire-surface test set is:

```text
data/test_data/
  test_3MW/
  test_6MW/
  test_8MW/
```

PyroSim `.psm` source files are kept separately from CSV model data:

```text
data/psm/
  train_data/
    a_1MW.psm ... j_1MW.psm
  test_data/
    fst_3MW.psm
    fst_6MW.psm
    fst_8MW.psm
```

## Evaluate TCN

```bash
python scripts/evaluate_tcn.py --dataset data/train_data --checkpoint checkpoints/TCN_base.pth
```

## Evaluate ConvLSTM

```bash
python scripts/evaluate_convlstm.py --dataset data/train_data --checkpoint checkpoints/convlstm_base.pth
python scripts/evaluate_convlstm.py --dataset data/test_data --checkpoint checkpoints/convlstm_base.pth
```

## Train

```bash
python scripts/train_tcn.py
python scripts/train_lstm.py
python scripts/train_convlstm.py
```

Training scripts use `data/train_data` by default.
