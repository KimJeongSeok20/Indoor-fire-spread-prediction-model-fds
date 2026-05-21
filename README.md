# Fire Field Prediction Models

Sensor-sequence models for predicting 2D fire-field outputs:

- `TEMPERATURE`
- `CO_FRACTION`
- `SOOT_VISIBILITY`

The repository contains TCN, LSTM, and ConvLSTM model code, pretrained checkpoints, and fitted scalers.

## Install

```bash
pip install -r requirements.txt
```

## Model Architecture

### ConvLSTM

![ConvLSTM model architecture](assets/CONVLSTM.png)

### TCN

![TCN model architecture](assets/TCN.png)

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

## Demo

https://github.com/user-attachments/assets/9708a141-80a6-41dc-9e02-68a5c68625e1

### Quantitative Results

Evaluation was run on `test_3MW` from 60s to 120s using the Temperature and CO Fraction outputs.

| Model | Temperature R2 | Temperature RMSE | CO Fraction R2 | CO Fraction RMSE | Inference FPS |
| --- | ---: | ---: | ---: | ---: | ---: |
| LSTM | 0.8581 | 79.0945 | 0.7897 | 0.003452 | 433.82 |
| ConvLSTM | 0.8654 | 77.0301 | 0.8169 | 0.003221 | 28.93 |
| TCN | 0.8987 | 66.8148 | 0.8475 | 0.002940 | 38.35 |
