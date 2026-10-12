# Traffic State Prediction with LSTM Networks

This project explores using LSTM-based recurrent neural networks to predict the
next traffic state (queue lengths / signal phase configuration) at a network of
intersections, as a step toward smarter, adaptive traffic light control.

The idea: instead of a fixed-time traffic signal plan, a network could predict
how traffic is about to evolve and adjust ahead of time. This repo contains the
data preprocessing, model definitions, and experiments used to test a few ways of
framing the prediction problem.

> Course project for **Artificial Intelligence**, Spring 2021.
> Author: Mohammad Abbasishahkoo · Instructor: Dr. Farnaz Mahan

## Problem setup

Each sample in the dataset is a row describing the traffic state of an
intersection at a point in time. The dataset is a sequence of such rows
(time-ordered), so the task is naturally framed as a sequence prediction problem
and modeled with an LSTM.

Three ways of framing the input/target pair were tried:

- **Type 1 — state only.** Input: a window of `n` consecutive states
  (columns 0-7). Target: the state at `n+1`. Straightforward next-step
  regression.
- **Type 2 — state + action (encoded).** Input: a window of state columns
  together with the corresponding action columns (last 4 columns of the raw
  data), with the action tuple encoded into a single integer id. Target: the
  next state only.
- **Type 3 — state + action, predict both.** Same encoding as Type 2, but the
  network is trained to predict both the next state *and* the next action.
  A separate script (`traffic_lstm_classification.py`) treats the action part
  of this as a multi-class classification problem (one-hot / softmax output)
  instead of regression.

All numeric inputs are scaled to `[0, 1]` with `MinMaxScaler` before training,
and predictions are inverse-transformed back to the original scale for
reporting RMSE.

## Models

Two small stacked-LSTM architectures were used for the regression variants
(Type 1 / Type 2), and a slightly deeper LSTM + dense stack with a softmax
head for the classification variant (Type 3).

| | Regression model (Type 1 / 2) | Classification model (Type 3) |
|---|---|---|
| Layer 1 | LSTM(6), sigmoid | LSTM(6), tanh |
| Layer 2 | LSTM(6), sigmoid | LSTM(6), tanh |
| Layer 3 | Dense(n_features) | Dense(10), tanh |
| Layer 4 | — | Dense(10), tanh |
| Layer 5 | — | Dense(8), tanh |
| Output | — | Dense(n_classes), softmax |
| Loss | MSE | Categorical cross-entropy |
| Optimizer | Adam | Adam (lr=0.04) |

Training used `EarlyStopping` (monitoring loss) and `ModelCheckpoint` (saving
the best model on validation loss), batch size 32, and window sizes of 5 and
10 time steps.

## Results

RMSE reported both on the normalized scale (0-1) and on the original scale,
for window sizes 5 and 10, across 3 datasets.

**Window size = 5**

| Dataset | Model 1 (norm.) Train/Test | Model 1 (raw) Train/Test | Model 2 (norm.) Train/Test | Model 2 (raw) Train/Test |
|---|---|---|---|---|
| 1 | 0.05 / 0.12 | 18.16 / 32.87 | 0.05 / 0.10 | 17.17 / 37.53 |
| 2 | 0.05 / 0.08 | 14.51 / 23.83 | 0.06 / 0.12 | 15.53 / 36.83 |
| 3 | 0.09 / 0.11 | 24.04 / 34.60 | 0.09 / 0.10 | 25.00 / 34.67 |

**Window size = 10**

| Dataset | Model 1 (norm.) Train/Test | Model 1 (raw) Train/Test | Model 2 (norm.) Train/Test | Model 2 (raw) Train/Test |
|---|---|---|---|---|
| 1 | 0.05 / 0.12 | 18.24 / 37.52 | 0.05 / 0.10 | 16.17 / 36.37 |
| 2 | 0.05 / 0.08 | 14.67 / 23.15 | 0.05 / 0.13 | 15.45 / 37.71 |
| 3 | 0.09 / 0.11 | 23.44 / 35.40 | 0.10 / 0.10 | 24.26 / 33.34 |

**Type 3 (classification of next action)**

| Dataset | Accuracy |
|---|---|
| 1 | 0.273 |
| 2 | 0.264 |
| 3 | 0.109 |

### Takeaways

- Window size (5 vs. 10) didn't make a big difference for this dataset — the
  network converges to roughly the same error either way, and training loss
  flattens out after a handful of epochs (hence early stopping kicking in
  early in several runs).
- The regression framings (Type 1/2, predicting the next state) worked
  reasonably well. The classification framing (Type 3, predicting the next
  discrete action out of the encoded action space) performed noticeably
  worse — the action space encoding is high-cardinality and the classes are
  imbalanced, which the accuracy numbers reflect.
- A run with window size 30 on dataset 1 didn't show a meaningful
  improvement over 5/10, but larger window sizes weren't explored further due
  to hardware limits.

## Repo structure

```
.
├── src/
│   ├── traffic_lstm_regression.py      # Type 1 & Type 2 framing, next-state regression
│   └── traffic_lstm_classification.py  # Type 3 framing, next-action classification
├── data/                               # place raw per-intersection CSV files here (not included)
├── results/                            # training logs / plots go here
└── requirements.txt
```

## Data

The raw data is a set of CSV files (one per data source) with no header,
where columns 0-7 describe the traffic state and the last 4 columns describe
the signal action/phase configuration. The scripts expect all CSV files for a
given dataset to live under a single folder and simply concatenate them in
file order — see `read_dataset()` in either script.

The dataset itself isn't included in this repo. Point `DIR_PATH` in the config
section at the top of each script to your local copy.

## Running

```bash
pip install -r requirements.txt
python src/traffic_lstm_regression.py
python src/traffic_lstm_classification.py
```

Both scripts have a small config block near the top (dataset path, problem
`TYPE`, window size, batch size, epochs) — that's the easiest way to switch
between experiments without touching the rest of the code.

## Background / related reading

This project was written after reading up on adaptive, learning-based
approaches to traffic signal control, in particular:

- M. Abdoos, N. Mozayani, A. L. C. Bazzan, "Traffic Light Control in
  Non-stationary Environments based on Multi Agent Q-learning," *IEEE ITSC*,
  2011.
- M. Abdoos, N. Mozayani, "Using reinforcement learning agents for smart
  urban traffic control" (ICIS 2013, in Persian).

Those papers use multi-agent Q-learning directly on the signal-control
problem; this project instead looks at the narrower sub-problem of
*predicting* traffic state transitions with a supervised model, which could
feed into a control policy but doesn't implement one.

## Acknowledgements

Thanks to Dr. Farnaz Mahan for supervising this project as part of the
Artificial Intelligence course.

## Notes

- Built with Python 3.5, TensorFlow/Keras.
- `traffic_lstm_classification.py` only implements the Type 3 framing (it was
  originally a separate notebook for that experiment).
