# Florida Temperature Forecasting with an LSTM

A stacked LSTM network that forecasts Florida's monthly average temperature from the previous 225 months of history. Built as part of a Machine Learning lab (Week 6).

## Results

Evaluated on the held-out test period, January 1995 to September 2013 (225 months):

| Metric | LSTM | Naive baseline (last month's value) |
|---|---|---|
| MAE | 1.24 °C | 2.31 °C |
| RMSE | 1.54 °C | 2.87 °C |
| R² | 0.900 | 0.653 |
| MAPE | 6.0% | n/a |

The model roughly halves the error of the naive baseline and tracks the seasonal cycle closely. Its main weakness is extremes: it under-predicts the hottest summers and over-predicts the coldest winters, and on average it under-predicts by about 0.9 °C.

## Approach

1. **Data:** 3,239 monthly readings for Florida (1743 to 2013), from Berkeley Earth's Earth Surface Temperature dataset (via Kaggle).
2. **Split:** chronological. The last 225 months are the test set and everything before is training data.
3. **Scaling:** `MinMaxScaler` fitted on the training data only, so no information leaks from the test set.
4. **Windows:** each sample is a 225-month sliding window, and the target is the month that follows it.
5. **Model:** three LSTM layers (128 units, ReLU), each followed by 20% Dropout, then a single Dense output.
6. **Training:** Adam optimiser, MSE loss, 25 epochs, batch size 32, 10% validation split.
7. **Evaluation:** MAE, MSE, RMSE, MAPE and R², compared against a naive baseline, plus charts of predicted vs actual values and errors over time.

## Setup

Requires Python 3.11.

```bash
git clone https://github.com/julianna162/florida-temperature-lstm.git
cd florida-temperature-lstm
pip install -r requirements.txt
```

## Usage

```bash
jupyter lab ML_wk6_lab.ipynb
```

Then choose **Kernel > Restart Kernel and Run All Cells**. Training takes about 7 minutes on a laptop CPU. Results will vary slightly between runs.

## Files

| File | Description |
|---|---|
| `ML_wk6_lab.ipynb` | Full notebook: data prep, model, training, evaluation and charts |
| `florida.csv` | Monthly Florida temperature data |
| `requirements.txt` | Python dependencies |
