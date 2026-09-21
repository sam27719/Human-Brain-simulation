# NeuroSim EEG backend

This project now includes a small FastAPI service for a baseline EEG change detector. It is intended for research and prototyping, not medical diagnosis.

## Run the API

```powershell
cd backend
python -m venv .venv
.venv\Scripts\Activate.ps1
pip install -r requirements.txt
uvicorn main:app --reload
```

The API is available at `http://localhost:8000`; interactive docs are at `/docs`.

## Dataset formats

- CSV: one row per trial, numeric signal/features in columns, plus a `label` column. Pass a different label name with `label_column`.
- NPZ: arrays named `signals` and `labels`. `signals` should be shaped `[trials, channels, time]` (or `[trials, time]`).
- NPY: reserved for signals only; use NPZ when labels are required.

The trainer creates time-domain features plus average FFT power in delta, theta, alpha, beta, and gamma bands. It stores the trained pipeline in `backend/eeg_model.joblib`. Use `/predict` with a `channels x time` signal after training.

`POST /train` accepts the dataset as the raw request body. Include `filename`, `label_column`, and `sampling_rate` as query parameters, for example `POST /train?filename=recordings.npz&sampling_rate=256`.

For real EEG work, keep subject-level train/test splits to avoid leakage, clean artifacts, and validate against a clinically appropriate labeled dataset before interpreting results.
