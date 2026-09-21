from __future__ import annotations

import io
import json
import re
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, classification_report
from database import create_simulation, delete_simulation, initialize_database, list_simulations
from hormone_model import predictor, HORMONES, SCENARIO_FEATURES, PHASE_NAMES, HORMONE_DISPLAY, analyze_situation, EMOTION_MAP

MODEL_PATH = Path(__file__).with_name("eeg_model.joblib")

app = FastAPI(title="NeuroSim API", version="2.0.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=False,
    allow_methods=["*"],
    allow_headers=["*"],
)

initialize_database()
# auto-load trained hormone model so health shows "Trained" immediately
try:
    if not predictor.is_trained:
        predictor.load()
except Exception:
    pass


class PredictionRequest(BaseModel):
    signal: list[list[float]] = Field(..., description="channels x time samples")
    sampling_rate: float = Field(default=256.0, gt=0)


class SimulationRequest(BaseModel):
    type: str = "scenario"
    title: str
    description: str
    icon: str = "brain"
    hormone: str
    profile: dict[str, Any]


class HormonePredictionRequest(BaseModel):
    scenario: str = ""
    emotion: str = ""
    features: dict[str, float] = Field(default_factory=dict)


class HormoneFeedbackRequest(BaseModel):
    features: dict[str, float]
    hormone_levels: dict[str, list[float]]
    rating: float = Field(default=1.0, ge=0.1, le=2.0)


def _bandpower(signal: np.ndarray, sampling_rate: float, low: float, high: float) -> np.ndarray:
    frequencies = np.fft.rfftfreq(signal.shape[-1], d=1.0 / sampling_rate)
    spectrum = np.abs(np.fft.rfft(signal - signal.mean(axis=-1, keepdims=True), axis=-1)) ** 2
    mask = (frequencies >= low) & (frequencies < high)
    return spectrum[..., mask].mean(axis=-1) if mask.any() else np.zeros(signal.shape[:-1])


def extract_features(signals: np.ndarray, sampling_rate: float) -> np.ndarray:
    signals = np.asarray(signals, dtype=np.float64)
    if signals.ndim == 2:
        signals = signals[:, np.newaxis, :]
    if signals.ndim != 3 or signals.shape[-1] < 8:
        raise ValueError("Signals must have shape [trials, channels, time] with at least 8 time samples")

    features = [
        signals.mean(axis=-1),
        signals.std(axis=-1),
        np.ptp(signals, axis=-1),
        np.mean(np.abs(np.diff(signals, axis=-1)), axis=-1),
    ]
    for low, high in ((1, 4), (4, 8), (8, 13), (13, 30), (30, 45)):
        features.append(_bandpower(signals, sampling_rate, low, high))
    return np.concatenate([item.reshape(signals.shape[0], -1) for item in features], axis=1)


def _load_dataset(filename: str, content: bytes, label_column: str) -> tuple[np.ndarray, np.ndarray]:
    suffix = Path(filename).suffix.lower()
    if suffix == ".csv":
        data = np.genfromtxt(io.BytesIO(content), delimiter=",", names=True, dtype=None, encoding="utf-8")
        if data.dtype.names is None or label_column not in data.dtype.names:
            raise ValueError(f"CSV must contain a '{label_column}' column")
        labels = np.asarray(data[label_column])
        feature_names = [name for name in data.dtype.names if name != label_column]
        signals = np.column_stack([np.asarray(data[name], dtype=float) for name in feature_names])
        return signals, labels
    if suffix == ".npy":
        signals = np.load(io.BytesIO(content), allow_pickle=False)
        labels = None
    elif suffix == ".npz":
        archive = np.load(io.BytesIO(content), allow_pickle=False)
        if "signals" not in archive or "labels" not in archive:
            raise ValueError("NPZ must contain 'signals' and 'labels' arrays")
        signals, labels = archive["signals"], archive["labels"]
    else:
        raise ValueError("Supported formats are .csv, .npy, and .npz")

    if labels is None:
        raise ValueError("NPY is supported for signals only; use NPZ with 'signals' and 'labels'")
    signals = np.asarray(signals, dtype=float)
    labels = np.asarray(labels)
    if len(signals) != len(labels):
        raise ValueError("The number of signals and labels must match")
    return signals, labels


def _make_model() -> Pipeline:
    return Pipeline([
        ("scale", StandardScaler()),
        ("classifier", RandomForestClassifier(n_estimators=250, random_state=42, class_weight="balanced")),
    ])


@app.get("/health")
def health() -> dict[str, Any]:
    return {
        "status": "ok",
        "model_ready": MODEL_PATH.exists(),
        "hormone_model_ready": predictor.is_trained,
        "database_ready": True,
        "eeg_data_ready": EEG_DATA_PATH.exists(),
    }


@app.get("/info")
def get_info() -> dict[str, Any]:
    info = predictor.get_info()
    if predictor.feature_importance:
        info["feature_importance"] = predictor.feature_importance
    return info


@app.get("/simulations")
def get_simulations() -> dict[str, Any]:
    return {"simulations": list_simulations()}


@app.post("/simulations")
def save_simulation(simulation: SimulationRequest) -> dict[str, Any]:
    # -- only proper simulations may be stored --
    sim_type = simulation.type.lower() if simulation.type else "scenario"
    text_to_validate = (simulation.description or simulation.title or "").strip()
    if sim_type == "scenario" and text_to_validate:
        analysis = analyze_situation(text_to_validate)
        if not analysis["valid"]:
            raise HTTPException(status_code=400, detail=f"Fake/invalid simulation rejected: {analysis['reasons'][0]}. Please describe a real emotional situation.")
    if sim_type == "emotion":
        # validate that the title/emotion is a known emotion
        title = simulation.title.strip() if simulation.title else ""
        # allow combinations like "Joy + Pride"
        parts = [p.strip() for p in title.replace("+", ",").split(",") if p.strip()]
        # check at least one known emotion or the description is a valid situation
        has_known = any(p in EMOTION_MAP for p in parts) or any(p.split()[0] in EMOTION_MAP for p in parts)
        # if title is a scenario description, validate it instead
        if not has_known and text_to_validate:
            analysis = analyze_situation(text_to_validate)
            if not analysis["valid"] and len(text_to_validate.split()) > 2:
                raise HTTPException(status_code=400, detail="Fake emotion simulation rejected. Use a valid emotion or describe a real emotional situation.")
    # validate profile has hormones structure to prevent empty/fake saves
    if not simulation.profile or "hormones" not in simulation.profile:
        raise HTTPException(status_code=400, detail="Invalid simulation profile: missing hormone data.")
    return create_simulation(simulation.model_dump())


@app.delete("/simulations/{simulation_id}")
def remove_simulation(simulation_id: int) -> dict[str, Any]:
    if not delete_simulation(simulation_id):
        raise HTTPException(status_code=404, detail="Simulation not found")
    return {"message": "Simulation deleted"}


@app.post("/train")
async def train(
    request: Request,
    filename: str = "dataset.csv",
    label_column: str = "label",
    sampling_rate: float = 256.0,
) -> dict[str, Any]:
    if sampling_rate <= 0:
        raise HTTPException(status_code=400, detail="sampling_rate must be greater than zero")
    try:
        signals, labels = _load_dataset(filename, await request.body(), label_column)
        features = extract_features(signals, sampling_rate)
        if len(np.unique(labels)) < 2:
            raise ValueError("The dataset must contain at least two classes")
        stratify = labels if np.min(np.unique(labels, return_counts=True)[1]) >= 2 else None
        x_train, x_test, y_train, y_test = train_test_split(
            features, labels, test_size=0.2, random_state=42, stratify=stratify
        )
        model = _make_model()
        model.fit(x_train, y_train)
        predictions = model.predict(x_test)
        MODEL_PATH.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump({"model": model, "sampling_rate": sampling_rate}, MODEL_PATH)
        report = classification_report(y_test, predictions, output_dict=True, zero_division=0)
        return {
            "message": "Model trained and saved",
            "samples": int(len(labels)),
            "classes": [str(value) for value in model.classes_],
            "accuracy": round(float(accuracy_score(y_test, predictions)), 4),
            "report": report,
        }
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/predict")
def predict(request: PredictionRequest) -> dict[str, Any]:
    if not MODEL_PATH.exists():
        raise HTTPException(status_code=409, detail="Train a model before predicting")
    try:
        saved = joblib.load(MODEL_PATH)
        features = extract_features(np.asarray([request.signal]), request.sampling_rate)
        prediction = saved["model"].predict(features)[0]
        probabilities = saved["model"].predict_proba(features)[0]
        return {
            "label": str(prediction),
            "confidence": round(float(np.max(probabilities)), 4),
            "probabilities": {
                str(label): round(float(probability), 4)
                for label, probability in zip(saved["model"].classes_, probabilities)
            },
        }
    except (ValueError, TypeError) as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/hormone/patterns")
def get_hormone_patterns() -> dict[str, Any]:
    return {"patterns": predictor.get_patterns(), "hormones": HORMONES, "features": SCENARIO_FEATURES}


@app.post("/hormone/predict")
def predict_hormones(request: HormonePredictionRequest) -> dict[str, Any]:
    try:
        if request.emotion:
            result = predictor.predict_from_emotion(request.emotion)
            if result.get("error"):
                raise HTTPException(status_code=400, detail=result["message"])
        elif request.scenario:
            result = predictor.predict_from_text(request.scenario)
            if result.get("error"):
                raise HTTPException(status_code=400, detail=result["message"])
        elif request.features:
            features = [request.features.get(f, 0.5) for f in SCENARIO_FEATURES]
            result = predictor.predict(features)
        else:
            raise HTTPException(status_code=400, detail="Provide scenario, emotion, or features")
        return result
    except HTTPException:
        raise
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.post("/hormone/train")
def train_hormone_model() -> dict[str, Any]:
    try:
        return predictor.train()
    except Exception as error:
        raise HTTPException(status_code=500, detail=str(error)) from error


@app.post("/hormone/feedback")
def hormone_feedback(request: HormoneFeedbackRequest) -> dict[str, Any]:
    try:
        features = [request.features.get(f, 0.5) for f in SCENARIO_FEATURES]
        predictor.add_feedback(features, request.hormone_levels, request.rating)
        return {"message": "Feedback recorded", "rating": request.rating}
    except Exception as error:
        raise HTTPException(status_code=400, detail=str(error)) from error


@app.get("/hormone/status")
def hormone_model_status() -> dict[str, Any]:
    return {"trained": predictor.is_trained, "model_exists": MODEL_DIR.exists()}


from hormone_model import MODEL_DIR
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

DATASET_PATH = Path(__file__).with_name("neuroscience_dataset.npz")
EEG_DATA_PATH = Path(__file__).with_name("real_eeg_signals.npz")

# Label-to-emotion mapping for EEG
EEG_LABEL_MAP = {
    "focused": "focused", "engaged": "engaged", "calm": "calm",
    "alert": "alert", "excited": "excited", "determined": "determined",
}

# Scenario/emotion keywords to EEG labels
SCENARIO_TO_EEG = {
    "accident|crash|near.miss|danger|threat|attack": "alert",
    "speak|speech|audience|stage|present": "engaged",
    "win|competition|victory|achieve|success": "excited",
    "rescue|hero|save|brave": "focused",
    "spider|phobia|fear|scared|terrified": "alert",
    "interview|job|exam|test|assessment": "focused",
    "lose|lost|missing|gone": "calm",
    "meditat|peace|calm|quiet|zen|mindful": "calm",
    "hug|reunite|partner|friend|love|hold": "engaged",
    "anxious|worry|nervous|stress|panic": "alert",
    "focus|concentrate|study|learn|think": "focused",
    "sleep|dream|rest|nap|tired": "calm",
    "exercise|run|gym|workout|sport|physical": "excited",
    "music|listen|song|melody|rhythm": "engaged",
    "walk|nature|garden|forest|outside": "calm",
    "read|book|story|novel|study": "focused",
    "argue|fight|conflict|dispute|yell": "alert",
    "gift|surprise|unexpected|present": "excited",
    "cook|kitchen|recipe|food|eat": "engaged",
}

EMOTION_TO_EEG = {
    "Joy": "excited", "Fear": "alert", "Anger": "alert", "Sadness": "calm",
    "Surprise": "excited", "Disgust": "alert", "Love": "engaged", "Anxiety": "alert",
    "Excitement": "excited", "Calm": "calm", "Guilt": "calm", "Jealousy": "alert",
    "Pride": "excited", "Shame": "calm", "Gratitude": "engaged", "Hope": "focused",
    "Loneliness": "calm", "Frustration": "alert", "Relief": "calm", "Nostalgia": "calm",
    "Empathy": "engaged", "Boredom": "calm", "Contentment": "calm", "Grief": "calm",
}


@app.get("/eeg/data")
def get_eeg_data(label: str = "calm") -> dict[str, Any]:
    """Return real EEG waveform data from PhysioNet dataset."""
    if not EEG_DATA_PATH.exists():
        raise HTTPException(status_code=404, detail="EEG dataset not generated. Run download_eeg_data.py")
    
    data = np.load(EEG_DATA_PATH, allow_pickle=True)
    signals = data["signals"]
    labels = list(data["labels"])
    band_powers = json.loads(str(data["band_powers_json"]))
    channels = list(data["channels"])
    fs = int(data["sampling_rate"])
    
    # Find matching samples
    label_lower = label.lower()
    matching = [i for i, l in enumerate(labels) if l == label_lower]
    
    if not matching:
        # Try partial match
        matching = [i for i, l in enumerate(labels) if label_lower in l or l in label_lower]
    
    if not matching:
        # Default to first sample
        matching = list(range(min(8, len(labels))))
    
    # Pick a random sample from matching
    idx = np.random.choice(matching)
    sample_signals = signals[idx]  # [10 x 320]
    sample_bands = band_powers[idx] if idx < len(band_powers) else {}
    
    # Convert to frontend format
    eeg_data = {}
    for ch_idx, ch_name in enumerate(channels):
        eeg_data[ch_name] = sample_signals[ch_idx].tolist()
    
    return {
        "data": eeg_data,
        "channels": channels,
        "sampling_rate": fs,
        "band_powers": sample_bands,
        "label": label_lower,
    }


@app.get("/eeg/scenario")
def get_eeg_for_scenario(scenario: str = "", emotion: str = "") -> dict[str, Any]:
    """Map a scenario/emotion to real EEG data."""
    label = "calm"
    
    if emotion and emotion in EMOTION_TO_EEG:
        label = EMOTION_TO_EEG[emotion]
    elif scenario:
        lower = scenario.lower()
        for regex, eeg_label in SCENARIO_TO_EEG.items():
            if re.search(regex, lower, re.IGNORECASE):
                label = eeg_label
                break
    
    return get_eeg_data(label)


@app.get("/eeg/info")
def eeg_info() -> dict[str, Any]:
    """EEG dataset metadata."""
    if not EEG_DATA_PATH.exists():
        return {"exists": False}
    data = np.load(EEG_DATA_PATH, allow_pickle=True)
    labels = list(data["labels"])
    return {
        "exists": True,
        "total_samples": len(labels),
        "labels": sorted(set(labels)),
        "channels": list(data["channels"]),
        "sampling_rate": int(data["sampling_rate"]),
        "source": "PhysioNet EEG Motor Imagery (eegbci)",
    }

# Serve frontend so file:// CORS issue disappears — visit http://localhost:8000/
FRONTEND_DIR = Path(__file__).resolve().parent.parent
if (FRONTEND_DIR / "index.html").exists():
    app.mount("/", StaticFiles(directory=str(FRONTEND_DIR), html=True), name="frontend")


@app.get("/dataset/download")
def download_dataset():
    if not DATASET_PATH.exists():
        raise HTTPException(status_code=404, detail="Dataset not generated yet")
    return FileResponse(
        DATASET_PATH,
        media_type="application/octet-stream",
        filename="neuroscience_dataset.npz",
    )


@app.get("/dataset/info")
def dataset_info():
    if not DATASET_PATH.exists():
        return {"exists": False}
    import numpy as np
    data = np.load(DATASET_PATH)
    return {
        "exists": True,
        "samples": int(data["features"].shape[0]),
        "features": int(data["features"].shape[1]),
        "hormones": [str(h) for h in data.files if h != "features"],
        "phases": int(data[data.files[1]].shape[1]) if len(data.files) > 1 else 0,
        "feature_names": list(SCENARIO_FEATURES),
    }
