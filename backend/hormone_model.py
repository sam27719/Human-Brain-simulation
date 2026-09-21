"""
Hormone response prediction model using neuroscience-based training data.

Uses Random Forest regressors trained on literature-derived hormone response curves
across 8 scenario types and 10 hormones over 16 time phases.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any

import joblib
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

MODEL_DIR = Path(__file__).with_name("models")
HORMONE_MODEL_PATH = MODEL_DIR / "hormone_model.joblib"
FEEDBACK_DB = Path(__file__).with_name("hormone_feedback.db")
DATASET_PATH = Path(__file__).with_name("neuroscience_dataset.npz")

HORMONES = [
    "Cortisol", "Epinephrine", "Dopamine", "Serotonin",
    "Norepinephrine", "Oxytocin", "EndogenousOpioids",
    "GABA", "Glutamate", "Endocannabinoids",
]

HORMONE_DISPLAY = {
    "Cortisol": {"name": "Cortisol", "type": "Hormone", "role": "Stress response, energy mobilization, immune regulation"},
    "Epinephrine": {"name": "Epinephrine", "type": "Hormone + Neurotransmitter", "role": "Fight-or-flight, heart rate, blood flow to muscles"},
    "Dopamine": {"name": "Dopamine", "type": "Neurotransmitter", "role": "Reward, motivation, learning, pleasure"},
    "Serotonin": {"name": "Serotonin", "type": "Neurotransmitter", "role": "Mood stability, sleep, appetite, social behavior"},
    "Norepinephrine": {"name": "Norepinephrine", "type": "Hormone + Neurotransmitter", "role": "Alertness, attention, arousal, vigilance"},
    "Oxytocin": {"name": "Oxytocin", "type": "Hormone", "role": "Social bonding, trust, empathy, attachment"},
    "EndogenousOpioids": {"name": "Endorphins", "type": "Neuropeptide", "role": "Pain relief, pleasure, stress buffering"},
    "GABA": {"name": "GABA", "type": "Neurotransmitter", "role": "Inhibition, calmness, anxiety reduction"},
    "Glutamate": {"name": "Glutamate", "type": "Neurotransmitter", "role": "Excitation, learning, memory formation"},
    "Endocannabinoids": {"name": "Endocannabinoids", "type": "Lipid Signal", "role": "Stress recovery, appetite, emotional regulation"},
}

PHASE_NAMES = [
    "Baseline", "Cue", "Orient", "Detect", "Appraise",
    "Decide", "Initiate", "Respond", "Peak", "Sustain",
    "Regulate", "Recover", "Consolidate", "Settle", "Review", "Return"
]

SCENARIO_FEATURES = [
    "stress", "threat", "reward", "social",
    "novelty", "cognitive", "physical", "emotion",
    "uncertainty", "loss", "safety", "time_pressure",
]

EMOTION_MAP = {
    "Joy":         {"stress": 0.15, "threat": 0.05, "reward": 0.85, "social": 0.6, "novelty": 0.5, "cognitive": 0.3, "physical": 0.3, "emotion": 0.85, "uncertainty": 0.2, "loss": 0.7, "safety": 0.85, "time_pressure": 0.1},
    "Fear":        {"stress": 0.85, "threat": 0.95, "reward": 0.1, "social": 0.2, "novelty": 0.6, "cognitive": 0.5, "physical": 0.7, "emotion": 0.9, "uncertainty": 0.85, "loss": -0.3, "safety": 0.1, "time_pressure": 0.8},
    "Anger":       {"stress": 0.8, "threat": 0.7, "reward": 0.1, "social": 0.3, "novelty": 0.2, "cognitive": 0.5, "physical": 0.6, "emotion": 0.9, "uncertainty": 0.3, "loss": -0.5, "safety": 0.3, "time_pressure": 0.6},
    "Sadness":     {"stress": 0.6, "threat": 0.2, "reward": -0.3, "social": 0.4, "novelty": 0.1, "cognitive": 0.5, "physical": 0.2, "emotion": 0.8, "uncertainty": 0.4, "loss": -0.9, "safety": 0.5, "time_pressure": 0.1},
    "Surprise":    {"stress": 0.4, "threat": 0.3, "reward": 0.5, "social": 0.3, "novelty": 0.95, "cognitive": 0.7, "physical": 0.3, "emotion": 0.8, "uncertainty": 0.9, "loss": 0.3, "safety": 0.5, "time_pressure": 0.4},
    "Disgust":     {"stress": 0.5, "threat": 0.5, "reward": -0.2, "social": 0.2, "novelty": 0.3, "cognitive": 0.4, "physical": 0.4, "emotion": 0.7, "uncertainty": 0.3, "loss": -0.2, "safety": 0.4, "time_pressure": 0.2},
    "Love":        {"stress": 0.1, "threat": 0.05, "reward": 0.8, "social": 0.95, "novelty": 0.3, "cognitive": 0.4, "physical": 0.4, "emotion": 0.9, "uncertainty": 0.1, "loss": 0.5, "safety": 0.9, "time_pressure": 0.05},
    "Anxiety":     {"stress": 0.75, "threat": 0.5, "reward": 0.1, "social": 0.3, "novelty": 0.4, "cognitive": 0.7, "physical": 0.4, "emotion": 0.8, "uncertainty": 0.9, "loss": -0.3, "safety": 0.3, "time_pressure": 0.6},
    "Excitement":  {"stress": 0.2, "threat": 0.05, "reward": 0.9, "social": 0.5, "novelty": 0.8, "cognitive": 0.5, "physical": 0.5, "emotion": 0.9, "uncertainty": 0.3, "loss": 0.6, "safety": 0.8, "time_pressure": 0.2},
    "Calm":        {"stress": 0.05, "threat": 0.02, "reward": 0.4, "social": 0.2, "novelty": 0.1, "cognitive": 0.2, "physical": 0.1, "emotion": 0.25, "uncertainty": 0.05, "loss": 0.3, "safety": 0.95, "time_pressure": 0.02},
    "Guilt":       {"stress": 0.55, "threat": 0.2, "reward": -0.3, "social": 0.6, "novelty": 0.1, "cognitive": 0.6, "physical": 0.2, "emotion": 0.7, "uncertainty": 0.4, "loss": -0.5, "safety": 0.5, "time_pressure": 0.3},
    "Jealousy":    {"stress": 0.65, "threat": 0.5, "reward": -0.2, "social": 0.7, "novelty": 0.2, "cognitive": 0.6, "physical": 0.3, "emotion": 0.8, "uncertainty": 0.6, "loss": -0.4, "safety": 0.35, "time_pressure": 0.4},
    # --- Expanded emotions: distinct hormone profiles ---
    "Pride":       {"stress": 0.25, "threat": 0.1, "reward": 0.85, "social": 0.45, "novelty": 0.2, "cognitive": 0.6, "physical": 0.25, "emotion": 0.8, "uncertainty": 0.2, "loss": 0.6, "safety": 0.75, "time_pressure": 0.2},
    "Shame":       {"stress": 0.6, "threat": 0.25, "reward": 0.1, "social": 0.65, "novelty": 0.15, "cognitive": 0.7, "physical": 0.2, "emotion": 0.85, "uncertainty": 0.45, "loss": -0.6, "safety": 0.35, "time_pressure": 0.25},
    "Gratitude":   {"stress": 0.15, "threat": 0.05, "reward": 0.75, "social": 0.85, "novelty": 0.25, "cognitive": 0.4, "physical": 0.25, "emotion": 0.8, "uncertainty": 0.15, "loss": 0.5, "safety": 0.9, "time_pressure": 0.05},
    "Hope":        {"stress": 0.25, "threat": 0.1, "reward": 0.7, "social": 0.4, "novelty": 0.4, "cognitive": 0.6, "physical": 0.2, "emotion": 0.7, "uncertainty": 0.5, "loss": 0.4, "safety": 0.65, "time_pressure": 0.2},
    "Loneliness":  {"stress": 0.55, "threat": 0.15, "reward": 0.1, "social": 0.1, "novelty": 0.1, "cognitive": 0.5, "physical": 0.15, "emotion": 0.75, "uncertainty": 0.5, "loss": -0.7, "safety": 0.3, "time_pressure": 0.1},
    "Frustration": {"stress": 0.7, "threat": 0.35, "reward": 0.15, "social": 0.25, "novelty": 0.2, "cognitive": 0.65, "physical": 0.45, "emotion": 0.8, "uncertainty": 0.5, "loss": -0.3, "safety": 0.4, "time_pressure": 0.6},
    "Relief":      {"stress": 0.2, "threat": 0.1, "reward": 0.6, "social": 0.3, "novelty": 0.3, "cognitive": 0.3, "physical": 0.2, "emotion": 0.65, "uncertainty": 0.2, "loss": 0.4, "safety": 0.85, "time_pressure": 0.1},
    "Nostalgia":   {"stress": 0.3, "threat": 0.05, "reward": 0.45, "social": 0.6, "novelty": 0.15, "cognitive": 0.65, "physical": 0.15, "emotion": 0.75, "uncertainty": 0.25, "loss": -0.5, "safety": 0.6, "time_pressure": 0.1},
    "Empathy":     {"stress": 0.35, "threat": 0.1, "reward": 0.3, "social": 0.9, "novelty": 0.2, "cognitive": 0.55, "physical": 0.2, "emotion": 0.8, "uncertainty": 0.3, "loss": -0.1, "safety": 0.65, "time_pressure": 0.15},
    "Boredom":     {"stress": 0.15, "threat": 0.05, "reward": 0.1, "social": 0.2, "novelty": 0.05, "cognitive": 0.25, "physical": 0.1, "emotion": 0.4, "uncertainty": 0.2, "loss": -0.2, "safety": 0.7, "time_pressure": 0.05},
    "Contentment": {"stress": 0.1, "threat": 0.03, "reward": 0.6, "social": 0.45, "novelty": 0.15, "cognitive": 0.25, "physical": 0.15, "emotion": 0.6, "uncertainty": 0.1, "loss": 0.4, "safety": 0.9, "time_pressure": 0.05},
    "Grief":       {"stress": 0.7, "threat": 0.15, "reward": 0.05, "social": 0.35, "novelty": 0.05, "cognitive": 0.6, "physical": 0.2, "emotion": 0.9, "uncertainty": 0.5, "loss": -0.9, "safety": 0.3, "time_pressure": 0.1},
}

SCENARIO_KEYWORDS = {
    "acute_threat": ["danger", "threat", "attack", "spider", "accident", "fire", "alarm", "loud noise", "alone at night", "confrontation", "rescue", "escape"],
    "chronic_stress": ["deadline", "work pressure", "financial", "relationship conflict", "exam", "worry", "overwhelmed", "burnout", "deadline pressure"],
    "reward": ["win", "won", "gift", "surprise", "achieved", "success", "promotion", "celebrate", "good news", "competition", "prize"],
    "social_bonding": ["hug", "friend", "family", "partner", "love", "wedding", "baby", "reunion", "deep conversation", "trust", "intimate"],
    "fear_anxiety": ["anxious", "nervous", "waiting for results", "fear of", "dark alley", "afraid", "scared", "worry", "panic", "dread"],
    "joy_excitement": ["concert", "adventure", "birthday", "excited", "amazing", "thrilled", "ecstatic", "party", "festival"],
    "loss_grief": ["lost", "death", "breakup", "failed", "grief", "sad", "miss", "gone", "end of", "funeral"],
    "calm_meditation": ["meditation", "yoga", "nature", "peaceful", "relax", "breathe", "calm", "serene", "tranquil", "quiet"],
    "physical_sports": ["race", "racing", "marathon", "sprint", "jog", "gym", "workout", "lift", "weight", "push-up", "pull-up", "sport", "football", "basketball", "tennis", "swim", "swimming", "cycling", "bike", "climb", "hiking", "trail", "surf", "surfing", "ski", "skating"],
    "driving_transport": ["drive", "driving", "road trip", "highway", "traffic", "commute", "parking", "car", "motorcycle", "taxi", "uber", "bus", "train", "flight", "airplane", "boarding"],
    "sleep_rest": ["sleep", "sleeping", "nap", "dream", "nightmare", "insomnia", "bed", "tired", "exhausted", "rest", "siesta", "doze", "snooze"],
    "creative_work": ["paint", "painting", "draw", "drawing", "write", "writing", "compose", "music", "guitar", "piano", "sing", "singing", "dance", "dancing", "photography", "film", "choreograph"],
    "food_social": ["cook", "cooking", "bake", "baking", "restaurant", "dinner", "lunch", "brunch", "feast", "bbq", "barbecue", "recipe", "kitchen", "chef", "meal"],
    "outdoor_nature": ["camp", "camping", "fishing", "hunt", "kayak", "canoe", "sail", "sailing", "garden", "gardening", "picnic", "beach", "mountain", "forest", "trail", "explore"],
    "learning_focus": ["study", "studying", "learn", "learning", "lecture", "class", "course", "read", "reading", "research", "practice", "train", "training", "certification", "exam prep"],
    "social_gathering": ["party", "聚会", "reunion", "gathering", "hang out", "hangout", "bar", "pub", "night out", "karaoke", "board game", "game night", "potluck"],
    "gaming_competition": ["video game", "gaming", "esports", "tournament", "ranked", "match", "boss fight", "speedrun", "puzzle", "strategy game", "card game", "chess"],
}

EMOTION_KEYWORDS = [
    "feel", "feeling", "felt", "emotion", "emotional", "mood", "afraid", "scared", "terrified",
    "happy", "joyful", "excited", "thrilled", "ecstatic", "grateful", "proud", "relieved",
    "angry", "furious", "frustrated", "irritated", "enraged", "annoyed",
    "sad", "depressed", "heartbroken", "grief", "mourning", "lonely", "hopeless",
    "surprised", "shocked", "astonished", "amazed", "stunned",
    "disgusted", "revolted", "nauseated", "repulsed",
    "loved", "affectionate", "attached", "caring", "compassionate",
    "anxious", "nervous", "worried", "panicked", "dread", "uneasy",
    "calm", "peaceful", "relaxed", "serene", "tranquil", "content",
    "guilty", "ashamed", "regretful", "remorseful", "embarrassed",
    "jealous", "envious", "resentful", "bitter",
    "stressed", "overwhelmed", "burned out", "exhausted", "drained",
    "confused", "lost", "uncertain", "conflicted", "torn",
    "proud", "accomplished", "satisfied", "fulfilled",
    "afraid", "fear", "phobia", "terror", "horror",
]

NON_EMOTIONAL_PATTERNS = [
    r"\b(what is|who is|where is|when is|how to|define|meaning of|definition of)\b",
    r"\b(weather|temperature|forecast)\b",
    r"\b(math|calculate|compute|equation|formula)\b",
    r"\b(recipe|ingredient|cook|bake)\b",
    r"\b(website|url|link|http|www)\b",
    r"\b(code|program|function|variable|debug)\b",
    r"\b(movie|song|book|author|actor)\b",
    r"\b(history|historical|century|era|war)\b",
    r"\b(science|physics|chemistry|biology|theorem)\b",
    r"\b(instructions|steps|guide|tutorial|manual)\b",
]


def analyze_situation(text: str) -> dict[str, Any]:
    """Analyze whether a text describes a proper emotional situation suitable for simulation."""
    import re
    text_lower = text.lower().strip()
    reasons = []
    score = 0.0

    if len(text_lower) < 3:
        return {"valid": False, "score": 0.0, "reasons": ["Input too short"], "category": "invalid"}

    for pattern in NON_EMOTIONAL_PATTERNS:
        if re.search(pattern, text_lower):
            reasons.append("Contains non-emotional content")
            score -= 0.3

    emotion_hits = sum(1 for kw in EMOTION_KEYWORDS if kw in text_lower)
    scenario_hits = 0
    matched_categories = []
    for category, keywords in SCENARIO_KEYWORDS.items():
        cat_hits = sum(1 for kw in keywords if kw in text_lower)
        if cat_hits > 0:
            scenario_hits += cat_hits
            matched_categories.append(category)

    has_second_person = any(w in text_lower for w in ["you ", "your ", "you're", "you'll", "yourself"])
    has_action = any(w in text_lower for w in [
        "went", "goes", "go", "running", "run", "walk", "walking", "drove", "drive",
        "faced", "face", "encounter", "met", "meet", "received", "receive",
        "lost", "find", "found", "saw", "see", "heard", "hear", "felt", "feel",
        "celebrated", "celebrate", "experienced", "experience", "witnessed", "witness",
        "attended", "attend", "participated", "participate", "survived", "survive",
        "escaped", "escape", "fought", "fight", "won", "lost", "saved", "rescued",
        "hugged", "hug", "kissed", "kiss", "cried", "laughed", "screamed", "ran",
        "slept", "sleep", "dream", "drove", "drive", "raced", "race", "swam", "swim",
        "cooked", "cook", "baked", "bake", "sang", "sing", "danced", "dance",
        "painted", "paint", "wrote", "write", "read", "studied", "study",
        "climbed", "climb", "cycled", "cycle", "fished", "fish", "camped", "camp",
        "surfed", "surf", "skied", "ski", "jogged", "jog", "trained", "exercise",
        "played", "play", "built", "build", "flew", "fly", "sailed", "sail",
        "drove", "drive", "raced", "race", "relaxed", "relax", "rested", "rest",
        "meditated", "meditate", "practiced", "practice", "performed", "perform",
        "explored", "explore", "hiked", "hike", "cooked", "cook", "ate", "eat",
    ])
    has_context = any(w in text_lower for w in [
        "before", "after", "during", "while", "when", "because", "since",
        "morning", "evening", "night", "day", "today", "yesterday", "tomorrow",
        "home", "work", "school", "office", "hospital", "park", "street",
        "friend", "family", "partner", "colleague", "stranger", "doctor",
        "gym", "track", "field", "court", "pool", "beach", "mountain",
        "restaurant", "kitchen", "bedroom", "living room", "outdoors",
        "classroom", "library", "studio", "stage", "arena", "stadium",
        "road", "highway", "bridge", "forest", "lake", "river", "campsite",
    ])

    if emotion_hits >= 2:
        score += 0.4
        reasons.append(f"Found {emotion_hits} emotion-related words")
    elif emotion_hits == 1:
        score += 0.2
        reasons.append("Found 1 emotion-related word")
    else:
        reasons.append("No emotion-related words found")

    if matched_categories:
        score += 0.35
        reasons.append(f"Matched {len(matched_categories)} scenario category(ies)")
    elif scenario_hits > 0:
        score += 0.15
        reasons.append("Partial scenario match")
    else:
        reasons.append("No scenario keywords matched")

    if has_second_person:
        score += 0.15
        reasons.append("Contains second-person perspective")
    if has_action:
        score += 0.1
        reasons.append("Contains action verbs")
    if has_context:
        score += 0.1
        reasons.append("Contains contextual details")

    word_count = len(text_lower.split())
    if word_count < 4:
        score -= 0.2
        reasons.append("Too few words to describe a situation")
    elif word_count > 3:
        score += 0.1

    if not has_action and not has_second_person and emotion_hits == 0 and scenario_hits == 0:
        score = max(score, 0.0)
        reasons.append("Does not describe an emotional situation")

    score = max(0.0, min(1.0, score))

    category = "invalid"
    if score >= 0.5:
        if matched_categories:
            category = matched_categories[0]
        elif emotion_hits >= 2:
            category = "emotional_experience"
        else:
            category = "mixed"
    elif score >= 0.4:
        category = "borderline"
    else:
        category = "invalid"

    return {
        "valid": score >= 0.5,
        "score": round(score, 3),
        "reasons": reasons,
        "category": category,
        "matched_categories": matched_categories,
    }


def _connect_feedback() -> sqlite3.Connection:
    connection = sqlite3.connect(FEEDBACK_DB)
    connection.row_factory = sqlite3.Row
    connection.execute("""
        CREATE TABLE IF NOT EXISTS feedback (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            features_json TEXT NOT NULL,
            hormone_levels_json TEXT NOT NULL,
            rating REAL DEFAULT 1.0,
            created_at TEXT NOT NULL
        )
    """)
    return connection


def _load_dataset() -> tuple[np.ndarray, dict[str, np.ndarray]]:
    if DATASET_PATH.exists():
        data = np.load(DATASET_PATH)
        features = data["features"]
        labels = {h: data[h] for h in HORMONES}
        return features, labels

    from generate_dataset import generate_dataset
    return generate_dataset(samples_per_type=200, noise_scale=0.04)


def _features_from_text(text: str) -> list[float]:
    text_lower = text.lower()
    scores = {feat: 0.5 for feat in SCENARIO_FEATURES}

    best_match = None
    best_count = 0
    for scenario, keywords in SCENARIO_KEYWORDS.items():
        count = sum(1 for kw in keywords if kw in text_lower)
        if count > best_count:
            best_count = count
            best_match = scenario

    if best_match:
        from generate_dataset import SCENARIO_TYPES
        base = SCENARIO_TYPES[best_match]["base_features"]
        for i, feat in enumerate(SCENARIO_FEATURES):
            scores[feat] = base[feat]

    threat_words = ["danger", "threat", "attack", "spider", "accident", "fire", "afraid", "scared"]
    stress_words = ["deadline", "pressure", "worried", "nervous", "overwhelmed", "exam"]
    reward_words = ["win", "gift", "achieved", "success", "celebrate", "good news"]
    social_words = ["friend", "family", "love", "partner", "wedding", "hug"]
    calm_words = ["meditation", "yoga", "nature", "peaceful", "relax", "calm"]

    if any(w in text_lower for w in threat_words):
        scores["threat"] = 0.9
        scores["stress"] = 0.8
        scores["safety"] = 0.15
    if any(w in text_lower for w in stress_words):
        scores["stress"] = 0.75
        scores["uncertainty"] = 0.7
        scores["time_pressure"] = 0.6
    if any(w in text_lower for w in reward_words):
        scores["reward"] = 0.9
        scores["emotion"] = 0.8
        scores["safety"] = 0.8
    if any(w in text_lower for w in social_words):
        scores["social"] = 0.85
        scores["emotion"] = 0.75
        scores["safety"] = 0.85
    if any(w in text_lower for w in calm_words):
        scores["stress"] = 0.05
        scores["threat"] = 0.02
        scores["safety"] = 0.95

    return [scores[feat] for feat in SCENARIO_FEATURES]


class HormonePredictor:
    def __init__(self):
        self.models: dict[str, Pipeline] = {}
        self.scaler = StandardScaler()
        self.is_trained = False
        self.patterns: dict[str, Any] = {}
        self.feature_importance: dict[str, Any] = {}

    def train(self, features: np.ndarray | None = None, labels: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
        if features is None or labels is None:
            features, labels = _load_dataset()

        with _connect_feedback() as conn:
            rows = conn.execute("SELECT features_json, hormone_levels_json, rating FROM feedback").fetchall()
            for row in rows:
                feat = np.array(json.loads(row["features_json"]))
                levels = json.loads(row["hormone_levels_json"])
                rating = row["rating"]
                features = np.vstack([features, feat.reshape(1, -1)])
                for h in HORMONES:
                    h_levels = np.array(levels[h]) * rating
                    labels[h] = np.vstack([labels[h], h_levels.reshape(1, -1)])

        self.scaler.fit(features)
        X = self.scaler.transform(features)

        results = {}
        for hormone in HORMONES:
            y = labels[hormone]
            model = RandomForestRegressor(
                n_estimators=200, max_depth=15, random_state=42, n_jobs=-1
            )
            model.fit(X, y)
            self.models[hormone] = model

            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:5]
            score = float(model.score(X, y))
            results[hormone] = {
                "r2_score": round(score, 4),
                "top_features": [
                    {"feature": SCENARIO_FEATURES[i], "importance": round(float(importances[i]), 4)}
                    for i in top_indices
                ],
            }
            self.feature_importance[hormone] = {
                SCENARIO_FEATURES[i]: round(float(importances[i]), 4)
                for i in np.argsort(importances)[::-1]
            }

        self._extract_patterns(features, labels)
        self.is_trained = True

        MODEL_DIR.mkdir(parents=True, exist_ok=True)
        joblib.dump({
            "models": self.models,
            "scaler": self.scaler,
            "patterns": self.patterns,
            "feature_importance": self.feature_importance,
        }, HORMONE_MODEL_PATH)

        return {"message": "Model trained on neuroscience dataset", "hormone_results": results, "samples": len(features)}

    def load(self) -> bool:
        if HORMONE_MODEL_PATH.exists():
            saved = joblib.load(HORMONE_MODEL_PATH)
            self.models = saved["models"]
            self.scaler = saved["scaler"]
            self.patterns = saved.get("patterns", {})
            self.feature_importance = saved.get("feature_importance", {})
            self.is_trained = True
            return True
        return False

    def predict_from_text(self, text: str) -> dict[str, Any]:
        analysis = analyze_situation(text)
        if not analysis["valid"]:
            return {
                "error": True,
                "valid": False,
                "message": "This doesn't appear to be an emotional situation. Please describe a scenario involving emotions, feelings, or a personal experience.",
                "analysis": analysis,
            }
        features = _features_from_text(text)
        result = self.predict(features)
        result["situation_analysis"] = analysis
        return result

    def predict_from_emotion(self, emotion: str) -> dict[str, Any]:
        if emotion not in EMOTION_MAP:
            return {
                "error": True,
                "valid": False,
                "message": f"Unknown emotion '{emotion}'. Please select a valid emotion from the Emotion Lab.",
                "valid_emotions": list(EMOTION_MAP.keys()),
            }
        if emotion in EMOTION_HORMONE_PROFILES:
            hormones = EMOTION_HORMONE_PROFILES[emotion]
            features = [EMOTION_MAP[emotion][f] for f in SCENARIO_FEATURES]
            primary = max(hormones, key=lambda h: float(np.mean(hormones[h])))
            stress, threat, reward, social = features[0], features[1], features[2], features[3]
            if threat > 0.6 and stress > 0.6:
                scenario_type = "acute_threat"
            elif stress > 0.6 and threat < 0.3:
                scenario_type = "chronic_stress"
            elif reward > 0.7:
                scenario_type = "reward"
            elif social > 0.7:
                scenario_type = "social_bonding"
            elif stress < 0.15 and threat < 0.1:
                scenario_type = "calm_meditation"
            else:
                scenario_type = "mixed"
            return {
                "hormones": hormones,
                "primary_hormone": primary,
                "pattern": self._match_pattern(features),
                "confidence": self._compute_confidence_from_features(features),
                "scenario_type": scenario_type,
                "phases": PHASE_NAMES,
                "features": {f: round(v, 3) for f, v in zip(SCENARIO_FEATURES, features)},
                "input_emotion": emotion,
            }
        features = [EMOTION_MAP[emotion][f] for f in SCENARIO_FEATURES]
        result = self.predict(features)
        result["input_emotion"] = emotion
        return result

    def predict(self, features: list[float]) -> dict[str, Any]:
        if not self.is_trained:
            if not self.load():
                self.train()

        X = self.scaler.transform(np.array(features).reshape(1, -1))

        curves = {}
        for hormone in HORMONES:
            prediction = self.models[hormone].predict(X)[0]
            curves[hormone] = [float(max(0.05, min(1.0, v))) for v in prediction]

        primary = self._detect_primary(features)
        pattern = self._match_pattern(features)
        confidence = self._compute_confidence(features)
        scenario_type = self._classify_scenario(features)

        return {
            "hormones": curves,
            "primary_hormone": primary,
            "pattern": pattern,
            "confidence": confidence,
            "scenario_type": scenario_type,
            "phases": PHASE_NAMES,
            "features": {f: round(v, 3) for f, v in zip(SCENARIO_FEATURES, features)},
        }

    def add_feedback(self, features: list[float], hormone_levels: dict[str, list[float]], rating: float = 1.0):
        with _connect_feedback() as conn:
            conn.execute(
                "INSERT INTO feedback (features_json, hormone_levels_json, rating, created_at) VALUES (?, ?, ?, ?)",
                (json.dumps(features), json.dumps(hormone_levels), rating,
                 __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat()),
            )

    def _detect_primary(self, features: list[float]) -> str:
        stress, threat, reward, social = features[0], features[1], features[2], features[3]
        scores = {}
        X = self.scaler.transform(np.array(features).reshape(1, -1))
        for hormone in HORMONES:
            model = self.models[hormone]
            pred = model.predict(X)[0]
            scores[hormone] = float(np.mean(pred))

        if threat > 0.6:
            scores["Epinephrine"] *= 1.25
            scores["Norepinephrine"] *= 1.15
        if stress > 0.6:
            scores["Cortisol"] *= 1.25
        if reward > 0.6:
            scores["Dopamine"] *= 1.25
        if social > 0.6:
            scores["Oxytocin"] *= 1.3
        if stress < 0.15 and threat < 0.1:
            scores["GABA"] *= 1.2
            scores["Serotonin"] *= 1.1

        return max(scores, key=scores.get)

    def _match_pattern(self, features: list[float]) -> str:
        stress, threat, reward, social, novelty, safety = features[0], features[1], features[2], features[3], features[4], features[10]
        if threat > 0.7 and stress > 0.6:
            return "Acute threat response: rapid catecholamine spike followed by HPA axis activation"
        if stress > 0.6 and threat < 0.3:
            return "Chronic stress pattern: sustained cortisol with gradual dopamine adaptation"
        if reward > 0.7 and novelty > 0.5:
            return "Reward prediction: dopamine surge with endorphin reinforcement"
        if social > 0.7 and reward > 0.4:
            return "Social bonding: oxytocin-driven with dopamine reward reinforcement"
        if threat > 0.4 and stress > 0.4:
            return "Anticipatory anxiety: sustained norepinephrine with cortisol support"
        if novelty > 0.7:
            return "Orienting response: rapid norepinephrine spike with glutamate-driven learning"
        if stress < 0.15 and threat < 0.1:
            return "Parasympathetic dominance: GABA-mediated calm with serotonin stability"
        if social > 0.5 and stress < 0.3:
            return "Social warmth: oxytocin-endorphin synergy with reduced stress markers"
        return "Context-dependent adaptive response"

    def _classify_scenario(self, features: list[float]) -> str:
        stress, threat, reward, social = features[0], features[1], features[2], features[3]
        if threat > 0.7 and stress > 0.6:
            return "acute_threat"
        if stress > 0.6 and threat < 0.3:
            return "chronic_stress"
        if reward > 0.7:
            return "reward"
        if social > 0.7:
            return "social_bonding"
        if stress < 0.15 and threat < 0.1:
            return "calm_meditation"
        return "mixed"

    def _compute_confidence(self, features: list[float]) -> float:
        variance_scores = []
        X = self.scaler.transform(np.array(features).reshape(1, -1))
        for hormone in HORMONES:
            model = self.models[hormone]
            tree_preds = np.array([tree.predict(X)[0] for tree in model.estimators_])
            tree_stds = np.std(tree_preds, axis=0).mean()
            variance_scores.append(1.0 - min(float(tree_stds), 0.3))
        return round(float(np.mean(variance_scores)), 4)

    def _compute_confidence_from_features(self, features: list[float]) -> float:
        variance = np.std(features)
        confidence = round(float(1.0 - min(variance / 0.5, 0.5)), 4)
        return confidence

    def _extract_patterns(self, features: np.ndarray, labels: dict[str, np.ndarray]):
        self.patterns = {}
        for hormone in HORMONES:
            model = self.models[hormone]
            importances = model.feature_importances_
            top_indices = np.argsort(importances)[::-1][:3]
            avg_curve = np.mean(labels[hormone], axis=0)
            self.patterns[hormone] = {
                "driving_features": [SCENARIO_FEATURES[i] for i in top_indices],
                "avg_curve": avg_curve.tolist(),
                "peak_phase": int(np.argmax(avg_curve)),
                "baseline": float(avg_curve[0]),
                "peak_value": float(np.max(avg_curve)),
            }

    def get_patterns(self) -> dict[str, Any]:
        if not self.patterns and self.is_trained:
            return self.patterns
        if not self.load():
            return {}
        return self.patterns

    def get_info(self) -> dict[str, Any]:
        return {
            "hormones": HORMONES,
            "hormone_info": HORMONE_DISPLAY,
            "features": SCENARIO_FEATURES,
            "phases": PHASE_NAMES,
            "emotions": list(EMOTION_MAP.keys()),
            "scenario_types": list(SCENARIO_KEYWORDS.keys()),
        }


EMOTION_HORMONE_PROFILES = {
    "Joy": {
        "Cortisol": [0.5551, 0.5003, 0.3704, 0.4099, 0.4471, 0.4378, 0.4293, 0.5245, 0.3717, 0.3605, 0.3858, 0.4259, 0.4332, 0.3469, 0.5041, 0.323],
        "Epinephrine": [0.3723, 0.3317, 0.3232, 0.2912, 0.2898, 0.3726, 0.3647, 0.3598, 0.3489, 0.3311, 0.3248, 0.3046, 0.3319, 0.3209, 0.3285, 0.317],
        "Dopamine": [0.15, 0.196, 0.2801, 0.389, 0.518, 0.5563, 0.4909, 0.4254, 0.36, 0.3273, 0.3273, 0.3273, 0.3273, 0.3273, 0.3273, 0.3273],
        "Serotonin": [0.4993, 0.5353, 0.4119, 0.4837, 0.3979, 0.4052, 0.4285, 0.362, 0.264, 0.5243, 0.4254, 0.4892, 0.4038, 0.541, 0.6606, 0.4867],
        "Norepinephrine": [0.15, 0.2054, 0.3068, 0.438, 0.5934, 0.6413, 0.5659, 0.4904, 0.415, 0.3773, 0.3773, 0.3773, 0.3773, 0.3773, 0.3773, 0.3773],
        "Oxytocin": [0.4339, 0.2529, 0.2469, 0.3316, 0.3566, 0.3889, 0.2886, 0.2587, 0.2479, 0.3196, 0.2718, 0.2883, 0.1914, 0.1863, 0.3135, 0.4838],
        "EndogenousOpioids": [0.15, 0.2072, 0.3117, 0.4471, 0.6074, 0.6572, 0.5799, 0.5025, 0.4252, 0.3866, 0.3866, 0.3866, 0.3866, 0.3866, 0.3866, 0.3866],
        "GABA": [0.15, 0.1943, 0.2754, 0.3804, 0.5047, 0.5413, 0.4777, 0.414, 0.3503, 0.3184, 0.3184, 0.3184, 0.3184, 0.3184, 0.3184, 0.3184],
        "Glutamate": [0.381, 0.37, 0.4257, 0.3886, 0.3445, 0.4636, 0.3902, 0.3565, 0.3911, 0.4112, 0.3338, 0.3647, 0.3448, 0.3843, 0.4102, 0.419],
        "Endocannabinoids": [0.15, 0.2061, 0.3088, 0.4417, 0.5991, 0.6477, 0.5715, 0.4953, 0.4191, 0.381, 0.381, 0.381, 0.381, 0.381, 0.381, 0.381],
    },
    "Jealousy": {
        "Cortisol": [0.15, 0.2185, 0.3187, 0.4357, 0.5653, 0.4521, 0.4569, 0.4567, 0.4517, 0.4418, 0.4275, 0.4093, 0.3878, 0.3638, 0.3381, 0.3116],
        "Epinephrine": [0.15, 0.2161, 0.3127, 0.4256, 0.5505, 0.4399, 0.4445, 0.4443, 0.4394, 0.4298, 0.4159, 0.3982, 0.3773, 0.3539, 0.3289, 0.3031],
        "Dopamine": [0.3423, 0.2978, 0.3892, 0.4013, 0.4288, 0.3504, 0.2525, 0.162, 0.3971, 0.3206, 0.2759, 0.4655, 0.4551, 0.357, 0.2723, 0.4803],
        "Serotonin": [0.3483, 0.3433, 0.4226, 0.4694, 0.4206, 0.4966, 0.4329, 0.4467, 0.2619, 0.3473, 0.367, 0.2975, 0.4103, 0.311, 0.4724, 0.4483],
        "Norepinephrine": [0.15, 0.2269, 0.3393, 0.4707, 0.6162, 0.4944, 0.4996, 0.4995, 0.4939, 0.4832, 0.4675, 0.4476, 0.4241, 0.3978, 0.3697, 0.3408],
        "Oxytocin": [0.15, 0.2301, 0.3472, 0.484, 0.6355, 0.5105, 0.5159, 0.5157, 0.51, 0.4989, 0.4827, 0.4622, 0.4379, 0.4108, 0.3817, 0.3518],
        "EndogenousOpioids": [0.3383, 0.4382, 0.4819, 0.3797, 0.4387, 0.3693, 0.3893, 0.3082, 0.2881, 0.2366, 0.3077, 0.3085, 0.2489, 0.1493, 0.3433, 0.3795],
        "GABA": [0.3497, 0.4162, 0.3925, 0.3898, 0.2671, 0.3172, 0.2336, 0.3306, 0.3249, 0.2945, 0.1712, 0.2773, 0.363, 0.2203, 0.2939, 0.2633],
        "Glutamate": [0.15, 0.2181, 0.3176, 0.4339, 0.5627, 0.45, 0.4547, 0.4546, 0.4495, 0.4397, 0.4255, 0.4074, 0.386, 0.3621, 0.3365, 0.3101],
        "Endocannabinoids": [0.4386, 0.3991, 0.4847, 0.3166, 0.267, 0.2747, 0.3702, 0.4111, 0.4444, 0.451, 0.5343, 0.48, 0.3129, 0.3286, 0.3268, 0.4616],
    },
    "Fear": {
        "Cortisol": [0.15, 0.1918, 0.2681, 0.367, 0.484, 0.5181, 0.4571, 0.3962, 0.3352, 0.3048, 0.3048, 0.3048, 0.3048, 0.3048, 0.3048, 0.3048],
        "Epinephrine": [0.15, 0.1973, 0.2837, 0.3957, 0.5283, 0.568, 0.5011, 0.4343, 0.3675, 0.3341, 0.3341, 0.3341, 0.3341, 0.3341, 0.3341, 0.3341],
        "Dopamine": [0.5032, 0.4696, 0.4418, 0.2094, 0.3041, 0.404, 0.4348, 0.4396, 0.3769, 0.5114, 0.5341, 0.4486, 0.4348, 0.4108, 0.2739, 0.4361],
        "Serotonin": [0.6027, 0.3986, 0.483, 0.5324, 0.563, 0.379, 0.5005, 0.2978, 0.3798, 0.4253, 0.3541, 0.3507, 0.2107, 0.3017, 0.4346, 0.4891],
        "Norepinephrine": [0.4791, 0.3219, 0.4074, 0.3922, 0.3926, 0.386, 0.2061, 0.3882, 0.3343, 0.5339, 0.5649, 0.4717, 0.4195, 0.3604, 0.3785, 0.3635],
        "Oxytocin": [0.15, 0.19, 0.2631, 0.3578, 0.4699, 0.5022, 0.4431, 0.384, 0.3249, 0.2954, 0.2954, 0.2954, 0.2954, 0.2954, 0.2954, 0.2954],
        "EndogenousOpioids": [0.3444, 0.331, 0.2801, 0.2688, 0.3313, 0.3814, 0.3805, 0.341, 0.2521, 0.2163, 0.2158, 0.3114, 0.1681, 0.343, 0.3585, 0.212],
        "GABA": [0.4116, 0.3375, 0.4665, 0.417, 0.2358, 0.2692, 0.5095, 0.1943, 0.213, 0.3187, 0.5018, 0.4023, 0.4632, 0.2588, 0.4109, 0.3103],
        "Glutamate": [0.3886, 0.4982, 0.36, 0.3901, 0.347, 0.3986, 0.2767, 0.3407, 0.5262, 0.3519, 0.3789, 0.4702, 0.2835, 0.3118, 0.3261, 0.3624],
        "Endocannabinoids": [0.15, 0.1876, 0.2563, 0.3452, 0.4506, 0.4804, 0.4239, 0.3674, 0.3109, 0.2826, 0.2826, 0.2826, 0.2826, 0.2826, 0.2826, 0.2826],
    },
    "Anger": {
        "Cortisol": [0.2869, 0.5668, 0.4203, 0.4998, 0.4709, 0.3974, 0.475, 0.4166, 0.333, 0.2552, 0.3724, 0.315, 0.3882, 0.3023, 0.3406, 0.199],
        "Epinephrine": [0.15, 0.2049, 0.3053, 0.4354, 0.5893, 0.6368, 0.5619, 0.487, 0.412, 0.3746, 0.3746, 0.3746, 0.3746, 0.3746, 0.3746, 0.3746],
        "Dopamine": [0.3616, 0.4931, 0.3903, 0.4103, 0.413, 0.462, 0.5078, 0.443, 0.3023, 0.2171, 0.3442, 0.3928, 0.3998, 0.459, 0.5195, 0.3501],
        "Serotonin": [0.5667, 0.5369, 0.5021, 0.6737, 0.5674, 0.2678, 0.3797, 0.395, 0.5599, 0.3764, 0.5717, 0.3219, 0.3958, 0.4841, 0.4566, 0.3666],
        "Norepinephrine": [0.3199, 0.2521, 0.3637, 0.4938, 0.357, 0.2795, 0.4226, 0.3363, 0.1448, 0.2314, 0.205, 0.3342, 0.363, 0.1956, 0.3229, 0.4074],
        "Oxytocin": [0.15, 0.2048, 0.3051, 0.435, 0.5888, 0.6361, 0.5613, 0.4865, 0.4116, 0.3742, 0.3742, 0.3742, 0.3742, 0.3742, 0.3742, 0.3742],
        "EndogenousOpioids": [0.4131, 0.1644, 0.5315, 0.2917, 0.4234, 0.4278, 0.4399, 0.3906, 0.411, 0.3675, 0.385, 0.3615, 0.3279, 0.4835, 0.314, 0.5017],
        "GABA": [0.4898, 0.4761, 0.3799, 0.5318, 0.6091, 0.5806, 0.3933, 0.3764, 0.3101, 0.2941, 0.4165, 0.3695, 0.5614, 0.4046, 0.5662, 0.3197],
        "Glutamate": [0.3635, 0.2807, 0.3178, 0.3315, 0.5518, 0.2802, 0.2491, 0.3015, 0.0618, 0.4053, 0.3169, 0.5235, 0.322, 0.3902, 0.3956, 0.468],
        "Endocannabinoids": [0.15, 0.2009, 0.2939, 0.4144, 0.557, 0.6003, 0.5297, 0.4591, 0.3885, 0.3531, 0.3531, 0.3531, 0.3531, 0.3531, 0.3531, 0.3531],
    },
    "Sadness": {
        "Cortisol": [0.15, 0.2336, 0.3421, 0.4626, 0.5914, 0.5177, 0.5161, 0.5009, 0.4732, 0.4349, 0.3888, 0.338, 0.2859, 0.236, 0.1913, 0.1546],
        "Epinephrine": [0.4215, 0.5039, 0.4858, 0.4704, 0.4307, 0.4223, 0.6121, 0.4685, 0.4453, 0.4545, 0.4918, 0.4592, 0.4422, 0.311, 0.4498, 0.4837],
        "Dopamine": [0.7851, 0.8357, 0.8833, 0.7886, 0.6531, 0.7059, 0.6943, 0.6434, 0.6624, 0.5978, 0.5739, 0.6059, 0.4972, 0.4766, 0.5177, 0.5264],
        "Serotonin": [0.334, 0.5627, 0.4316, 0.5938, 0.3813, 0.3955, 0.4039, 0.426, 0.5053, 0.6652, 0.5144, 0.3053, 0.2878, 0.2997, 0.4985, 0.3924],
        "Norepinephrine": [0.5389, 0.4408, 0.4786, 0.6634, 0.5029, 0.2817, 0.4776, 0.4008, 0.3713, 0.3463, 0.51, 0.4895, 0.5292, 0.6434, 0.4718, 0.5512],
        "Oxytocin": [0.5195, 0.2585, 0.4069, 0.3752, 0.5376, 0.4431, 0.1931, 0.5264, 0.3336, 0.1997, 0.2439, 0.3238, 0.3682, 0.4838, 0.519, 0.2834],
        "EndogenousOpioids": [0.4464, 0.5567, 0.5862, 0.7067, 0.3738, 0.3548, 0.3429, 0.4695, 0.3731, 0.5613, 0.5025, 0.3712, 0.438, 0.4025, 0.4383, 0.3987],
        "GABA": [0.3646, 0.3315, 0.5048, 0.4185, 0.5574, 0.3557, 0.2911, 0.3832, 0.2896, 0.4303, 0.404, 0.4195, 0.3553, 0.2377, 0.4295, 0.3975],
        "Glutamate": [0.1462, 0.269, 0.2096, 0.3451, 0.2298, 0.05, 0.219, 0.2897, 0.3333, 0.05, 0.4421, 0.3865, 0.3747, 0.4325, 0.1786, 0.2643],
        "Endocannabinoids": [0.3727, 0.2688, 0.3809, 0.393, 0.2017, 0.2275, 0.2764, 0.224, 0.3806, 0.309, 0.3797, 0.4039, 0.3121, 0.3351, 0.514, 0.2657],
    },
    "Surprise": {
        "Cortisol": [0.15, 0.2379, 0.352, 0.4786, 0.6141, 0.5382, 0.5366, 0.5207, 0.4919, 0.4521, 0.4042, 0.3514, 0.2972, 0.2453, 0.1989, 0.1607],
        "Epinephrine": [0.15, 0.1924, 0.27, 0.3704, 0.4894, 0.5241, 0.4624, 0.4008, 0.3391, 0.3083, 0.3083, 0.3083, 0.3083, 0.3083, 0.3083, 0.3083],
        "Dopamine": [0.15, 0.2391, 0.3546, 0.4828, 0.62, 0.5437, 0.542, 0.526, 0.4968, 0.4567, 0.4082, 0.3549, 0.3002, 0.2478, 0.2009, 0.1623],
        "Serotonin": [0.3069, 0.3619, 0.3898, 0.4527, 0.401, 0.448, 0.3287, 0.3402, 0.278, 0.2066, 0.2493, 0.3329, 0.205, 0.2823, 0.2668, 0.3019],
        "Norepinephrine": [0.15, 0.1968, 0.2823, 0.3931, 0.5242, 0.5634, 0.4971, 0.4308, 0.3645, 0.3314, 0.3314, 0.3314, 0.3314, 0.3314, 0.3314, 0.3314],
        "Oxytocin": [0.2162, 0.2448, 0.3919, 0.2785, 0.3845, 0.2785, 0.3371, 0.2813, 0.3834, 0.395, 0.3595, 0.4044, 0.2827, 0.2774, 0.4131, 0.462],
        "EndogenousOpioids": [0.15, 0.2407, 0.3585, 0.4891, 0.629, 0.5517, 0.55, 0.5338, 0.5042, 0.4634, 0.4143, 0.3602, 0.3047, 0.2515, 0.2039, 0.1647],
        "GABA": [0.3036, 0.3617, 0.4964, 0.4214, 0.4033, 0.318, 0.2557, 0.3026, 0.3203, 0.5362, 0.4221, 0.5227, 0.3483, 0.3382, 0.3682, 0.5085],
        "Glutamate": [0.15, 0.2352, 0.3458, 0.4685, 0.5998, 0.5253, 0.5237, 0.5082, 0.48, 0.4412, 0.3944, 0.3429, 0.2901, 0.2394, 0.1941, 0.1568],
        "Endocannabinoids": [0.1442, 0.3033, 0.3404, 0.3517, 0.4442, 0.3342, 0.3105, 0.1665, 0.2598, 0.2417, 0.3911, 0.4196, 0.3161, 0.3279, 0.3203, 0.3668],
    },
    "Disgust": {
        "Cortisol": [0.307, 0.3997, 0.4072, 0.3244, 0.5314, 0.4266, 0.1362, 0.3439, 0.2748, 0.355, 0.3837, 0.5233, 0.4472, 0.4448, 0.3925, 0.4356],
        "Epinephrine": [0.3703, 0.5388, 0.424, 0.5193, 0.5139, 0.4192, 0.363, 0.3067, 0.3821, 0.427, 0.3602, 0.4092, 0.5564, 0.4157, 0.518, 0.4623],
        "Dopamine": [0.2833, 0.4905, 0.3712, 0.4231, 0.4634, 0.274, 0.5338, 0.2611, 0.1662, 0.3431, 0.1843, 0.2736, 0.2734, 0.3067, 0.3356, 0.3091],
        "Serotonin": [0.3795, 0.6168, 0.4052, 0.5244, 0.4315, 0.3033, 0.2397, 0.2621, 0.2056, 0.4446, 0.3475, 0.3755, 0.3709, 0.4099, 0.3415, 0.2782],
        "Norepinephrine": [0.2691, 0.4016, 0.3818, 0.5324, 0.3917, 0.3324, 0.3628, 0.2838, 0.2831, 0.5192, 0.4131, 0.4207, 0.4215, 0.4429, 0.4639, 0.2162],
        "Oxytocin": [0.6212, 0.4015, 0.4456, 0.4604, 0.5313, 0.484, 0.4322, 0.4236, 0.3835, 0.3485, 0.3322, 0.3849, 0.3928, 0.4473, 0.5591, 0.3814],
        "EndogenousOpioids": [0.3997, 0.264, 0.3423, 0.6053, 0.5781, 0.3554, 0.3986, 0.369, 0.3084, 0.4222, 0.443, 0.4561, 0.3125, 0.4001, 0.3502, 0.3466],
        "GABA": [0.4202, 0.2727, 0.4994, 0.3871, 0.4028, 0.4155, 0.4098, 0.3688, 0.3376, 0.3749, 0.389, 0.256, 0.3065, 0.2733, 0.3759, 0.3765],
        "Glutamate": [0.2316, 0.4828, 0.5538, 0.3109, 0.1768, 0.283, 0.2225, 0.262, 0.3136, 0.3362, 0.4514, 0.3982, 0.3215, 0.3277, 0.4133, 0.1876],
        "Endocannabinoids": [0.421, 0.3256, 0.313, 0.3335, 0.4728, 0.2519, 0.2835, 0.407, 0.3528, 0.3625, 0.4154, 0.4335, 0.2726, 0.356, 0.3465, 0.3824],
    },
    "Love": {
        "Cortisol": [0.262, 0.2856, 0.3624, 0.2852, 0.4347, 0.3065, 0.3524, 0.4179, 0.2756, 0.3131, 0.2269, 0.2241, 0.1304, 0.1646, 0.3286, 0.2603],
        "Epinephrine": [0.4757, 0.4848, 0.4868, 0.528, 0.3657, 0.2485, 0.2709, 0.4853, 0.4574, 0.4559, 0.4731, 0.6238, 0.4147, 0.4042, 0.4197, 0.5256],
        "Dopamine": [0.15, 0.2011, 0.2944, 0.4153, 0.5585, 0.602, 0.5312, 0.4604, 0.3895, 0.3541, 0.3541, 0.3541, 0.3541, 0.3541, 0.3541, 0.3541],
        "Serotonin": [0.4034, 0.4166, 0.3545, 0.4368, 0.5502, 0.475, 0.4961, 0.4436, 0.4157, 0.4708, 0.328, 0.3078, 0.4158, 0.1624, 0.5151, 0.4537],
        "Norepinephrine": [0.3421, 0.3434, 0.4574, 0.5472, 0.5347, 0.4379, 0.5827, 0.3642, 0.3316, 0.5289, 0.3436, 0.434, 0.4, 0.4786, 0.2491, 0.4497],
        "Oxytocin": [0.15, 0.2293, 0.3322, 0.4464, 0.5686, 0.497, 0.4955, 0.4809, 0.4542, 0.4175, 0.3732, 0.3245, 0.2745, 0.2265, 0.1836, 0.1484],
        "EndogenousOpioids": [0.15, 0.1974, 0.2839, 0.3961, 0.5288, 0.5686, 0.5017, 0.4348, 0.3679, 0.3345, 0.3345, 0.3345, 0.3345, 0.3345, 0.3345, 0.3345],
        "GABA": [0.4486, 0.532, 0.4252, 0.472, 0.5478, 0.414, 0.3477, 0.5961, 0.29, 0.3177, 0.4093, 0.3794, 0.3971, 0.3712, 0.3675, 0.4137],
        "Glutamate": [0.3469, 0.2009, 0.3515, 0.4741, 0.2964, 0.2567, 0.3325, 0.3298, 0.2538, 0.3586, 0.4246, 0.2722, 0.3901, 0.2551, 0.3406, 0.3435],
        "Endocannabinoids": [0.3304, 0.3169, 0.2212, 0.355, 0.3552, 0.4423, 0.3939, 0.4157, 0.5004, 0.2386, 0.3707, 0.4531, 0.2425, 0.4164, 0.6143, 0.3732],
    },
    "Anxiety": {
        "Cortisol": [0.15, 0.1881, 0.2577, 0.3479, 0.4547, 0.485, 0.428, 0.3709, 0.3138, 0.2853, 0.2853, 0.2853, 0.2853, 0.2853, 0.2853, 0.2853],
        "Epinephrine": [0.15, 0.241, 0.359, 0.49, 0.6302, 0.5529, 0.5512, 0.5349, 0.5053, 0.4644, 0.4152, 0.3609, 0.3053, 0.252, 0.2043, 0.1651],
        "Dopamine": [0.5071, 0.2376, 0.4477, 0.4671, 0.3856, 0.4321, 0.4285, 0.2869, 0.3747, 0.5025, 0.4275, 0.4003, 0.5063, 0.3977, 0.4464, 0.4373],
        "Serotonin": [0.4046, 0.3713, 0.4562, 0.6225, 0.4742, 0.6171, 0.4386, 0.4332, 0.4221, 0.2362, 0.1247, 0.3085, 0.2798, 0.4262, 0.4652, 0.3398],
        "Norepinephrine": [0.3354, 0.5796, 0.5157, 0.3625, 0.5098, 0.3471, 0.3431, 0.272, 0.2013, 0.371, 0.4459, 0.4237, 0.425, 0.4271, 0.3277, 0.3772],
        "Oxytocin": [0.15, 0.2002, 0.2921, 0.4111, 0.5519, 0.5946, 0.5247, 0.4547, 0.3848, 0.3498, 0.3498, 0.3498, 0.3498, 0.3498, 0.3498, 0.3498],
        "EndogenousOpioids": [0.4101, 0.4526, 0.3831, 0.2943, 0.3804, 0.358, 0.3491, 0.2724, 0.3223, 0.2242, 0.2225, 0.3619, 0.3908, 0.3255, 0.3976, 0.3826],
        "GABA": [0.1689, 0.1946, 0.4163, 0.3844, 0.3981, 0.3312, 0.2821, 0.3633, 0.2249, 0.1995, 0.1475, 0.3169, 0.424, 0.2692, 0.2995, 0.2694],
        "Glutamate": [0.3303, 0.3725, 0.5504, 0.3958, 0.5164, 0.4469, 0.4679, 0.3442, 0.367, 0.308, 0.479, 0.3008, 0.39, 0.3135, 0.364, 0.365],
        "Endocannabinoids": [0.15, 0.2298, 0.3334, 0.4484, 0.5714, 0.4996, 0.498, 0.4833, 0.4566, 0.4196, 0.3752, 0.3261, 0.2759, 0.2277, 0.1846, 0.1491],
    },
    "Excitement": {
        "Cortisol": [0.15, 0.2486, 0.3764, 0.5183, 0.6702, 0.5891, 0.5873, 0.5699, 0.5384, 0.4948, 0.4424, 0.3846, 0.3253, 0.2685, 0.2177, 0.1759],
        "Epinephrine": [0.15, 0.2011, 0.2946, 0.4157, 0.559, 0.6026, 0.5317, 0.4608, 0.3899, 0.3545, 0.3545, 0.3545, 0.3545, 0.3545, 0.3545, 0.3545],
        "Dopamine": [0.15, 0.1962, 0.2806, 0.3899, 0.5194, 0.5579, 0.4923, 0.4267, 0.361, 0.3282, 0.3282, 0.3282, 0.3282, 0.3282, 0.3282, 0.3282],
        "Serotonin": [0.3901, 0.5655, 0.5642, 0.3713, 0.6763, 0.4154, 0.5027, 0.6253, 0.3852, 0.4029, 0.5919, 0.2494, 0.471, 0.4361, 0.4575, 0.4555],
        "Norepinephrine": [0.15, 0.2023, 0.2979, 0.4217, 0.5684, 0.6131, 0.541, 0.4689, 0.3967, 0.3607, 0.3607, 0.3607, 0.3607, 0.3607, 0.3607, 0.3607],
        "Oxytocin": [0.4445, 0.2061, 0.3535, 0.2547, 0.4327, 0.417, 0.3048, 0.4622, 0.4932, 0.4165, 0.2265, 0.516, 0.1868, 0.368, 0.3969, 0.3307],
        "EndogenousOpioids": [0.15, 0.2307, 0.3355, 0.4517, 0.5761, 0.5039, 0.5023, 0.4875, 0.4605, 0.4232, 0.3784, 0.3289, 0.2783, 0.2297, 0.1862, 0.1504],
        "GABA": [0.15, 0.2035, 0.3014, 0.4282, 0.5784, 0.6244, 0.5509, 0.4775, 0.404, 0.3673, 0.3673, 0.3673, 0.3673, 0.3673, 0.3673, 0.3673],
        "Glutamate": [0.3323, 0.5294, 0.5239, 0.6168, 0.39, 0.3769, 0.3729, 0.5299, 0.3203, 0.5332, 0.2505, 0.5272, 0.3815, 0.3054, 0.5007, 0.2283],
        "Endocannabinoids": [0.15, 0.225, 0.3224, 0.4304, 0.546, 0.4766, 0.4751, 0.4611, 0.4355, 0.4003, 0.3579, 0.3111, 0.2632, 0.2172, 0.1761, 0.1423],
    },
    "Calm": {
        "Cortisol": [0.8381, 0.8106, 0.783, 0.7555, 0.728, 0.7005, 0.673, 0.6454, 0.6179, 0.5904, 0.5629, 0.5353, 0.5078, 0.4803, 0.4528, 0.4252],
        "Epinephrine": [0.943, 0.9113, 0.8795, 0.8478, 0.8161, 0.7844, 0.7527, 0.7209, 0.6892, 0.6575, 0.6258, 0.5941, 0.5623, 0.5306, 0.4989, 0.4672],
        "Dopamine": [0.528, 0.5653, 0.5084, 0.4157, 0.4612, 0.5633, 0.4697, 0.3704, 0.3506, 0.4791, 0.4591, 0.3922, 0.5306, 0.4556, 0.4998, 0.529],
        "Serotonin": [0.2405, 0.377, 0.3269, 0.2963, 0.3978, 0.3918, 0.4444, 0.303, 0.4511, 0.2521, 0.2516, 0.2717, 0.1794, 0.3908, 0.4378, 0.3038],
        "Norepinephrine": [0.9419, 0.9102, 0.8785, 0.8468, 0.8152, 0.7835, 0.7518, 0.7201, 0.6885, 0.6568, 0.6251, 0.5934, 0.5618, 0.5301, 0.4984, 0.4667],
        "Oxytocin": [0.15, 0.2462, 0.371, 0.5095, 0.6577, 0.5778, 0.576, 0.559, 0.528, 0.4853, 0.4339, 0.3772, 0.3191, 0.2633, 0.2135, 0.1725],
        "EndogenousOpioids": [0.15, 0.1891, 0.2607, 0.3534, 0.4632, 0.4946, 0.4364, 0.3782, 0.3201, 0.291, 0.291, 0.291, 0.291, 0.291, 0.291, 0.291],
        "GABA": [0.2, 0.2589, 0.345, 0.4457, 0.5572, 0.606, 0.5854, 0.5649, 0.5444, 0.5238, 0.4976, 0.4714, 0.4452, 0.4191, 0.3929, 0.3667],
        "Glutamate": [0.3981, 0.5761, 0.4649, 0.4824, 0.5247, 0.541, 0.5392, 0.3723, 0.2738, 0.2931, 0.4486, 0.2658, 0.3098, 0.3253, 0.3406, 0.3776],
        "Endocannabinoids": [0.15, 0.2483, 0.3759, 0.5175, 0.669, 0.588, 0.5862, 0.5689, 0.5374, 0.4939, 0.4416, 0.3839, 0.3247, 0.268, 0.2173, 0.1756],
    },
    "Guilt": {
        "Cortisol": [0.15, 0.2447, 0.3675, 0.5037, 0.6496, 0.5704, 0.5686, 0.5519, 0.5213, 0.4791, 0.4283, 0.3724, 0.315, 0.26, 0.2108, 0.1703],
        "Epinephrine": [0.4163, 0.3617, 0.4659, 0.3685, 0.4063, 0.2315, 0.3707, 0.2974, 0.333, 0.4082, 0.3114, 0.1854, 0.243, 0.2808, 0.05, 0.4042],
        "Dopamine": [0.4798, 0.5124, 0.3799, 0.4073, 0.3463, 0.4699, 0.5161, 0.4552, 0.3844, 0.5279, 0.4044, 0.3808, 0.5403, 0.4062, 0.42, 0.3792],
        "Serotonin": [0.4836, 0.3845, 0.55, 0.6209, 0.5432, 0.281, 0.4739, 0.3849, 0.3298, 0.3773, 0.4421, 0.3005, 0.5839, 0.4305, 0.5051, 0.5032],
        "Norepinephrine": [0.4546, 0.4982, 0.5369, 0.477, 0.4616, 0.3654, 0.4113, 0.5717, 0.3619, 0.3183, 0.3649, 0.4592, 0.3559, 0.448, 0.4643, 0.5885],
        "Oxytocin": [0.2713, 0.3234, 0.6739, 0.465, 0.4677, 0.4235, 0.2765, 0.2776, 0.2158, 0.2832, 0.3934, 0.4798, 0.5045, 0.2967, 0.2979, 0.3811],
        "EndogenousOpioids": [0.3829, 0.2904, 0.5007, 0.3238, 0.29, 0.3869, 0.3195, 0.3997, 0.4114, 0.1902, 0.2175, 0.2823, 0.2575, 0.2312, 0.2485, 0.2414],
        "GABA": [0.3928, 0.4102, 0.4742, 0.3557, 0.2876, 0.2357, 0.2825, 0.399, 0.3989, 0.5669, 0.495, 0.3734, 0.4694, 0.3954, 0.3367, 0.2358],
        "Glutamate": [0.3903, 0.2921, 0.3433, 0.5266, 0.4992, 0.4868, 0.4243, 0.3151, 0.4165, 0.3711, 0.246, 0.3022, 0.3922, 0.548, 0.4632, 0.4238],
        "Endocannabinoids": [0.3982, 0.5369, 0.5936, 0.4779, 0.4957, 0.3643, 0.5411, 0.368, 0.3568, 0.3116, 0.3591, 0.5169, 0.3807, 0.4438, 0.4907, 0.3292],
    },
    "Pride": {
        "Cortisol": [0.4694, 0.5595, 0.5296, 0.5888, 0.5256, 0.6234, 0.4528, 0.4456, 0.4836, 0.3165, 0.4244, 0.2874, 0.2759, 0.3677, 0.2564, 0.4003],
        "Epinephrine": [0.4928, 0.53, 0.4927, 0.466, 0.3269, 0.3363, 0.3801, 0.2664, 0.4274, 0.5417, 0.2851, 0.5028, 0.4923, 0.4073, 0.3868, 0.5356],
        "Dopamine": [0.3057, 0.612, 0.4194, 0.4426, 0.4908, 0.2666, 0.3405, 0.4439, 0.3238, 0.273, 0.3911, 0.3116, 0.4772, 0.5992, 0.4451, 0.4748],
        "Serotonin": [0.2, 0.2518, 0.3274, 0.4159, 0.5138, 0.5563, 0.5374, 0.5185, 0.4997, 0.4808, 0.4568, 0.4327, 0.4087, 0.3847, 0.3606, 0.3366],
        "Norepinephrine": [0.3349, 0.5392, 0.3452, 0.5261, 0.4615, 0.4098, 0.5463, 0.4887, 0.3586, 0.318, 0.342, 0.4762, 0.3711, 0.4275, 0.4389, 0.3717],
        "Oxytocin": [0.2748, 0.3574, 0.4987, 0.4118, 0.3374, 0.3643, 0.349, 0.2555, 0.4109, 0.3401, 0.2765, 0.4865, 0.4432, 0.4885, 0.2822, 0.314],
        "EndogenousOpioids": [0.3605, 0.3184, 0.2473, 0.2422, 0.3122, 0.3438, 0.4126, 0.2177, 0.308, 0.2077, 0.2898, 0.3361, 0.2227, 0.2449, 0.2463, 0.3578],
        "GABA": [0.4843, 0.5026, 0.5178, 0.5227, 0.2463, 0.5805, 0.3957, 0.4352, 0.3583, 0.3195, 0.1918, 0.3445, 0.3312, 0.3213, 0.3225, 0.3326],
        "Glutamate": [0.4845, 0.3395, 0.456, 0.3444, 0.4461, 0.4722, 0.3607, 0.3975, 0.4219, 0.4131, 0.367, 0.4203, 0.3174, 0.3786, 0.4178, 0.4528],
        "Endocannabinoids": [0.5857, 0.3971, 0.4226, 0.4969, 0.4198, 0.3753, 0.2741, 0.3341, 0.297, 0.1787, 0.4396, 0.4445, 0.3729, 0.4295, 0.3681, 0.5572],
    },
    "Shame": {
        "Cortisol": [0.15, 0.2274, 0.3278, 0.4392, 0.5584, 0.4878, 0.4863, 0.472, 0.4458, 0.4098, 0.3663, 0.3185, 0.2694, 0.2223, 0.1802, 0.1456],
        "Epinephrine": [0.4676, 0.3619, 0.2681, 0.3888, 0.242, 0.2107, 0.2337, 0.2276, 0.3323, 0.3899, 0.3069, 0.3261, 0.4532, 0.3407, 0.1897, 0.3228],
        "Dopamine": [0.3524, 0.3055, 0.4249, 0.4601, 0.2805, 0.4126, 0.4711, 0.3881, 0.355, 0.2881, 0.3834, 0.2038, 0.2344, 0.3392, 0.3149, 0.3428],
        "Serotonin": [0.3502, 0.4989, 0.3657, 0.446, 0.5139, 0.4009, 0.4224, 0.4185, 0.5066, 0.3336, 0.3136, 0.3652, 0.3896, 0.3118, 0.3808, 0.4421],
        "Norepinephrine": [0.487, 0.3765, 0.6303, 0.4276, 0.5628, 0.374, 0.4337, 0.3853, 0.3674, 0.3879, 0.4243, 0.4434, 0.5938, 0.4364, 0.5676, 0.3705],
        "Oxytocin": [0.4452, 0.5449, 0.4426, 0.5365, 0.5002, 0.4042, 0.4518, 0.4447, 0.3489, 0.3216, 0.4092, 0.4277, 0.4841, 0.6033, 0.5013, 0.4423],
        "EndogenousOpioids": [0.439, 0.3822, 0.497, 0.4704, 0.3527, 0.4188, 0.5111, 0.5008, 0.3274, 0.5016, 0.2562, 0.5147, 0.2172, 0.2784, 0.5118, 0.4196],
        "GABA": [0.5041, 0.4176, 0.4241, 0.4324, 0.3629, 0.3333, 0.3697, 0.1631, 0.3815, 0.5295, 0.3834, 0.605, 0.3472, 0.4607, 0.2577, 0.4266],
        "Glutamate": [0.5238, 0.5083, 0.6066, 0.491, 0.4793, 0.4744, 0.6163, 0.4382, 0.3521, 0.4175, 0.5191, 0.3778, 0.4494, 0.3544, 0.4921, 0.3363],
        "Endocannabinoids": [0.5313, 0.52, 0.3109, 0.4629, 0.4938, 0.4424, 0.4709, 0.4127, 0.2075, 0.3593, 0.2837, 0.4634, 0.4643, 0.4908, 0.4812, 0.3979],
    },
    "Gratitude": {
        "Cortisol": [0.6828, 0.7045, 0.6787, 0.7101, 0.6147, 0.6065, 0.7059, 0.6213, 0.6577, 0.5791, 0.5356, 0.5336, 0.5269, 0.541, 0.4956, 0.5168],
        "Epinephrine": [0.7748, 0.7842, 0.7533, 0.7311, 0.75, 0.641, 0.7819, 0.6869, 0.6579, 0.6091, 0.5636, 0.5402, 0.6046, 0.4696, 0.5589, 0.4995],
        "Dopamine": [0.2, 0.2565, 0.3391, 0.4357, 0.5426, 0.5893, 0.5693, 0.5493, 0.5294, 0.5094, 0.4839, 0.4584, 0.433, 0.4075, 0.382, 0.3566],
        "Serotonin": [0.2, 0.2585, 0.3439, 0.4438, 0.5544, 0.6028, 0.5824, 0.5619, 0.5415, 0.5211, 0.495, 0.469, 0.4429, 0.4169, 0.3908, 0.3647],
        "Norepinephrine": [0.2, 0.2589, 0.345, 0.4457, 0.5571, 0.606, 0.5854, 0.5649, 0.5443, 0.5238, 0.4976, 0.4714, 0.4452, 0.419, 0.3929, 0.3667],
        "Oxytocin": [0.7951, 0.9197, 0.9183, 0.7916, 0.7014, 0.7762, 0.7076, 0.7648, 0.5973, 0.7901, 0.6229, 0.6826, 0.6042, 0.5739, 0.5401, 0.4894],
        "EndogenousOpioids": [0.2, 0.2539, 0.3327, 0.4248, 0.5268, 0.5712, 0.5518, 0.5325, 0.5131, 0.4937, 0.4691, 0.4444, 0.4197, 0.395, 0.3703, 0.3456],
        "GABA": [0.2, 0.2567, 0.3396, 0.4365, 0.5438, 0.5907, 0.5707, 0.5506, 0.5306, 0.5106, 0.4851, 0.4595, 0.434, 0.4085, 0.3829, 0.3574],
        "Glutamate": [0.2187, 0.5067, 0.5103, 0.5416, 0.5358, 0.1891, 0.3629, 0.4472, 0.1995, 0.3602, 0.4502, 0.3556, 0.3811, 0.2568, 0.2529, 0.4989],
        "Endocannabinoids": [0.2, 0.26, 0.3476, 0.4501, 0.5635, 0.6133, 0.5925, 0.5717, 0.5509, 0.5301, 0.5036, 0.4771, 0.4506, 0.4241, 0.3976, 0.3711],
    },
    "Hope": {
        "Cortisol": [0.3647, 0.4895, 0.4636, 0.3928, 0.3047, 0.4487, 0.2252, 0.1985, 0.235, 0.1385, 0.2739, 0.4314, 0.3408, 0.5155, 0.3294, 0.1623],
        "Epinephrine": [0.374, 0.5, 0.3713, 0.5407, 0.4052, 0.3416, 0.3597, 0.2965, 0.4283, 0.5067, 0.5165, 0.4773, 0.5049, 0.337, 0.4178, 0.4656],
        "Dopamine": [0.4074, 0.4254, 0.6445, 0.3343, 0.3325, 0.2914, 0.3339, 0.443, 0.4354, 0.38, 0.4395, 0.4446, 0.3951, 0.4519, 0.462, 0.3416],
        "Serotonin": [0.2, 0.2505, 0.3243, 0.4106, 0.5062, 0.5475, 0.529, 0.5104, 0.4918, 0.4733, 0.4496, 0.426, 0.4023, 0.3786, 0.355, 0.3313],
        "Norepinephrine": [0.4666, 0.4258, 0.6132, 0.4234, 0.5173, 0.3396, 0.246, 0.3762, 0.4762, 0.3337, 0.3227, 0.483, 0.3735, 0.5505, 0.4849, 0.1627],
        "Oxytocin": [0.3494, 0.423, 0.4237, 0.5047, 0.5102, 0.4538, 0.5731, 0.3729, 0.3724, 0.2765, 0.3535, 0.1383, 0.2985, 0.2476, 0.3522, 0.2456],
        "EndogenousOpioids": [0.5044, 0.4679, 0.4258, 0.5222, 0.438, 0.2814, 0.3774, 0.4524, 0.4821, 0.5383, 0.5927, 0.3119, 0.3131, 0.3396, 0.4828, 0.4148],
        "GABA": [0.2959, 0.2689, 0.3623, 0.4339, 0.3766, 0.2909, 0.4222, 0.1684, 0.4313, 0.3836, 0.4713, 0.1973, 0.3118, 0.335, 0.3081, 0.4336],
        "Glutamate": [0.2938, 0.2878, 0.4843, 0.4424, 0.3786, 0.3814, 0.3565, 0.3987, 0.3727, 0.4116, 0.2941, 0.3883, 0.2315, 0.3141, 0.3396, 0.4359],
        "Endocannabinoids": [0.4177, 0.4164, 0.4312, 0.6216, 0.4205, 0.4694, 0.5232, 0.3455, 0.4549, 0.471, 0.2597, 0.0942, 0.2958, 0.2255, 0.2102, 0.3991],
    },
    "Loneliness": {
        "Cortisol": [0.2, 0.2584, 0.3437, 0.4434, 0.5538, 0.6022, 0.5817, 0.5613, 0.5409, 0.5205, 0.4945, 0.4685, 0.4424, 0.4164, 0.3904, 0.3644],
        "Epinephrine": [0.3611, 0.312, 0.3861, 0.5157, 0.4466, 0.396, 0.3785, 0.4565, 0.3462, 0.2367, 0.3451, 0.291, 0.3031, 0.3417, 0.2106, 0.2842],
        "Dopamine": [0.5033, 0.482, 0.3826, 0.4396, 0.5422, 0.4416, 0.4535, 0.5131, 0.5228, 0.3603, 0.3663, 0.2476, 0.3118, 0.282, 0.4956, 0.272],
        "Serotonin": [0.3396, 0.4279, 0.4051, 0.3982, 0.5165, 0.3835, 0.18, 0.4165, 0.5168, 0.2948, 0.3897, 0.3633, 0.2011, 0.3297, 0.3496, 0.3748],
        "Norepinephrine": [0.3926, 0.4598, 0.4499, 0.4752, 0.4105, 0.4306, 0.3948, 0.4376, 0.297, 0.5105, 0.4823, 0.2742, 0.3119, 0.5168, 0.5024, 0.3655],
        "Oxytocin": [0.3572, 0.5228, 0.5233, 0.2337, 0.3832, 0.3276, 0.3714, 0.36, 0.3514, 0.3308, 0.4401, 0.434, 0.4162, 0.3327, 0.4409, 0.3861],
        "EndogenousOpioids": [0.2434, 0.2085, 0.4039, 0.3027, 0.2228, 0.2586, 0.3672, 0.1544, 0.2924, 0.216, 0.3971, 0.3731, 0.3292, 0.3762, 0.1337, 0.2201],
        "GABA": [0.4958, 0.5187, 0.524, 0.6093, 0.4437, 0.4852, 0.5577, 0.3556, 0.3398, 0.4473, 0.3562, 0.2956, 0.2783, 0.3646, 0.5467, 0.4752],
        "Glutamate": [0.4066, 0.4577, 0.5495, 0.4163, 0.398, 0.4197, 0.2414, 0.3761, 0.3214, 0.4688, 0.4103, 0.5039, 0.4411, 0.4937, 0.32, 0.2927],
        "Endocannabinoids": [0.4768, 0.5642, 0.6443, 0.4408, 0.4811, 0.4712, 0.379, 0.3901, 0.5642, 0.5821, 0.527, 0.391, 0.4829, 0.1842, 0.4468, 0.5425],
    },
    "Frustration": {
        "Cortisol": [0.15, 0.2484, 0.376, 0.5177, 0.6693, 0.5882, 0.5864, 0.5691, 0.5376, 0.4941, 0.4417, 0.384, 0.3249, 0.2681, 0.2173, 0.1756],
        "Epinephrine": [0.15, 0.2484, 0.376, 0.5177, 0.6693, 0.5883, 0.5865, 0.5691, 0.5376, 0.4941, 0.4417, 0.384, 0.3249, 0.2681, 0.2174, 0.1756],
        "Dopamine": [0.15, 0.2319, 0.3381, 0.4559, 0.582, 0.5092, 0.5077, 0.4927, 0.4654, 0.4278, 0.3824, 0.3324, 0.2812, 0.2321, 0.1882, 0.152],
        "Serotonin": [0.3097, 0.3414, 0.4014, 0.4023, 0.4405, 0.1802, 0.2038, 0.3546, 0.2912, 0.5003, 0.4737, 0.2317, 0.2809, 0.389, 0.1334, 0.4039],
        "Norepinephrine": [0.539, 0.4602, 0.6617, 0.5839, 0.3727, 0.4086, 0.375, 0.3173, 0.4264, 0.324, 0.4862, 0.5497, 0.4913, 0.436, 0.4294, 0.4502],
        "Oxytocin": [0.15, 0.1953, 0.2781, 0.3854, 0.5124, 0.5501, 0.4853, 0.4206, 0.3559, 0.3236, 0.3236, 0.3236, 0.3236, 0.3236, 0.3236, 0.3236],
        "EndogenousOpioids": [0.382, 0.4802, 0.4651, 0.4113, 0.45, 0.3314, 0.2987, 0.2703, 0.4018, 0.4165, 0.4403, 0.4602, 0.5094, 0.5146, 0.1551, 0.335],
        "GABA": [0.5706, 0.4375, 0.4295, 0.4742, 0.3728, 0.2851, 0.3093, 0.377, 0.2935, 0.5743, 0.5163, 0.5142, 0.4361, 0.2867, 0.3245, 0.6381],
        "Glutamate": [0.3811, 0.4331, 0.3837, 0.4743, 0.4488, 0.3828, 0.3512, 0.3042, 0.447, 0.4193, 0.4908, 0.3714, 0.2064, 0.4224, 0.5683, 0.433],
        "Endocannabinoids": [0.15, 0.2393, 0.3551, 0.4836, 0.6212, 0.5447, 0.543, 0.527, 0.4978, 0.4575, 0.409, 0.3556, 0.3008, 0.2483, 0.2013, 0.1626],
    },
    "Relief": {
        "Cortisol": [0.936, 0.9046, 0.8732, 0.8417, 0.8103, 0.7788, 0.7474, 0.7159, 0.6845, 0.6531, 0.6216, 0.5902, 0.5587, 0.5273, 0.4959, 0.4644],
        "Epinephrine": [0.7685, 0.7438, 0.719, 0.6943, 0.6696, 0.6448, 0.6201, 0.5953, 0.5706, 0.5459, 0.5211, 0.4964, 0.4716, 0.4469, 0.4221, 0.3974],
        "Dopamine": [0.15, 0.2309, 0.3359, 0.4524, 0.5771, 0.5048, 0.5032, 0.4884, 0.4613, 0.424, 0.379, 0.3295, 0.2788, 0.2301, 0.1865, 0.1507],
        "Serotonin": [0.15, 0.2375, 0.351, 0.477, 0.6119, 0.5362, 0.5346, 0.5188, 0.4901, 0.4504, 0.4027, 0.3501, 0.2961, 0.2444, 0.1981, 0.1601],
        "Norepinephrine": [0.2662, 0.4467, 0.5655, 0.5054, 0.431, 0.2342, 0.2583, 0.243, 0.2238, 0.2782, 0.4088, 0.3158, 0.258, 0.455, 0.3295, 0.3022],
        "Oxytocin": [0.8613, 0.8329, 0.8044, 0.776, 0.7475, 0.7191, 0.6906, 0.6622, 0.6337, 0.6053, 0.5768, 0.5483, 0.5199, 0.4914, 0.463, 0.4345],
        "EndogenousOpioids": [0.15, 0.2359, 0.3474, 0.4711, 0.6034, 0.5286, 0.527, 0.5114, 0.4831, 0.444, 0.3969, 0.3451, 0.2919, 0.2409, 0.1953, 0.1578],
        "GABA": [0.15, 0.2321, 0.3386, 0.4567, 0.5832, 0.5103, 0.5087, 0.4937, 0.4663, 0.4286, 0.3832, 0.3331, 0.2818, 0.2326, 0.1885, 0.1523],
        "Glutamate": [0.15, 0.2336, 0.3421, 0.4625, 0.5913, 0.5176, 0.516, 0.5008, 0.4731, 0.4348, 0.3887, 0.3379, 0.2859, 0.2359, 0.1913, 0.1545],
        "Endocannabinoids": [0.4667, 0.5334, 0.471, 0.556, 0.6597, 0.4216, 0.3298, 0.3634, 0.3762, 0.513, 0.4056, 0.4763, 0.4948, 0.5294, 0.4299, 0.5027],
    },
    "Nostalgia": {
        "Cortisol": [0.4045, 0.5314, 0.456, 0.4022, 0.2397, 0.2309, 0.3523, 0.2691, 0.3664, 0.2683, 0.3113, 0.5081, 0.3845, 0.4171, 0.3773, 0.246],
        "Epinephrine": [0.3558, 0.361, 0.4897, 0.4773, 0.3441, 0.3549, 0.4377, 0.4396, 0.3503, 0.4021, 0.4156, 0.3951, 0.4247, 0.3923, 0.3655, 0.3016],
        "Dopamine": [0.342, 0.3813, 0.3742, 0.3904, 0.4499, 0.2377, 0.3034, 0.3089, 0.3867, 0.2797, 0.2228, 0.226, 0.5298, 0.5785, 0.3294, 0.29],
        "Serotonin": [0.2, 0.2514, 0.3265, 0.4144, 0.5116, 0.5538, 0.535, 0.5162, 0.4974, 0.4787, 0.4547, 0.4308, 0.4069, 0.3829, 0.359, 0.3351],
        "Norepinephrine": [0.3234, 0.292, 0.4031, 0.5181, 0.508, 0.401, 0.3109, 0.3063, 0.4032, 0.3759, 0.3215, 0.5391, 0.3889, 0.3685, 0.4354, 0.312],
        "Oxytocin": [0.2779, 0.3949, 0.5673, 0.4313, 0.3992, 0.4031, 0.3833, 0.2007, 0.4473, 0.3382, 0.258, 0.411, 0.4513, 0.5054, 0.3453, 0.3895],
        "EndogenousOpioids": [0.2, 0.2636, 0.3567, 0.4654, 0.5858, 0.6388, 0.6172, 0.5955, 0.5739, 0.5522, 0.5246, 0.497, 0.4694, 0.4418, 0.4142, 0.3866],
        "GABA": [0.2, 0.2629, 0.3548, 0.4622, 0.5812, 0.6335, 0.612, 0.5906, 0.5691, 0.5476, 0.5202, 0.4928, 0.4655, 0.4381, 0.4107, 0.3833],
        "Glutamate": [0.4512, 0.4933, 0.4675, 0.5485, 0.314, 0.3674, 0.4036, 0.2637, 0.4261, 0.6547, 0.2279, 0.4248, 0.3512, 0.4451, 0.1163, 0.3233],
        "Endocannabinoids": [0.3267, 0.519, 0.4507, 0.3076, 0.4356, 0.4575, 0.2976, 0.3172, 0.2081, 0.3736, 0.2983, 0.3836, 0.5074, 0.33, 0.3759, 0.4491],
    },
    "Empathy": {
        "Cortisol": [0.4118, 0.479, 0.3754, 0.4829, 0.5118, 0.4167, 0.4973, 0.3303, 0.3727, 0.4664, 0.3061, 0.3935, 0.422, 0.5244, 0.4112, 0.38],
        "Epinephrine": [0.3377, 0.3393, 0.4378, 0.576, 0.4384, 0.494, 0.3765, 0.4206, 0.2393, 0.3635, 0.3089, 0.3604, 0.2357, 0.4348, 0.4217, 0.4668],
        "Dopamine": [0.397, 0.4376, 0.2484, 0.331, 0.5783, 0.3316, 0.5431, 0.4817, 0.2636, 0.486, 0.2541, 0.406, 0.2763, 0.4893, 0.4018, 0.2373],
        "Serotonin": [0.331, 0.2755, 0.284, 0.2837, 0.369, 0.0533, 0.2726, 0.2639, 0.3877, 0.2053, 0.2795, 0.405, 0.3539, 0.3589, 0.3244, 0.2959],
        "Norepinephrine": [0.3443, 0.5194, 0.5448, 0.4181, 0.4423, 0.6335, 0.4443, 0.5059, 0.3548, 0.3848, 0.2787, 0.4013, 0.3702, 0.3219, 0.4389, 0.299],
        "Oxytocin": [0.2, 0.2502, 0.3237, 0.4095, 0.5046, 0.5457, 0.5272, 0.5087, 0.4902, 0.4717, 0.4481, 0.4245, 0.4009, 0.3774, 0.3538, 0.3302],
        "EndogenousOpioids": [0.5021, 0.3258, 0.2445, 0.4216, 0.3385, 0.2345, 0.2888, 0.2895, 0.2863, 0.2438, 0.4242, 0.4748, 0.3897, 0.5654, 0.4159, 0.3227],
        "GABA": [0.1317, 0.3688, 0.3735, 0.3017, 0.1923, 0.3006, 0.215, 0.3072, 0.2633, 0.5071, 0.3611, 0.3957, 0.3508, 0.2396, 0.3206, 0.4099],
        "Glutamate": [0.3732, 0.5544, 0.4838, 0.4382, 0.3531, 0.2145, 0.3246, 0.302, 0.3567, 0.5176, 0.5053, 0.3897, 0.2647, 0.2994, 0.4411, 0.5202],
        "Endocannabinoids": [0.3209, 0.3761, 0.4461, 0.2436, 0.482, 0.3405, 0.5098, 0.3725, 0.2489, 0.5483, 0.3464, 0.4794, 0.4487, 0.4818, 0.3463, 0.2755],
    },
    "Boredom": {
        "Cortisol": [0.4002, 0.2712, 0.4881, 0.3616, 0.2061, 0.3286, 0.2329, 0.3281, 0.2538, 0.318, 0.2975, 0.5429, 0.324, 0.3857, 0.413, 0.4141],
        "Epinephrine": [0.3958, 0.4641, 0.4428, 0.3152, 0.5267, 0.3845, 0.3498, 0.468, 0.3815, 0.2452, 0.3378, 0.316, 0.1687, 0.4341, 0.2797, 0.4457],
        "Dopamine": [0.4884, 0.4371, 0.3814, 0.4166, 0.302, 0.4681, 0.6251, 0.4683, 0.426, 0.476, 0.5331, 0.2476, 0.3711, 0.267, 0.4322, 0.4133],
        "Serotonin": [0.6622, 0.6183, 0.6048, 0.5801, 0.5947, 0.6438, 0.6434, 0.5588, 0.5563, 0.5086, 0.5613, 0.576, 0.5085, 0.5677, 0.4997, 0.4439],
        "Norepinephrine": [0.4207, 0.3985, 0.426, 0.3388, 0.2661, 0.4238, 0.3532, 0.3539, 0.4584, 0.4842, 0.3924, 0.4459, 0.3619, 0.4022, 0.3173, 0.4209],
        "Oxytocin": [0.4112, 0.3681, 0.3865, 0.4051, 0.2615, 0.4015, 0.3907, 0.4318, 0.4612, 0.3341, 0.3617, 0.2722, 0.4546, 0.444, 0.2532, 0.3856],
        "EndogenousOpioids": [0.3307, 0.3064, 0.4435, 0.4499, 0.44, 0.4674, 0.3051, 0.4157, 0.338, 0.2948, 0.4765, 0.4012, 0.3652, 0.4149, 0.4422, 0.2461],
        "GABA": [0.4042, 0.4709, 0.3404, 0.2173, 0.3088, 0.2089, 0.3196, 0.3684, 0.2999, 0.3424, 0.3285, 0.3527, 0.2185, 0.2824, 0.3433, 0.3821],
        "Glutamate": [0.3776, 0.4064, 0.3162, 0.4353, 0.5143, 0.2402, 0.3369, 0.4075, 0.3946, 0.578, 0.3905, 0.4828, 0.2586, 0.2753, 0.2234, 0.5105],
        "Endocannabinoids": [0.4547, 0.3497, 0.3943, 0.5064, 0.3299, 0.5023, 0.4077, 0.3335, 0.4111, 0.4252, 0.2965, 0.4279, 0.2862, 0.2607, 0.3477, 0.4157],
    },
    "Contentment": {
        "Cortisol": [0.7066, 0.651, 0.6826, 0.6524, 0.6443, 0.5857, 0.6229, 0.6313, 0.5516, 0.5618, 0.526, 0.5681, 0.5041, 0.5752, 0.4869, 0.4579],
        "Epinephrine": [0.755, 0.7249, 0.6789, 0.7363, 0.6547, 0.6432, 0.6405, 0.6132, 0.5473, 0.6019, 0.5344, 0.5242, 0.5452, 0.4979, 0.3902, 0.5194],
        "Dopamine": [0.2354, 0.2808, 0.3973, 0.3471, 0.501, 0.343, 0.2916, 0.2913, 0.2471, 0.3889, 0.3542, 0.4961, 0.4229, 0.4226, 0.4884, 0.3096],
        "Serotonin": [0.3166, 0.3134, 0.1625, 0.3807, 0.3622, 0.5702, 0.4466, 0.5094, 0.1955, 0.2825, 0.4582, 0.1512, 0.36, 0.3695, 0.357, 0.4915],
        "Norepinephrine": [0.7657, 0.7789, 0.7199, 0.7321, 0.7553, 0.7197, 0.6636, 0.6852, 0.6905, 0.6257, 0.6082, 0.5816, 0.6029, 0.5526, 0.5767, 0.5915],
        "Oxytocin": [0.6697, 0.6996, 0.6035, 0.6027, 0.5818, 0.6301, 0.5893, 0.5957, 0.5308, 0.5452, 0.5644, 0.504, 0.5379, 0.5022, 0.515, 0.4318],
        "EndogenousOpioids": [0.15, 0.2446, 0.3672, 0.5034, 0.6491, 0.57, 0.5682, 0.5514, 0.5209, 0.4788, 0.428, 0.3721, 0.3148, 0.2598, 0.2106, 0.1702],
        "GABA": [0.2, 0.2536, 0.332, 0.4236, 0.525, 0.5691, 0.5498, 0.5305, 0.5112, 0.4919, 0.4673, 0.4427, 0.4181, 0.3935, 0.3689, 0.3443],
        "Glutamate": [0.2, 0.261, 0.3502, 0.4545, 0.5699, 0.6206, 0.5995, 0.5785, 0.5575, 0.5364, 0.5096, 0.4828, 0.456, 0.4291, 0.4023, 0.3755],
        "Endocannabinoids": [0.2, 0.2566, 0.3394, 0.4361, 0.5432, 0.59, 0.57, 0.55, 0.53, 0.51, 0.4845, 0.459, 0.4335, 0.408, 0.3825, 0.357],
    },
    "Grief": {
        "Cortisol": [0.15, 0.2329, 0.3406, 0.46, 0.5878, 0.5145, 0.5129, 0.4977, 0.4702, 0.4321, 0.3863, 0.3358, 0.2841, 0.2345, 0.1901, 0.1536],
        "Epinephrine": [0.2765, 0.3635, 0.4099, 0.3705, 0.484, 0.2727, 0.2381, 0.3171, 0.3728, 0.2478, 0.3323, 0.3359, 0.3304, 0.3203, 0.3852, 0.3774],
        "Dopamine": [0.3869, 0.3486, 0.3958, 0.5737, 0.4069, 0.4062, 0.1656, 0.364, 0.4268, 0.4423, 0.288, 0.3695, 0.5463, 0.5773, 0.3209, 0.473],
        "Serotonin": [0.9284, 0.7994, 0.8563, 0.8125, 0.7769, 0.6716, 0.71, 0.6689, 0.7638, 0.6081, 0.6478, 0.6065, 0.4573, 0.5909, 0.5129, 0.3843],
        "Norepinephrine": [0.4184, 0.3913, 0.4324, 0.5296, 0.467, 0.408, 0.3684, 0.249, 0.1657, 0.3255, 0.297, 0.3681, 0.4572, 0.3728, 0.443, 0.3317],
        "Oxytocin": [0.4976, 0.4066, 0.518, 0.3393, 0.4646, 0.478, 0.4758, 0.3081, 0.364, 0.4238, 0.3626, 0.5029, 0.3687, 0.5714, 0.5135, 0.3439],
        "EndogenousOpioids": [0.3675, 0.5137, 0.4199, 0.5153, 0.5075, 0.5467, 0.5184, 0.4208, 0.5189, 0.3848, 0.2939, 0.4069, 0.4475, 0.4203, 0.3706, 0.3873],
        "GABA": [0.8362, 0.8817, 0.7602, 0.8273, 0.7042, 0.8535, 0.7032, 0.726, 0.6892, 0.7521, 0.6936, 0.6498, 0.6428, 0.5723, 0.4389, 0.5432],
        "Glutamate": [0.2801, 0.4132, 0.3133, 0.4414, 0.4402, 0.2701, 0.3023, 0.1991, 0.2559, 0.3501, 0.2448, 0.2866, 0.3256, 0.3554, 0.3164, 0.3984],
        "Endocannabinoids": [0.3521, 0.5041, 0.483, 0.5681, 0.393, 0.3793, 0.267, 0.3496, 0.4796, 0.3994, 0.3904, 0.3404, 0.3831, 0.4952, 0.2308, 0.508],
    },
}

predictor = HormonePredictor()
