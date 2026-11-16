"""MediPredict - Disease Prediction System.

Flask application that serves the website and exposes a prediction API backed
by a trained Random Forest classifier.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
from flask import Flask, jsonify, render_template, request

BASE_DIR = Path(__file__).resolve().parent
BUNDLE_PATH = BASE_DIR / "model" / "model_bundle.joblib"

MIN_SYMPTOMS = 3
MAX_SYMPTOMS = 20
TOP_PREDICTIONS = 3

EMERGENCY_SYMPTOMS = {
    "chest_pain",
    "shortness_of_breath",
    "cough_with_blood",
    "confusion",
    "fainting",
    "neck_stiffness",
}

app = Flask(__name__)

if not BUNDLE_PATH.exists():
    raise RuntimeError(
        "Model bundle not found. Run 'python generate_dataset.py' and "
        "'python train_model.py' before starting the app."
    )

BUNDLE = joblib.load(BUNDLE_PATH)
MODEL = BUNDLE["model"]
SYMPTOMS: list[str] = BUNDLE["symptoms"]
SYMPTOM_LABELS: dict[str, str] = BUNDLE["symptom_labels"]
SYMPTOM_GROUPS: dict[str, list[str]] = BUNDLE["symptom_groups"]
DISEASES: dict[str, dict] = BUNDLE["diseases"]
METRICS: dict = BUNDLE["metrics"]
KNOWN_SYMPTOMS = set(SYMPTOMS)


def symptom_catalog() -> list[dict]:
    """Grouped symptom data used by the website and the API."""
    catalog = []
    for group_name, keys in SYMPTOM_GROUPS.items():
        catalog.append(
            {
                "name": group_name,
                "symptoms": [
                    {"key": key, "label": SYMPTOM_LABELS[key]} for key in keys if key in KNOWN_SYMPTOMS
                ],
            }
        )
    return catalog


def display_metrics() -> dict:
    accuracy = float(METRICS.get("holdout_accuracy", 0)) * 100
    cv = float(METRICS.get("cv_accuracy_mean", 0)) * 100
    return {
        "algorithm": METRICS.get("algorithm", "Random Forest"),
        "accuracy": f"{accuracy:.1f}",
        "cv_accuracy": f"{cv:.1f}",
        "cv_std": f"{float(METRICS.get('cv_accuracy_std', 0)) * 100:.1f}",
        "dataset_rows": f"{int(METRICS.get('dataset_rows', 0)):,}",
        "disease_count": len(DISEASES),
        "symptom_count": len(SYMPTOMS),
        "top_features": METRICS.get("top_features", [])[:6],
        "comparison": METRICS.get("comparison", {}),
        "trained_at": METRICS.get("trained_at", ""),
    }


def parse_prediction_request(payload: dict) -> tuple[list[str], str | None]:
    raw = payload.get("symptoms")
    if not isinstance(raw, list):
        return [], "Send a JSON object with a 'symptoms' list."
    if not all(isinstance(item, str) for item in raw):
        return [], "Each symptom must be a string."

    # Only accept known symptom keys, remove duplicates, preserve order.
    cleaned = list(dict.fromkeys(item.strip() for item in raw if item.strip()))
    unknown = [item for item in cleaned if item not in KNOWN_SYMPTOMS]
    if unknown:
        return [], f"Unknown symptom(s): {', '.join(unknown[:5])}."
    if len(cleaned) < MIN_SYMPTOMS:
        return [], f"Please select at least {MIN_SYMPTOMS} symptoms for a meaningful prediction."
    if len(cleaned) > MAX_SYMPTOMS:
        return [], f"Please select no more than {MAX_SYMPTOMS} symptoms."
    return cleaned, None


def predict(symptoms: list[str]) -> list[dict]:
    vector = np.array([[1 if symptom in symptoms else 0 for symptom in SYMPTOMS]])
    probabilities = MODEL.predict_proba(vector)[0]
    order = np.argsort(probabilities)[::-1][:TOP_PREDICTIONS]

    selected = set(symptoms)
    results = []
    for index in order:
        name = str(MODEL.classes_[index])
        info = DISEASES.get(name, {})
        core = info.get("symptoms", [])
        matched = [symptom for symptom in core if symptom in selected]
        results.append(
            {
                "name": name,
                "confidence": round(float(probabilities[index]) * 100, 1),
                "description": info.get("description", ""),
                "precautions": info.get("precautions", []),
                "specialist": info.get("specialist", "General Physician"),
                "matched_symptoms": [SYMPTOM_LABELS[s] for s in matched],
                "matched_count": len(matched),
                "total_known_symptoms": len(core),
                "coverage": round(len(matched) / len(core) * 100) if core else 0,
            }
        )
    return results


@app.route("/")
def home():
    return render_template(
        "index.html",
        catalog=symptom_catalog(),
        metrics=display_metrics(),
        min_symptoms=MIN_SYMPTOMS,
        max_symptoms=MAX_SYMPTOMS,
    )


@app.get("/api/symptoms")
def api_symptoms():
    return jsonify({"groups": symptom_catalog(), "min_symptoms": MIN_SYMPTOMS})


@app.get("/api/model-info")
def api_model_info():
    return jsonify(display_metrics())


@app.post("/api/predict")
def api_predict():
    payload = request.get_json(silent=True) or {}
    symptoms, error = parse_prediction_request(payload)
    if error:
        return jsonify({"error": error}), 400

    try:
        predictions = predict(symptoms)
    except Exception:
        app.logger.exception("Prediction failed")
        return jsonify({"error": "Could not generate a prediction. Please try again."}), 500

    selected = set(symptoms)
    emergency = sorted(SYMPTOM_LABELS[key] for key in EMERGENCY_SYMPTOMS if key in selected)
    top_confidence = predictions[0]["confidence"] if predictions else 0.0
    second_confidence = predictions[1]["confidence"] if len(predictions) > 1 else 0.0
    urgent_condition = bool(predictions and predictions[0]["name"] == "Meningitis")

    return jsonify(
        {
            "predictions": predictions,
            "symptom_count": len(symptoms),
            "low_confidence": top_confidence < 35 or (top_confidence - second_confidence) < 8,
            "emergency_symptoms": emergency,
            "urgent_condition": urgent_condition,
            "disclaimer": (
                "This tool is for educational purposes only and is not a medical diagnosis. "
                "Always consult a qualified doctor about your symptoms."
            ),
        }
    )


@app.errorhandler(404)
def not_found(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Not found"}), 404
    return render_template(
        "index.html",
        catalog=symptom_catalog(),
        metrics=display_metrics(),
        min_symptoms=MIN_SYMPTOMS,
        max_symptoms=MAX_SYMPTOMS,
    ), 404


@app.errorhandler(405)
def method_not_allowed(error):
    if request.path.startswith("/api/"):
        return jsonify({"error": "Method not allowed"}), 405
    return jsonify({"error": "Method not allowed"}), 405


if __name__ == "__main__":
    app.run(debug=True, host="127.0.0.1", port=5000)
