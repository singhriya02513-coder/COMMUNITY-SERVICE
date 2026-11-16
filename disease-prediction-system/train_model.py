"""Train and evaluate the disease prediction model.

Outputs:
  model/model_bundle.joblib  - model + knowledge base used by the Flask app
  model/metrics.json         - evaluation metrics shown on the website
  model/classification_report.txt
"""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score
from sklearn.model_selection import cross_val_score, train_test_split
from sklearn.neighbors import KNeighborsClassifier

from generate_dataset import OUTPUT_PATH, load_knowledge_base, main as build_dataset

BASE_DIR = Path(__file__).resolve().parent
MODEL_DIR = BASE_DIR / "model"
BUNDLE_PATH = MODEL_DIR / "model_bundle.joblib"
METRICS_PATH = MODEL_DIR / "metrics.json"
REPORT_PATH = MODEL_DIR / "classification_report.txt"

RANDOM_SEED = 42


def build_models() -> dict:
    return {
        "Random Forest": RandomForestClassifier(
            n_estimators=400,
            min_samples_leaf=1,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Extra Trees": ExtraTreesClassifier(
            n_estimators=400,
            random_state=RANDOM_SEED,
            n_jobs=-1,
        ),
        "Logistic Regression": LogisticRegression(
            max_iter=3000,
            random_state=RANDOM_SEED,
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(n_neighbors=7),
    }


def evaluate_candidates(X_train, X_test, y_train, y_test) -> tuple[dict, dict, dict, str]:
    results: dict[str, dict] = {}
    fitted_models: dict[str, object] = {}
    predictions: dict[str, np.ndarray] = {}

    for name, model in build_models().items():
        model.fit(X_train, y_train)
        predicted = model.predict(X_test)
        accuracy = accuracy_score(y_test, predicted)
        weighted_f1 = f1_score(y_test, predicted, average="weighted")
        results[name] = {
            "accuracy": round(float(accuracy), 4),
            "weighted_f1": round(float(weighted_f1), 4),
        }
        fitted_models[name] = model
        predictions[name] = predicted
        print(f"{name:22s} accuracy={accuracy:.4f}  weighted_f1={weighted_f1:.4f}")

    best_name = max(results, key=lambda candidate: (results[candidate]["weighted_f1"], results[candidate]["accuracy"]))

    # Prefer Random Forest when its score is within 0.5% of the leader.
    # It produces smoother probabilities for short symptom lists and exposes
    # feature importances, which makes the project easier to explain and use.
    forest_score = results["Random Forest"]["weighted_f1"]
    if forest_score >= results[best_name]["weighted_f1"] - 0.005:
        best_name = "Random Forest"

    return results, fitted_models, predictions, best_name


def main() -> None:
    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    knowledge = load_knowledge_base()
    if not OUTPUT_PATH.exists():
        print("Training data not found. Generating it first...")
        build_dataset()

    frame = pd.read_csv(OUTPUT_PATH)
    symptoms = list(knowledge["symptom_labels"].keys())
    diseases = knowledge["diseases"]

    missing_columns = set(symptoms) - set(frame.columns)
    if missing_columns:
        raise ValueError(f"Dataset is missing symptom columns: {sorted(missing_columns)}")
    if set(frame["disease"].unique()) != set(diseases):
        raise ValueError("Dataset diseases do not match the knowledge base. Re-run generate_dataset.py")

    X = frame[symptoms].astype(int).to_numpy()
    y = frame["disease"].to_numpy()

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.20, random_state=RANDOM_SEED, stratify=y
    )

    print(f"\nDataset: {len(frame)} samples | {len(diseases)} diseases | {len(symptoms)} symptoms")
    print(f"Train: {len(X_train)} | Test: {len(X_test)}\n")

    comparisons, fitted_models, holdout_predictions, best_name = evaluate_candidates(
        X_train, X_test, y_train, y_test
    )
    best_model = fitted_models[best_name]
    best_predictions = holdout_predictions[best_name]
    print(f"\nSelected model: {best_name}")

    holdout_accuracy = accuracy_score(y_test, best_predictions)
    holdout_f1 = f1_score(y_test, best_predictions, average="weighted")

    # 5-fold cross-validation gives a more stable estimate of generalisation.
    cv_scores = cross_val_score(
        clone(best_model),
        X,
        y,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
    )

    report = classification_report(y_test, best_predictions, digits=4)
    REPORT_PATH.write_text(f"Model: {best_name}\n\n{report}", encoding="utf-8")

    confusion = confusion_matrix(y_test, best_predictions, labels=best_model.classes_)
    confusion_pairs = []
    for actual_idx, actual in enumerate(best_model.classes_):
        for predicted_idx, predicted in enumerate(best_model.classes_):
            count = int(confusion[actual_idx, predicted_idx])
            if actual != predicted and count:
                confusion_pairs.append(
                    {"actual": str(actual), "predicted": str(predicted), "count": count}
                )
    confusion_pairs.sort(key=lambda item: item["count"], reverse=True)

    # Refit the winning algorithm on all collected data for deployment.
    final_model = clone(best_model)
    final_model.fit(X, y)

    importance_pairs = []
    if hasattr(final_model, "feature_importances_"):
        importance_pairs = sorted(
            zip(symptoms, final_model.feature_importances_),
            key=lambda pair: pair[1],
            reverse=True,
        )[:15]
    elif hasattr(final_model, "coef_"):
        importance_pairs = sorted(
            zip(symptoms, np.abs(final_model.coef_).mean(axis=0)),
            key=lambda pair: pair[1],
            reverse=True,
        )[:15]

    metrics = {
        "algorithm": best_name,
        "holdout_accuracy": round(float(holdout_accuracy), 4),
        "holdout_weighted_f1": round(float(holdout_f1), 4),
        "cv_accuracy_mean": round(float(cv_scores.mean()), 4),
        "cv_accuracy_std": round(float(cv_scores.std()), 4),
        "cv_folds": [round(float(score), 4) for score in cv_scores],
        "comparison": comparisons,
        "confusions": confusion_pairs[:8],
        "top_features": [
            {"symptom": symptom, "label": knowledge["symptom_labels"][symptom], "importance": round(float(score), 4)}
            for symptom, score in importance_pairs
        ],
        "dataset_rows": int(len(frame)),
        "samples_per_disease": int(len(frame) // len(diseases)),
        "trained_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
    }

    bundle = {
        "model": final_model,
        "symptoms": symptoms,
        "symptom_labels": knowledge["symptom_labels"],
        "symptom_groups": knowledge["symptom_groups"],
        "diseases": diseases,
        "metrics": metrics,
        "version": "1.0.0",
    }
    joblib.dump(bundle, BUNDLE_PATH, compress=3)
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    print(f"\nBest model: {best_name}")
    print(f"Holdout accuracy: {holdout_accuracy:.4f} | Weighted F1: {holdout_f1:.4f}")
    print(f"5-fold CV accuracy: {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}")
    print(f"Saved bundle -> {BUNDLE_PATH}")
    print(f"Saved metrics -> {METRICS_PATH}")
    print(f"Saved report  -> {REPORT_PATH}")

    demo_profiles = [
        ["fever", "chills", "muscle_pain", "headache", "dry_cough"],
        ["excessive_thirst", "frequent_urination", "increased_hunger", "weight_loss"],
        ["sneezing", "runny_nose", "watery_eyes", "itchy_skin"],
        ["yellow_skin", "yellow_eyes", "dark_urine", "fatigue"],
    ]
    print("\nSample predictions:")
    for profile in demo_profiles:
        vector = np.array([[1 if symptom in profile else 0 for symptom in symptoms]])
        probabilities = final_model.predict_proba(vector)[0]
        top = sorted(zip(final_model.classes_, probabilities), key=lambda x: x[1], reverse=True)[:3]
        formatted = ", ".join(f"{name} ({prob * 100:.1f}%)" for name, prob in top)
        print(f"  {' + '.join(profile)}\n    -> {formatted}")


if __name__ == "__main__":
    main()
