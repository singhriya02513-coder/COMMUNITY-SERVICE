"""Generate a reproducible symptom-disease training dataset.

The dataset is synthesised from the curated knowledge base in
``data/disease_data.json``. Each disease has a set of core symptoms. For every
synthetic patient we keep most core symptoms, randomly drop a few, and add a
small amount of realistic noise (shared symptoms such as fatigue or headache).

Run:  python generate_dataset.py
"""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

BASE_DIR = Path(__file__).resolve().parent
DATA_PATH = BASE_DIR / "data" / "disease_data.json"
OUTPUT_PATH = BASE_DIR / "data" / "training_data.csv"

SAMPLES_PER_DISEASE = 90
RANDOM_SEED = 42

BACKGROUND_SYMPTOMS = [
    "fatigue",
    "headache",
    "fever",
    "nausea",
    "weakness",
    "loss_of_appetite",
    "muscle_pain",
    "dizziness",
    "cough",
    "sore_throat",
]


def load_knowledge_base(path: Path = DATA_PATH) -> dict:
    with path.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    validate_knowledge_base(data)
    return data


def validate_knowledge_base(data: dict) -> None:
    labels = data["symptom_labels"]
    grouped = {symptom for group in data["symptom_groups"].values() for symptom in group}
    diseases = data["diseases"]

    missing_labels = grouped - set(labels)
    if missing_labels:
        raise ValueError(f"Symptoms missing display labels: {sorted(missing_labels)}")

    ungrouped = set(labels) - grouped
    if ungrouped:
        raise ValueError(f"Symptoms missing a category: {sorted(ungrouped)}")

    if len(diseases) < 10:
        raise ValueError("At least 10 diseases are required")
    if not 30 <= len(labels) <= 200:
        raise ValueError("Symptom count looks unrealistic")

    for name, info in diseases.items():
        symptoms = info.get("symptoms", [])
        if len(symptoms) < 5:
            raise ValueError(f"{name} needs at least 5 symptoms")
        unknown = set(symptoms) - set(labels)
        if unknown:
            raise ValueError(f"{name} uses unknown symptoms: {sorted(unknown)}")
        if len(symptoms) != len(set(symptoms)):
            raise ValueError(f"{name} contains duplicate symptoms")
        if len(info.get("precautions", [])) < 3:
            raise ValueError(f"{name} needs at least 3 precautions")


def generate_rows(data: dict, rng: random.Random) -> list[dict]:
    all_symptoms = list(data["symptom_labels"].keys())
    rows: list[dict] = []

    for disease, info in data["diseases"].items():
        core = info["symptoms"]
        for _ in range(SAMPLES_PER_DISEASE):
            active: set[str] = set()

            for symptom in core:
                # Most patients report most core symptoms, but not all of them.
                if rng.random() < rng.uniform(0.84, 0.95):
                    active.add(symptom)

            # Real patients often mention common background complaints.
            roll = rng.random()
            noise_count = 1 if roll < 0.55 else (2 if roll < 0.20 else 0)
            noise_pool = [s for s in BACKGROUND_SYMPTOMS if s not in active]
            active.update(rng.sample(noise_pool, min(noise_count, len(noise_pool))))

            # Occasional unrelated symptom, simulating imperfect self-reporting.
            if rng.random() < 0.30:
                extra_pool = [s for s in all_symptoms if s not in active and s not in core]
                if extra_pool:
                    active.add(rng.choice(extra_pool))

            # Some patients report only a few symptoms. This teaches the model
            # to cope with short, incomplete symptom lists from the website.
            if rng.random() < 0.30 and len(active) > 4:
                drop_count = rng.randint(1, max(1, len(active) // 2))
                for symptom in rng.sample(sorted(active), drop_count):
                    if len(active) <= 3:
                        break
                    active.discard(symptom)

            # Guarantee enough signal and keep the profile realistic.
            if len(active) < 3:
                active.update(rng.sample(core, 3 - len(active)))
            while len(active) > 12:
                removable = [s for s in active if s not in core]
                active.remove(rng.choice(removable or list(active)))

            row = {symptom: int(symptom in active) for symptom in all_symptoms}
            row["disease"] = disease
            rows.append(row)

    rng.shuffle(rows)
    return rows


def main() -> None:
    data = load_knowledge_base()
    rng = random.Random(RANDOM_SEED)
    rows = generate_rows(data, rng)

    columns = ["disease"] + list(data["symptom_labels"].keys())
    frame = pd.DataFrame(rows, columns=columns)
    frame.to_csv(OUTPUT_PATH, index=False)

    print(f"Saved {len(frame)} rows to {OUTPUT_PATH}")
    print(f"Diseases: {frame['disease'].nunique()} | Symptoms: {len(columns) - 1}")
    print(f"Average symptoms reported: {frame.drop(columns='disease').sum(axis=1).mean():.2f}")


if __name__ == "__main__":
    main()
