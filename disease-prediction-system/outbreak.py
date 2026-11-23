"""Syndromic outbreak early-warning engine.

Every anonymous prediction is logged with a ward code and a timestamp. This
module compares the last 7 days of predictions per ward against that ward's own
4-week baseline and raises an alert when a disease climbs far above it.

Privacy: only the ward code, the predicted disease, a confidence score and the
symptom keys are stored. No name, phone number, address or device identifier.

Commands:
    python outbreak.py seed     # reset + load a demo Dengue outbreak in WARD-7
    python outbreak.py reset    # delete all logged predictions
    python outbreak.py stats    # print the current regional summary
"""

from __future__ import annotations

import json
import random
import sqlite3
import statistics
from collections import Counter
from datetime import datetime, timedelta, timezone
from functools import lru_cache
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR / "data" / "outbreak.db"
KNOWLEDGE_PATH = BASE_DIR / "data" / "disease_data.json"

# Anonymous ward codes only. This is what a hospital desk would actually use.
REGIONS: dict[str, str] = {
    "WARD-1": "Ward 1 - North district",
    "WARD-2": "Ward 2 - East district",
    "WARD-3": "Ward 3 - South district",
    "WARD-7": "Ward 7 - Riverside",
    "CITY": "City centre",
}
DEFAULT_REGION = "WARD-1"

WINDOW_DAYS = 7
BASELINE_WEEKS = 4
MIN_CASES = 10
Z_THRESHOLD = 3.0
WATCH_Z = 2.0
PRIOR_BOOST = 6.0
PRIOR_WATCH = 1.8

STATUS_ORDER = {"alert": 0, "watch": 1, "normal": 2}


def _utc_now() -> datetime:
    return datetime.now(timezone.utc).replace(microsecond=0)


def _parse(value: str) -> datetime:
    return datetime.fromisoformat(value)


def _connect() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    connection = sqlite3.connect(DB_PATH)
    connection.row_factory = sqlite3.Row
    return connection


def init_db() -> None:
    with _connect() as connection:
        connection.execute(
            """
            CREATE TABLE IF NOT EXISTS predictions (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                region TEXT NOT NULL,
                disease TEXT NOT NULL,
                confidence REAL NOT NULL,
                symptoms TEXT NOT NULL,
                created_at TEXT NOT NULL
            )
            """
        )
        connection.execute(
            "CREATE INDEX IF NOT EXISTS idx_predictions_region_time "
            "ON predictions (region, created_at)"
        )


def log_prediction(region: str, disease: str, confidence: float, symptoms: list[str]) -> None:
    """Store one anonymous prediction for outbreak surveillance."""
    if region not in REGIONS:
        region = DEFAULT_REGION
    init_db()
    with _connect() as connection:
        connection.execute(
            "INSERT INTO predictions (region, disease, confidence, symptoms, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            (region, disease, float(confidence), json.dumps(sorted(symptoms)), _utc_now().isoformat()),
        )


def clear_data() -> None:
    init_db()
    with _connect() as connection:
        connection.execute("DELETE FROM predictions")


def _rows_between(region: str, start: datetime, end: datetime) -> list[sqlite3.Row]:
    with _connect() as connection:
        return connection.execute(
            "SELECT disease, created_at FROM predictions "
            "WHERE region = ? AND created_at >= ? AND created_at <= ?",
            (region, start.isoformat(), end.isoformat()),
        ).fetchall()


def _status(current: int, z_score: float) -> str:
    if current >= MIN_CASES and z_score >= Z_THRESHOLD:
        return "alert"
    if current >= 3 and z_score >= WATCH_Z:
        return "watch"
    return "normal"


def region_stats(region: str, now: datetime | None = None) -> list[dict]:
    """Per-disease counts for a ward: last week vs its own 4-week baseline."""
    now = now or _utc_now()
    window_start = now - timedelta(days=WINDOW_DAYS)
    baseline_start = window_start - timedelta(weeks=BASELINE_WEEKS)

    rows = _rows_between(region, baseline_start, now)
    current: Counter[str] = Counter()
    weekly: list[Counter[str]] = [Counter() for _ in range(BASELINE_WEEKS)]
    daily: dict[str, Counter[str]] = {}

    for row in rows:
        timestamp = _parse(row["created_at"])
        disease = row["disease"]
        if timestamp >= window_start:
            current[disease] += 1
            offset = (timestamp - window_start).days
            day = min(max(offset, 0), WINDOW_DAYS - 1)
            daily.setdefault(disease, Counter())[day] += 1
        else:
            week = int((timestamp - baseline_start).total_seconds() // (7 * 86400))
            weekly[min(max(week, 0), BASELINE_WEEKS - 1)][disease] += 1

    diseases = set(current)
    for bucket in weekly:
        diseases.update(bucket)

    results: list[dict] = []
    for disease in diseases:
        series = [bucket.get(disease, 0) for bucket in weekly]
        baseline_mean = statistics.fmean(series) if series else 0.0
        baseline_std = statistics.pstdev(series) if len(series) > 1 else 0.0
        z_score = (current[disease] - baseline_mean) / max(baseline_std, 1.0)
        day_counts = daily.get(disease, Counter())
        results.append(
            {
                "disease": disease,
                "current": current[disease],
                "baseline_mean": round(baseline_mean, 2),
                "baseline_std": round(baseline_std, 2),
                "z_score": round(z_score, 2),
                "status": _status(current[disease], z_score),
                "daily": [day_counts.get(day, 0) for day in range(WINDOW_DAYS)],
            }
        )

    results.sort(key=lambda item: (STATUS_ORDER[item["status"]], -item["current"], item["disease"]))
    return results


def detect_alerts(now: datetime | None = None) -> list[dict]:
    """All wards with an active alert, strongest signal first."""
    now = now or _utc_now()
    alerts: list[dict] = []
    for region, label in REGIONS.items():
        for row in region_stats(region, now):
            if row["status"] == "alert":
                alerts.append(
                    {
                        "region": region,
                        "region_label": label,
                        "disease": row["disease"],
                        "current": row["current"],
                        "baseline_mean": row["baseline_mean"],
                        "z_score": row["z_score"],
                    }
                )
    alerts.sort(key=lambda item: item["z_score"], reverse=True)
    return alerts


def regional_weights(region: str, now: datetime | None = None) -> dict[str, float]:
    """Probability multipliers applied to the classifier for a ward."""
    if not region:
        return {}
    weights: dict[str, float] = {}
    for row in region_stats(region, now):
        if row["status"] == "alert":
            weights[row["disease"]] = PRIOR_BOOST
        elif row["status"] == "watch":
            weights[row["disease"]] = PRIOR_WATCH
    return weights


def banner_for(region: str, now: datetime | None = None) -> dict | None:
    """Compact outbreak notice for the website, or None when things look normal."""
    now = now or _utc_now()
    alerts = [
        row for row in region_stats(region, now) if row["status"] in {"alert", "watch"}
    ]
    if not alerts:
        return None
    disease_rows = [
        {
            "disease": row["disease"],
            "current": row["current"],
            "baseline_mean": row["baseline_mean"],
            "z_score": row["z_score"],
            "status": row["status"],
        }
        for row in alerts[:3]
    ]
    return {
        "region": region,
        "region_label": REGIONS.get(region, region),
        "window_days": WINDOW_DAYS,
        "level": "alert" if alerts[0]["status"] == "alert" else "watch",
        "diseases": disease_rows,
    }


def top_diseases(limit: int = 6, now: datetime | None = None) -> list[dict]:
    now = now or _utc_now()
    window_start = now - timedelta(days=WINDOW_DAYS)
    totals: Counter[str] = Counter()
    for region in REGIONS:
        for row in _rows_between(region, window_start, now):
            totals[row["disease"]] += 1
    return [{"disease": name, "count": count} for name, count in totals.most_common(limit)]


def counted_predictions() -> int:
    init_db()
    with _connect() as connection:
        return int(connection.execute("SELECT COUNT(*) FROM predictions").fetchone()[0])


def dashboard_data() -> dict:
    now = _utc_now()
    alerts = detect_alerts(now)
    regions = []
    for region, label in REGIONS.items():
        stats = region_stats(region, now)
        regions.append(
            {
                "region": region,
                "label": label,
                "total": sum(row["current"] for row in stats),
                "alert_count": sum(1 for row in stats if row["status"] == "alert"),
                "stats": stats,
            }
        )
    return {
        "generated_at": now.isoformat(timespec="seconds"),
        "window_days": WINDOW_DAYS,
        "baseline_weeks": BASELINE_WEEKS,
        "z_threshold": Z_THRESHOLD,
        "min_cases": MIN_CASES,
        "total_predictions": counted_predictions(),
        "regions": regions,
        "alerts": alerts,
        "top_diseases": top_diseases(now=now),
    }


@lru_cache(maxsize=1)
def _disease_symptoms() -> dict[str, list[str]]:
    if not KNOWLEDGE_PATH.exists():
        return {}
    with KNOWLEDGE_PATH.open("r", encoding="utf-8") as handle:
        data = json.load(handle)
    return {
        name: list(info.get("symptoms", [])) for name, info in data.get("diseases", {}).items()
    }


# Ward baselines: expected cases per day for each disease.
_SEED_BASELINES: dict[str, dict[str, float]] = {
    "WARD-1": {"Influenza": 1.2, "Gastroenteritis": 0.8, "Dengue": 0.3},
    "WARD-2": {"Influenza": 1.0, "Allergic Rhinitis": 0.9, "Typhoid": 0.3},
    "WARD-3": {"Gastroenteritis": 1.1, "Typhoid": 0.5, "Influenza": 0.6},
    "WARD-7": {"Dengue": 0.4, "Influenza": 0.9, "Malaria": 0.4},
    "CITY": {"Influenza": 2.0, "COVID-19": 0.8, "Dengue": 0.4},
}

# A Dengue surge that started 5 days ago in WARD-7.
_SEED_SPIKE = {"WARD-7": {"Dengue": (5, 10)}}
_SEED_WATCH = {"WARD-3": {"Typhoid": (7, 0.5)}}


def seed_demo_data(days: int = 35, seed: int = 42) -> dict:
    """Reset the log and load a realistic baseline plus a Dengue outbreak."""
    init_db()
    rng = random.Random(seed)
    now = _utc_now()
    symptoms_by_disease = _disease_symptoms()

    def pick_symptoms(disease: str) -> list[str]:
        core = symptoms_by_disease.get(disease, [])
        active = [symptom for symptom in core if rng.random() < 0.85]
        return active or core[:3]

    rows: list[tuple[str, str, float, str, str]] = []

    for region, baselines in _SEED_BASELINES.items():
        for day in range(days, -1, -1):
            day_start = now - timedelta(days=day)
            for disease, rate in baselines.items():
                expect = max(0.0, rng.gauss(rate, rate ** 0.5 + 0.2))
                count = int(round(expect))
                for _ in range(count):
                    stamp = day_start - timedelta(
                        hours=rng.randint(0, 20), minutes=rng.randint(0, 59)
                    )
                    rows.append(
                        (
                            region,
                            disease,
                            round(rng.uniform(58, 94), 1),
                            json.dumps(pick_symptoms(disease)),
                            stamp.isoformat(),
                        )
                    )

    for region, surges in _SEED_SPIKE.items():
        for disease, (spike_days, per_day) in surges.items():
            for day in range(spike_days):
                day_start = now - timedelta(days=day)
                for _ in range(per_day):
                    stamp = day_start - timedelta(
                        hours=rng.randint(0, 20), minutes=rng.randint(0, 59)
                    )
                    rows.append(
                        (
                            region,
                            disease,
                            round(rng.uniform(70, 97), 1),
                            json.dumps(pick_symptoms(disease)),
                            stamp.isoformat(),
                        )
                    )

    for region, bumps in _SEED_WATCH.items():
        for disease, (spike_days, per_day) in bumps.items():
            for day in range(spike_days):
                if rng.random() > per_day:
                    continue
                stamp = (now - timedelta(days=day)).replace(
                    hour=rng.randint(8, 18), minute=rng.randint(0, 59)
                )
                rows.append(
                    (
                        region,
                        disease,
                        round(rng.uniform(55, 88), 1),
                        json.dumps(pick_symptoms(disease)),
                        stamp.isoformat(),
                    )
                )

    with _connect() as connection:
        connection.execute("DELETE FROM predictions")
        connection.executemany(
            "INSERT INTO predictions (region, disease, confidence, symptoms, created_at) "
            "VALUES (?, ?, ?, ?, ?)",
            rows,
        )

    alerts = detect_alerts(now)
    return {"inserted": len(rows), "alerts": alerts}


def _print_stats() -> None:
    data = dashboard_data()
    print(f"Anonymous predictions logged: {data['total_predictions']}")
    print(f"Window: last {data['window_days']} days | threshold z >= {data['z_threshold']}\n")
    for region in data["regions"]:
        print(f"{region['region']}  {region['label']}  (last 7 days: {region['total']})")
        for row in region["stats"]:
            if row["status"] == "normal":
                continue
            print(
                f"   [{row['status'].upper()}] {row['disease']}: "
                f"{row['current']} vs baseline {row['baseline_mean']}/week "
                f"(z={row['z_score']})"
            )
    if not data["alerts"]:
        print("No active alerts.")


if __name__ == "__main__":
    import sys

    command = sys.argv[1] if len(sys.argv) > 1 else "stats"
    if command == "seed":
        summary = seed_demo_data()
        print(f"Seeded {summary['inserted']} anonymous predictions.")
        for alert in summary["alerts"]:
            print(
                f"ALERT: {alert['disease']} in {alert['region']} "
                f"({alert['current']} cases vs {alert['baseline_mean']}/week, z={alert['z_score']})"
            )
    elif command == "reset":
        clear_data()
        print("All outbreak predictions cleared.")
    elif command == "stats":
        _print_stats()
    else:
        print(__doc__)
