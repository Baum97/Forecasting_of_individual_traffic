"""
Eine Person hat genau eine der drei Schichten (Früh/Spät/Nacht), schläft
~9 h zu Hause und unternimmt 0-3 Freizeitfahrten pro Tag außerhalb der
Arbeitszeit. Die persönliche Abfahrtszeit wird einmal pro Person gezogen
und variiert danach nur um wenige Minuten, damit der Alltag realistisch
bleibt (wer um 5 losfährt, fährt auch morgen um ~5 los).
"""

from __future__ import annotations

import argparse
import csv
import random
from datetime import date, datetime, time, timedelta
from pathlib import Path

SHIFTS = {
    #          depart-Fenster   Arbeitsdauer   Passtime-Fenster(er) im Tag
    "early": {"depart": (5.0, 8.0),   "work_h": 8.5, "pass_win": [(15.0, 21.0)]},
    "late":  {"depart": (12.0, 14.0), "work_h": 8.5, "pass_win": [(8.0, 11.5)]},
    "night": {"depart": (19.0, 22.0), "work_h": 9.0, "pass_win": [(15.0, 18.5)]},
}

WEEKEND_PASS_WINDOW = (9.0, 21.0)
PASSTIME_KINDS = ["shopping", "groceries", "leisure", "visit",
                  "errand", "sport", "family", "doctor"]


def _dt(day: date, hours: float) -> datetime:
    """Datum + Stunden (float, darf > 24 sein) -> datetime."""
    return datetime.combine(day, time(0)) + timedelta(hours=hours)


def pick_anchor(shift: str, rng: random.Random) -> float:
    """Persönliche Abfahrtszeit (1x pro Person), z.B. 5.17 -> ~05:10."""
    lo, hi = SHIFTS[shift]["depart"]
    return rng.uniform(lo, hi)


def work_shift(day: date, shift: str, anchor_h: float,
               rng: random.Random) -> list[dict]:
    """
    Liefert Hin- und Rückfahrt zur Arbeit als zwei Trip-Events.
    Abfahrt streut nur ±20 Min um den Anker -> bleibt nahe am persönlichen
    Rhythmus.
    """
    depart_h = anchor_h + rng.uniform(-20, 20) / 60.0
    commute_out = rng.randint(15, 50)
    work_h = SHIFTS[shift]["work_h"] + rng.uniform(-0.3, 0.6)
    commute_back = rng.randint(15, 55)

    depart = _dt(day, depart_h)
    arrive_work = depart + timedelta(minutes=commute_out)
    leave_work = arrive_work + timedelta(hours=work_h)
    arrive_home = leave_work + timedelta(minutes=commute_back)

    return [
        _row(depart, arrive_work, "work_out", shift),
        _row(leave_work, arrive_home, "work_home", shift),
    ]


def passtime(day: date, windows: list[tuple[float, float]],
             busy: list[tuple[datetime, datetime]],
             rng: random.Random, max_trips: int = 3) -> list[dict]:
    """
    0-`max_trips` Freizeitfahrten (10 Min - 4 h) in den erlaubten Fenstern,
    ohne Überschneidung mit bestehenden busy-Fenstern (z.B. Arbeit) oder
    zwischen einander.
    """
    n = rng.randint(0, max_trips)
    trips: list[dict] = []
    attempts = 0

    while len(trips) < n and attempts < 50:
        attempts += 1
        win_lo, win_hi = rng.choice(windows)
        if win_hi - win_lo < 0.25:
            continue
        duration_h = rng.uniform(10 / 60, 4.0)
        latest_start = win_hi - duration_h
        if latest_start <= win_lo:
            continue
        start_h = rng.uniform(win_lo, latest_start)
        start = _dt(day, start_h)
        end = start + timedelta(hours=duration_h)
        existing = busy + [(t["start"], t["end"]) for t in trips]
        if _overlaps(start, end, existing):
            continue
        trips.append(_row(start, end, "passtime", rng.choice(PASSTIME_KINDS)))

    return trips


def _row(start: datetime, end: datetime, category: str, detail: str) -> dict:
    return {
        "start": start,
        "end": end,
        "duration_min": int(round((end - start).total_seconds() / 60)),
        "category": category,
        "detail": detail,
    }


def _overlaps(start: datetime, end: datetime,
              windows: list[tuple[datetime, datetime]]) -> bool:
    return any(start < we and end > ws for ws, we in windows)


def generate(days: int | None = None, shift: str | None = None,
             seed: int | None = None, start_date: date | None = None,
             out_path: str | Path = "routine.csv") -> Path:
    rng = random.Random(seed)
    if days is None:
        days = rng.randint(50, 100)
    if not 50 <= days <= 100:
        raise ValueError("days must be between 50 and 100")

    shift = shift or rng.choice(list(SHIFTS))
    anchor_h = pick_anchor(shift, rng)
    start_date = start_date or date(2026, 1, 5)  # ein Montag

    rows: list[dict] = []
    for i in range(days):
        day = start_date + timedelta(days=i)
        is_weekend = day.weekday() >= 5
        busy: list[tuple[datetime, datetime]] = []
        day_rows: list[dict] = []

        if not is_weekend:
            work_trips = work_shift(day, shift, anchor_h, rng)
            day_rows.extend(work_trips)
            # komplette Arbeitsphase (Hinfahrt -> Rückfahrt) sperren,
            # damit keine Passtime zwischendrin geplant wird
            busy.append((work_trips[0]["start"], work_trips[-1]["end"]))
            pass_win = SHIFTS[shift]["pass_win"]
        else:
            pass_win = [WEEKEND_PASS_WINDOW]

        day_rows.extend(passtime(day, pass_win, busy, rng))
        day_rows.sort(key=lambda r: r["start"])
        rows.extend(day_rows)

    out = Path(out_path)
    if out.suffix.lower() != ".csv":
        out = out / "routine.csv"
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["date", "weekday", "start_min", "end_min",
                    "duration_min", "category", "detail", "shift"])
        for r in rows:
            base = datetime.combine(r["start"].date(), time(0))
            start_min = int(round((r["start"] - base).total_seconds() / 60))
            end_min = int(round((r["end"] - base).total_seconds() / 60))
            w.writerow([
                r["start"].date().isoformat(),
                r["start"].strftime("%A"),
                start_min,
                end_min,
                r["duration_min"],
                r["category"],
                r["detail"],
                shift,
            ])

    print(f"{len(rows)} Zeilen -> {out.resolve()} "
          f"(days={days}, shift={shift}, anchor={int(round(anchor_h * 60))} min)")
    return out


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--days", type=int, help="50-100 (default: zufällig)")
    p.add_argument("--shift", choices=list(SHIFTS),
                   help="früh/spät/nacht -- default: zufällig")
    p.add_argument("--seed", type=int, help="für reproduzierbare Läufe")
    p.add_argument("--out", default="data/generated_trips/routine.csv")
    a = p.parse_args()
    generate(days=a.days, shift=a.shift, seed=a.seed, out_path=a.out)
