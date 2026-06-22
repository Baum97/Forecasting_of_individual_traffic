"""
Unified data adapters for all four data sources.

Every loader returns an hourly DataFrame with three columns:
    vehicle_id : str
    timestamp  : pd.Timestamp (hourly grid, no gaps within a vehicle)
    driving    : int (0 = parked, 1 = driving)

Source mapping:
    emobpy        -> data/emobpy-vehicles/vehicle_*.csv         (200 vehicles, 1 year)
    real_world_ev -> data/cars-real-world-electric/car_full_timeline.csv
    ved           -> data/VED/VED_DynamicData_Part{1,2}/*.csv  (DayNum-based)
    routine       -> data/generated_trips/routine.csv          (single synthetic person)
    yjmob         -> data/yjmob100k/task2_dataset_kotae.csv    (human GPS traces; driving = movement proxy)

Hinweis zu yjmob: Smartphone-Bewegungsdaten von Personen (kein Fahrzeug,
kein Verkehrstraeger). ``driving`` wird hier als Bewegungs-Proxy abgeleitet
(Rasterzelle gewechselt = aktiv) und bedeutet "in Bewegung", nicht woertlich
"faehrt".
"""
from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

DATA_ROOT = Path(__file__).resolve().parents[2] / "data"


# emobpy ---------------------------------------------------------------------

def load_emobpy(limit_vehicles: Optional[int] = None) -> pd.DataFrame:
    src = DATA_ROOT / "emobpy-vehicles"
    files = sorted(src.glob("vehicle_*.csv"))
    if limit_vehicles is not None:
        files = files[:limit_vehicles]

    frames = []
    for f in files:
        df = pd.read_csv(f, parse_dates=["datetime"], usecols=["datetime", "in_use"])
        df = df.rename(columns={"datetime": "timestamp", "in_use": "driving"})
        df["vehicle_id"] = f.stem
        frames.append(df)

    out = pd.concat(frames, ignore_index=True)
    out["driving"] = pd.to_numeric(out["driving"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    return out[["vehicle_id", "timestamp", "driving"]]


# real_world_ev --------------------------------------------------------------

def load_real_world_ev() -> pd.DataFrame:
    src = DATA_ROOT / "cars-real-world-electric" / "car_full_timeline.csv"
    df = pd.read_csv(src, parse_dates=["datetime"], usecols=["datetime", "in_use"])
    df = df.rename(columns={"datetime": "timestamp", "in_use": "driving"})
    df["vehicle_id"] = "real_world_ev_v1"
    df["driving"] = pd.to_numeric(df["driving"], errors="coerce").fillna(0).clip(0, 1).astype(int)
    return df[["vehicle_id", "timestamp", "driving"]]


# generated routine ----------------------------------------------------------

def load_routine() -> pd.DataFrame:
    src = DATA_ROOT / "generated_trips" / "routine.csv"
    df = pd.read_csv(src, parse_dates=["date"])

    start = df["date"].min()
    end = df["date"].max() + pd.Timedelta(days=1)
    grid = pd.date_range(start, end, freq="h", inclusive="left")

    out = pd.DataFrame({"timestamp": grid})
    out["vehicle_id"] = "routine_person_1"
    out["driving"] = 0

    grid_end = out["timestamp"] + pd.Timedelta(hours=1)
    for _, row in df.iterrows():
        trip_start = row["date"] + pd.Timedelta(minutes=int(row["start_min"]))
        trip_end = row["date"] + pd.Timedelta(minutes=int(row["end_min"]))
        mask = (out["timestamp"] < trip_end) & (grid_end > trip_start)
        out.loc[mask, "driving"] = 1

    return out[["vehicle_id", "timestamp", "driving"]]


# VED ------------------------------------------------------------------------

VED_EPOCH = pd.Timestamp("2017-11-01")


def load_ved(
    limit_vehicles: Optional[int] = 20,
    limit_files: Optional[int] = 6,
) -> pd.DataFrame:
    """VED logs ~1 Hz only while driving; gaps imply parked.

    DayNum is the decimal number of days since 2017-11-01 00:00 (DayNum=1.0).
    """
    files = sorted((DATA_ROOT / "VED" / "VED_DynamicData_Part1").glob("VED_*.csv"))
    files += sorted((DATA_ROOT / "VED" / "VED_DynamicData_Part2").glob("VED_*.csv"))
    if limit_files is not None:
        files = files[:limit_files]

    frames = []
    for f in files:
        df = pd.read_csv(f, usecols=["DayNum", "VehId"])
        df["timestamp"] = VED_EPOCH + pd.to_timedelta(df["DayNum"] - 1.0, unit="D")
        df["timestamp"] = df["timestamp"].dt.floor("h")
        frames.append(df[["VehId", "timestamp"]].drop_duplicates())

    drives = pd.concat(frames, ignore_index=True).drop_duplicates()

    if limit_vehicles is not None:
        top = (
            drives.groupby("VehId").size().sort_values(ascending=False).head(limit_vehicles).index
        )
        drives = drives[drives["VehId"].isin(top)]

    parts = []
    for vid, grp in drives.groupby("VehId"):
        ts_min = grp["timestamp"].min().normalize()
        ts_max = (grp["timestamp"].max() + pd.Timedelta(hours=1)).normalize()
        full = pd.date_range(ts_min, ts_max, freq="h", inclusive="left")
        d = pd.DataFrame({"timestamp": full})
        d["vehicle_id"] = f"ved_{vid}"
        d["driving"] = d["timestamp"].isin(set(grp["timestamp"])).astype(int)
        parts.append(d)

    return pd.concat(parts, ignore_index=True)[["vehicle_id", "timestamp", "driving"]]


# YJMob100K (human mobility, smartphone GPS) ---------------------------------

YJMOB_EPOCH = pd.Timestamp("2023-01-01")  # synthetisch; Datumsangaben anonymisiert


def load_yjmob(
    limit_vehicles: Optional[int] = 50,
    src: Optional[Path] = None,
) -> pd.DataFrame:
    """YJMob100K Smartphone-Standortspuren (KEINE Fahrzeugdaten).

    Rohspalten: uid, d (Tag), t (30-Min-Slot 0..47), x, y (200x200-Rasterzelle).
    Es gibt kein driving-Label und kein Fahrzeug; ``driving`` wird als
    Bewegungs-Proxy abgeleitet: Ein Slot gilt als aktiv, wenn sich die
    Rasterzelle (x, y) gegenueber dem vorigen *beobachteten* Slot derselben
    Person geaendert hat. Anschliessend auf ein Stundenraster aggregiert
    (Stunde aktiv, wenn mind. ein 30-Min-Slot Bewegung zeigt). Stunden ohne
    Beobachtung gelten als stationaer (0) -- eine bewusste Annahme.
    """
    if src is None:
        src = DATA_ROOT / "yjmob100k" / "task2_dataset_kotae.csv"

    df = pd.read_csv(src, usecols=["uid", "d", "t", "x", "y"])

    if limit_vehicles is not None:
        keep = sorted(df["uid"].unique())[:limit_vehicles]
        df = df[df["uid"].isin(keep)]

    df = df.sort_values(["uid", "d", "t"], kind="stable")

    # Bewegungs-Proxy: Zelle gegenueber vorigem beobachteten Slot geaendert?
    prev_x = df.groupby("uid")["x"].shift()
    prev_y = df.groupby("uid")["y"].shift()
    df["moved"] = (((df["x"] != prev_x) | (df["y"] != prev_y)) & prev_x.notna()).astype(int)

    # 30-Min-Slot -> Stunden-Zeitstempel
    df["timestamp"] = (
        YJMOB_EPOCH
        + pd.to_timedelta(df["d"], unit="D")
        + pd.to_timedelta(df["t"].astype("int64") * 30, unit="min")
    ).dt.floor("h")

    # Stunde aktiv, wenn ein beobachteter Slot darin Bewegung zeigt
    active = df.groupby(["uid", "timestamp"])["moved"].max().reset_index()

    parts = []
    for uid, grp in active.groupby("uid"):
        ts_min = grp["timestamp"].min().normalize()
        ts_max = (grp["timestamp"].max() + pd.Timedelta(hours=1)).normalize()
        full = pd.date_range(ts_min, ts_max, freq="h", inclusive="left")
        d = pd.DataFrame({"timestamp": full})
        d["vehicle_id"] = f"yjmob_{uid}"
        act = dict(zip(grp["timestamp"], grp["moved"]))
        d["driving"] = d["timestamp"].map(act).fillna(0).astype(int)
        parts.append(d)

    return pd.concat(parts, ignore_index=True)[["vehicle_id", "timestamp", "driving"]]


# Registry -------------------------------------------------------------------

SOURCES = {
    "emobpy": load_emobpy,
    "real_world_ev": load_real_world_ev,
    "ved": load_ved,
    "routine": load_routine,
    "yjmob": load_yjmob,
}


def load_source(name: str, **kwargs) -> pd.DataFrame:
    if name not in SOURCES:
        raise ValueError(f"unknown source '{name}', expected one of {list(SOURCES)}")
    return SOURCES[name](**kwargs)