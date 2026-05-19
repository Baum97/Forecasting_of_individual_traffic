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


# Registry -------------------------------------------------------------------

SOURCES = {
    "emobpy": load_emobpy,
    "real_world_ev": load_real_world_ev,
    "ved": load_ved,
    "routine": load_routine,
}


def load_source(name: str, **kwargs) -> pd.DataFrame:
    if name not in SOURCES:
        raise ValueError(f"unknown source '{name}', expected one of {list(SOURCES)}")
    return SOURCES[name](**kwargs)
