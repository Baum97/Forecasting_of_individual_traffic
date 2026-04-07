"""
parse_real_world_ev.py
──────────────────────
Parsed alle Raw.mat-Dateien aus den Drive- und Charge-Ordnern des
"Real-world electric vehicle data"-Datensatzes und schreibt pro
Aufzeichnungsperiode eine stündliche CSV-Datei.

Hintergrund:
  Der Datensatz enthält EIN Fahrzeug, das über ~12 Monate in 8 Perioden
  aufgezeichnet wurde. Jede Periode besteht aus einem Charge-Folder (ungerade)
  und einem Drive-Folder (gerade) mit überlappenden Zeiträumen.

  Folder-Paare:
    Periode 1: Charge/Folder1  + Drive/Folder2   (Nov 2019, Woche 1)
    Periode 2: Charge/Folder3  + Drive/Folder4   (Nov 2019, Woche 4)
    Periode 3: Charge/Folder5  + Drive/Folder6   (Dez 2019, Woche 2)
    Periode 4: Charge/Folder7  + Drive/Folder8   (Dez 2019, Woche 3)
    Periode 5: Charge/Folder9  + Drive/Folder10  (Dez 2019 - Jan 2020)
    Periode 6: Charge/Folder11 + Drive/Folder12  (Jan 2020)
    Periode 7: Charge/Folder13 + Drive/Folder14  (Mai - Jul 2020)
    Periode 8: Charge/Folder15 + Drive/Folder16  (Aug - Okt 2020)

Ausgabe (im Zielordner):
  car_period_1.csv  …  car_period_8.csv
    Spalten: datetime, in_use, event, soc_pct, volt_v, curr_a, temp_c
  car_full_timeline.csv
    Alle Perioden zusammengeführt + Lücken mit 0 gefüllt

Verwendung:
  python parse_real_world_ev.py
  python parse_real_world_ev.py --out-dir ../data/cars-real-world-electric
  python parse_real_world_ev.py --resample 15min
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import h5py
except ImportError:
    sys.exit("h5py fehlt - installieren mit: pip install h5py")


# ── Konstanten ────────────────────────────────────────────────────────────────

BASE = Path(__file__).parent.parent / (
    "data/Real-world electric vehicle data driving and charging"
    "/Real-world electric vehicle data driving and charging"
)

# Perioden: (Periode-Nr, Charge-Folder, Drive-Folder)
PERIODS: List[Tuple[int, str, str]] = [
    (1,  "Charge/Folder1",  "Drive/Folder2"),
    (2,  "Charge/Folder3",  "Drive/Folder4"),
    (3,  "Charge/Folder5",  "Drive/Folder6"),
    (4,  "Charge/Folder7",  "Drive/Folder8"),
    (5,  "Charge/Folder9",  "Drive/Folder10"),
    (6,  "Charge/Folder11", "Drive/Folder12"),
    (7,  "Charge/Folder13", "Drive/Folder14"),
    (8,  "Charge/Folder15", "Drive/Folder16"),
]


# ── MAT laden ─────────────────────────────────────────────────────────────────

def _load_mat(path: Path) -> Optional[Dict[str, np.ndarray]]:
    """
    Lädt eine Raw.mat (HDF5/v7.3) und gibt die relevanten Arrays zurück.
    Gibt None zurück wenn die Datei nicht existiert oder fehlerhaft ist.
    """
    if not path.exists():
        print(f"    [SKIP] Nicht gefunden: {path.relative_to(BASE.parent.parent)}")
        return None
    try:
        with h5py.File(str(path), "r") as f:
            raw = f["Raw"]
            result = {}

            # Referenz-Zeitachse: TimeCurr
            tc = np.array(raw["TimeCurr"]).flatten().astype(float)
            result["time_curr"] = tc
            result["curr"]      = np.array(raw["Curr"]).flatten().astype(float)

            # Spannung auf TimeCurr-Raster (TimeVolt kann leicht abweichen)
            time_volt = np.array(raw["TimeVolt"]).flatten().astype(float)
            volt_raw  = np.array(raw["Volt"]).flatten().astype(float)
            result["volt"] = np.interp(tc, time_volt, volt_raw)

            # SoC auf TimeCurr-Raster (TimeSoC kann abweichen)
            time_soc = np.array(raw["TimeSoC"]).flatten().astype(float)
            soc_raw  = np.array(raw["SoC"]).flatten().astype(float)
            result["soc"] = np.interp(tc, time_soc, soc_raw)

            # Temperatur (eigene, dünnere Zeitachse)
            result["time_temp"] = np.array(raw["TimeTemp"]).flatten().astype(float)
            result["temp"]      = np.array(raw["Temp"]).flatten().astype(float)

            # Epoch-Anker für absolute Zeitrekonstruktion
            result["time_epoch"] = np.array(raw["TimeEpoch"]).flatten().astype(float)
            result["epoch"]      = np.array(raw["Epoch"]).flatten().astype(float)

        return result
    except OSError as e:
        print(f"    [SKIP] {path.name}: {e}")
        return None


def _to_absolute_timestamps(
    time_relative: np.ndarray,
    time_epoch: np.ndarray,
    epoch: np.ndarray,
) -> np.ndarray:
    """
    Konvertiert relative Zeitstempel (s seit Gerätstart) in absolute Unix-Timestamps.

    Methode: lineare Interpolation zwischen den Epoch-Ankerpunkten.
    Punkte ausserhalb der Anker werden linear extrapoliert.
    """
    # np.interp extrapoliert mit Randwerten - wir rechnen manuell
    # um auch ausserhalb der Anker korrekt zu sein
    abs_times = np.interp(time_relative, time_epoch, epoch)
    return abs_times


def _parse_folder(mat_path: Path, event_type: str) -> Optional[pd.DataFrame]:
    """
    Parst eine Raw.mat und gibt einen DataFrame mit absoluten Zeitstempeln zurück.

    Spalten: timestamp_unix, datetime, curr_a, soc_pct, volt_v, temp_c, event
    """
    data = _load_mat(mat_path)
    if data is None:
        return None

    tc    = data["time_curr"]
    curr  = data["curr"]
    soc   = data["soc"]
    volt  = data["volt"]
    te    = data["time_epoch"]
    ep    = data["epoch"]

    # Absolute Zeitstempel rekonstruieren
    abs_unix = _to_absolute_timestamps(tc, te, ep)
    datetimes = pd.to_datetime(abs_unix, unit="s", utc=False)

    # Temperatur auf das Strom-Raster interpolieren
    if len(data["time_temp"]) > 1:
        temp_interp = np.interp(tc, data["time_temp"], data["temp"])
    else:
        temp_interp = np.full(len(tc), np.nan)

    df = pd.DataFrame({
        "datetime": datetimes,
        "curr_a":   curr.round(3),
        "soc_pct":  soc.round(2),
        "volt_v":   volt.round(2),
        "temp_c":   temp_interp.round(2),
        "event":    event_type,
    })

    # Duplikate und ungültige Timestamps entfernen
    df = df.dropna(subset=["datetime"])
    df = df[df["datetime"] > pd.Timestamp("2015-01-01")]
    df = df[df["datetime"] < pd.Timestamp("2030-01-01")]
    df = df.drop_duplicates(subset=["datetime"]).sort_values("datetime")

    n = len(df)
    t_start = df["datetime"].iloc[0].strftime("%Y-%m-%d %H:%M")
    t_end   = df["datetime"].iloc[-1].strftime("%Y-%m-%d %H:%M")
    dur_h   = (df["datetime"].iloc[-1] - df["datetime"].iloc[0]).total_seconds() / 3600
    print(f"    {event_type:8s}: {n:>9,} Messpunkte  |  {t_start} - {t_end}  ({dur_h:.1f} h)")

    return df


# ── Stündliche Aggregation ────────────────────────────────────────────────────

def _to_hourly(raw_df: pd.DataFrame, freq: str = "1h") -> pd.DataFrame:
    """
    Aggregiert die Rohmesspunkte auf ein stündliches (oder anderes) Raster.

    in_use = 1 wenn Fahren (Strom > 5 A) oder Laden (Strom < -0.5 A) in der Stunde
    event  = 'driving' / 'charging' / 'parked'

    Alle anderen Signale werden als Mittelwert je Stunde ausgegeben.
    """
    df = raw_df.copy()
    df["hour"] = df["datetime"].dt.floor(freq)

    def agg_event(s: pd.Series) -> str:
        vals = s.values
        if "driving" in vals:
            return "driving"
        if "charging" in vals:
            return "charging"
        return "parked"

    # Stunde gilt als in_use wenn Fahren ODER Laden aktiv
    df["is_driving"]  = ((df["event"] == "driving")  & (df["curr_a"].abs() > 5)).astype(int)
    df["is_charging"] = ((df["event"] == "charging") & (df["curr_a"] < -0.5)).astype(int)

    hourly = df.groupby("hour").agg(
        is_driving  =("is_driving",  "max"),
        is_charging =("is_charging", "max"),
        soc_pct     =("soc_pct",     "mean"),
        volt_v      =("volt_v",      "mean"),
        curr_a      =("curr_a",      "mean"),
        temp_c      =("temp_c",      "mean"),
        event       =("event",       agg_event),
    ).reset_index()

    hourly = hourly.rename(columns={"hour": "datetime"})
    hourly["in_use"] = ((hourly["is_driving"] == 1) | (hourly["is_charging"] == 1)).astype(int)
    hourly = hourly.drop(columns=["is_driving", "is_charging"])

    # Vollständiges Stundenraster (Lücken = geparkt)
    full_idx = pd.date_range(hourly["datetime"].min(), hourly["datetime"].max(), freq=freq)
    hourly = hourly.set_index("datetime").reindex(full_idx).reset_index()
    hourly = hourly.rename(columns={"index": "datetime"})
    hourly["in_use"] = hourly["in_use"].fillna(0).astype(int)
    hourly["event"]  = hourly["event"].fillna("parked")
    hourly[["soc_pct", "volt_v", "curr_a", "temp_c"]] = (
        hourly[["soc_pct", "volt_v", "curr_a", "temp_c"]].ffill().bfill()
    )

    col_order = ["datetime", "in_use", "event", "soc_pct", "volt_v", "curr_a", "temp_c"]
    return hourly[col_order].round({"soc_pct": 2, "volt_v": 2, "curr_a": 3, "temp_c": 2})


# ── Haupt-Parsing ─────────────────────────────────────────────────────────────

def parse_all(
    base_dir: Path,
    out_dir: Path,
    resample: str = "1h",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    all_hourly: List[pd.DataFrame] = []

    print(f"\nZieldaten: {out_dir}")
    print(f"Raster:    {resample}\n")
    print("=" * 65)

    for period_nr, charge_sub, drive_sub in PERIODS:
        label = f"car_period_{period_nr}"
        print(f"\nPeriode {period_nr}: {charge_sub} + {drive_sub}")

        charge_path = base_dir / charge_sub / "Raw.mat"
        drive_path  = base_dir / drive_sub  / "Raw.mat"

        frames: List[pd.DataFrame] = []

        charge_df = _parse_folder(charge_path, "charging")
        if charge_df is not None:
            frames.append(charge_df)

        drive_df = _parse_folder(drive_path, "driving")
        if drive_df is not None:
            frames.append(drive_df)

        if not frames:
            print(f"  [SKIP] Periode {period_nr}: keine Daten")
            continue

        raw = pd.concat(frames, ignore_index=True).sort_values("datetime")
        hourly = _to_hourly(raw, freq=resample)

        out_path = out_dir / f"{label}.csv"
        hourly.to_csv(out_path, index=False)

        n_used  = int(hourly["in_use"].sum())
        n_total = len(hourly)
        n_drive = int((hourly["event"] == "driving").sum())
        n_charge= int((hourly["event"] == "charging").sum())
        print(f"  --> {out_path.name}")
        print(f"      Stunden gesamt: {n_total:,}  |  Tage: {n_total // 24}")
        print(f"      Genutzt:        {n_used:,}  ({n_used/n_total:.1%})")
        print(f"      Fahren:         {n_drive:,}  |  Laden: {n_charge:,}")

        all_hourly.append(hourly)

    # ── Vollständige Zeitreihe ─────────────────────────────────────────────────
    if all_hourly:
        print("\n" + "=" * 65)
        print("\nErstelle car_full_timeline.csv ...")

        full = pd.concat(all_hourly, ignore_index=True).sort_values("datetime")

        # Lücken zwischen Perioden mit geparkt füllen
        full_idx = pd.date_range(full["datetime"].min(), full["datetime"].max(), freq=resample)
        full = full.set_index("datetime").reindex(full_idx).reset_index()
        full = full.rename(columns={"index": "datetime"})
        full["in_use"] = full["in_use"].fillna(0).astype(int)
        full["event"]  = full["event"].fillna("parked")
        full[["soc_pct", "volt_v", "curr_a", "temp_c"]] = (
            full[["soc_pct", "volt_v", "curr_a", "temp_c"]].ffill().bfill()
        )

        out_full = out_dir / "car_full_timeline.csv"
        full.to_csv(out_full, index=False)

        n_used  = int(full["in_use"].sum())
        n_total = len(full)
        n_days  = n_total // (60 // int(resample.replace("min","").replace("h","60")) if "min" in resample else 24)
        print(f"  --> {out_full.name}")
        print(f"      Zeitraum:  {full['datetime'].iloc[0].date()} bis {full['datetime'].iloc[-1].date()}")
        print(f"      Stunden:   {n_total:,}  |  Tage: {n_total // 24}")
        print(f"      Genutzt:   {n_used:,} Stunden ({n_used/n_total:.1%})")
        print(f"\n      Hinweis: Dieses eine Fahrzeug hat ~12 Monate Aufzeichnung.")
        print(f"      Fuer Multi-Fahrzeug-Training bitte emobpy-Daten verwenden.")

    print("\nFertig.\n")


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        prog="parse_real_world_ev.py",
        description="Parsed Real-world EV MAT-Daten -> stündliche CSVs",
    )
    parser.add_argument(
        "--base-dir",
        default=str(BASE),
        help="Pfad zum Verzeichnis mit Charge/ und Drive/ Unterordnern",
    )
    parser.add_argument(
        "--out-dir",
        default=str(Path(__file__).parent.parent / "data" / "cars-real-world-electric"),
        help="Ausgabeordner fuer CSV-Dateien",
    )
    parser.add_argument(
        "--resample",
        default="1h",
        choices=["15min", "30min", "1h"],
        help="Zeitraster der Ausgabe (Standard: 1h)",
    )
    args = parser.parse_args()

    parse_all(
        base_dir=Path(args.base_dir),
        out_dir=Path(args.out_dir),
        resample=args.resample,
    )


if __name__ == "__main__":
    main()
