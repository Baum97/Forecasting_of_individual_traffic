"""
prepare_emobpy_vehicles.py
──────────────────────────
Konvertiert die emobpy_timeseries_original.csv (200 Fahrzeuge, 15-min-Raster,
gestapelt) in 200 einzelne stündliche CSV-Dateien, die direkt von
time_pattern_forecaster.py (fit-multi) verwendet werden können.

Datenstruktur der Quelle:
  - 7.008.000 Zeilen = 200 Fahrzeuge × 365 Tage × 96 Zeitschritte (15 min)
  - Fahrzeuge stehen untereinander (Long-Format nach ID)
  - Spalten: datetime, ID, Location, Distance_km, Consumption_kWh,
             ChargingStation, PowerRating_kW, ...

Ausgabe (im Zielordner, z.B. data/emobpy-vehicles/):
  vehicle_000.csv  ...  vehicle_199.csv
    Spalten: datetime, in_use, location, distance_km, consumption_kwh

Verwendung:
  python prepare_emobpy_vehicles.py
  python prepare_emobpy_vehicles.py --out-dir ../data/emobpy-vehicles
  python prepare_emobpy_vehicles.py --resample 1h --n-vehicles 50
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

SRC_DEFAULT = (
    Path(__file__).parent.parent
    / "data/4514928/emobpy_timeseries_original/emobpy_timeseries_original.csv"
)
OUT_DEFAULT = Path(__file__).parent.parent / "data/emobpy-vehicles"

# Anzahl 15-min-Schritte pro Fahrzeug (1 Jahr = 365 Tage × 96)
STEPS_PER_VEHICLE = 365 * 96   # = 35.040


def _load_one_vehicle(
    path: Path,
    vehicle_index: int,
    resample: str = "1h",
) -> pd.DataFrame:
    """
    Liest die Zeilen eines einzelnen Fahrzeugs direkt per skiprows/nrows.

    Rückgabe: DataFrame [datetime, in_use, location, distance_km, consumption_kwh]
              auf stündlichem (oder gewähltem) Raster.
    """
    skip = 3 + vehicle_index * STEPS_PER_VEHICLE   # 3 Header-Zeilen
    df = pd.read_csv(
        path,
        skiprows=skip,
        nrows=STEPS_PER_VEHICLE,
        header=None,
        usecols=[0, 1, 2, 3, 4],
        names=["datetime", "ID", "location", "distance_km", "consumption_kwh"],
        parse_dates=["datetime"],
    )

    df["distance_km"]    = pd.to_numeric(df["distance_km"],    errors="coerce").fillna(0.0)
    df["consumption_kwh"]= pd.to_numeric(df["consumption_kwh"],errors="coerce").fillna(0.0)

    # in_use = 1 wenn das Fahrzeug fährt (Location == 'driving' ODER Distance > 0)
    df["in_use"] = ((df["location"] == "driving") | (df["distance_km"] > 0)).astype(int)

    # Auf gewünschtes Zeitraster aggregieren
    df = df.set_index("datetime")
    hourly = pd.DataFrame({
        "in_use":          df["in_use"].resample(resample).max(),
        "location":        df["location"].resample(resample).agg(
                               lambda s: "driving" if "driving" in s.values
                               else (s.mode()[0] if len(s) > 0 else "home")
                           ),
        "distance_km":     df["distance_km"].resample(resample).sum(),
        "consumption_kwh": df["consumption_kwh"].resample(resample).sum(),
    }).reset_index()
    hourly = hourly.rename(columns={"datetime": "datetime"})
    hourly["in_use"] = hourly["in_use"].fillna(0).astype(int)

    return hourly[["datetime", "in_use", "location", "distance_km", "consumption_kwh"]]


def prepare(
    src: Path,
    out_dir: Path,
    resample: str = "1h",
    n_vehicles: int = 200,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    # Gesamtanzahl Fahrzeuge aus Dateigröße schätzen
    with open(src, "r") as f:
        total_rows = sum(1 for _ in f) - 3   # 3 Header-Zeilen
    total_vehicles = total_rows // STEPS_PER_VEHICLE
    n = min(n_vehicles, total_vehicles)

    print(f"\nQuelle:    {src.name}")
    print(f"Ziel:      {out_dir}")
    print(f"Raster:    {resample}")
    print(f"Fahrzeuge: {n} von {total_vehicles} verfügbaren")
    print("=" * 55)

    stats = []

    for i in range(n):
        df = _load_one_vehicle(src, vehicle_index=i, resample=resample)

        usage_rate  = df["in_use"].mean()
        total_km    = df["distance_km"].sum()
        drive_days  = int((df.groupby(df["datetime"].dt.date)["in_use"].max() == 1).sum())
        total_days  = int(df["datetime"].dt.date.nunique())

        out_path = out_dir / f"vehicle_{i:03d}.csv"
        df.to_csv(out_path, index=False)

        stats.append({
            "vehicle_id":   i,
            "file":         out_path.name,
            "days":         total_days,
            "drive_days":   drive_days,
            "usage_rate":   round(usage_rate, 4),
            "total_km":     round(total_km, 1),
        })

        if i % 20 == 0 or i == n - 1:
            print(f"  [{i+1:>3}/{n}] {out_path.name}  "
                  f"Nutzung: {usage_rate:.1%}  "
                  f"Fahrtage: {drive_days}/{total_days}  "
                  f"km: {total_km:.0f}")

    # Übersichts-CSV
    stats_df = pd.DataFrame(stats)
    stats_path = out_dir / "vehicles_summary.csv"
    stats_df.to_csv(stats_path, index=False)

    print("\n" + "=" * 55)
    print(f"Fertig: {n} Fahrzeuge -> {out_dir}")
    print(f"Zusammenfassung: {stats_path.name}")
    print()
    print("Statistiken über alle Fahrzeuge:")
    print(f"  Mittlere Nutzungsrate:  {stats_df['usage_rate'].mean():.1%}")
    print(f"  Mittlere Fahrtage/Jahr: {stats_df['drive_days'].mean():.0f}")
    print(f"  Mittlere km/Jahr:       {stats_df['total_km'].mean():.0f}")
    print()
    print("Nächster Schritt – Modell trainieren:")
    print(f"  python time_pattern_forecaster.py fit-multi \\")
    print(f"    --vehicle-dir {out_dir} \\")
    print(f"    --n-train 140 --n-val 40 \\")
    print(f"    --model-out ./models/emobpy_global.joblib \\")
    print(f"    --metrics-out ./predictions/emobpy_metrics.csv")


def main() -> None:
    parser = argparse.ArgumentParser(
        prog="prepare_emobpy_vehicles.py",
        description="emobpy 200-Fahrzeuge CSV -> 200 Einzel-CSVs für time_pattern_forecaster",
    )
    parser.add_argument("--src", default=str(SRC_DEFAULT),
                        help="Pfad zur emobpy_timeseries_original.csv")
    parser.add_argument("--out-dir", default=str(OUT_DEFAULT),
                        help="Ausgabeordner (wird erstellt falls nicht vorhanden)")
    parser.add_argument("--resample", default="1h", choices=["15min", "30min", "1h"],
                        help="Zeitraster der Ausgabe (Standard: 1h)")
    parser.add_argument("--n-vehicles", type=int, default=200,
                        help="Wie viele Fahrzeuge konvertieren (Standard: alle 200)")
    args = parser.parse_args()

    prepare(
        src=Path(args.src),
        out_dir=Path(args.out_dir),
        resample=args.resample,
        n_vehicles=args.n_vehicles,
    )


if __name__ == "__main__":
    main()
