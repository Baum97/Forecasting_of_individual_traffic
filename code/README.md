# E-Mobility Time Series Forecasting (1-7 Tage)

Dieses Setup trainiert ein globales ML-Modell über viele Fahrzeuge und erstellt Vorhersagen für die nächsten 1 bis 7 Tage auf Basis von ca. 100 vergangenen Tagen.

## Modellidee

- Direkte Multi-Horizon-Prognose: ein Modell pro Horizont (Tag +1 bis Tag +7)
- Features aus den letzten 100 Tagen:
  - Lags (`lag_1` bis `lag_100`)
  - Rolling-Statistiken (Mittelwert, Std, Min/Max)
  - Wochenzyklus des Prognosetages (sin/cos)
- Modelltyp: `RandomForestRegressor`

Funktioniert gut, wenn viele Fahrzeuge für Training vorhanden sind und für einzelne Person stabile Prognose benötigt wird.

## Datenformate

### 1) Long-Format (empfohlen)
CSV mit mindestens:
- `vehicle_id`
- `date` (Zeitstempel)
- Zielspalte, z. B. `distance_km`

### 2) emobpy-Format
Direkt die emobpy-Zeitreihen-Datei (`emobpy_timeseries_hourly.csv` oder `emobpy_timeseries_original.csv`).
Header-Zeilen + Metazeile werden automatisch verarbeitet.

## Installation

```powershell
cd d:\Projects\Forecasting_of_individual_traffic\code
pip install -r requirements.txt
```

## Training

### Long-Format

```powershell
python emobility_forecaster.py train `
  --data-path ..\data\my_long_timeseries.csv `
  --data-format long `
  --vehicle-col vehicle_id `
  --date-col date `
  --target-col distance_km `
  --history-days 100 `
  --horizons 1,2,3,4,5,6,7 `
  --model-out .\models\emobility_forecaster.joblib `
  --metrics-out .\models\validation_metrics.csv
```

### emobpy-Datei

Beispiel für stundenbasierte Distanz (`VehicleMobility__Distance_km`):

```powershell
python emobility_forecaster.py train `
  --data-path ..\data\4514928\emobpy_timeseries_hourly\emobpy_timeseries_hourly.csv `
  --data-format emobpy `
  --vehicle-col ID__ID `
  --date-col date `
  --target-col VehicleMobility__Distance_km `
  --agg sum `
  --history-days 100 `
  --horizons 1,2,3,4,5,6,7 `
  --model-out .\models\emobility_forecaster.joblib `
  --metrics-out .\models\validation_metrics.csv
```

## Forecast für eine Person

Eingabe ist eine CSV mit täglichen Werten aus mindestens 100 Tagen, z. B.:
- `date`
- `target` (z. B. Distanz in km)
- optional `energy_kwh` pro Tag oder `soc_used_percent` pro Tag

```powershell
python emobility_forecaster.py predict `
  --model-in .\models\emobility_forecaster.joblib `
  --history-path .\examples\person_100_days.csv `
  --date-col date `
  --target-col target `
  --horizons 1,2,3,4,5,6,7 `
  --summary-days 3 `
  --energy-col energy_kwh `
  --battery-capacity-kwh 55 `
  --current-soc-percent 30 `
  --pred-out .\predictions\person_forecast.csv `
  --plot-out .\predictions\person_forecast_uncertainty.png
```

Konsolen-Ausgabe enthält automatisch:
- Gesamtdistanz für die nächsten 3 Tage (inkl. Unsicherheitsband)
- Geschätzten Akku-Bedarf in % (bei gleichbleibendem Fahrverhalten)
- Optional verbleibenden SoC, wenn `--current-soc-percent` gesetzt ist

CSV-Ausgabe:
- `forecast_day` (z. B. 1,2,...,7)
- `date`
- `prediction`
- `p10`, `p50`, `p90`, `std`

Typischer Satz im Output:
- `Geschätzter Akku-Bedarf nächste 3 Tage: ca. 30.0% (...)`

## Forecast für mehrere Personen (mehrere History-Dateien)

Wenn du mehrere Personen/Fahrzeuge gleichzeitig aus einzelnen 100-Tage-History-Dateien vorhersagen willst, nutze `predict-batch`.

Manifest-Datei (Template):
- `examples/predict_batch_manifest_template.csv`
- Pflichtspalte: `history_path`
- Optional: `label`, `current_soc_percent`, `battery_capacity_kwh`, `capacity_ah`, `nominal_voltage`

```powershell
python emobility_forecaster.py predict-batch `
  --model-in .\models\emobility_forecaster.joblib `
  --batch-path .\examples\predict_batch_manifest_template.csv `
  --history-col history_path `
  --label-col label `
  --date-col date `
  --target-col target `
  --horizons 1,2,3,4,5,6,7 `
  --summary-days 3 `
  --energy-col energy_kwh `
  --plot-out .\predictions\predict_batch_5_boxplots.png `
  --stats-out .\predictions\predict_batch_5_stats.csv `
  --pred-out-dir .\predictions\persons
```

Ergebnis:
- Ein gemeinsamer Graph mit mehreren Boxplots (eine Box pro Person)
- Jeder Boxplot zeigt die Unsicherheitsverteilung der Gesamtdistanz für die nächsten `summary_days` Tage
- Optional Akku-Bedarf in % je Person in der Stats-CSV (wenn Verbrauchsspalten vorhanden sind)

Validierung (nur prüfen, ohne Forecast):

```powershell
python emobility_forecaster.py predict-batch `
  --model-in .\models\emobility_forecaster.joblib `
  --batch-path .\examples\predict_batch_manifest_template.csv `
  --plot-out .\predictions\predict_batch_5_boxplots.png `
  --validate-only
```

Geprüft werden u. a.:
- Existenz der Manifest- und History-Dateien
- Pflichtspalten (`date`, `target` bzw. konfiguriert)
- Gültige Datum-/Target-Werte
- Mindestanzahl gültiger Historien-Tage (mindestens `history_days`)
- Gültiger `current_soc_percent` im Bereich 0 bis 100 (falls gesetzt)

## Was bedeuten P10, P50, P90?

Ja, das sind Perzentile:
- `P10`: 10. Perzentil, d. h. 10% der möglichen Werte liegen darunter
- `P50`: 50. Perzentil (Median)
- `P90`: 90. Perzentil, d. h. 90% der möglichen Werte liegen darunter

Interpretation im Kontext der Prognose:
- Ein Bereich `P10-P90` ist ein Unsicherheitsband, das etwa die mittleren 80% plausibler Verläufe abdeckt.

## Direkte Reichweiten-Schätzung mit Unsicherheit

Wenn du sofort eine sinnvolle Reichweiten-Aussage aus Akkuwerten willst (inkl. Unsicherheits-Graph), nutze den `range`-Befehl.

Beispiel passend zu deinem Szenario:

```powershell
python emobility_forecaster.py range `
  --soc-percent 30 `
  --capacity-ah 33 `
  --nominal-voltage 400 `
  --bad-km 20 `
  --avg-min-km 35 `
  --avg-max-km 50 `
  --good-km 70 `
  --plot-out .\predictions\range_uncertainty.png `
  --stats-out .\predictions\range_stats.csv
```

Konsolen-Ausgabe (Beispiel):
- Akku: `30.0% / 33.0Ah`
- Nutzbare Energie: `x.xx kWh`
- Distanz (schlechtes Fahrverhalten): `20.0 km`
- Distanz (Durchschnitt): `35.0-50.0 km`
- Distanz (gutes Fahrverhalten): `70.0 km`
- Erwartungswert und P10/P50/P90

Dateien:
- Plot: `predictions/range_uncertainty.png`
- Statistik: `predictions/range_stats.csv`

Hinweis: Die Szenario-Werte (`bad/avg/good`) werden auf andere Akku-Zustände skaliert. Wenn du z. B. `--soc-percent 60` setzt, steigen die Distanzen entsprechend.

## Mehrere Datensätze als mehrere Boxplots

Wenn du mehrere Reihen gleichzeitig plotten willst (z. B. 5 Personen/Fahrzeuge), nutze `range-batch`.

Beispiel-CSV:
- `examples/range_batch_5_cases.csv`
- Pflichtspalten: `soc_percent`, `capacity_ah`
- Optional: `label`, `nominal_voltage`, `reserve_percent`, `bad_km`, `avg_min_km`, `avg_max_km`, `good_km`, `ref_soc_percent`, `ref_capacity_ah`, `samples`

```powershell
python emobility_forecaster.py range-batch `
  --batch-path .\examples\range_batch_5_cases.csv `
  --label-col label `
  --plot-out .\predictions\range_batch_5_boxplots.png `
  --stats-out .\predictions\range_batch_5_stats.csv
```

Ergebnis:
- Ein gemeinsamer Graph mit mehreren horizontalen Reichweiten-Boxplots (z. B. 5 Boxen)
- CSV mit Kennzahlen je Zeile (Mean, P10, P50, P90 usw.)

## Hinweise

- Wenn Werte in untertäglichen Zeitreihen vorliegen (z. B. 15 Minuten oder 1 Stunde), werden diese zürst auf Tageswerte aggregiert (`sum` oder `mean`).
- Für robuste Modelle pro Fahrzeug sollten deutlich mehr als 100 Tage Trainingsdaten je Fahrzeug vorliegen.
- Wenn du willst, kann ich als nächsten Schritt noch einen Backtesting-Workflow (rolling origin) und einen Baseline-Vergleich (naiver Wochenzyklus) einbauen.
