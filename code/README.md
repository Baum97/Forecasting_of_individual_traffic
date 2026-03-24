# E-Mobility Time Series Forecasting (1-7 Tage)

Dieses Setup trainiert ein globales ML-Modell ueber viele Fahrzeuge und erstellt Vorhersagen fuer die naechsten 1 bis 7 Tage auf Basis von ca. 100 vergangenen Tagen.

## Modellidee

- Direkte Multi-Horizon-Prognose: ein Modell pro Horizont (Tag +1 bis Tag +7)
- Features aus den letzten 100 Tagen:
  - Lags (`lag_1` bis `lag_100`)
  - Rolling-Statistiken (Mittelwert, Std, Min/Max)
  - Wochenzyklus des Prognosetages (sin/cos)
- Modelltyp: `RandomForestRegressor`

Das funktioniert gut, wenn viele Fahrzeuge fuer das Training vorhanden sind und fuer eine einzelne Person eine stabile Kurzfristprognose benoetigt wird.

## Datenformate

### 1) Long-Format (empfohlen)
CSV mit mindestens:
- `vehicle_id`
- `date` (Zeitstempel)
- Zielspalte, z. B. `distance_km`

### 2) emobpy-Format
Direkt die emobpy-Zeitreihen-Datei (`emobpy_timeseries_hourly.csv` oder `emobpy_timeseries_original.csv`).
Das Skript verarbeitet die 2 Header-Zeilen + Metazeile automatisch.

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

Beispiel fuer stundenbasierte Distanz (`VehicleMobility__Distance_km`):

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

## Forecast fuer eine Person

Eingabe ist eine CSV mit taeglichen Werten aus mindestens 100 Tagen, z. B.:
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

Konsolen-Ausgabe enthaelt automatisch:
- Gesamtdistanz fuer die naechsten 3 Tage (inkl. Unsicherheitsband)
- Geschaetzten Akku-Bedarf in % (bei gleichbleibendem Fahrverhalten)
- Optional verbleibenden SoC, wenn `--current-soc-percent` gesetzt ist

CSV-Ausgabe:
- `forecast_day` (z. B. 1,2,...,7)
- `date`
- `prediction`
- `p10`, `p50`, `p90`, `std`

Typischer Satz im Output:
- `Geschaetzter Akku-Bedarf naechste 3 Tage: ca. 30.0% (...)`

## Direkte Reichweiten-Schaetzung mit Unsicherheit

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

Hinweis: Die Szenario-Werte (`bad/avg/good`) werden auf andere Akku-Zustaende skaliert. Wenn du z. B. `--soc-percent 60` setzt, steigen die Distanzen entsprechend.

## Hinweise

- Wenn Werte in untertaeglichen Zeitreihen vorliegen (z. B. 15 Minuten oder 1 Stunde), werden diese zuerst auf Tageswerte aggregiert (`sum` oder `mean`).
- Fuer robuste Modelle pro Fahrzeug sollten deutlich mehr als 100 Tage Trainingsdaten je Fahrzeug vorliegen.
- Wenn du willst, kann ich als naechsten Schritt noch einen Backtesting-Workflow (rolling origin) und einen Baseline-Vergleich (naiver Wochenzyklus) einbauen.
