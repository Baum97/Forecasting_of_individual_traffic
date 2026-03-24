# E-Mobility Time Series Forecasting (2-7 Tage)

Dieses Setup trainiert ein globales ML-Modell ueber viele Fahrzeuge und erstellt Vorhersagen fuer die naechsten 2 bis 7 Tage auf Basis von ca. 100 vergangenen Tagen.

## Modellidee

- Direkte Multi-Horizon-Prognose: ein Modell pro Horizont (Tag +2 bis Tag +7)
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
  --horizons 2,3,4,5,6,7 `
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
  --horizons 2,3,4,5,6,7 `
  --model-out .\models\emobility_forecaster.joblib `
  --metrics-out .\models\validation_metrics.csv
```

## Forecast fuer eine Person

Eingabe ist eine CSV mit taeglichen Werten aus mindestens 100 Tagen, z. B.:
- `date`
- `target`

```powershell
python emobility_forecaster.py predict `
  --model-in .\models\emobility_forecaster.joblib `
  --history-path .\examples\person_100_days.csv `
  --date-col date `
  --target-col target `
  --horizons 2,3,4,5,6,7 `
  --pred-out .\predictions\person_forecast.csv
```

Ausgabe:
- `forecast_day` (z. B. 2,3,...,7)
- `date`
- `prediction`

## Hinweise

- Wenn Werte in untertaeglichen Zeitreihen vorliegen (z. B. 15 Minuten oder 1 Stunde), werden diese zuerst auf Tageswerte aggregiert (`sum` oder `mean`).
- Fuer robuste Modelle pro Fahrzeug sollten deutlich mehr als 100 Tage Trainingsdaten je Fahrzeug vorliegen.
- Wenn du willst, kann ich als naechsten Schritt noch einen Backtesting-Workflow (rolling origin) und einen Baseline-Vergleich (naiver Wochenzyklus) einbauen.
