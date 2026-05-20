export type SimulationType = 'predict' | 'predict-batch' | 'fit-multi' | 'range';

export interface SimulationDefinition {
  id: string;
  name: string;
  command: SimulationType;
  historyDays: number;
  horizons: string;
  targetColumn: string;
  createdAt: string;
  notes?: string;
}

// ── Forecast-Ergebnisse ──────────────────────────────────────────────────────

export interface DayForecast {
  forecastDay: number;
  date: string;
  weekday: string;
  pUsed: number;
  depEst: number | null;
  depP10: number | null;
  depP90: number | null;
  retEst: number | null;
  retP10: number | null;
  retP90: number | null;
  hourProfile: number[];
}

export interface ForecastResult {
  modelId: string;
  generatedAt: string;
  days: DayForecast[];
  rollingUsage: { date: string; roll7: number; roll14: number }[];
  hourlyProfile: number[][];
  metrics?: { accuracy: number; maeDepH: number; maeRetH: number };
  /** Server-seitiger Pfad zur generierten Run-CSV (predictions/<model>/<dataset>_<N>d_T<DD-MM-YYYY>/forecast.csv). */
  runCsvPath?: string;
}

// ── Simulation Run (gespeicherte Ausführung) ─────────────────────────────────

/**
 * Ein Artefakt aus dem Run-Verzeichnis predictions/<model>/<run>/.
 * `url` ist relativ zum Backend (z.B. "/runs/car_full_forecaster/foo/forecast.csv").
 */
export interface SimulationArtifact {
  name: string;
  kind: 'csv' | 'image' | 'text' | 'json' | 'other' | 'error';
  size_kb: number;
  url: string;
  error?: string;
}

export interface SimulationRun {
  id: string;
  /** Automatisch generierter Name: z.B. "emobpy_003-7d_demo" */
  name: string;
  modelId: string;
  modelLabel: string;
  inputMode: 'csv' | 'emobpy';
  horizons: number;
  historyDays: number;
  createdAt: string;
  result: ForecastResult;

  // ── Optional: Simulation als Verzeichnis-Objekt ────────────────────────────
  /** Relativer Pfad <model>/<run>, z.B. "car_full_forecaster/car_full_7d_T19-05-2026". */
  runDir?: string;
  /** Datensatz-Label aus dem Verzeichnisnamen. */
  dataset?: string;
  /** Datum aus dem Verzeichnisnamen (DD-MM-YYYY). */
  runDate?: string;
  /** Alle Dateien im Run-Verzeichnis (CSV, PNG, TXT, ...). */
  artifacts?: SimulationArtifact[];
  /** Inhalt einer optionalen notes.txt. */
  notes?: string;
  /** "imported" wenn aus predictions/ geladen, sonst "live". */
  origin?: 'live' | 'imported';
}

/** Schlüssel zur Deduplizierung: gleiche Parameter → gleicher Key */
export function runKey(modelId: string, inputMode: string, horizons: number, historyDays: number): string {
  return `${modelId}|${inputMode}|${horizons}|${historyDays}`;
}

// ── Modell-Auswahl ───────────────────────────────────────────────────────────
// Die verfuegbaren Modelle werden zur Laufzeit vom Backend (/api/models)
// geladen. Es gibt deshalb keine statische Liste mehr.

export interface AvailableModel {
  id: string;
  label: string;
  description: string;
  type: string;
  path?: string;
  size_kb?: number;
  modified?: string;
}
