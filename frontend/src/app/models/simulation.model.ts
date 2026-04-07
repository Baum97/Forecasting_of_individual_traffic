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
}

// ── Simulation Run (gespeicherte Ausführung) ─────────────────────────────────

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
}

/** Schlüssel zur Deduplizierung: gleiche Parameter → gleicher Key */
export function runKey(modelId: string, inputMode: string, horizons: number, historyDays: number): string {
  return `${modelId}|${inputMode}|${horizons}|${historyDays}`;
}

// ── Modell-Auswahl ───────────────────────────────────────────────────────────

export interface AvailableModel {
  id: string;
  label: string;
  description: string;
  type: 'time_pattern' | 'emobpy_global' | 'real_world';
}

export const AVAILABLE_MODELS: AvailableModel[] = [
  {
    id: 'time_pattern_real',
    label: 'Time Pattern - Real World',
    description: '1 Fahrzeug, 347 Tage echte Messdaten (Nov 2019 - Okt 2020)',
    type: 'real_world',
  },
  {
    id: 'emobpy_global',
    label: 'emobpy Global - 200 Fahrzeuge',
    description: 'Globales Modell trainiert auf 200 simulierten Fahrzeugen (emobpy)',
    type: 'emobpy_global',
  },
  {
    id: 'time_pattern_demo',
    label: 'Demo-Modell',
    description: 'Vorberechnete Demo-Daten ohne Abhängigkeit von lokalem Python-Backend',
    type: 'time_pattern',
  },
];
