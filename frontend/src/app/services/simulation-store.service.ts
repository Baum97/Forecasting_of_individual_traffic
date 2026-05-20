import { Injectable, signal } from '@angular/core';
import {
  DayForecast, ForecastResult,
  SimulationArtifact, SimulationDefinition, SimulationRun, runKey
} from '../models/simulation.model';

const KEY_DEFS = 'forecasting.simulations.v1';
const KEY_RUNS = 'forecasting.runs.v2';

@Injectable({ providedIn: 'root' })
export class SimulationStoreService {

  // ── Alte SimulationDefinitions (Rückwärtskompatibilität) ──────────────────
  readonly simulations = signal<SimulationDefinition[]>(this.loadDefs());

  addSimulation(payload: Omit<SimulationDefinition, 'id' | 'createdAt'>): void {
    const next: SimulationDefinition = {
      ...payload,
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString(),
    };
    this.simulations.update((list) => {
      const updated = [next, ...list];
      localStorage.setItem(KEY_DEFS, JSON.stringify(updated));
      return updated;
    });
  }

  removeSimulation(id: string): void {
    this.simulations.update((list) => {
      const updated = list.filter((item) => item.id !== id);
      localStorage.setItem(KEY_DEFS, JSON.stringify(updated));
      return updated;
    });
  }

  // ── Simulation Runs (mit Ergebnissen) ────────────────────────────────────
  readonly runs = signal<SimulationRun[]>(this.loadRuns());

  /**
   * Prüft ob ein Run mit identischen Parametern bereits existiert.
   * Gibt den vorhandenen Run zurück, oder null.
   */
  findDuplicate(
    modelId: string, inputMode: string, horizons: number, historyDays: number
  ): SimulationRun | null {
    const key = runKey(modelId, inputMode, horizons, historyDays);
    return this.runs().find(r => runKey(r.modelId, r.inputMode, r.horizons, r.historyDays) === key) ?? null;
  }

  /**
   * Speichert einen neuen Run.
   * Name-Schema: {inputMode}_{counter:03d}-{horizons}d_{modelId}
   * Beispiel: emobpy_003-7d_time_pattern_demo
   */
  addRun(run: Omit<SimulationRun, 'id' | 'name' | 'createdAt'>): SimulationRun {
    const counter = this.runs().length + 1;
    const shortModel = run.modelId.replace('time_pattern_', 'tp_').replace('emobpy_global', 'emobpy');
    const name = `${run.inputMode}_${String(counter).padStart(3, '0')}-${run.horizons}d_${shortModel}`;

    const next: SimulationRun = {
      ...run,
      id: crypto.randomUUID(),
      name,
      createdAt: new Date().toISOString(),
    };

    this.runs.update((list) => {
      const updated = [next, ...list];
      this.persistRuns(updated);
      return updated;
    });

    return next;
  }

  /**
   * Importiert einen Run aus dem Backend (predictions/<model>/<run>/) in den Store.
   * Erwartet das Payload von GET /api/runs/{model}/{run}.
   */
  importRun(payload: {
    model: string;
    run: string;
    dataset: string;
    days: number | null;
    date: string | null;
    artifacts: SimulationArtifact[];
    days_rows: Record<string, any>[];
    notes: string;
  }): SimulationRun {
    const days: DayForecast[] = (payload.days_rows ?? []).map((r, i) => ({
      forecastDay: Number(r['forecastDay'] ?? r['day'] ?? i + 1),
      date: String(r['date'] ?? ''),
      weekday: String(r['weekday'] ?? ''),
      pUsed: Number(r['pUsed'] ?? r['p_used'] ?? 0),
      depEst: r['depEst'] != null ? Number(r['depEst']) : null,
      depP10: r['depP10'] != null ? Number(r['depP10']) : null,
      depP90: r['depP90'] != null ? Number(r['depP90']) : null,
      retEst: r['retEst'] != null ? Number(r['retEst']) : null,
      retP10: r['retP10'] != null ? Number(r['retP10']) : null,
      retP90: r['retP90'] != null ? Number(r['retP90']) : null,
      hourProfile: Array.from({ length: 24 }, () => 0),
    }));

    const result: ForecastResult = {
      modelId: payload.model,
      generatedAt: payload.date ?? new Date().toISOString(),
      days,
      rollingUsage: [],
      hourlyProfile: Array.from({ length: 7 }, () => Array(24).fill(0)),
    };

    const horizons = payload.days ?? days.length ?? 1;
    const counter = this.runs().length + 1;
    const name = `import_${String(counter).padStart(3, '0')}-${horizons}d_${payload.model}`;

    const next: SimulationRun = {
      id: crypto.randomUUID(),
      name,
      modelId: payload.model,
      modelLabel: payload.model,
      inputMode: 'csv',
      horizons,
      historyDays: 0,
      createdAt: new Date().toISOString(),
      result,
      runDir: `${payload.model}/${payload.run}`,
      dataset: payload.dataset,
      runDate: payload.date ?? undefined,
      artifacts: payload.artifacts,
      notes: payload.notes,
      origin: 'imported',
    };

    this.runs.update((list) => {
      const updated = [next, ...list];
      this.persistRuns(updated);
      return updated;
    });

    return next;
  }

  removeRun(id: string): void {
    this.runs.update((list) => {
      const updated = list.filter((r) => r.id !== id);
      this.persistRuns(updated);
      return updated;
    });
  }

  getRunById(id: string): SimulationRun | undefined {
    return this.runs().find(r => r.id === id);
  }

  // ── Persistenz ────────────────────────────────────────────────────────────

  private loadDefs(): SimulationDefinition[] {
    try {
      const raw = localStorage.getItem(KEY_DEFS);
      if (!raw) return [];
      const parsed = JSON.parse(raw) as SimulationDefinition[];
      return Array.isArray(parsed) ? parsed : [];
    } catch { return []; }
  }

  private loadRuns(): SimulationRun[] {
    try {
      const raw = localStorage.getItem(KEY_RUNS);
      if (!raw) return [];
      const parsed = JSON.parse(raw) as SimulationRun[];
      return Array.isArray(parsed) ? parsed : [];
    } catch { return []; }
  }

  private persistRuns(list: SimulationRun[]): void {
    // Nur Config + Metadaten pro Run in Index speichern;
    // vollständige Ergebnisse in separaten Keys (Quota-Schutz)
    try {
      // Versuche alles zu speichern
      localStorage.setItem(KEY_RUNS, JSON.stringify(list));
    } catch {
      // Falls localStorage voll: älteste Runs entfernen
      const trimmed = list.slice(0, 20);
      localStorage.setItem(KEY_RUNS, JSON.stringify(trimmed));
    }
  }
}
