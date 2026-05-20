import { HttpClient } from '@angular/common/http';
import { Injectable, inject, signal } from '@angular/core';

import { SimulationArtifact } from '../models/simulation.model';

const API_BASE = 'http://localhost:8000';

export interface RunSummary {
  model: string;
  run: string;
  dataset: string;
  days: number | null;
  date: string | null;
  artifactCount: number;
  modified: string;
}

export interface RunDetail {
  model: string;
  run: string;
  dataset: string;
  /** Anzahl Forecast-Tage aus dem Verzeichnisnamen (kann null sein). */
  days: number | null;
  date: string | null;
  artifacts: SimulationArtifact[];
  /** Forecast-Zeilen aus forecast.csv (Schluessel = CSV-Header). */
  rows: Record<string, any>[];
  notes: string;
}

@Injectable({ providedIn: 'root' })
export class RunCatalogService {
  private readonly http = inject(HttpClient);

  readonly runs = signal<RunSummary[]>([]);
  readonly state = signal<'idle' | 'loading' | 'ready' | 'error'>('idle');
  readonly errorMessage = signal<string | null>(null);

  refresh(): void {
    this.state.set('loading');
    this.errorMessage.set(null);
    this.http.get<RunSummary[]>(`${API_BASE}/api/runs`).subscribe({
      next: (list) => {
        this.runs.set(list ?? []);
        this.state.set('ready');
      },
      error: (err) => {
        this.runs.set([]);
        this.errorMessage.set(err?.message ?? 'Backend nicht erreichbar.');
        this.state.set('error');
      },
    });
  }

  load(model: string, run: string) {
    return this.http.get<RunDetail>(`${API_BASE}/api/runs/${model}/${run}`);
  }

  /** Absoluter Browser-URL fuer ein Artefakt. */
  artifactUrl(artifact: SimulationArtifact): string {
    return `${API_BASE}${artifact.url}`;
  }
}
