import { HttpClient } from '@angular/common/http';
import { Injectable, inject, signal } from '@angular/core';

import { AvailableModel } from '../models/simulation.model';

const API_BASE = 'http://localhost:8000';

@Injectable({ providedIn: 'root' })
export class ModelCatalogService {
  private readonly http = inject(HttpClient);

  readonly models = signal<AvailableModel[]>([]);
  readonly state = signal<'idle' | 'loading' | 'ready' | 'error'>('idle');
  readonly errorMessage = signal<string | null>(null);

  refresh(): void {
    this.state.set('loading');
    this.errorMessage.set(null);
    this.http.get<AvailableModel[]>(`${API_BASE}/api/models`).subscribe({
      next: (list) => {
        this.models.set(list ?? []);
        this.state.set('ready');
      },
      error: (err) => {
        this.models.set([]);
        this.errorMessage.set(
          err?.message ?? 'Backend nicht erreichbar (erwartet auf :8000).'
        );
        this.state.set('error');
      },
    });
  }
}
