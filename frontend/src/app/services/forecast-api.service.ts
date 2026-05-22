import { HttpClient, HttpParams } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

import { ForecastResult } from '../models/simulation.model';

const API_BASE = 'http://localhost:8000';

@Injectable({ providedIn: 'root' })
export class ForecastApiService {
  private readonly http = inject(HttpClient);

  /**
   * Ruft das Backend mit einem bereits gespeicherten Modell auf.
   * Schreibt zusaetzlich eine forecast.csv ins Run-Verzeichnis.
   */
  runForecast(
    modelId: string,
    horizons: number,
    dataset: string
  ): Observable<ForecastResult> {
    const params = new HttpParams()
      .set('horizons', String(horizons))
      .set('dataset', dataset);
    return this.http.post<ForecastResult>(
      `${API_BASE}/api/forecast/${modelId}`,
      null,
      { params }
    );
  }
}
