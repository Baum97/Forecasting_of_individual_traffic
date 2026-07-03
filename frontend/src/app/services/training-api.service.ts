import { HttpClient } from '@angular/common/http';
import { Injectable, inject } from '@angular/core';
import { Observable } from 'rxjs';

const API_BASE = 'http://localhost:8000';

export interface TrainStartRequest {
  source: 'emobpy' | 'realworldev' | 'routine' | 'ved' | 'yjmob';
  algo: 'rf' | 'lgbm' | 'lstm';
  limit_vehicles?: number;
  history_days?: number;
}

export interface TrainStartResponse {
  job_id: string;
  command: string;
}

export interface TrainStatus {
  status: 'running' | 'done' | 'error';
  exit_code: number | null;
  command: string;
  source: string;
  algo: string;
  log: string[];
  log_offset: number;
}

@Injectable({ providedIn: 'root' })
export class TrainingApiService {
  private readonly http = inject(HttpClient);

  start(req: TrainStartRequest): Observable<TrainStartResponse> {
    return this.http.post<TrainStartResponse>(`${API_BASE}/api/train`, req);
  }

  status(jobId: string, offset: number = 0): Observable<TrainStatus> {
    const params = { offset: String(offset) };
    return this.http.get<TrainStatus>(`${API_BASE}/api/train/${jobId}`, { params });
  }
}
