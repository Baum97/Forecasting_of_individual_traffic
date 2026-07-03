import { Component, OnDestroy, computed, inject, signal } from '@angular/core';
import { FormsModule } from '@angular/forms';

import {
  TrainStartRequest,
  TrainStatus,
  TrainingApiService,
} from '../../services/training-api.service';

interface SourceOption {
  id: TrainStartRequest['source'];
  label: string;
  description: string;
  supportsLimitVehicles: boolean;
  defaultHistoryDays: number;
  defaultLimitVehicles?: number;
}

interface AlgoOption {
  id: TrainStartRequest['algo'];
  label: string;
  description: string;
}

const SOURCES: SourceOption[] = [
  {
    id: 'realworldev',
    label: 'Real-World EV',
    description:
      'Echte Messdaten eines einzelnen Elektrofahrzeugs ueber ca. 347 Tage (Nov 2019 - Okt 2020). Hoechste Realitaetsnaehe, aber nur eine Person.',
    supportsLimitVehicles: false,
    defaultHistoryDays: 60,
  },
  {
    id: 'emobpy',
    label: 'emobpy',
    description:
      'Ca. 200 simulierte Fahrzeuge aus dem emobpy-Framework, ein Jahr stuendliche Mobilitaet. Synthetisch, aber statistisch realistisch und vielfaeltig.',
    supportsLimitVehicles: true,
    defaultHistoryDays: 100,
    defaultLimitVehicles: 30,
  },
  {
    id: 'ved',
    label: 'VED',
    description:
      'Vehicle Energy Dataset der Univ. of Michigan: ~1 Hz Logs vieler Fahrzeuge, aber nur waehrend Fahrten. Luecken werden als geparkt rekonstruiert.',
    supportsLimitVehicles: true,
    defaultHistoryDays: 100,
    defaultLimitVehicles: 15,
  },
  {
    id: 'routine',
    label: 'Routine (synthetisch)',
    description:
      'Regelbasiert generierte Pendler-Routine eines einzelnen Fahrers. Dient als Sanity-Check mit klarer Erwartung an die Vorhersage.',
    supportsLimitVehicles: false,
    defaultHistoryDays: 60,
  },
  {
    id: 'yjmob',
    label: 'YJMob100K',
    description:
      'Anonymisierte Smartphone-Bewegungsdaten von Personen (kein Fahrzeug, kein Verkehrstraeger). driving ist ein abgeleiteter Bewegungs-Proxy; dient als Kontrast-Quelle mit deutlich hoeherer Aktiv-Rate.',
    supportsLimitVehicles: true,
    defaultHistoryDays: 60,
    defaultLimitVehicles: 50,
  },
];

const ALGOS: AlgoOption[] = [
  {
    id: 'rf',
    label: 'RandomForest',
    description:
      'Baseline-Ensemble aus 200 unabhaengigen Entscheidungsbaeumen, robust und ohne Tuning solide. Quantile (P10/P90) werden aus der Varianz ueber Baeume geschaetzt.',
  },
  {
    id: 'lgbm',
    label: 'LightGBM',
    description:
      'Gradient Boosting mit sequenziellen Baeumen. Schnelleres Training, oft +2-5 Punkte F1 gegenueber RF. Liefert native Quantil-Regression (alpha=0.1/0.5/0.9).',
  },
  {
    id: 'lstm',
    label: 'LSTM',
    description:
      'Rekurrentes neuronales Netz (PyTorch). Die 7 Tages-Lags werden als Sequenz gefuettert, die uebrigen Features als statischer Kontext. Quantile via Pinball-Loss (alpha=0.1/0.5/0.9). Deutlich laengeres Training auf CPU.',
  },
];

@Component({
  selector: 'app-training-page',
  standalone: true,
  imports: [FormsModule],
  templateUrl: './training-page.component.html',
  styleUrl: './training-page.component.css',
})
export class TrainingPageComponent implements OnDestroy {
  private readonly api = inject(TrainingApiService);

  readonly sources = SOURCES;
  readonly algos = ALGOS;

  selectedSourceId = signal<TrainStartRequest['source']>('realworldev');
  selectedAlgoId = signal<TrainStartRequest['algo']>('rf');
  historyDays = signal<number>(60);
  limitVehicles = signal<number>(30);

  selectedSource = computed<SourceOption>(
    () => this.sources.find((s) => s.id === this.selectedSourceId()) ?? this.sources[0]
  );
  selectedAlgo = computed<AlgoOption>(
    () => this.algos.find((a) => a.id === this.selectedAlgoId()) ?? this.algos[0]
  );

  cliPreview = computed(() => {
    const src = this.selectedSource();
    const parts = [
      'python',
      `code\\model_scripts\\forecast\\train_${src.id}_forecaster.py`,
      '--model',
      this.selectedAlgoId(),
      '--history-days',
      String(this.historyDays()),
    ];
    if (src.supportsLimitVehicles) {
      parts.push('--limit-vehicles', String(this.limitVehicles()));
    }
    return parts.join(' ');
  });

  jobId = signal<string | null>(null);
  jobStatus = signal<TrainStatus | null>(null);
  startError = signal<string | null>(null);
  private logOffset = 0;
  private pollHandle: number | null = null;

  selectSource(src: SourceOption): void {
    this.selectedSourceId.set(src.id);
    this.historyDays.set(src.defaultHistoryDays);
    if (src.defaultLimitVehicles != null) {
      this.limitVehicles.set(src.defaultLimitVehicles);
    }
  }

  selectAlgo(algo: AlgoOption): void {
    this.selectedAlgoId.set(algo.id);
  }

  startTraining(): void {
    this.startError.set(null);
    this.jobStatus.set(null);
    this.logOffset = 0;

    const src = this.selectedSource();
    const req: TrainStartRequest = {
      source: src.id,
      algo: this.selectedAlgoId(),
      history_days: this.historyDays(),
    };
    if (src.supportsLimitVehicles) {
      req.limit_vehicles = this.limitVehicles();
    }

    this.api.start(req).subscribe({
      next: (resp) => {
        this.jobId.set(resp.job_id);
        this.startPolling();
      },
      error: (err) => {
        const detail = err?.error?.detail ?? err?.message ?? 'Start fehlgeschlagen.';
        this.startError.set(typeof detail === 'string' ? detail : JSON.stringify(detail));
      },
    });
  }

  private startPolling(): void {
    this.stopPolling();
    this.pollHandle = window.setInterval(() => this.poll(), 1000);
    this.poll();
  }

  private poll(): void {
    const id = this.jobId();
    if (!id) return;
    this.api.status(id, this.logOffset).subscribe({
      next: (status) => {
        const prev = this.jobStatus();
        const mergedLog = (prev?.log ?? []).concat(status.log);
        this.jobStatus.set({ ...status, log: mergedLog });
        this.logOffset = status.log_offset;
        if (status.status !== 'running') {
          this.stopPolling();
        }
      },
      error: () => {
        this.stopPolling();
      },
    });
  }

  private stopPolling(): void {
    if (this.pollHandle != null) {
      window.clearInterval(this.pollHandle);
      this.pollHandle = null;
    }
  }

  ngOnDestroy(): void {
    this.stopPolling();
  }
}
