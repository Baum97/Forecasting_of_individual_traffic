import { Component, OnInit, inject, signal } from '@angular/core';

import { RunCatalogService, RunSummary } from '../../services/run-catalog.service';
import { SimulationStoreService } from '../../services/simulation-store.service';
import { SimulationArtifact } from '../../models/simulation.model';

@Component({
  selector: 'app-settings-page',
  standalone: true,
  imports: [],
  templateUrl: './settings-page.component.html',
  styleUrl: './settings-page.component.css'
})
export class SettingsPageComponent implements OnInit {
  private readonly catalog = inject(RunCatalogService);
  private readonly store = inject(SimulationStoreService);

  readonly runs = this.catalog.runs;
  readonly state = this.catalog.state;
  readonly error = this.catalog.errorMessage;

  // Detail-Preview eines selektierten Runs
  selected = signal<RunSummary | null>(null);
  detailLoading = signal(false);
  detailArtifacts = signal<SimulationArtifact[]>([]);
  detailNotes = signal<string>('');
  detailRowCount = signal<number>(0);

  // Letzter Importstatus
  importInfo = signal<string | null>(null);
  importError = signal<string | null>(null);

  ngOnInit(): void {
    this.catalog.refresh();
  }

  refresh(): void {
    this.catalog.refresh();
    this.selected.set(null);
    this.detailArtifacts.set([]);
    this.detailNotes.set('');
    this.detailRowCount.set(0);
  }

  select(run: RunSummary): void {
    this.selected.set(run);
    this.importInfo.set(null);
    this.importError.set(null);
    this.detailLoading.set(true);
    this.catalog.load(run.model, run.run).subscribe({
      next: (detail) => {
        this.detailArtifacts.set(detail.artifacts ?? []);
        this.detailNotes.set(detail.notes ?? '');
        this.detailRowCount.set((detail.rows ?? []).length);
        this.detailLoading.set(false);
      },
      error: (err) => {
        this.detailArtifacts.set([]);
        this.detailNotes.set('');
        this.detailRowCount.set(0);
        this.detailLoading.set(false);
        this.importError.set(err?.message ?? 'Run konnte nicht geladen werden.');
      },
    });
  }

  artifactUrl(a: SimulationArtifact): string {
    return this.catalog.artifactUrl(a);
  }

  importSelected(): void {
    const run = this.selected();
    if (!run) return;
    this.importInfo.set(null);
    this.importError.set(null);
    this.catalog.load(run.model, run.run).subscribe({
      next: (detail) => {
        const imported = this.store.importRun({
          model: detail.model,
          run: detail.run,
          dataset: detail.dataset,
          days: detail.days ?? null,
          date: detail.date,
          artifacts: detail.artifacts ?? [],
          days_rows: detail.rows ?? [],
          notes: detail.notes ?? '',
        });
        this.importInfo.set(`Importiert als "${imported.name}".`);
      },
      error: (err) => {
        this.importError.set(err?.message ?? 'Import fehlgeschlagen.');
      },
    });
  }
}
