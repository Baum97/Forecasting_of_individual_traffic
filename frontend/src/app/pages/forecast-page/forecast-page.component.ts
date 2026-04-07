import {
  AfterViewInit, Component, computed, ElementRef,
  inject, OnDestroy, signal, ViewChild
} from '@angular/core';
import { FormsModule } from '@angular/forms';
import { DecimalPipe, NgClass, NgFor, NgIf, PercentPipe, SlicePipe } from '@angular/common';
import {
  AVAILABLE_MODELS, DayForecast, ForecastResult, SimulationRun
} from '../../models/simulation.model';
import { SimulationStoreService } from '../../services/simulation-store.service';
import { Chart, registerables } from 'chart.js';

Chart.register(...registerables);

const WEEKDAY_DE = ['Mo', 'Di', 'Mi', 'Do', 'Fr', 'Sa', 'So'];

// ── Demo-Datengenerator ──────────────────────────────────────────────────────
function generateDemoForecast(horizons: number): ForecastResult {
  const today = new Date();
  const days: DayForecast[] = [];

  for (let d = 1; d <= horizons; d++) {
    const dt = new Date(today);
    dt.setDate(today.getDate() + d);
    const wd = dt.getDay() === 0 ? 6 : dt.getDay() - 1;
    const isWeekend = wd >= 5;
    const pUsed = isWeekend
      ? 0.25 + Math.random() * 0.3
      : 0.6 + Math.random() * 0.35;
    const depEst = isWeekend ? 9 + Math.random() * 3 : 7 + Math.random() * 1.5;
    const retEst = isWeekend ? 15 + Math.random() * 4 : 16.5 + Math.random() * 2;

    const hourProfile = Array.from({ length: 24 }, (_, h) => {
      if (isWeekend) return h >= 9 && h <= 19 ? 0.3 + Math.random() * 0.25 : Math.random() * 0.08;
      const morning = h >= 7 && h <= 9 ? 0.65 + Math.random() * 0.2 : 0;
      const evening = h >= 16 && h <= 18 ? 0.6 + Math.random() * 0.25 : 0;
      return Math.max(morning, evening, Math.random() * 0.05);
    });

    days.push({
      forecastDay: d,
      date: dt.toISOString().slice(0, 10),
      weekday: WEEKDAY_DE[wd],
      pUsed: +pUsed.toFixed(3),
      depEst: +depEst.toFixed(1),
      depP10: +(depEst - 1.2).toFixed(1),
      depP90: +(depEst + 1.2).toFixed(1),
      retEst: +retEst.toFixed(1),
      retP10: +(retEst - 1.5).toFixed(1),
      retP90: +(retEst + 1.5).toFixed(1),
      hourProfile,
    });
  }

  const rollingUsage = Array.from({ length: 90 }, (_, i) => {
    const d = new Date(today);
    d.setDate(today.getDate() - 90 + i);
    return {
      date: d.toISOString().slice(0, 10),
      roll7:  +(0.45 + Math.sin(i / 7) * 0.12 + Math.random() * 0.06).toFixed(3),
      roll14: +(0.47 + Math.sin(i / 14) * 0.09 + Math.random() * 0.04).toFixed(3),
    };
  });

  const hourlyProfile: number[][] = Array.from({ length: 7 }, (_, wd) =>
    Array.from({ length: 24 }, (_, h) => {
      const isWE = wd >= 5;
      if (!isWE && h >= 7 && h <= 8)   return 0.6 + Math.random() * 0.2;
      if (!isWE && h >= 16 && h <= 18) return 0.55 + Math.random() * 0.2;
      if (isWE && h >= 10 && h <= 17)  return 0.2 + Math.random() * 0.15;
      return Math.random() * 0.05;
    })
  );

  return {
    modelId: 'demo',
    generatedAt: new Date().toISOString(),
    days,
    rollingUsage,
    hourlyProfile,
    metrics: { accuracy: 0.812, maeDepH: 1.34, maeRetH: 1.61 },
  };
}

// ────────────────────────────────────────────────────────────────────────────

@Component({
  selector: 'app-forecast-page',
  standalone: true,
  imports: [FormsModule, NgIf, NgFor, NgClass, PercentPipe, DecimalPipe, SlicePipe],
  templateUrl: './forecast-page.component.html',
  styleUrl: './forecast-page.component.css',
})
export class ForecastPageComponent implements AfterViewInit, OnDestroy {
  @ViewChild('chartUsage')   chartUsageRef!:   ElementRef<HTMLCanvasElement>;
  @ViewChild('chartTime')    chartTimeRef!:    ElementRef<HTMLCanvasElement>;
  @ViewChild('chartHeatmap') chartHeatmapRef!: ElementRef<HTMLCanvasElement>;
  @ViewChild('chartTrend')   chartTrendRef!:   ElementRef<HTMLCanvasElement>;

  private readonly store = inject(SimulationStoreService);

  readonly models = AVAILABLE_MODELS;

  // ── Parameter ────────────────────────────────────────────────────────────
  selectedModelId = signal<string>('time_pattern_demo');
  horizons        = signal<number>(7);
  historyDays     = signal<number>(100);
  inputMode       = signal<'csv' | 'emobpy'>('emobpy');

  // ── State ─────────────────────────────────────────────────────────────────
  runState      = signal<'idle' | 'running' | 'done' | 'error'>('idle');
  activeRun     = signal<SimulationRun | null>(null);
  activeDay     = signal<DayForecast | null>(null);
  dupWarning    = signal<SimulationRun | null>(null);   // vorhandener Duplikat-Run

  // ── History aus Store ─────────────────────────────────────────────────────
  readonly runs = this.store.runs;

  selectedModel = computed(() =>
    this.models.find(m => m.id === this.selectedModelId()) ?? this.models[0]
  );

  private charts: Chart[] = [];

  // ── Parameter-Änderung → Duplikat-Check ──────────────────────────────────

  onParamChange(): void {
    const dup = this.store.findDuplicate(
      this.selectedModelId(), this.inputMode(), this.horizons(), this.historyDays()
    );
    this.dupWarning.set(dup);
  }

  // ── Forecast starten ─────────────────────────────────────────────────────

  runForecast(): void {
    this.runState.set('running');
    this.destroyCharts();

    // Simuliert API-Aufruf (in echter App: HTTP → Python-Backend)
    setTimeout(() => {
      const result = generateDemoForecast(this.horizons());
      const model = this.selectedModel();

      const run = this.store.addRun({
        modelId: this.selectedModelId(),
        modelLabel: model.label,
        inputMode: this.inputMode(),
        horizons: this.horizons(),
        historyDays: this.historyDays(),
        result,
      });

      this.activeRun.set(run);
      this.activeDay.set(result.days[0]);
      this.dupWarning.set(null);
      this.runState.set('done');

      setTimeout(() => this.buildCharts(), 80);
    }, 900);
  }

  /** Lädt einen bestehenden Run aus der History */
  loadRun(run: SimulationRun): void {
    this.destroyCharts();
    this.activeRun.set(run);
    this.activeDay.set(run.result.days[0]);
    this.runState.set('done');
    setTimeout(() => this.buildCharts(), 80);
  }

  /** Löscht einen Run aus der History */
  deleteRun(id: string, event: Event): void {
    event.stopPropagation();
    if (this.activeRun()?.id === id) {
      this.activeRun.set(null);
      this.activeDay.set(null);
      this.runState.set('idle');
      this.destroyCharts();
    }
    this.store.removeRun(id);
  }

  selectDay(day: DayForecast): void {
    this.activeDay.set(day);
  }

  backToList(): void {
    this.activeRun.set(null);
    this.activeDay.set(null);
    this.runState.set('idle');
    this.destroyCharts();
  }

  // ── Charts ────────────────────────────────────────────────────────────────

  ngAfterViewInit(): void {}

  private buildCharts(): void {
    const r = this.activeRun()?.result;
    if (!r) return;
    this.buildUsageChart(r.days);
    this.buildTimeChart(r.days);
    this.buildHeatmapChart(r.hourlyProfile);
    this.buildTrendChart(r.rollingUsage);
  }

  private buildUsageChart(days: DayForecast[]): void {
    if (!this.chartUsageRef) return;
    const ctx = this.chartUsageRef.nativeElement.getContext('2d')!;
    const labels = days.map(d => `${d.weekday}\n${d.date.slice(5)}`);
    const values = days.map(d => +(d.pUsed * 100).toFixed(1));
    const colors = values.map(v => v >= 70 ? '#1a73e8' : v >= 40 ? '#fbbc04' : '#34a853');

    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [{
          label: 'P(Nutzung) %',
          data: values,
          backgroundColor: colors,
          borderRadius: 6,
          borderSkipped: false,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: (items) => {
                if (!items.length) return '';
                const d = days[items[0].dataIndex];
                return d ? `${d.weekday}, ${d.date}` : '';
              },
              label: (item) => `Nutzungswahrscheinlichkeit: ${item.formattedValue}%`,
              afterLabel: (item) => {
                const d = days[item.dataIndex];
                const lines: string[] = [];
                if (d.depEst) lines.push(`Abfahrt: ~${d.depEst.toFixed(0)}:00 Uhr`);
                if (d.retEst) lines.push(`Rückkehr: ~${d.retEst.toFixed(0)}:00 Uhr`);
                return lines;
              },
            },
            backgroundColor: '#202124', titleColor: '#fff',
            bodyColor: 'rgba(255,255,255,.85)', padding: 12, cornerRadius: 8,
          },
        },
        scales: {
          y: {
            min: 0, max: 100,
            ticks: { callback: (v) => `${v}%`, color: '#5f6368', font: { size: 12 } },
            grid: { color: '#f1f3f4' }, border: { display: false },
          },
          x: {
            ticks: { color: '#5f6368', font: { size: 12 } },
            grid: { display: false }, border: { display: false },
          },
        },
        onClick: (_evt, elements) => {
          if (elements.length) this.selectDay(days[elements[0].index]);
        },
      },
    });
    this.charts.push(chart);
  }

  private buildTimeChart(days: DayForecast[]): void {
    if (!this.chartTimeRef) return;
    const ctx = this.chartTimeRef.nativeElement.getContext('2d')!;
    const labels = days.map(d => `${d.weekday} ${d.date.slice(5)}`);
    const depBar   = days.map(d => d.depP90! - d.depP10!);
    const depStart = days.map(d => d.depP10!);
    const retBar   = days.map(d => d.retP90! - d.retP10!);
    const retStart = days.map(d => d.retP10!);

    const chart = new Chart(ctx, {
      type: 'bar',
      data: {
        labels,
        datasets: [
          { label: 'Abfahrt P10', data: depStart, backgroundColor: 'transparent', stack: 'dep', borderSkipped: false },
          { label: 'Abfahrt Fenster', data: depBar, backgroundColor: 'rgba(26,115,232,.75)', stack: 'dep', borderRadius: 4, borderSkipped: false },
          { label: 'Rückkehr P10', data: retStart, backgroundColor: 'transparent', stack: 'ret', borderSkipped: false },
          { label: 'Rückkehr Fenster', data: retBar, backgroundColor: 'rgba(52,168,83,.75)', stack: 'ret', borderRadius: 4, borderSkipped: false },
        ],
      },
      options: {
        indexAxis: 'y',
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: {
            labels: { filter: (item) => !item.text.includes('P10'), color: '#5f6368', font: { size: 12 } },
          },
          tooltip: {
            filter: (item) => !item.dataset.label!.includes('P10'),
            callbacks: {
              title: (items) => {
                if (!items.length) return '';
                const d = days[items[0].dataIndex];
                return d ? `${d.weekday}, ${d.date}` : '';
              },
              label: (item) => {
                const d = days[item.dataIndex];
                if (item.dataset.label?.includes('Abfahrt'))
                  return `Abfahrt: ${d.depP10}:00 - ${d.depP90}:00 Uhr (erw. ${d.depEst}:00)`;
                if (item.dataset.label?.includes('Rückkehr'))
                  return `Rückkehr: ${d.retP10}:00 - ${d.retP90}:00 Uhr (erw. ${d.retEst}:00)`;
                return '';
              },
            },
            backgroundColor: '#202124', padding: 12, cornerRadius: 8,
          },
        },
        scales: {
          x: {
            min: 0, max: 24,
            ticks: { callback: (v) => `${v}:00`, color: '#5f6368', stepSize: 4, font: { size: 11 } },
            grid: { color: '#f1f3f4' }, border: { display: false },
          },
          y: {
            ticks: { color: '#5f6368', font: { size: 12 } },
            grid: { display: false }, border: { display: false },
          },
        },
        onClick: (_evt, elements) => {
          if (elements.length) this.selectDay(days[elements[0].index]);
        },
      },
    });
    this.charts.push(chart);
  }

  private buildHeatmapChart(profile: number[][]): void {
    if (!this.chartHeatmapRef) return;
    const ctx = this.chartHeatmapRef.nativeElement.getContext('2d')!;
    const data: { x: number; y: number; v: number }[] = [];
    for (let wd = 0; wd < 7; wd++) {
      for (let h = 0; h < 24; h++) {
        data.push({ x: wd, y: h, v: profile[wd][h] });
      }
    }

    const chart = new Chart(ctx, {
      type: 'scatter',
      data: {
        datasets: [{
          label: 'P(in_use)',
          data: data as any,
          backgroundColor: (ctx) => {
            const v = (ctx.raw as any)?.v ?? 0;
            const alpha = 0.15 + v * 0.85;
            if (v > 0.6) return `rgba(26,115,232,${alpha})`;
            if (v > 0.3) return `rgba(251,188,4,${alpha})`;
            return `rgba(52,168,83,${alpha})`;
          },
          pointStyle: 'rect', pointRadius: 13, pointHoverRadius: 15,
        }],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        plugins: {
          legend: { display: false },
          tooltip: {
            callbacks: {
              title: () => '',
              label: (item) => {
                const d = item.raw as { x: number; y: number; v: number };
                return `${WEEKDAY_DE[d.x]}, ${String(d.y).padStart(2, '0')}:00  →  ${(d.v * 100).toFixed(0)}%`;
              },
            },
            backgroundColor: '#202124', padding: 10, cornerRadius: 8,
          },
        },
        scales: {
          x: {
            min: -0.5, max: 6.5,
            ticks: { callback: (v) => WEEKDAY_DE[+v] ?? '', stepSize: 1, color: '#5f6368', font: { size: 12 } },
            grid: { display: false }, border: { display: false },
          },
          y: {
            min: -0.5, max: 23.5,
            ticks: { callback: (v) => +v % 2 === 0 ? `${String(+v).padStart(2, '0')}:00` : '', stepSize: 1, color: '#5f6368', font: { size: 11 } },
            grid: { color: '#f1f3f4' }, border: { display: false },
          },
        },
      },
    });
    this.charts.push(chart);
  }

  private buildTrendChart(rolling: ForecastResult['rollingUsage']): void {
    if (!this.chartTrendRef) return;
    const ctx = this.chartTrendRef.nativeElement.getContext('2d')!;
    const labels = rolling.map(r => r.date.slice(5));

    const chart = new Chart(ctx, {
      type: 'line',
      data: {
        labels,
        datasets: [
          {
            label: '7-Tage',
            data: rolling.map(r => +(r.roll7 * 100).toFixed(1)),
            borderColor: '#1a73e8', backgroundColor: 'rgba(26,115,232,.08)',
            borderWidth: 2, pointRadius: 0, pointHoverRadius: 5, tension: 0.35, fill: true,
          },
          {
            label: '14-Tage',
            data: rolling.map(r => +(r.roll14 * 100).toFixed(1)),
            borderColor: '#ea4335', backgroundColor: 'transparent',
            borderWidth: 2, borderDash: [4, 4], pointRadius: 0, pointHoverRadius: 5, tension: 0.35,
          },
        ],
      },
      options: {
        responsive: true,
        maintainAspectRatio: false,
        interaction: { mode: 'index', intersect: false },
        plugins: {
          legend: { labels: { color: '#5f6368', font: { size: 12 }, boxWidth: 16 } },
          tooltip: {
            callbacks: { label: (item) => `${item.dataset.label}: ${item.formattedValue}%` },
            backgroundColor: '#202124', padding: 12, cornerRadius: 8,
          },
        },
        scales: {
          y: {
            min: 0, max: 100,
            ticks: { callback: (v) => `${v}%`, color: '#5f6368', font: { size: 12 } },
            grid: { color: '#f1f3f4' }, border: { display: false },
          },
          x: {
            ticks: { color: '#5f6368', font: { size: 11 }, maxTicksLimit: 10, maxRotation: 0 },
            grid: { display: false }, border: { display: false },
          },
        },
      },
    });
    this.charts.push(chart);
  }

  private destroyCharts(): void {
    this.charts.forEach(c => c.destroy());
    this.charts = [];
  }

  ngOnDestroy(): void { this.destroyCharts(); }

  // ── Template Helpers ──────────────────────────────────────────────────────

  pUsedClass(p: number): string {
    if (p >= 0.7) return 'chip-blue';
    if (p >= 0.4) return 'chip-yellow';
    return 'chip-green';
  }

  pUsedLabel(p: number): string {
    if (p >= 0.7) return 'Wahrscheinlich';
    if (p >= 0.4) return 'Möglich';
    return 'Unwahrscheinlich';
  }

  hourLabel(h: number): string {
    return `${String(Math.floor(h)).padStart(2, '0')}:00`;
  }

  avgPUsed(run: SimulationRun): number {
    const d = run.result.days;
    return d.reduce((s, r) => s + r.pUsed, 0) / d.length;
  }
}
