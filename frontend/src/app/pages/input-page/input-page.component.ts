import { DatePipe, NgClass, NgFor, NgIf } from '@angular/common';
import { Component, computed, inject, signal } from '@angular/core';
import { FormBuilder, ReactiveFormsModule, Validators } from '@angular/forms';

import { SimulationType } from '../../models/simulation.model';
import { SimulationStoreService } from '../../services/simulation-store.service';

@Component({
  selector: 'app-input-page',
  standalone: true,
  imports: [ReactiveFormsModule, DatePipe, NgClass, NgIf, NgFor],
  templateUrl: './input-page.component.html',
  styleUrl: './input-page.component.css'
})
export class InputPageComponent {
  private readonly fb = inject(FormBuilder);
  private readonly store = inject(SimulationStoreService);

  protected readonly simulations = this.store.simulations;
  protected readonly filterText = signal('');
  protected readonly createState = signal<'idle' | 'success'>('idle');

  protected readonly form = this.fb.nonNullable.group({
    name: ['', [Validators.required, Validators.minLength(3)]],
    command: ['predict' as SimulationType, Validators.required],
    historyDays: [100, [Validators.required, Validators.min(20), Validators.max(365)]],
    horizons: ['1,2,3,4,5,6,7', [Validators.required]],
    targetColumn: ['target', [Validators.required]],
    notes: ['']
  });

  protected readonly filtered = computed(() => {
    const query = this.filterText().trim().toLowerCase();
    const list = this.simulations();
    if (!query) {
      return list;
    }
    return list.filter((item) => {
      return (
        item.name.toLowerCase().includes(query) ||
        item.command.toLowerCase().includes(query) ||
        item.targetColumn.toLowerCase().includes(query)
      );
    });
  });

  protected submitSimulation(): void {
    if (this.form.invalid) {
      this.form.markAllAsTouched();
      return;
    }

    const values = this.form.getRawValue();
    this.store.addSimulation({
      name: values.name,
      command: values.command,
      historyDays: values.historyDays,
      horizons: values.horizons,
      targetColumn: values.targetColumn,
      notes: values.notes || undefined
    });

    this.form.patchValue({ name: '', notes: '' });
    this.createState.set('success');
    window.setTimeout(() => this.createState.set('idle'), 2200);
  }

  protected updateFilter(raw: string): void {
    this.filterText.set(raw);
  }
}
