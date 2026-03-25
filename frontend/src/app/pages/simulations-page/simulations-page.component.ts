import { DatePipe, NgFor, NgIf } from '@angular/common';
import { Component, computed, inject } from '@angular/core';

import { SimulationStoreService } from '../../services/simulation-store.service';

@Component({
  selector: 'app-simulations-page',
  standalone: true,
  imports: [DatePipe, NgIf, NgFor],
  templateUrl: './simulations-page.component.html',
  styleUrl: './simulations-page.component.css'
})
export class SimulationsPageComponent {
  private readonly store = inject(SimulationStoreService);

  protected readonly simulations = this.store.simulations;

  protected readonly commandCounts = computed(() => {
    const grouped = new Map<string, number>();
    for (const sim of this.simulations()) {
      grouped.set(sim.command, (grouped.get(sim.command) ?? 0) + 1);
    }
    return Array.from(grouped.entries()).sort((a, b) => a[0].localeCompare(b[0]));
  });

  protected removeSimulation(id: string): void {
    this.store.removeSimulation(id);
  }
}
