import { Injectable, signal } from '@angular/core';

import { SimulationDefinition } from '../models/simulation.model';

const STORAGE_KEY = 'forecasting.simulations.v1';

@Injectable({ providedIn: 'root' })
export class SimulationStoreService {
  readonly simulations = signal<SimulationDefinition[]>(this.load());

  addSimulation(payload: Omit<SimulationDefinition, 'id' | 'createdAt'>): void {
    const next: SimulationDefinition = {
      ...payload,
      id: crypto.randomUUID(),
      createdAt: new Date().toISOString()
    };

    this.simulations.update((list) => {
      const updated = [next, ...list];
      this.persist(updated);
      return updated;
    });
  }

  removeSimulation(id: string): void {
    this.simulations.update((list) => {
      const updated = list.filter((item) => item.id !== id);
      this.persist(updated);
      return updated;
    });
  }

  private load(): SimulationDefinition[] {
    try {
      const raw = localStorage.getItem(STORAGE_KEY);
      if (!raw) {
        return [];
      }
      const parsed = JSON.parse(raw) as SimulationDefinition[];
      return Array.isArray(parsed) ? parsed : [];
    } catch {
      return [];
    }
  }

  private persist(list: SimulationDefinition[]): void {
    localStorage.setItem(STORAGE_KEY, JSON.stringify(list));
  }
}
