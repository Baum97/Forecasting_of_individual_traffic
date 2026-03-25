export type SimulationType = 'predict' | 'predict-batch' | 'range' | 'range-batch';

export interface SimulationDefinition {
  id: string;
  name: string;
  command: SimulationType;
  historyDays: number;
  horizons: string;
  targetColumn: string;
  createdAt: string;
  notes?: string;
}
