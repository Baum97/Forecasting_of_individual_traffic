import { Routes } from '@angular/router';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'forecast' },
  {
    path: 'forecast',
    loadComponent: () =>
      import('./pages/forecast-page/forecast-page.component').then(
        (m) => m.ForecastPageComponent
      ),
  },
  {
    path: 'training',
    loadComponent: () =>
      import('./pages/training-page/training-page.component').then(
        (m) => m.TrainingPageComponent
      ),
  },
  {
    path: 'einstellungen',
    loadComponent: () =>
      import('./pages/settings-page/settings-page.component').then(
        (m) => m.SettingsPageComponent
      ),
  },
  { path: '**', redirectTo: 'forecast' },
];
