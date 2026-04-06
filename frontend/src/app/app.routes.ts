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
    path: 'simulationen',
    loadComponent: () =>
      import('./pages/simulations-page/simulations-page.component').then(
        (m) => m.SimulationsPageComponent
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
