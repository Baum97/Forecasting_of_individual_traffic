import { Routes } from '@angular/router';

import { InputPageComponent } from './pages/input-page/input-page.component';
import { SettingsPageComponent } from './pages/settings-page/settings-page.component';
import { SimulationsPageComponent } from './pages/simulations-page/simulations-page.component';

export const routes: Routes = [
  { path: '', pathMatch: 'full', redirectTo: 'input' },
  { path: 'input', component: InputPageComponent },
  { path: 'simulationen', component: SimulationsPageComponent },
  { path: 'einstellungen', component: SettingsPageComponent },
  { path: '**', redirectTo: 'input' }
];
