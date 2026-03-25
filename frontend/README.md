# Forecasting WebUI (Angular)

Single-Page-Anwendung fuer die Eingabe und Verwaltung von Simulationen.

## Start

```powershell
cd d:\Projects\Forecasting_of_individual_traffic\frontend
npm install
npm start
```

Danach ist die UI unter `http://localhost:4200` erreichbar.

## Enthaltene Bildschirme

- Input: Neue Simulation anlegen und vorhandene Simulationen durchsuchen
- Simulationen: Tabellenansicht mit Uebersicht und Entfernen-Funktion
- Einstellungen: Platzhalter fuer API-Pfade und Default-Parameter

## Hinweis zur Datenhaltung

Die Simulationen werden aktuell im Browser (`localStorage`) gespeichert.
Das ist bewusst als erster UI-Schritt umgesetzt und kann spaeter mit einem Python-Backend verbunden werden.
