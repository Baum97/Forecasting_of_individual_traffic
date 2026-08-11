from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional, Sequence, Tuple

import numpy as np
import pandas as pd

try:
    import torch
    from torch import nn
except ImportError as e:  # pragma: no cover
    raise ImportError(
        "torch ist nicht installiert. Bitte 'pip install torch' ausfuehren."
    ) from e

from model_scripts.base import FEATURE_COLS

# Lag-Spalten, aeltester Zeitschritt zuerst -- so entsteht aus der flachen
# Feature-Matrix eine chronologische Sequenz.
_LAG_ORDER = ["lag_168", "lag_24", "lag_2", "lag_1"]


@dataclass
class LSTMConfig:
    """Hyperparameter beider LSTM-Varianten.

    Bewusst klein gehalten: die Aktiv-Rate liegt je nach Quelle bei 5-10 %,
    groessere Netze passen sofort ueber. max_train_samples deckelt die
    Laufzeit auf CPU deterministisch (Zufallsstichprobe mit festem Seed).
    """
    hidden: int = 32
    dropout: float = 0.1
    lr: float = 1e-3
    batch_size: int = 512
    max_epochs: int = 15
    patience: int = 3
    val_frac: float = 0.1
    window: int = 168
    max_train_samples: int = 60_000
    random_state: int = 42


class _SeqNet(nn.Module):
    """LSTM ueber die Sequenz, statischer Kontext erst am MLP-Kopf."""

    def __init__(self, n_channels: int, n_static: int, hidden: int, dropout: float):
        super().__init__()
        self.lstm = nn.LSTM(n_channels, hidden, batch_first=True)
        self.head = nn.Sequential(
            nn.Linear(hidden + n_static, hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden, 1),
        )

    def forward(self, seq: "torch.Tensor", static: "torch.Tensor") -> "torch.Tensor":
        _, (h_n, _) = self.lstm(seq)
        z = torch.cat([h_n[-1], static], dim=1)
        return self.head(z).squeeze(1)


class _SeqTrainer:
    """Trainingsschleife mit Standardisierung des statischen Kontexts,
    pos_weight gegen die Klassen-Imbalance und Early Stopping auf einem
    zeitlichen Validierungs-Tail (die letzten val_frac der Zeilen -- die
    Reihen sind nach vehicle_id/timestamp sortiert, ein zufaelliger Split
    wuerde durch die Fensterueberlappung lecken).
    """

    def __init__(self, n_channels: int, n_static: int, cfg: LSTMConfig):
        self.cfg = cfg
        torch.manual_seed(cfg.random_state)
        self.net = _SeqNet(n_channels, n_static, cfg.hidden, cfg.dropout)
        self.stat_mean: Optional[np.ndarray] = None
        self.stat_std: Optional[np.ndarray] = None

    def _scale(self, static: np.ndarray) -> np.ndarray:
        return (static - self.stat_mean) / self.stat_std

    def fit(self, seq: np.ndarray, static: np.ndarray, y: np.ndarray) -> "_SeqTrainer":
        cfg = self.cfg
        self.stat_mean = static.mean(axis=0).astype(np.float32)
        std = static.std(axis=0).astype(np.float32)
        std[std == 0] = 1.0
        self.stat_std = std

        n = len(y)
        n_val = int(n * cfg.val_frac)
        idx_tr = np.arange(0, n - n_val)
        idx_va = np.arange(n - n_val, n) if n_val >= 50 else None

        seq_t = torch.from_numpy(seq.astype(np.float32))
        stat_t = torch.from_numpy(self._scale(static).astype(np.float32))
        y_t = torch.from_numpy(y.astype(np.float32))

        pos = float(y[idx_tr].sum())
        neg = float(len(idx_tr) - pos)
        pos_weight = torch.tensor(neg / pos if pos > 0 else 1.0, dtype=torch.float32)
        loss_fn = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

        opt = torch.optim.Adam(self.net.parameters(), lr=cfg.lr)
        best_loss, best_state, bad = float("inf"), None, 0
        gen = torch.Generator().manual_seed(cfg.random_state)

        # Lernkurve fuer die Overfitting-Kontrolle: mittlerer Trainings-Loss je
        # Epoche und Validierungs-Loss auf dem zeitlichen Tail. best_epoch ist
        # der von Early Stopping gewaehlte Zustand (0-basiert).
        self.history = {"train_loss": [], "val_loss": [], "best_epoch": 0}

        for epoch in range(cfg.max_epochs):
            self.net.train()
            perm = idx_tr[torch.randperm(len(idx_tr), generator=gen).numpy()]
            ep_loss, ep_n = 0.0, 0
            for i in range(0, len(perm), cfg.batch_size):
                b = perm[i:i + cfg.batch_size]
                opt.zero_grad()
                loss = loss_fn(self.net(seq_t[b], stat_t[b]), y_t[b])
                loss.backward()
                opt.step()
                ep_loss += float(loss) * len(b)
                ep_n += len(b)
            self.history["train_loss"].append(ep_loss / max(ep_n, 1))

            if idx_va is None:
                self.history["val_loss"].append(float("nan"))
                continue
            self.net.eval()
            with torch.no_grad():
                val_loss = float(loss_fn(self.net(seq_t[idx_va], stat_t[idx_va]),
                                         y_t[idx_va]))
            self.history["val_loss"].append(val_loss)
            if val_loss < best_loss - 1e-4:
                best_loss, bad = val_loss, 0
                best_state = {k: v.clone() for k, v in self.net.state_dict().items()}
                self.history["best_epoch"] = epoch
            else:
                bad += 1
                if bad >= cfg.patience:
                    print(f"    early stop @ epoch {epoch + 1} (val {best_loss:.4f})")
                    break

        if best_state is not None:
            self.net.load_state_dict(best_state)
        self.net.eval()
        return self

    def predict_proba(self, seq: np.ndarray, static: np.ndarray) -> np.ndarray:
        seq_t = torch.from_numpy(seq.astype(np.float32))
        stat_t = torch.from_numpy(self._scale(static).astype(np.float32))
        out = np.empty(len(seq_t), dtype=np.float32)
        with torch.no_grad():
            for i in range(0, len(seq_t), 4096):
                sl = slice(i, i + 4096)
                out[sl] = torch.sigmoid(self.net(seq_t[sl], stat_t[sl])).numpy()
        return out


def _subsample(n: int, limit: int, seed: int) -> np.ndarray:
    """Sortierte Zufallsstichprobe der Zeilenindizes (Reihenfolge bleibt
    chronologisch, damit der Validierungs-Tail ein Tail bleibt)."""
    if n <= limit:
        return np.arange(n)
    rng = np.random.default_rng(seed)
    return np.sort(rng.choice(n, size=limit, replace=False))


# ── Variante A: identischer Feature-Satz ─────────────────────────────────────

class LSTMTabularClassifier:
    """LSTM auf derselben Feature-Matrix wie RF/LGBM (sklearn-API).

    Die vorhandenen Lag-Spalten bilden die Sequenz, alle uebrigen Spalten den
    statischen Kontext. Fehlen die Lags vollstaendig (Ablationsvariante
    "nur_kalender"), degeneriert das Netz zu einem MLP -- das ist gewollt und
    macht die Variante ablationsfaehig.
    """

    needs_frame = False

    def __init__(self, cfg: Optional[LSTMConfig] = None,
                 feature_names: Optional[Sequence[str]] = None):
        self.cfg = cfg or LSTMConfig()
        self.feature_names: List[str] = list(feature_names or FEATURE_COLS)
        self.model: Optional[_SeqTrainer] = None

    def set_feature_names(self, names: Sequence[str]) -> None:
        """Muss vor fit() aufgerufen werden, wenn X nicht FEATURE_COLS ist
        (z. B. in der Ablationsstudie mit Spalten-Teilmengen)."""
        self.feature_names = list(names)

    def _split(self, X: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
        X = np.asarray(X, dtype=np.float32)
        if X.shape[1] != len(self.feature_names):
            raise ValueError(
                f"X hat {X.shape[1]} Spalten, feature_names {len(self.feature_names)}. "
                f"set_feature_names() vor fit() aufrufen."
            )
        pos = {name: i for i, name in enumerate(self.feature_names)}
        lag_idx = [pos[c] for c in _LAG_ORDER if c in pos]
        stat_idx = [i for i, name in enumerate(self.feature_names)
                    if name not in _LAG_ORDER]

        seq = (X[:, lag_idx][:, :, None] if lag_idx
               else np.zeros((len(X), 1, 1), dtype=np.float32))
        static = (X[:, stat_idx] if stat_idx
                  else np.zeros((len(X), 1), dtype=np.float32))
        return np.ascontiguousarray(seq), np.ascontiguousarray(static)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "LSTMTabularClassifier":
        seq, static = self._split(X)
        y = np.asarray(y, dtype=np.float32)
        keep = _subsample(len(y), self.cfg.max_train_samples, self.cfg.random_state)
        self.model = _SeqTrainer(seq.shape[2], static.shape[1], self.cfg).fit(
            seq[keep], static[keep], y[keep]
        )
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        seq, static = self._split(X)
        p1 = self.model.predict_proba(seq, static)
        return np.column_stack([1.0 - p1, p1]).astype(float)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)

    @property
    def history(self) -> Optional[dict]:
        """Lernkurve (train/val Loss je Epoche) nach fit(), sonst None."""
        return getattr(self.model, "history", None)


# ── Variante B: native Sequenz ───────────────────────────────────────────────

class LSTMSequenceClassifier:
    """LSTM auf einem Fenster der letzten `window` Stunden der rohen Reihe.

    Braucht den DataFrame statt einer Feature-Matrix (`needs_frame = True`);
    base.train_model()/evaluate() routen entsprechend.

    Kanaele je Zeitschritt: driving, sin/cos der Tagesstunde.
    Statischer Kontext der Zielstunde: sin/cos Stunde, sin/cos Wochentag,
    is_weekend. Rolling-Means werden bewusst *nicht* uebergeben -- sie stehen
    im Fenster und sollen vom Netz selbst abgeleitet werden.

    Randfall: Zeilen am Anfang eines Fensters, deren Vorgeschichte nicht im
    uebergebenen Frame liegt (Testsplit beginnt mitten in der Reihe), werden
    links mit Nullen aufgefuellt. Das betrifft die ersten `window` Zeilen je
    Fahrzeug und wird bewusst in Kauf genommen, damit die Zeilenmenge exakt
    der von RF/LGBM entspricht und die Matrix vergleichbar bleibt.
    """

    needs_frame = True

    def __init__(self, cfg: Optional[LSTMConfig] = None):
        self.cfg = cfg or LSTMConfig()
        self.model: Optional[_SeqTrainer] = None

    def _index(self, frame: pd.DataFrame):
        """Vorbereitung ohne die Fenster zu materialisieren.

        Liefert (flat_drive, flat_hour, start, static, y) fuer genau die
        Zeilen, die auch RF/LGBM sehen (dropna ueber FEATURE_COLS), in
        identischer Reihenfolge. Die Fenster selbst entstehen erst in
        _gather() -- bei 200k Zeilen waeren 168x3 float32 sonst ~400 MB.
        """
        L = self.cfg.window
        frame = frame.sort_values(["vehicle_id", "timestamp"])
        valid = frame[FEATURE_COLS].notna().all(axis=1).to_numpy()

        # Alle Fahrzeuge in ein zusammenhaengendes Array; je Fahrzeug L Nullen
        # davor, damit ein Fenster nie ueber eine Fahrzeuggrenze laeuft.
        drive_parts, hour_parts, starts = [], [], []
        offset = 0
        codes = frame["vehicle_id"].to_numpy()
        drive_all = frame["driving"].to_numpy(dtype=np.float32)
        hour_all = frame["timestamp"].dt.hour.to_numpy(dtype=np.float32)
        bounds = np.flatnonzero(np.r_[True, codes[1:] != codes[:-1], True])

        for a, b in zip(bounds[:-1], bounds[1:]):
            pad = np.zeros(L, dtype=np.float32)
            drive_parts.append(np.concatenate([pad, drive_all[a:b]]))
            hour_parts.append(np.concatenate([pad, hour_all[a:b]]))
            # Fensterstart je Zeile: absolute Position im konkatenierten Array.
            starts.append(offset + np.arange(b - a, dtype=np.int64))
            offset += (b - a) + L

        flat_drive = np.concatenate(drive_parts)
        flat_hour = np.concatenate(hour_parts)
        start = np.concatenate(starts)[valid]

        sub = frame[valid]
        h = sub["timestamp"].dt.hour.to_numpy(dtype=np.float32)
        wd = sub["timestamp"].dt.weekday.to_numpy(dtype=np.float32)
        static = np.column_stack([
            np.sin(2 * np.pi * h / 24), np.cos(2 * np.pi * h / 24),
            np.sin(2 * np.pi * wd / 7), np.cos(2 * np.pi * wd / 7),
            (wd >= 5).astype(np.float32),
        ]).astype(np.float32)

        y = sub["driving"].astype(int).to_numpy().astype(np.float32)
        return flat_drive, flat_hour, start, static, y

    def _gather(self, flat_drive: np.ndarray, flat_hour: np.ndarray,
                start: np.ndarray) -> np.ndarray:
        """Fenster (n, L, 3) fuer die uebergebenen Startpositionen."""
        idx = start[:, None] + np.arange(self.cfg.window, dtype=np.int64)[None, :]
        ang = 2.0 * np.pi * flat_hour[idx] / 24.0
        return np.ascontiguousarray(
            np.stack([flat_drive[idx], np.sin(ang), np.cos(ang)], axis=2)
        )

    def fit_frame(self, frame: pd.DataFrame) -> "LSTMSequenceClassifier":
        flat_drive, flat_hour, start, static, y = self._index(frame)
        if len(np.unique(y)) < 2:
            raise ValueError("training data has only one class — cannot fit classifier")
        keep = _subsample(len(y), self.cfg.max_train_samples, self.cfg.random_state)
        print(f"    lstm_seq: {len(y):,} Fenster -> {len(keep):,} genutzt, "
              f"L={self.cfg.window}, Aktiv-Rate {y.mean():.3f}", flush=True)
        seq = self._gather(flat_drive, flat_hour, start[keep])
        self.model = _SeqTrainer(seq.shape[2], static.shape[1], self.cfg).fit(
            seq, static[keep], y[keep]
        )
        return self

    @property
    def history(self) -> Optional[dict]:
        """Lernkurve (train/val Loss je Epoche) nach fit_frame(), sonst None."""
        return getattr(self.model, "history", None)

    def predict_proba_frame(self, frame: pd.DataFrame) -> np.ndarray:
        flat_drive, flat_hour, start, static, _ = self._index(frame)
        out = np.empty(len(start), dtype=np.float32)
        # Blockweise, damit auch grosse Testframes nicht das Fenster-Array
        # komplett materialisieren muessen.
        for i in range(0, len(start), 20_000):
            sl = slice(i, i + 20_000)
            seq = self._gather(flat_drive, flat_hour, start[sl])
            out[sl] = self.model.predict_proba(seq, static[sl])
        return np.column_stack([1.0 - out, out]).astype(float)
