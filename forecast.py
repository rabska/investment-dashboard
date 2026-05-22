"""
forecast.py — per-asset price forecasting pipeline
====================================================

Model selection
---------------
Two complementary architectures are tried per ticker; the one with lower
validation loss is kept for inference.

  PatchTST  (Nie et al., 2023)
      Patch-based Transformer encoder.  Chosen when the series is long enough
      to fill the training set with many windows.  Right-sized for per-asset
      training: d_model=64, n_layers=2, so the flat dimension is
      N_PATCHES * 64 = 63 * 64 = 4 032 — manageable even with a few hundred
      windows.

  N-BEATS  (Oreshkin et al., 2019)
      Doubly residual MLP stack with trend + seasonality basis expansion.
      Preferred when the series is short or PatchTST overfits.  Trained on
      the same log-return windows; output is the 63-step ahead forecast
      directly (no recursion).

Both models are trained to predict log-returns over the full 3-month (63
trading-day) horizon in a single forward pass (non-recursive direct
multi-step output).  The 1-month forecast is simply the first 21 steps
of the same prediction.

Horizons
---------
  "1mo"  →  21 trading days  (first 21 steps of the 63-step prediction)
  "3mo"  →  63 trading days  (full prediction)

Leak-free chronological split
-------------------------------
  log_ret has length N = len(prices) - 1.

  train_end = int(N * 0.80)
  val_end   = int(N * 0.90)

  TRAIN windows:
    context ends at or before train_end.
    target ends at or before train_end.
    (so max context start = train_end − CONTEXT_LEN − PRED_LEN)

  VAL windows:
    context may reach back into the train region.
    target starts at or after train_end, ends at or before val_end.

  TEST windows (walk-forward eval only, never seen during training):
    context may reach back into the val region.
    target starts at or after val_end, ends at or before N.

  The test condition is relaxed from the original code:
    old:  val_end + CONTEXT_LEN + PRED_LEN > N → flat fallback
    new:  requires at least ONE test window, i.e.
          N − val_end >= PRED_LEN (there are ≥63 unseen returns after val_end)

Hyper-parameters
----------------
  CONTEXT_LEN  = 252   1 year lookback — keeps the flat dimension manageable
                        and still captures annual seasonality
  PATCH_LEN    = 12    ~2.5-week patch
  STRIDE       = 6     50 % overlap
  N_PATCHES    = (252 − 12) // 6 + 1 = 41
  D_MODEL      = 64    smaller than original to avoid overfit on few windows
  N_HEADS      = 4
  N_LAYERS     = 2
  FFN_DIM      = 128
  DROPOUT      = 0.10
  EPOCHS       = 80    more epochs to compensate for smaller model
  PATIENCE     = 20
  BATCH_SIZE   = 32
  LR           = 3e-4  AdamW
  WEIGHT_DECAY = 1e-4
  HUBER_DELTA  = 0.01

N-BEATS hyper-parameters
  NBEATS_STACKS     = 3   (trend stack + seasonality stack + generic stack)
  NBEATS_BLOCKS     = 3   blocks per stack
  NBEATS_HIDDEN     = 256 units per fully-connected layer
  NBEATS_BASIS_POLY = 4   polynomial degree (trend basis)
  NBEATS_HARMONICS  = 3   Fourier harmonics (seasonality basis)
"""

from __future__ import annotations

import math
import os
import warnings
from datetime import datetime
from pathlib import Path
from typing import Literal

import numpy as np
import pandas as pd

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")
warnings.filterwarnings("ignore", category=UserWarning)

from data_manager import DataManager

# ============================================================
# HORIZONS
# ============================================================

HORIZONS: dict[str, int] = {
    "1mo": 21,   # first 21 steps of the 63-step model output
    "3mo": 63,   # full direct multi-step prediction
}

PRED_LEN: int = 63   # single model output length (covers both horizons)

# ============================================================
# SHARED HYPER-PARAMETERS
# ============================================================

CONTEXT_LEN  : int   = 252    # lookback window (≈1 trading year)
PATCH_LEN    : int   = 12     # patch size
STRIDE       : int   = 6      # patch stride (50 % overlap)
N_PATCHES    : int   = (CONTEXT_LEN - PATCH_LEN) // STRIDE + 1  # = 41

EPOCHS       : int   = 80
PATIENCE     : int   = 20
BATCH_SIZE   : int   = 32
LR           : float = 3e-4
WEIGHT_DECAY : float = 1e-4
HUBER_DELTA  : float = 0.01

TRAIN_RATIO  : float = 0.80

# Minimum rows to attempt any training
MIN_ROWS     : int   = CONTEXT_LEN + PRED_LEN + 60   # ≈375 rows (~1.5yr)

# ── PatchTST ────────────────────────────────────────────────
D_MODEL  : int   = 64
N_HEADS  : int   = 4
N_LAYERS : int   = 2
FFN_DIM  : int   = 128
DROPOUT  : float = 0.10

# ── N-BEATS ─────────────────────────────────────────────────
NBEATS_STACKS     : int = 3
NBEATS_BLOCKS     : int = 3
NBEATS_HIDDEN     : int = 256
NBEATS_BASIS_POLY : int = 4    # polynomial degree for trend basis
NBEATS_HARMONICS  : int = 3    # Fourier harmonics for seasonality basis


# ============================================================
# DATASET BUILDER  (leak-free)
# ============================================================

def _make_dataset(
    returns    : np.ndarray,
    context_len: int,
    pred_len   : int,
    ctx_start  : int = 0,
    ctx_end    : int | None = None,
    tgt_start  : int | None = None,
    tgt_end    : int | None = None,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build sliding-window arrays from a 1-D log-return array.

    Window i covers:
      context → returns[i : i + context_len]
      target  → returns[i + context_len : i + context_len + pred_len]

    Constraints (all indices into the returns array):
      i >= ctx_start
      i + context_len          <= ctx_end    (context fits in allowed region)
      i + context_len          >= tgt_start  (target doesn't start before split)
      i + context_len + pred_len <= tgt_end  (target fits in allowed region)

    Returns
    -------
    X : (N, context_len)  float32   — flat context (log-returns)
    Y : (N, pred_len)     float32   — targets (log-returns)
    """
    T = len(returns)

    if ctx_end  is None: ctx_end  = T
    if tgt_end  is None: tgt_end  = T
    if tgt_start is None: tgt_start = 0

    # Derive valid range of window start indices
    i_min = max(ctx_start, tgt_start - context_len)
    i_max = min(ctx_end, tgt_end - pred_len) - context_len  # inclusive

    if i_max < i_min:
        return (
            np.empty((0, context_len), dtype=np.float32),
            np.empty((0, pred_len),    dtype=np.float32),
        )

    indices = np.arange(i_min, i_max + 1)
    N = len(indices)

    X = np.empty((N, context_len), dtype=np.float32)
    Y = np.empty((N, pred_len),    dtype=np.float32)

    for s, i in enumerate(indices):
        X[s] = returns[i : i + context_len]
        Y[s] = returns[i + context_len : i + context_len + pred_len]

    return X, Y


# ============================================================
# INSTANCE NORMALISATION (RevIN — applied to flat context)
# ============================================================

def _revin(X: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Per-sample zero-mean unit-variance normalisation of the context window.

    X : (B, context_len)
    Returns normalised X, mean (B,1), std (B,1).
    """
    mean = X.mean(axis=1, keepdims=True)
    std  = X.std(axis=1,  keepdims=True) + 1e-6
    return ((X - mean) / std).astype(np.float32), mean, std


def _revin_single(x: np.ndarray) -> np.ndarray:
    """Normalise a single (1, context_len) array; return normalised copy."""
    normed, _, _ = _revin(x)
    return normed


# ============================================================
# MODEL 1: PatchTST  (right-sized for per-asset training)
# ============================================================

def _build_patchtst(n_patches: int, patch_len: int, pred_len: int):
    """
    PatchTST encoder → flat → single linear head for pred_len outputs.

    Input shape : (B, N_patches, patch_len)
    Output shape: (B, pred_len)
    """
    import torch
    import torch.nn as nn

    class SinusoidalPE(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            max_len = n_patches + 16
            pe  = torch.zeros(max_len, D_MODEL)
            pos = torch.arange(max_len).unsqueeze(1).float()
            div = torch.exp(
                torch.arange(0, D_MODEL, 2).float()
                * (-math.log(10000.0) / D_MODEL)
            )
            pe[:, 0::2] = torch.sin(pos * div)
            pe[:, 1::2] = torch.cos(pos * div)
            self.register_buffer("pe", pe.unsqueeze(0))

        def forward(self, x):
            return x + self.pe[:, : x.size(1)]

    class PatchTST(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            self.patch_embed  = nn.Linear(patch_len, D_MODEL)
            self.pos_enc      = SinusoidalPE()
            encoder_layer     = nn.TransformerEncoderLayer(
                d_model=D_MODEL,
                nhead=N_HEADS,
                dim_feedforward=FFN_DIM,
                dropout=DROPOUT,
                batch_first=True,
                norm_first=True,
            )
            self.encoder = nn.TransformerEncoder(encoder_layer,
                                                 num_layers=N_LAYERS)
            self.drop    = nn.Dropout(DROPOUT)
            flat_dim     = n_patches * D_MODEL          # 41 * 64 = 2 624
            # Two-layer MLP head
            hidden = max(pred_len * 4, flat_dim // 4)
            self.head = nn.Sequential(
                nn.Linear(flat_dim, hidden),
                nn.GELU(),
                nn.Dropout(DROPOUT),
                nn.Linear(hidden, pred_len),
            )

        def forward(self, x):
            # x: (B, n_patches, patch_len)
            x = self.patch_embed(x)
            x = self.pos_enc(x)
            x = self.encoder(x)
            x = self.drop(x)
            x = x.flatten(1)           # (B, n_patches * D_MODEL)
            return self.head(x)        # (B, pred_len)

    return PatchTST()


def _patched_context(flat_ctx: np.ndarray) -> np.ndarray:
    """
    Convert a flat (B, context_len) array to (B, N_PATCHES, PATCH_LEN)
    by striding PATCH_LEN windows with stride STRIDE.
    """
    B, L = flat_ctx.shape
    out = np.empty((B, N_PATCHES, PATCH_LEN), dtype=np.float32)
    for p in range(N_PATCHES):
        s = p * STRIDE
        out[:, p, :] = flat_ctx[:, s : s + PATCH_LEN]
    return out


# ============================================================
# MODEL 2: N-BEATS  (robust for shorter series)
# ============================================================

def _build_nbeats(context_len: int, pred_len: int):
    """
    N-BEATS: doubly residual MLP stack with trend + seasonality + generic blocks.

    Input : (B, context_len)  normalised log-returns
    Output: (B, pred_len)

    Each block produces:
      backcast: (B, context_len) — residual subtracted from input
      forecast: (B, pred_len)   — accumulated into the final prediction

    Stack types
    -----------
    trend      : forecast = polynomial basis (degree NBEATS_BASIS_POLY)
    seasonality: forecast = Fourier basis    (NBEATS_HARMONICS harmonics)
    generic    : forecast = linear projection (no interpretability constraint)
    """
    import torch
    import torch.nn as nn

    def _poly_basis(pred_len: int, degree: int) -> torch.Tensor:
        """(pred_len, degree+1) Vandermonde matrix on [0,1]."""
        t = torch.linspace(0, 1, pred_len).unsqueeze(1)    # (T, 1)
        p = torch.arange(degree + 1).unsqueeze(0).float()  # (1, deg+1)
        return t ** p                                        # (T, deg+1)

    def _fourier_basis(pred_len: int, harmonics: int) -> torch.Tensor:
        """(pred_len, 2*harmonics) sin/cos basis."""
        t = torch.linspace(0, 1, pred_len)                 # (T,)
        freqs = torch.arange(1, harmonics + 1).float()     # (H,)
        angles = 2 * math.pi * freqs.unsqueeze(0) * t.unsqueeze(1)  # (T, H)
        return torch.cat([torch.sin(angles), torch.cos(angles)], dim=1)  # (T, 2H)

    class NBEATSBlock(nn.Module):
        def __init__(
            self,
            context_len: int,
            pred_len: int,
            stack_type: Literal["trend", "seasonality", "generic"],
        ) -> None:
            super().__init__()
            self.stack_type  = stack_type
            self.context_len = context_len
            self.pred_len    = pred_len

            # Shared FC stack (same for all block types)
            layers = []
            in_dim = context_len
            for _ in range(4):
                layers += [nn.Linear(in_dim, NBEATS_HIDDEN), nn.ReLU()]
                in_dim  = NBEATS_HIDDEN
            self.fc = nn.Sequential(*layers)

            # Basis-specific theta projections
            if stack_type == "trend":
                deg = NBEATS_BASIS_POLY
                self.theta_f_dim = deg + 1
                self.theta_b_dim = deg + 1
                fwd_basis = _poly_basis(pred_len, deg)
                bwd_basis = _poly_basis(context_len, deg)
            elif stack_type == "seasonality":
                h = NBEATS_HARMONICS
                self.theta_f_dim = 2 * h
                self.theta_b_dim = 2 * h
                fwd_basis = _fourier_basis(pred_len, h)
                bwd_basis = _fourier_basis(context_len, h)
            else:  # generic
                self.theta_f_dim = pred_len
                self.theta_b_dim = context_len
                fwd_basis = torch.eye(pred_len)
                bwd_basis = torch.eye(context_len)

            self.register_buffer("fwd_basis", fwd_basis)  # (pred_len, theta_f)
            self.register_buffer("bwd_basis", bwd_basis)  # (ctx_len, theta_b)

            self.proj_f = nn.Linear(NBEATS_HIDDEN, self.theta_f_dim, bias=False)
            self.proj_b = nn.Linear(NBEATS_HIDDEN, self.theta_b_dim, bias=False)

        def forward(self, x):
            h        = self.fc(x)                          # (B, hidden)
            theta_f  = self.proj_f(h)                      # (B, theta_f)
            theta_b  = self.proj_b(h)                      # (B, theta_b)
            forecast  = theta_f @ self.fwd_basis.T         # (B, pred_len)
            backcast  = theta_b @ self.bwd_basis.T         # (B, ctx_len)
            return backcast, forecast

    class NBEATS(nn.Module):
        def __init__(self) -> None:
            super().__init__()
            stack_types = ["trend", "seasonality", "generic"]
            blocks = []
            for s in range(NBEATS_STACKS):
                st = stack_types[s % len(stack_types)]
                for _ in range(NBEATS_BLOCKS):
                    blocks.append(
                        NBEATSBlock(context_len, pred_len, st)
                    )
            self.blocks = nn.ModuleList(blocks)

        def forward(self, x):
            residual  = x
            forecast  = torch.zeros(
                x.size(0), self.blocks[0].pred_len,
                device=x.device, dtype=x.dtype
            )
            for block in self.blocks:
                backcast, block_fc = block(residual)
                residual = residual - backcast
                forecast = forecast + block_fc
            return forecast

    return NBEATS()


# ============================================================
# GENERIC TRAINING LOOP
# ============================================================

def _train_model(
    model,
    X_tr: np.ndarray,
    Y_tr: np.ndarray,
    X_va: np.ndarray,
    Y_va: np.ndarray,
    device,
    ticker: str,
    model_name: str,
    patchtst_mode: bool = False,
) -> tuple[dict, float]:
    """
    Train *model* and return (best_state_dict, best_val_loss).

    patchtst_mode=True: converts flat X to patched format before forward pass.
    """
    import torch
    import torch.nn as nn
    from torch.utils.data import DataLoader, TensorDataset

    criterion = nn.HuberLoss(delta=HUBER_DELTA)

    def _prepare(X: np.ndarray, Y: np.ndarray):
        if patchtst_mode:
            Xp = _patched_context(X)
            tx = torch.tensor(Xp, dtype=torch.float32)
        else:
            tx = torch.tensor(X, dtype=torch.float32)
        ty = torch.tensor(Y, dtype=torch.float32)
        return tx, ty

    tx_tr, ty_tr = _prepare(X_tr, Y_tr)
    tx_va, ty_va = _prepare(X_va, Y_va)

    train_dl = DataLoader(
        TensorDataset(tx_tr, ty_tr),
        batch_size=BATCH_SIZE,
        shuffle=True,
        drop_last=False,
    )
    val_dl = DataLoader(
        TensorDataset(tx_va, ty_va),
        batch_size=BATCH_SIZE,
        shuffle=False,
    )

    model = model.to(device)
    opt   = torch.optim.AdamW(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(
        opt, T_max=EPOCHS, eta_min=LR * 0.05
    )

    best_val   = math.inf
    best_state : dict = {}
    no_improve  = 0

    for epoch in range(1, EPOCHS + 1):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(device), yb.to(device)
            opt.zero_grad()
            pred = model(xb)
            loss = criterion(pred, yb)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            opt.step()
        sched.step()

        model.eval()
        v_loss, n_v = 0.0, 0
        with torch.no_grad():
            for xb, yb in val_dl:
                xb, yb = xb.to(device), yb.to(device)
                bl = criterion(model(xb), yb)
                v_loss += bl.item() * xb.size(0)
                n_v    += xb.size(0)
        v_loss /= max(n_v, 1)

        if v_loss < best_val:
            best_val   = v_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            no_improve  = 0
        else:
            no_improve += 1

        if no_improve >= PATIENCE:
            print(f"    [{ticker}/{model_name}] Early stop epoch {epoch} "
                  f"(val loss={best_val:.6f})")
            break

    return best_state, best_val


# ============================================================
# WALK-FORWARD TEST EVALUATION
# ============================================================

def _evaluate_test(
    model,
    device,
    X_te      : np.ndarray,
    Y_te      : np.ndarray,
    prices    : np.ndarray,
    i_min_test: int,
    ticker    : str,
    model_name: str,
    patchtst_mode: bool = False,
) -> None:
    """Print RMSE / MAE / MAPE / R² for the 3mo horizon on test windows."""
    import torch

    if len(X_te) == 0:
        print(f"  [{ticker}/{model_name}] No test windows.")
        return

    def _prepare_x(X):
        if patchtst_mode:
            return torch.tensor(_patched_context(X), dtype=torch.float32)
        return torch.tensor(X, dtype=torch.float32)

    model.eval()
    preds_list, trues_list = [], []

    with torch.no_grad():
        for b_start in range(0, len(X_te), BATCH_SIZE):
            xb    = _prepare_x(X_te[b_start : b_start + BATCH_SIZE]).to(device)
            out   = model(xb).cpu().numpy()            # (B, PRED_LEN)
            preds_list.append(out)

    pred_arr = np.concatenate(preds_list, axis=0)      # (n_test, PRED_LEN)
    n_px     = len(prices)

    for s_idx in range(len(X_te)):
        i            = i_min_test + s_idx
        anchor_idx   = i + CONTEXT_LEN
        if anchor_idx >= n_px:
            continue
        anchor        = prices[anchor_idx]
        pred_ret      = pred_arr[s_idx]
        true_ret      = Y_te[s_idx]
        pred_px       = anchor * np.exp(np.cumsum(pred_ret))
        true_px       = anchor * np.exp(np.cumsum(true_ret))
        preds_list.append(pred_px)
        trues_list.append(true_px)

    # Re-collect price-level arrays
    preds_px: list[np.ndarray] = []
    trues_px: list[np.ndarray] = []
    for s_idx in range(len(X_te)):
        i          = i_min_test + s_idx
        anchor_idx = i + CONTEXT_LEN
        if anchor_idx >= n_px:
            continue
        anchor      = prices[anchor_idx]
        preds_px.append(prices[anchor_idx] * np.exp(np.cumsum(pred_arr[s_idx])))
        trues_px.append(prices[anchor_idx] * np.exp(np.cumsum(Y_te[s_idx])))

    if not preds_px:
        return

    p_all = np.concatenate(preds_px)
    t_all = np.concatenate(trues_px)

    rmse  = float(np.sqrt(np.mean((t_all - p_all) ** 2)))
    mae   = float(np.mean(np.abs(t_all - p_all)))
    mask  = t_all != 0
    mape  = float(np.mean(np.abs((t_all[mask] - p_all[mask]) / t_all[mask])) * 100) \
            if mask.any() else float("nan")
    ss_r  = float(np.sum((t_all - p_all) ** 2))
    ss_t  = float(np.sum((t_all - t_all.mean()) ** 2))
    r2    = (1.0 - ss_r / ss_t) if ss_t > 0 else float("nan")

    print(
        f"  [{ticker}/{model_name}] Test ({len(X_te)} windows) — "
        f"RMSE: {rmse:.4f} | MAE: {mae:.4f} | MAPE: {mape:.2f}% | R²: {r2:.4f}"
    )


# ============================================================
# FORECAST GENERATOR
# ============================================================

class ForecastGenerator:

    def __init__(self) -> None:
        self.dm = DataManager()
        # cache: ticker → (model, device, prices_arr, log_ret_arr, patchtst_mode)
        self._model_cache: dict[str, tuple | None] = {}

    # ========================================================
    # CLOSE EXTRACTION
    # ========================================================

    @staticmethod
    def _extract_close(df: pd.DataFrame) -> pd.Series:
        if df is None or df.empty:
            return pd.Series(dtype="float64")

        frame = df.copy()
        if isinstance(frame.columns, pd.MultiIndex):
            frame.columns = [
                "_".join(str(x) for x in col
                         if str(x) not in {"", "nan", "None"}).strip("_")
                for col in frame.columns
            ]

        for candidate in (["Close"] +
                          [c for c in frame.columns
                           if c != "Close" and "close" in c.lower()]):
            if candidate not in frame.columns:
                continue
            obj = frame[candidate]
            if isinstance(obj, pd.DataFrame):
                obj = obj.iloc[:, 0]
            s = pd.to_numeric(obj, errors="coerce").dropna()
            if s.empty:
                continue
            s.index = pd.to_datetime(s.index).tz_localize(None)
            s.name  = "Close"
            return s.sort_index()

        return pd.Series(dtype="float64")

    # ========================================================
    # FLAT FALLBACK
    # ========================================================

    @staticmethod
    def _flat_fallback(last_val: float, periods: int) -> np.ndarray:
        return np.full(periods, last_val, dtype="float64")

    # ========================================================
    # TRAIN (per-asset, model competition)
    # ========================================================

    def _train(self, close: pd.Series, ticker: str) -> tuple | None:
        """
        Train PatchTST and N-BEATS on ticker's history.
        Keep whichever model achieves lower validation loss.

        Returns (model, device, prices_arr, log_ret_arr, patchtst_mode)
        or None if the series is too short.
        """
        import torch

        prices = close.values.astype("float64")
        n_px   = len(prices)

        if n_px < MIN_ROWS:
            print(f"  [{ticker}] Only {n_px} rows (need ≥ {MIN_ROWS}) — flat fallback.")
            return None

        # Log-returns
        log_ret = np.diff(np.log(prices)).astype("float32")
        N       = len(log_ret)

        train_end = int(N * TRAIN_RATIO)               # 80 %
        val_end   = int(N * (TRAIN_RATIO + 0.10))      # 90 %

        # Require at least one complete test window
        if N - val_end < PRED_LEN:
            print(f"  [{ticker}] Test region too short "
                  f"({N - val_end} < {PRED_LEN}) — flat fallback.")
            return None

        # ── Build datasets ─────────────────────────────────────────
        # TRAIN: context and target both end at or before train_end
        X_tr_raw, Y_tr = _make_dataset(
            log_ret, CONTEXT_LEN, PRED_LEN,
            ctx_start=0,
            ctx_end=train_end,
            tgt_start=CONTEXT_LEN,  # target begins right after context
            tgt_end=train_end,       # target ends within train region
        )

        # VAL: target strictly within [train_end, val_end)
        X_va_raw, Y_va = _make_dataset(
            log_ret, CONTEXT_LEN, PRED_LEN,
            ctx_start=0,
            ctx_end=val_end,
            tgt_start=train_end,
            tgt_end=val_end,
        )

        if len(X_tr_raw) < 10:
            print(f"  [{ticker}] Too few training windows ({len(X_tr_raw)}) — flat fallback.")
            return None
        if len(X_va_raw) < 2:
            print(f"  [{ticker}] Too few validation windows ({len(X_va_raw)}) — flat fallback.")
            return None

        # ── RevIN normalisation ────────────────────────────────────
        X_tr, _, _ = _revin(X_tr_raw)
        X_va, _, _ = _revin(X_va_raw)

        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        print(f"  [{ticker}] Training on {device} — "
              f"train={len(X_tr)} / val={len(X_va)} windows")

        # ── PatchTST ───────────────────────────────────────────────
        ptst  = _build_patchtst(N_PATCHES, PATCH_LEN, PRED_LEN)
        st_p, vl_p = _train_model(
            ptst, X_tr, Y_tr, X_va, Y_va, device, ticker,
            "PatchTST", patchtst_mode=True,
        )

        # ── N-BEATS ────────────────────────────────────────────────
        nbeats = _build_nbeats(CONTEXT_LEN, PRED_LEN)
        st_n, vl_n = _train_model(
            nbeats, X_tr, Y_tr, X_va, Y_va, device, ticker,
            "N-BEATS", patchtst_mode=False,
        )

        # ── Select winner ──────────────────────────────────────────
        if vl_p <= vl_n:
            print(f"  [{ticker}] PatchTST wins (val={vl_p:.6f} vs {vl_n:.6f})")
            winner       = ptst
            best_state   = st_p
            patchtst_mode = True
        else:
            print(f"  [{ticker}] N-BEATS wins (val={vl_n:.6f} vs {vl_p:.6f})")
            winner       = nbeats
            best_state   = st_n
            patchtst_mode = False

        winner.load_state_dict(best_state)
        winner.to(device)

        # ── Walk-forward test evaluation ───────────────────────────
        X_te_raw, Y_te = _make_dataset(
            log_ret, CONTEXT_LEN, PRED_LEN,
            ctx_start=0,
            ctx_end=N,
            tgt_start=val_end,
            tgt_end=N,
        )
        X_te, _, _ = _revin(X_te_raw)
        i_min_test  = max(0, val_end - CONTEXT_LEN)

        _evaluate_test(
            winner, device, X_te, Y_te, prices,
            i_min_test, ticker,
            "PatchTST" if patchtst_mode else "N-BEATS",
            patchtst_mode=patchtst_mode,
        )

        return winner, device, prices, log_ret, patchtst_mode

    # ========================================================
    # INFERENCE
    # ========================================================

    def _predict(
        self,
        model,
        device,
        log_ret     : np.ndarray,
        last_price  : float,
        pred_len    : int,
        patchtst_mode: bool,
    ) -> np.ndarray:
        """
        One forward pass using the last CONTEXT_LEN log-returns.
        Returns pred_len price-level predictions anchored on last_price.
        """
        import torch

        ctx = log_ret[-CONTEXT_LEN:].reshape(1, -1).astype("float32")
        ctx_n = _revin_single(ctx)                # (1, CONTEXT_LEN)

        if patchtst_mode:
            x_in = torch.tensor(
                _patched_context(ctx_n), dtype=torch.float32
            ).to(device)
        else:
            x_in = torch.tensor(ctx_n, dtype=torch.float32).to(device)

        model.eval()
        with torch.no_grad():
            pred_ret = model(x_in).cpu().numpy().flatten()[:pred_len]

        return (last_price * np.exp(np.cumsum(pred_ret))).astype("float64")

    # ========================================================
    # GET-OR-TRAIN
    # ========================================================

    def _get_model(self, close: pd.Series, ticker: str) -> tuple | None:
        if ticker not in self._model_cache:
            self._model_cache[ticker] = self._train(close, ticker)
        return self._model_cache[ticker]

    # ========================================================
    # BUILD FORECAST DATAFRAME
    # ========================================================

    def build_forecast_frame(self, ticker: str, horizon: str) -> pd.DataFrame:
        """
        Returns an OHLCV DataFrame for the forecast period.
        Close values come from the winning model or flat fallback.
        """
        if horizon not in HORIZONS:
            raise ValueError(f"horizon must be one of {list(HORIZONS)}")

        price_path = self.dm.get_price_path(ticker)
        if not price_path.exists():
            print(f"  [{ticker}] No price file — skipping.")
            return pd.DataFrame()

        df    = pd.read_parquet(price_path)
        close = self._extract_close(df)
        if close.empty:
            print(f"  [{ticker}] Empty Close — skipping.")
            return pd.DataFrame()

        pred_len  = HORIZONS[horizon]
        last_val  = float(close.iloc[-1])
        last_date = close.index.max()

        cache = self._get_model(close, ticker)

        if cache is not None:
            model, device, prices, log_ret, patchtst_mode = cache
            try:
                # Always run the full PRED_LEN prediction; slice to horizon
                preds = self._predict(
                    model, device, log_ret, last_val, PRED_LEN, patchtst_mode
                )[:pred_len]
            except Exception as exc:
                print(f"  [{ticker}] Prediction error ({exc}) — flat fallback.")
                preds = self._flat_fallback(last_val, pred_len)
        else:
            preds = self._flat_fallback(last_val, pred_len)

        future_dates = pd.bdate_range(
            start=last_date + pd.offsets.BDay(1),
            periods=pred_len,
        )

        return pd.DataFrame(
            {
                "Open"  : preds,
                "High"  : preds,
                "Low"   : preds,
                "Close" : preds,
                "Volume": np.zeros(pred_len, dtype="float64"),
            },
            index=future_dates,
        )

    # ========================================================
    # CACHE VALIDATION
    # ========================================================

    def _forecast_is_valid(self, ticker: str, horizon: str) -> bool:
        fcast_path  = self.dm.get_forecast_path(ticker, horizon)
        meta        = self.dm.load_forecast_metadata(ticker, horizon)
        price_state = self.dm.get_price_state(ticker)

        if not fcast_path.exists() or meta is None or price_state is None:
            return False

        return (
            str(meta.get("price_hash",  ""))    == str(price_state["price_hash"])
            and float(meta.get("price_mtime", -1)) == float(price_state["price_mtime"])
            and str(meta.get("horizon",     ""))    == horizon
            and str(meta.get("method",      ""))    == "patchtst_nbeats"
        )

    # ========================================================
    # PERSIST
    # ========================================================

    def _save_forecast(self, ticker: str, horizon: str,
                       forecast_df: pd.DataFrame) -> None:
        path = self.dm.get_forecast_path(ticker, horizon)
        path.parent.mkdir(parents=True, exist_ok=True)
        forecast_df.to_parquet(path)

        price_state = self.dm.get_price_state(ticker)

        # Determine which model was used
        cache = self._model_cache.get(ticker)
        if cache is not None:
            _, _, _, _, ptst_mode = cache
            method = "patchtst_nbeats"
            model_used = "patchtst" if ptst_mode else "nbeats"
        else:
            method     = "patchtst_nbeats"
            model_used = "flat_fallback"

        metadata = {
            "ticker"        : ticker,
            "horizon"       : horizon,
            "training_date" : datetime.now().isoformat(timespec="seconds"),
            "method"        : method,
            "model_used"    : model_used,
            "context_len"   : CONTEXT_LEN,
            "patch_len"     : PATCH_LEN,
            "stride"        : STRIDE,
            "d_model"       : D_MODEL,
            "n_heads"       : N_HEADS,
            "n_layers"      : N_LAYERS,
            "ffn_dim"       : FFN_DIM,
            "epochs"        : EPOCHS,
            "pred_len"      : HORIZONS[horizon],
            "last_real_date": (
                pd.to_datetime(price_state["last_date"]).isoformat()
                if price_state else None
            ),
            "price_hash"    : price_state["price_hash"]  if price_state else None,
            "price_mtime"   : price_state["price_mtime"] if price_state else None,
        }
        self.dm.save_forecast_metadata(ticker, horizon, metadata)
        print(f"  [{ticker}] Saved forecast ({horizon}) — {len(forecast_df)} rows.")

    # ========================================================
    # PUBLIC API
    # ========================================================

    def generate_forecast(self, ticker: str, horizon: str) -> pd.DataFrame:
        """Force-rebuild one forecast."""
        forecast_df = self.build_forecast_frame(ticker, horizon)
        if forecast_df.empty:
            return pd.DataFrame()
        self._save_forecast(ticker, horizon, forecast_df)
        return forecast_df

    def generate_forecast_if_needed(self, ticker: str,
                                    horizon: str) -> pd.DataFrame:
        """Rebuild only when cache is stale or missing."""
        if self._forecast_is_valid(ticker, horizon):
            path = self.dm.get_forecast_path(ticker, horizon)
            print(f"  [{ticker}] ({horizon}) cache valid — skipping.")
            return pd.read_parquet(path)
        return self.generate_forecast(ticker, horizon)

    def generate_forecasts(self, tickers: list[str]) -> None:
        """
        Train once per ticker (model competition), write parquets for all horizons.
        Both horizon parquets come from the same trained model — 1mo is the
        first 21 steps of the 63-step prediction.
        """
        for ticker in tickers:
            # Evict stale cache so each ticker gets a fresh competition
            self._model_cache.pop(ticker, None)
            for horizon in HORIZONS:
                try:
                    self.generate_forecast_if_needed(ticker, horizon)
                except Exception as exc:
                    print(f"  [{ticker}] ERROR ({horizon}): {exc}")


# ============================================================
# STANDALONE ENTRY
# ============================================================

if __name__ == "__main__":
    TICKERS = """
    SPY QQQ IWM VXUS
    TLT IEF SHY TIP VTIP LQD HYG BND
    GLD SLV USO
    XLV XLU XLP VDC
    USMV SPLV
    SCHD DGRO VIG
    VNQ
    DBC PDBC
    DBMF KMLM
    ARKK
    """.split()

    fg = ForecastGenerator()
    fg.generate_forecasts(TICKERS)