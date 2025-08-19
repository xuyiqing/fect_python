from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, List, Optional, Tuple, Dict, Any

import numpy as np
import pandas as pd
try:
    from scipy.stats import norm as _scipy_norm
except Exception:  # scipy optional
    _scipy_norm = None


@dataclass
class FectResult:
    # Core data
    Y_dat: np.ndarray
    Y0_dat: Optional[np.ndarray]
    D_dat: np.ndarray
    I_dat: np.ndarray

    # Names/meta
    Y: str
    D: str
    X: List[str]
    index: Tuple[str, str]
    id: List[Any]
    rawtime: List[Any]

    # Estimates
    eff: np.ndarray  # ATT matrix T x N (NaNs where not defined)
    eff_calendar: pd.DataFrame  # columns: ["ATT-calendar", "count"], index=time
    hasRevs: bool

    # For effect()
    method: str
    binary: bool

    # Optional fields to be closer to R output
    beta: Optional[np.ndarray] = None
    beta_se: Optional[np.ndarray] = None
    beta_names: Optional[List[str]] = None
    # Original (unfiltered) grids for plotting consistency
    Y_orig: Optional[np.ndarray] = None
    D_orig: Optional[np.ndarray] = None
    I_orig: Optional[np.ndarray] = None
    est_avg: Optional[np.ndarray] = None

    # Storage for bootstrap (not implemented yet)
    eff_boot: Optional[np.ndarray] = None
    D_boot: Optional[np.ndarray] = None
    I_boot: Optional[np.ndarray] = None
    att_avg_boot: Optional[np.ndarray] = None
    call: Optional[Dict[str, Any]] = None
    # Event-study uncertainty (optional)
    est_att_df: Optional[pd.DataFrame] = None  # columns: Period, ATT, SE, CI.lower, CI.upper, count
    # Overall uncertainty (optional) from bootstrap/jackknife
    est_summary: Optional[Dict[str, float]] = None  # keys: se_obs, se_unit

    # Minimal plotting API: support only 'gap' and 'calendar' as in R
    def plot(
        self,
        type: str = "gap",
        show_count: bool = True,
        proportion: float = 0.3,
        main: Optional[str] = None,
        ylab: Optional[str] = None,
        cex_main: Optional[float] = None,
        cex_lab: Optional[float] = None,
        cex_axis: Optional[float] = None,
        cex_text: Optional[float] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
    ):
        from .plot import plot_result as _plot_result
        return _plot_result(
            self,
            type=type,
            show_count=show_count,
            proportion=proportion,
            main=main,
            ylab=ylab,
            cex_main=cex_main,
            cex_lab=cex_lab,
            cex_axis=cex_axis,
            cex_text=cex_text,
            xlim=xlim,
            ylim=ylim,
        )

    # Pretty print to mirror R's print(out)
    def __str__(self) -> str:  # noqa: D401
        from .print import format_summary as _format_summary
        return _format_summary(self)

    def __repr__(self) -> str:
        return self.__str__()


def _check_and_prepare(
    data: pd.DataFrame,
    Y: str,
    D: str,
    X: Optional[Iterable[str]],
    index: Tuple[str, str],
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, List[Any], List[Any], List[str]]:
    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data)

    id_col, time_col = index
    req_cols = [Y, D, id_col, time_col]
    if X:
        req_cols.extend(list(X))
    for c in req_cols:
        if c not in data.columns:
            raise ValueError(f"Column '{c}' not in data.")

    # Drop rows with missing Y or D or index
    df = data.loc[:, req_cols].copy()

    # Sort by id, time
    df.sort_values([id_col, time_col], inplace=True)

    # Encode id and time order
    id_vals = pd.Index(sorted(df[id_col].unique()))
    time_vals = pd.Index(sorted(df[time_col].unique()))

    N = len(id_vals)
    T = len(time_vals)
    p = len(X) if X else 0

    # Reindex to a balanced grid (like data_ub_adj in R)
    g = (
        df.set_index([time_col, id_col])
        .reindex(pd.MultiIndex.from_product([time_vals, id_vals], names=[time_col, id_col]))
        .reset_index()
    )

    Y_mat = g[Y].to_numpy().reshape(T, N)
    D_mat = g[D].to_numpy().reshape(T, N)

    # I_mat indicates non-missing Y
    I_mat = (~pd.isna(Y_mat)).astype(float)

    # Replace NaNs with zeros for matrix ops; keep I_mat to mask
    Y_mat = np.where(np.isfinite(Y_mat), Y_mat, 0.0)
    D_mat = np.where(np.isfinite(D_mat), D_mat, 0.0)

    X_arr = np.zeros((T, N, p), dtype=float)
    X_cols: List[str] = []
    if p > 0 and X is not None:
        X_cols = list(X)
        for j, name in enumerate(X_cols):
            X_mat = g[name].to_numpy().reshape(T, N)
            X_mat = np.where(np.isfinite(X_mat), X_mat, 0.0)
            X_arr[:, :, j] = X_mat

    return Y_mat, D_mat, I_mat, X_arr, id_vals.tolist(), time_vals.tolist(), X_cols


# Native alternating-projections FE solver removed; FE estimation is implemented in effect._predict_counterfactual


# Native masked OLS helper removed; standard errors are obtained from the chosen FE implementation


def _predict_counterfactual(
    Y: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    X: np.ndarray,
    force: str,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    # Delegate to FE implementation in effect.py
    from .effect import _predict_counterfactual as __predict
    return __predict(Y, D, I, X, force)
def _event_study_from_mats(
    Y: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    Y0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from .effect import _event_study_from_mats as __ev
    return __ev(Y, D, I, Y0)


def _compute_event_study_se(
    data: pd.DataFrame,
    Y: str,
    D: str,
    X: Optional[Iterable[str]],
    index: Tuple[str, str],
    vartype: str,
    nboots: int,
    seed: Optional[int] = None,
    min_T0: int = 5,
) -> pd.DataFrame:
    from .effect import _compute_event_study_se as __se
    return __se(data, Y, D, X, index, vartype, nboots, seed, min_T0)


def _att_from_diff(Y: np.ndarray, Y0: np.ndarray, D: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    from .effect import _att_from_diff as __att
    return __att(Y, Y0, D, I)


def fect(
    data: pd.DataFrame,
    Y: str,
    D: str,
    X: Optional[Iterable[str]] = None,
    index: Tuple[str, str] = ("unit", "time"),
    method: str = "fe",
    binary: bool = False,
    force: str = "two-way",
    se: bool = False,
    vartype: str = "bootstrap",
    nboots: int = 200,
    seed: Optional[int] = None,
    keep_sims: bool = False,
    **kwargs: Any,
) -> FectResult:
    if binary:
        raise NotImplementedError("binary=True not yet supported in Python port.")
    if method not in {"fe", "ife"}:
        raise NotImplementedError("Only method='fe' (r=0) is supported in this version.")

    Y_mat, D_mat, I_mat, X_arr, id_vals, time_vals, X_cols = _check_and_prepare(
        data, Y, D, X, index
    )
    # Keep originals for plotting-level counts (R uses original grids)
    Y_orig_store = Y_mat.copy()
    D_orig_store = D_mat.copy()
    I_orig_store = I_mat.copy()

    # Preprocess to mirror key FEct steps
    # 1) Mask treated as missing for fitting
    II = I_mat.copy()
    II[D_mat > 0] = 0.0

    # 2) Drop units with too few untreated observations (min_T0 default 5 per docs)
    min_T0 = int(kwargs.get("min_T0", 5) or 5)
    T0 = II.sum(axis=0)
    keep_units = T0 >= min_T0
    if not np.all(keep_units):
        Y_mat = Y_mat[:, keep_units]
        D_mat = D_mat[:, keep_units]
        I_mat = I_mat[:, keep_units]
        II = II[:, keep_units]
        if X_arr.shape[2] > 0:
            X_arr = X_arr[:, keep_units, :]
        id_vals = [v for v, k in zip(id_vals, keep_units) if k]

    # 3) Drop periods with no controls (no untreated obs in II)
    I_use = II.sum(axis=1)
    keep_times = I_use > 0
    if not np.all(keep_times):
        Y_mat = Y_mat[keep_times, :]
        D_mat = D_mat[keep_times, :]
        I_mat = I_mat[keep_times, :]
        II = II[keep_times, :]
        if X_arr.shape[2] > 0:
            X_arr = X_arr[keep_times, :, :]
        time_vals = [v for v, k in zip(time_vals, keep_times) if k]

    # Fit counterfactual using two-way FE on untreated observations after filtering
    Y0, II, beta, beta_se = _predict_counterfactual(Y_mat, D_mat, I_mat, X_arr, force)

    # Effects
    eff, eff_calendar = _att_from_diff(Y_mat, Y0, D_mat, I_mat)

    # Optionally compute uncertainty estimates via bootstrap/jackknife on original data grid
    est_att_df: Optional[pd.DataFrame] = None
    if se:
        try:
            est_att_df = _compute_event_study_se(
                data, Y, D, X, index, vartype=vartype, nboots=int(nboots), seed=seed, min_T0=min_T0
            )
        except Exception:
            est_att_df = None

    out = FectResult(
        Y_dat=Y_mat,
        Y0_dat=Y0,
        D_dat=D_mat,
        I_dat=I_mat,
        Y_orig=Y_orig_store,
        D_orig=D_orig_store,
        I_orig=I_orig_store,
        Y=Y,
        D=D,
        X=X_cols,
        index=index,
        id=id_vals,
        rawtime=time_vals,
        eff=eff,
        eff_calendar=eff_calendar,
        hasRevs=bool(np.any(np.diff((D_mat > 0).astype(int), axis=0) < 0)),
        method="ife" if method == "fe" else method,
        binary=False,
        beta=beta,
        beta_se=beta_se,
        beta_names=X_cols if (beta is not None and len(X_cols) == (0 if beta is None else len(beta))) else X_cols,
        est_avg=None,
        eff_boot=None,
        D_boot=None,
        I_boot=None,
        att_avg_boot=None,
        call={"method": method, "binary": binary, "index": index, "Y": Y, "D": D, "X": X_cols, "seed": seed, "vartype": vartype, "nboots": nboots, "se": bool(se)},
        est_att_df=est_att_df,
        est_summary={
            "se_obs": (float(est_att_df.attrs.get("se_obs")) if est_att_df is not None and "se_obs" in est_att_df.attrs else float("nan")),
            "se_unit": (float(est_att_df.attrs.get("se_unit")) if est_att_df is not None and "se_unit" in est_att_df.attrs else float("nan")),
        },
    )
    return out
