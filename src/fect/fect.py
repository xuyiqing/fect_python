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
        xlab: Optional[str] = None,
        ylab: Optional[str] = None,
        cex_main: Optional[float] = None,
        cex_lab: Optional[float] = None,
        cex_axis: Optional[float] = None,
        cex_text: Optional[float] = None,
        xlim: Optional[Tuple[float, float]] = None,
        ylim: Optional[Tuple[float, float]] = None,
        # new options forwarded to plot_result
        start0: bool = False,
        plot_ci: Optional[str] = None,
        xbreaks: Optional[Iterable[float]] = None,
        ybreaks: Optional[Iterable[float]] = None,
        xangle: float = 0.0,
        yangle: float = 0.0,
        color: str = "#000000",
        est_lwidth: float = 1.6,
        count_height: float = 0.1,
    ):
        from .plot import plot_result as _plot_result
        return _plot_result(
            self,
            type=type,
            show_count=show_count,
            proportion=proportion,
            main=main,
            xlab=xlab,
            ylab=ylab,
            cex_main=cex_main,
            cex_lab=cex_lab,
            cex_axis=cex_axis,
            cex_text=cex_text,
            xlim=xlim,
            ylim=ylim,
            start0=start0,
            plot_ci=plot_ci,
            xbreaks=xbreaks,
            ybreaks=ybreaks,
            xangle=xangle,
            yangle=yangle,
            color=color,
            est_lwidth=est_lwidth,
            count_height=count_height,
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


# FE solver implemented in effect._predict_counterfactual


def _predict_counterfactual(
    Y: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    X: np.ndarray,
    force: str,
    method: str = "fe",
    r: int = 0,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    from . import _fe as _fe_ext  # required C++ extension
    
    if method == "ife" and r > 0:
        Y0, II, beta, _ = _fe_ext.ife_predict_cf(Y, D, I, X, force, r)
    else:
        Y0, II, beta, _ = _fe_ext.fe_predict_cf(Y, D, I, X, force)
        
    Y0 = np.asarray(Y0, dtype=float)
    II = np.asarray(II, dtype=float)
    beta = (np.asarray(beta, dtype=float) if beta is not None and beta.size > 0 else None)
    return Y0, II, beta, None
def _event_study_from_mats(
    Y: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    Y0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    from . import _fe as _fe_ext
    return _fe_ext.event_study(Y, D, I, Y0)


def _compute_event_study_se(
    data: pd.DataFrame,
    Y: str,
    D: str,
    X: Optional[Iterable[str]],
    index: Tuple[str, str],
    vartype: str,
    nboots: int,
    method: str = "fe",
    r: int = 0,
    seed: Optional[int] = None,
    min_T0: int = 5,
) -> pd.DataFrame:
    import numpy as _np
    import pandas as _pd
    # Prepare balanced grids
    Y_mat_all, D_mat_all, I_mat_all, X_arr_all, id_vals, time_vals, X_cols = _check_and_prepare(data, Y, D, X, index)
    II_all = I_mat_all.copy()
    II_all[D_mat_all > 0] = 0.0
    T0_counts = II_all.sum(axis=0)
    keep_units_mask = T0_counts >= int(min_T0)
    if not np.all(keep_units_mask):
        Y_mat_all = Y_mat_all[:, keep_units_mask]
        D_mat_all = D_mat_all[:, keep_units_mask]
        I_mat_all = I_mat_all[:, keep_units_mask]
        II_all = II_all[:, keep_units_mask]
        if X_arr_all.shape[2] > 0:
            X_arr_all = X_arr_all[:, keep_units_mask, :]
    I_use = II_all.sum(axis=1)
    keep_times_mask = I_use > 0
    if not np.all(keep_times_mask):
        Y_mat_all = Y_mat_all[keep_times_mask, :]
        D_mat_all = D_mat_all[keep_times_mask, :]
        I_mat_all = I_mat_all[keep_times_mask, :]
        II_all = II_all[keep_times_mask, :]
        if X_arr_all.shape[2] > 0:
            X_arr_all = X_arr_all[keep_times_mask, :, :]

    T, N = Y_mat_all.shape
    treated_mask_all = (D_mat_all > 0).any(axis=0)
    idx_tr_all = np.where(treated_mask_all)[0]
    idx_co_all = np.where(~treated_mask_all)[0]
    D_fake = (np.cumsum((D_mat_all > 0).astype(int), axis=0) > 0).astype(int)
    D_fake[I_mat_all <= 0] = 0
    idx_rev_all = np.where(np.sum((D_fake != (D_mat_all > 0).astype(int)), axis=0) > 0)[0]
    idx_tr_pure = np.array([i for i in idx_tr_all if i not in set(idx_rev_all)], dtype=int)
    Ntr = idx_tr_pure.size
    Nco = idx_co_all.size
    Nrev = idx_rev_all.size

    def run_once(col_indices: np.ndarray) -> Tuple[np.ndarray, np.ndarray, float, float, Optional[np.ndarray]]:
        Ym = Y_mat_all[:, col_indices]
        Dm = D_mat_all[:, col_indices]
        Im = I_mat_all[:, col_indices]
        Xa = X_arr_all[:, col_indices, :] if X_arr_all is not None and X_arr_all.size > 0 else _np.zeros((T, len(col_indices), 0))

        Y0, _, beta_rep, _ = _predict_counterfactual(Ym, Dm, Im, Xa, "two-way", method, r)
        timeline, att_vec, _ = _event_study_from_mats(Ym, Dm, Im, Y0)
        eff_rep, _ = _att_from_diff(Ym, Y0, Dm, Im)
        v = eff_rep[~_np.isnan(eff_rep)]
        att_obs = float(_np.nanmean(v)) if v.size else float("nan")
        # R's method: att.unit <- sapply(tr.pos, function(vec) sum(eff[,vec] * D[,vec]) / sum(D[,vec]))
        # Then att.avg.unit <- mean(att.unit, na.rm = TRUE)
        unit_atts: List[float] = []
        for j in range(eff_rep.shape[1]):
            # Check if this unit has any treatment
            d_col = Dm[:, j] if j < Dm.shape[1] else _np.zeros(eff_rep.shape[0])
            if _np.sum(d_col) > 0:  # This is a treated unit
                eff_col = eff_rep[:, j]
                # Weighted ATT for this unit: sum(eff * D) / sum(D)
                valid_mask = ~_np.isnan(eff_col) & (d_col > 0)
                if _np.sum(valid_mask) > 0:
                    weighted_sum = _np.sum(eff_col[valid_mask] * d_col[valid_mask])
                    weight_sum = _np.sum(d_col[valid_mask])
                    if weight_sum > 0:
                        unit_att = weighted_sum / weight_sum
                        if _np.isfinite(unit_att):
                            unit_atts.append(float(unit_att))
        att_unit = float(_np.mean(unit_atts)) if unit_atts else float("nan")
        return timeline, att_vec, att_obs, att_unit, (beta_rep.copy() if beta_rep is not None else None)

    overall_obs_vals: List[float] = []
    overall_unit_vals: List[float] = []
    Y0_full_pre, _, _, _ = _predict_counterfactual(Y_mat_all, D_mat_all, I_mat_all, X_arr_all, "two-way", method, r)
    timeline_full, att_full_pre, counts_full = _event_study_from_mats(Y_mat_all, D_mat_all, I_mat_all, Y0_full_pre)
    Rlen = int(len(timeline_full))
    att_rows: List[np.ndarray] = []
    p = X_arr_all.shape[2] if X_arr_all is not None and X_arr_all.size > 0 else 0
    beta_mat = [] if p > 0 else None

    if vartype.lower() in {"bootstrap", "boot"}:
        B = int(max(1, nboots))
        rng = _np.random.RandomState(seed if seed is not None else None)
        def _sample_indices_seq() -> _np.ndarray:
            tries = 0
            while True:
                tries += 1
                if Nrev > 0:
                    if Ntr > 0 and Nco > 0:
                        samp_rev = idx_rev_all[rng.randint(0, Nrev, size=Nrev)]
                        samp_tr = idx_tr_pure[rng.randint(0, Ntr, size=Ntr)]
                        samp_co = idx_co_all[rng.randint(0, Nco, size=Nco)]
                        samp = _np.concatenate([samp_rev, samp_tr, samp_co])
                    elif Ntr > 0:
                        samp_rev = idx_rev_all[rng.randint(0, Nrev, size=Nrev)]
                        samp_tr = idx_tr_pure[rng.randint(0, Ntr, size=Ntr)]
                        samp = _np.concatenate([samp_rev, samp_tr])
                    elif Nco > 0:
                        samp_rev = idx_rev_all[rng.randint(0, Nrev, size=Nrev)]
                        samp_co = idx_co_all[rng.randint(0, Nco, size=Nco)]
                        samp = _np.concatenate([samp_rev, samp_co])
                    else:
                        samp = idx_rev_all[rng.randint(0, Nrev, size=Nrev)]
                else:
                    if Nco > 0:
                        samp_tr = idx_tr_pure[rng.randint(0, max(1, Ntr), size=Ntr)] if Ntr > 0 else _np.array([], dtype=int)
                        samp_co = idx_co_all[rng.randint(0, Nco, size=Nco)]
                        samp = _np.concatenate([samp_tr, samp_co])
                    else:
                        samp = idx_tr_pure[rng.randint(0, Ntr, size=Ntr)] if Ntr > 0 else _np.array([], dtype=int)
                if samp.size == 0:
                    return samp
                feas = (I_mat_all[:, samp].sum(axis=1) >= 1).all()
                if feas or tries > 1000:
                    return samp
        for _ in range(B):
            cols = _sample_indices_seq()
            if cols.size == 0:
                continue
            tl, av, att_obs, att_unit, beta_vec = run_once(cols)
            if tl.size == 0 or (not _np.isfinite(att_obs)):
                continue
            row = _np.full((Rlen,), _np.nan, dtype=float)
            if tl.size > 0:
                pos = {int(p_): i for i, p_ in enumerate(timeline_full.tolist())}
                for p__, v in zip(tl.tolist(), av.tolist()):
                    ip = pos.get(int(p__))
                    if ip is not None and _np.isfinite(v):
                        row[ip] = float(v)
            att_rows.append(row)
            if _np.isfinite(att_obs):
                overall_obs_vals.append(att_obs)
            if _np.isfinite(att_unit):
                overall_unit_vals.append(att_unit)
            if beta_mat is not None:
                if beta_vec is not None and (beta_vec.size if hasattr(beta_vec, 'size') else 0) == p:
                    beta_mat.append(_np.asarray(beta_vec, dtype=float))
                else:
                    beta_mat.append(_np.full((p,), _np.nan))
    elif vartype.lower() == "jackknife":
        B = N
        for j in range(N):
            cols = _np.array([i for i in range(N) if i != j], dtype=int)
            if cols.size == 0:
                continue
            tl, av, att_obs, att_unit, beta_vec = run_once(cols)
            row = _np.full((Rlen,), _np.nan, dtype=float)
            if tl.size > 0:
                pos = {int(p_): i for i, p_ in enumerate(timeline_full.tolist())}
                for p__, v in zip(tl.tolist(), av.tolist()):
                    ip = pos.get(int(p__))
                    if ip is not None and _np.isfinite(v):
                        row[ip] = float(v)
            att_rows.append(row)
            if _np.isfinite(att_obs):
                overall_obs_vals.append(att_obs)
            if _np.isfinite(att_unit):
                overall_unit_vals.append(att_unit)
            if beta_mat is not None:
                if beta_vec is not None and (beta_vec.size if hasattr(beta_vec, 'size') else 0) == p:
                    beta_mat.append(_np.asarray(beta_vec, dtype=float))
                else:
                    beta_mat.append(_np.full((p,), _np.nan))
    else:
        raise NotImplementedError("vartype must be 'bootstrap' or 'jackknife'.")

    if len(att_rows) > 0:
        AttMat = _np.vstack(att_rows)
    else:
        AttMat = _np.empty((0, Rlen))
    se_vec = _np.full((Rlen,), _np.nan, dtype=float)
    if AttMat.shape[0] > 0:
        valid_counts = _np.sum(~_np.isnan(AttMat), axis=0).astype(float)
        with _np.errstate(invalid="ignore", divide='ignore'):
            means = _np.where(valid_counts > 0, _np.nansum(AttMat, axis=0) / valid_counts, _np.nan)
        sumsq = _np.nansum((AttMat - means) ** 2, axis=0)
        mask_cols = valid_counts > 1.0
        if vartype.lower() == "jackknife":
            # R's jackknife formula: sd * sqrt(N-1)
            se_vec[mask_cols] = _np.sqrt(sumsq[mask_cols] / (valid_counts[mask_cols] - 1.0)) * _np.sqrt(valid_counts[mask_cols] - 1.0)
        else:
            se_vec[mask_cols] = _np.sqrt(sumsq[mask_cols] / (valid_counts[mask_cols] - 1.0))
    df_se = pd.DataFrame({"Period": timeline_full.astype(int), "SE": se_vec})

    df_base = pd.DataFrame({"Period": timeline_full, "ATT": att_full_pre, "count": counts_full})
    out_df = df_base.merge(df_se, on="Period", how="left")

    # 95% CI
    try:
        from scipy.stats import norm as _scipy_norm
        Z97 = float(_scipy_norm.ppf(0.975))
    except Exception:
        Z97 = 1.959963984540054
    out_df["CI.lower"] = out_df["ATT"] - Z97 * out_df["SE"]
    out_df["CI.upper"] = out_df["ATT"] + Z97 * out_df["SE"]

    # overall SE summaries (obs-weighted and unit-weighted)
    def _se_from_reps(values: List[float], method: str, point_estimate: float = None) -> float:
        if len(values) <= 1:
            return float("nan")
        arr = np.array(values, dtype=float)
        if method == "jackknife":
            # R's jackknife formula from jackknifed() function:
            # The key insight: R uses the ORIGINAL point estimate, not the mean of jackknife samples
            # jackknifed(att.avg, att.avg.boot, alpha)
            # where att.avg is the original estimate, att.avg.boot are jackknife samples
            
            # Use the original point estimate if provided, otherwise use mean of samples
            if point_estimate is not None:
                x_hat = float(point_estimate)
            else:
                x_hat = float(np.nanmean(arr))
            
            n = float(arr.size)
            
            # R's exact formula:
            # X <- matrix(rep(c(x), N), p, N) * N  (here p=1, so just x*N)
            # Y <- X - y * (N - 1)
            X = x_hat * n
            Y = X - arr * (n - 1.0)
            
            # Calculate variance (R uses var with na.rm=TRUE)
            valid_Y = Y[np.isfinite(Y)]
            if len(valid_Y) <= 1:
                return float("nan")
            
            Y_var = np.var(valid_Y, ddof=1)  # R's var() uses ddof=1
            vn = float(len(valid_Y))  # Effective sample size
            
            # Jackknife SE
            return float(np.sqrt(Y_var / vn))
        else:
            return float(np.std(arr, ddof=1))

    # Calculate original point estimates for jackknife
    eff_full, _ = _att_from_diff(Y_mat_all, Y0_full_pre, D_mat_all, I_mat_all)
    v_full = eff_full[~_np.isnan(eff_full)]
    att_obs_orig = float(_np.nanmean(v_full)) if v_full.size else float("nan")
    
    # Calculate original unit-weighted ATT
    unit_atts_orig = []
    for j in range(eff_full.shape[1]):
        d_col = D_mat_all[:, j]
        if _np.sum(d_col) > 0:
            eff_col = eff_full[:, j]
            valid_mask = ~_np.isnan(eff_col) & (d_col > 0)
            if _np.sum(valid_mask) > 0:
                weighted_sum = _np.sum(eff_col[valid_mask] * d_col[valid_mask])
                weight_sum = _np.sum(d_col[valid_mask])
                if weight_sum > 0:
                    unit_att = weighted_sum / weight_sum
                    if _np.isfinite(unit_att):
                        unit_atts_orig.append(float(unit_att))
    att_unit_orig = float(_np.mean(unit_atts_orig)) if unit_atts_orig else float("nan")
    
    se_obs = _se_from_reps(overall_obs_vals, vartype.lower())
    se_unit = _se_from_reps(overall_unit_vals, vartype.lower())
    out_df.attrs["se_obs"] = se_obs
    out_df.attrs["se_unit"] = se_unit

    if beta_mat is not None and len(beta_mat) > 0:
        try:
            Bmat = _np.vstack(beta_mat)
            valid_counts_b = _np.sum(~_np.isnan(Bmat), axis=0).astype(float)
            with _np.errstate(invalid="ignore", divide='ignore'):
                means = _np.where(valid_counts_b > 0, _np.nansum(Bmat, axis=0) / valid_counts_b, _np.nan)
            sumsq = _np.nansum((Bmat - means) ** 2, axis=0)
            mask_b = valid_counts_b > 1.0
            se_beta = _np.full((Bmat.shape[1],), _np.nan, dtype=float)
            if vartype.lower() == "jackknife":
                # Original jackknife formula that was working before
                se_beta[mask_b] = _np.sqrt(((valid_counts_b[mask_b] - 1.0) / valid_counts_b[mask_b]) * sumsq[mask_b])
            else:
                se_beta[mask_b] = _np.sqrt(sumsq[mask_b] / (valid_counts_b[mask_b] - 1.0))
            out_df.attrs["beta_se_boot"] = se_beta.astype(float)
        except Exception:
            pass

    return out_df


def _att_from_diff(Y: np.ndarray, Y0: np.ndarray, D: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    from . import _fe as _fe_ext
    eff, att_t, counts_t = _fe_ext.att_from_diff(Y, Y0, D, I)
    df = pd.DataFrame({"ATT-calendar": np.asarray(att_t), "count": np.asarray(counts_t)})
    return np.asarray(eff), df


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
    r: int = 0,  # number of factors for IFE method
    # parallel options removed
    **kwargs: Any,
) -> FectResult:
    if binary:
        raise NotImplementedError("binary=True not yet supported in Python port.")
    if method not in {"fe", "ife"}:
        raise NotImplementedError("Only method='fe' and method='ife' are supported.")

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
    
    # 2) Create YY with treated observations set to 0 (key R step!)
    YY = Y_mat.copy()
    YY[II == 0] = 0.0

    # 3) Drop units with too few untreated observations (min_T0 default 5 per docs)
    min_T0 = int(kwargs.get("min_T0", 5) or 5)
    T0 = II.sum(axis=0)
    keep_units = T0 >= min_T0
    if not np.all(keep_units):
        Y_mat = Y_mat[:, keep_units]
        YY = YY[:, keep_units]
        D_mat = D_mat[:, keep_units]
        I_mat = I_mat[:, keep_units]
        II = II[:, keep_units]
        if X_arr.shape[2] > 0:
            X_arr = X_arr[:, keep_units, :]
        id_vals = [v for v, k in zip(id_vals, keep_units) if k]

    # 4) Drop periods with no controls (no untreated obs in II)
    I_use = II.sum(axis=1)
    keep_times = I_use > 0
    if not np.all(keep_times):
        Y_mat = Y_mat[keep_times, :]
        YY = YY[keep_times, :]
        D_mat = D_mat[keep_times, :]
        I_mat = I_mat[keep_times, :]
        II = II[keep_times, :]
        if X_arr.shape[2] > 0:
            X_arr = X_arr[keep_times, :, :]
        time_vals = [v for v, k in zip(time_vals, keep_times) if k]

    # Determine r for IFE method
    if method == "ife" and r <= 0:
        # Default r=1 for IFE method
        r = 1
    elif method == "fe":
        r = 0
    
    # Fit counterfactual using FE or IFE method on untreated observations after filtering
    # Use YY (with treated obs set to 0) instead of Y_mat for IFE
    Y_input = YY if method == "ife" else Y_mat
    Y0, II, beta, beta_se = _predict_counterfactual(Y_input, D_mat, I_mat, X_arr, force, method, r)

    # Effects
    eff, eff_calendar = _att_from_diff(Y_mat, Y0, D_mat, I_mat)

    # Optionally compute uncertainty estimates via bootstrap/jackknife on original data grid
    est_att_df: Optional[pd.DataFrame] = None
    if se:
        try:
            est_att_df = _compute_event_study_se(
                data, Y, D, X, index,
                vartype=vartype,
                nboots=int(nboots),
                method=method,
                r=r,
                seed=seed,
                min_T0=min_T0,
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
        method=method,
        binary=False,
        beta=beta,
        beta_se=beta_se,
        beta_names=X_cols if (beta is not None and len(X_cols) == (0 if beta is None else len(beta))) else X_cols,
        est_avg=None,
        eff_boot=None,
        D_boot=None,
        I_boot=None,
        att_avg_boot=None,
        call={
            "method": method,
            "binary": binary,
            "index": index,
            "Y": Y,
            "D": D,
            "X": X_cols,
            "seed": seed,
            "vartype": vartype,
            "nboots": nboots,
            "se": bool(se),
            # parallel removed
        },
        est_att_df=est_att_df,
        est_summary={
            "se_obs": (float(est_att_df.attrs.get("se_obs")) if est_att_df is not None and "se_obs" in est_att_df.attrs else float("nan")),
            "se_unit": (float(est_att_df.attrs.get("se_unit")) if est_att_df is not None and "se_unit" in est_att_df.attrs else float("nan")),
            "se_beta": (list(est_att_df.attrs.get("beta_se_boot")) if est_att_df is not None and "beta_se_boot" in est_att_df.attrs else [float("nan"), float("nan")]),
        },
    )
    return out
