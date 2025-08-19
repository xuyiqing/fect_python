from __future__ import annotations

from typing import Optional, Iterable, Tuple, List

import numpy as np
import pandas as pd

from .fect import FectResult


def _event_study_from_mats(
    Y: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    Y0: np.ndarray,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    import numpy as _np
    T, N = Y.shape
    first_on = _np.full(N, _np.nan)
    for j in range(N):
        dcol = (D[:, j] > 0).astype(int)
        idx = _np.where(dcol == 1)[0]
        if idx.size > 0:
            first_on[j] = idx[0]
    treated = _np.where(_np.isfinite(first_on))[0]
    if treated.size == 0:
        return _np.array([], dtype=int), _np.array([], dtype=float), _np.array([], dtype=int)

    T0_counts = _np.full(N, _np.nan)
    for j in treated:
        T0_counts[j] = int(first_on[j]) - 1
    T0_valid = T0_counts[_np.isfinite(T0_counts)].astype(int)
    if T0_valid.size == 0:
        return _np.array([], dtype=int), _np.array([], dtype=float), _np.array([], dtype=int)
    rmin = 1 - int(_np.max(T0_valid))
    rmax = T - int(_np.min(T0_valid))
    timeline = _np.arange(rmin, rmax + 1, dtype=int)
    R = timeline.size

    Y_tr_aug = _np.full((R, N), _np.nan)
    Y_ct_aug = _np.full((R, N), _np.nan)
    for j in treated:
        t0c = int(T0_counts[j])
        for t in range(T):
            if I[t, j] != 1:
                continue
            r = t - t0c
            ridx = r - rmin
            if ridx < 0 or ridx >= R:
                continue
            include = (r <= 0) or ((r > 0) and (D[t, j] == 1))
            if include:
                Y_tr_aug[ridx, j] = Y[t, j]
                Y_ct_aug[ridx, j] = Y0[t, j]

    Y_tr_aug[_np.isnan(Y_ct_aug)] = _np.nan
    tr_cnt = _np.sum(~_np.isnan(Y_tr_aug), axis=1).astype(float)
    ct_cnt = _np.sum(~_np.isnan(Y_ct_aug), axis=1).astype(float)
    tr_sum = _np.nansum(Y_tr_aug, axis=1)
    ct_sum = _np.nansum(Y_ct_aug, axis=1)
    with _np.errstate(invalid="ignore", divide='ignore'):
        tr_bar = _np.where(tr_cnt > 0, tr_sum / tr_cnt, _np.nan)
        ct_bar = _np.where(ct_cnt > 0, ct_sum / ct_cnt, _np.nan)
    att = tr_bar - ct_bar
    counts = _np.sum(_np.isfinite(Y_tr_aug), axis=1).astype(int)
    return timeline, att, counts


def _att_from_diff(Y: np.ndarray, Y0: np.ndarray, D: np.ndarray, I: np.ndarray) -> Tuple[np.ndarray, pd.DataFrame]:
    T, N = Y.shape
    treated_mask = (D > 0) & (I > 0)
    eff = np.full((T, N), np.nan, dtype=float)
    diff_mat = (Y - Y0)
    eff[treated_mask] = diff_mat[treated_mask]

    count_t = np.sum(~np.isnan(eff), axis=1)
    att_t = np.zeros(Y.shape[0], dtype=float)
    att_t[:] = np.nan
    valid = count_t > 0
    if np.any(valid):
        att_t[valid] = np.nanmean(eff[valid, :], axis=1)
    df = pd.DataFrame({"ATT-calendar": att_t, "count": count_t})
    return eff, df


def _predict_counterfactual(
    Y: np.ndarray,
    D: np.ndarray,
    I: np.ndarray,
    X: np.ndarray,
    force: str,
):
    """FE on control-only observations using linearmodels.PanelOLS (no dense dummies).

    Returns Y0 (T x N), II (mask of control-only rows used in fitting), beta (p,), beta_se (p,).
    """
    T, N = Y.shape
    p = X.shape[2] if X is not None and X.size > 0 else 0

    # mask treated as not usable for fitting
    II = I.copy()
    II[D > 0] = 0.0

    # Helper: fast two-way demeaning to recover FE parts on possibly unbalanced mask
    def _recover_additive_fe(resid_mat: np.ndarray, weight_mat: np.ndarray, fe_mode: str) -> tuple[float, np.ndarray, np.ndarray]:
        wm = (weight_mat > 0).astype(float)
        total_w = wm.sum()
        if total_w == 0:
            return 0.0, np.zeros(N, float), np.zeros(T, float)
        mu_local = float((resid_mat * wm).sum() / total_w)
        alpha_local = np.zeros(N, float)
        xi_local = np.zeros(T, float)

        if fe_mode == "none":
            return mu_local, alpha_local, xi_local
        if fe_mode == "unit":
            # alpha_i = mean_t (resid - mu)
            denom = wm.sum(axis=0)
            denom[denom == 0] = 1.0
            alpha_local = ((resid_mat - mu_local) * wm).sum(axis=0) / denom
            return mu_local, alpha_local, xi_local
        if fe_mode == "time":
            denom = wm.sum(axis=1)
            denom[denom == 0] = 1.0
            xi_local = ((resid_mat - mu_local) * wm).sum(axis=1) / denom
            return mu_local, alpha_local, xi_local

        # two-way: alternating projections
        max_iter = 50
        tol = 1e-10
        for _ in range(max_iter):
            prev_alpha = alpha_local.copy()
            prev_xi = xi_local.copy()
            # update alpha
            denom_a = wm.sum(axis=0)
            denom_a[denom_a == 0] = 1.0
            alpha_local = ((resid_mat - mu_local - xi_local[:, None]) * wm).sum(axis=0) / denom_a
            # center alpha to satisfy identifiability approximately
            alpha_local -= alpha_local.mean()
            # update xi
            denom_x = wm.sum(axis=1)
            denom_x[denom_x == 0] = 1.0
            xi_local = ((resid_mat - mu_local - alpha_local[None, :]) * wm).sum(axis=1) / denom_x
            xi_local -= xi_local.mean()
            # small mu adjustment
            mu_local = float(((resid_mat - alpha_local[None, :] - xi_local[:, None]) * wm).sum() / total_w)
            if np.max(np.abs(alpha_local - prev_alpha)) < tol and np.max(np.abs(xi_local - prev_xi)) < tol:
                break
        return mu_local, alpha_local, xi_local

    # If no covariates, just recover FE parts from Y on controls
    fe_mode = {
        "none": "none",
        "unit": "unit",
        "time": "time",
        "two-way": "two-way",
    }.get(force, "two-way")

    if p == 0:
        mu, alpha, xi = _recover_additive_fe(Y, II, fe_mode)
        Y0 = mu + alpha[None, :] + xi[:, None]
        return Y0, II, None, None

    # With covariates: fast within transformation + small OLS (no pandas, no dense dummies)
    def _within_tilde(mat: np.ndarray, wm: np.ndarray, mode: str) -> np.ndarray:
        m, a, x = _recover_additive_fe(mat, wm, mode)
        return mat - (m + a[None, :] + x[:, None])

    # Initialize
    beta = np.zeros(p, dtype=float)
    max_iter = 15
    tol = 1e-8
    mask_vec = (II > 0)

    for _ in range(max_iter):
        # Residual after removing covariate contribution
        covar_fit_iter = np.zeros((T, N), dtype=float)
        for k in range(p):
            covar_fit_iter += X[:, :, k] * beta[k]
        resid = Y - covar_fit_iter

        # Update FE by demeaning residuals on controls
        mu, alpha, xi = _recover_additive_fe(resid, II, fe_mode)
        y_tilde = resid - (mu + alpha[None, :] + xi[:, None])

        # Build small normal equations X'X and X'y using demeaned X on controls
        Xty = np.zeros(p, dtype=float)
        XtX = np.zeros((p, p), dtype=float)
        y_vec = y_tilde[mask_vec]
        # Precompute demeaned X columns lazily to save memory
        X_tilde_cols = []
        for k in range(p):
            Xk_tilde = _within_tilde(X[:, :, k], II, fe_mode)
            xk_vec = Xk_tilde[mask_vec]
            X_tilde_cols.append(xk_vec)
            Xty[k] = float(np.dot(xk_vec, y_vec))
        # Fill XtX (symmetric)
        for i in range(p):
            xi_c = X_tilde_cols[i]
            for j in range(i, p):
                xj_c = X_tilde_cols[j]
                v = float(np.dot(xi_c, xj_c))
                XtX[i, j] = v
                XtX[j, i] = v
        # Ridge to stabilize if near-singular
        lam = 1e-12
        for d in range(p):
            XtX[d, d] += lam
        try:
            new_beta = np.linalg.solve(XtX, Xty)
        except np.linalg.LinAlgError:
            new_beta, *_ = np.linalg.lstsq(XtX, Xty, rcond=None)

        if np.all(np.isfinite(new_beta)):
            if np.linalg.norm(new_beta - beta, ord=np.inf) < tol:
                beta = new_beta
                break
            beta = new_beta
        else:
            break

    # Final FE using final beta
    covar_fit = np.zeros((T, N), dtype=float)
    for k in range(p):
        covar_fit += X[:, :, k] * beta[k]
    mu, alpha, xi = _recover_additive_fe(Y - covar_fit, II, fe_mode)
    Y0 = mu + alpha[None, :] + xi[:, None] + covar_fit
    beta_se = None  # not computed in fast path
    return Y0, II, beta, beta_se


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
    import numpy as _np
    import pandas as _pd
    # local imports to avoid circular during module import
    from .fect import _check_and_prepare

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

        Y0, _, beta_rep, _ = _predict_counterfactual(Ym, Dm, Im, Xa, "two-way")
        timeline, att_vec, _ = _event_study_from_mats(Ym, Dm, Im, Y0)
        eff_rep, _ = _att_from_diff(Ym, Y0, Dm, Im)
        v = eff_rep[~_np.isnan(eff_rep)]
        att_obs = float(_np.nanmean(v)) if v.size else float("nan")
        unit_means: List[float] = []
        for j in range(eff_rep.shape[1]):
            col = eff_rep[:, j]
            if _np.all(_np.isnan(col)):
                continue
            m = _np.nanmean(col)
            if _np.isfinite(m):
                unit_means.append(float(m))
        att_unit = float(_np.nanmean(unit_means)) if unit_means else float("nan")
        return timeline, att_vec, att_obs, att_unit, (beta_rep.copy() if beta_rep is not None else None)

    overall_obs_vals: List[float] = []
    overall_unit_vals: List[float] = []
    Y0_full_pre, _, _, _ = _predict_counterfactual(Y_mat_all, D_mat_all, I_mat_all, X_arr_all, "two-way")
    timeline_full, att_full_pre, counts_full = _event_study_from_mats(Y_mat_all, D_mat_all, I_mat_all, Y0_full_pre)
    Rlen = int(len(timeline_full))
    att_rows: List[np.ndarray] = []
    p = X_arr_all.shape[2] if X_arr_all is not None and X_arr_all.size > 0 else 0
    beta_mat = None
    if p > 0:
        beta_mat = []

    if vartype.lower() in {"bootstrap", "boot"}:
        B = int(max(1, nboots))
        rng = np.random.RandomState(seed if seed is not None else None)
        for _ in range(B):
            tries = 0
            while True:
                tries += 1
                if Nrev > 0:
                    if Ntr > 0 and Nco > 0:
                        samp_rev = idx_rev_all[rng.randint(0, Nrev, size=Nrev)]
                        samp_tr = idx_tr_pure[rng.randint(0, Ntr, size=Ntr)]
                        samp_co = idx_co_all[rng.randint(0, Nco, size=Nco)]
                        samp = np.concatenate([samp_rev, samp_tr, samp_co])
                    elif Ntr > 0:
                        samp_rev = idx_rev_all[rng.randint(0, Nrev, size=Nrev)]
                        samp_tr = idx_tr_pure[rng.randint(0, Ntr, size=Ntr)]
                        samp = np.concatenate([samp_rev, samp_tr])
                    elif Nco > 0:
                        samp_rev = idx_rev_all[rng.randint(0, Nrev, size=Nrev)]
                        samp_co = idx_co_all[rng.randint(0, Nco, size=Nco)]
                        samp = np.concatenate([samp_rev, samp_co])
                    else:
                        samp = idx_rev_all[rng.randint(0, Nrev, size=Nrev)]
                else:
                    if Nco > 0:
                        samp_tr = idx_tr_pure[rng.randint(0, max(1, Ntr), size=Ntr)] if Ntr > 0 else np.array([], dtype=int)
                        samp_co = idx_co_all[rng.randint(0, Nco, size=Nco)]
                        samp = np.concatenate([samp_tr, samp_co])
                    else:
                        samp = idx_tr_pure[rng.randint(0, Ntr, size=Ntr)] if Ntr > 0 else np.array([], dtype=int)

                if samp.size == 0:
                    break
                feas = (I_mat_all[:, samp].sum(axis=1) >= 1).all()
                if feas or tries > 1000:
                    break
            tl, av, att_obs, att_unit, beta_vec = run_once(samp)
            if tl.size == 0 or (not _np.isfinite(att_obs)):
                continue
            row = _np.full((Rlen,), _np.nan, dtype=float)
            if tl.size > 0:
                pos = {int(p): i for i, p in enumerate(timeline_full.tolist())}
                for p_, v in zip(tl.tolist(), av.tolist()):
                    ip = pos.get(int(p_))
                    if ip is not None and _np.isfinite(v):
                        row[ip] = float(v)
            att_rows.append(row)
            if _np.isfinite(att_obs):
                overall_obs_vals.append(att_obs)
            if _np.isfinite(att_unit):
                overall_unit_vals.append(att_unit)
            if beta_mat is not None:
                if beta_vec is not None and beta_vec.size == p:
                    beta_mat.append(beta_vec.astype(float))
                else:
                    beta_mat.append(_np.full((p,), _np.nan))
    elif vartype.lower() == "jackknife":
        B = N
        for j in range(N):
            cols = np.array([i for i in range(N) if i != j], dtype=int)
            if cols.size == 0:
                continue
            tl, av, att_obs, att_unit, beta_vec = run_once(cols)
            row = _np.full((Rlen,), _np.nan, dtype=float)
            if tl.size > 0:
                pos = {int(p): i for i, p in enumerate(timeline_full.tolist())}
                for p_, v in zip(tl.tolist(), av.tolist()):
                    ip = pos.get(int(p_))
                    if ip is not None and np.isfinite(v):
                        row[ip] = float(v)
            att_rows.append(row)
            if _np.isfinite(att_obs):
                overall_obs_vals.append(att_obs)
            if _np.isfinite(att_unit):
                overall_unit_vals.append(att_unit)
            if beta_mat is not None:
                if beta_vec is not None and beta_vec.size == p:
                    beta_mat.append(beta_vec.astype(float))
                else:
                    beta_mat.append(_np.full((p,), _np.nan))
    else:
        raise NotImplementedError("vartype must be 'bootstrap' or 'jackknife' in Python port.")

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
            se_vec[mask_cols] = _np.sqrt(((valid_counts[mask_cols] - 1.0) / valid_counts[mask_cols]) * sumsq[mask_cols])
        else:
            se_vec[mask_cols] = _np.sqrt(sumsq[mask_cols] / (valid_counts[mask_cols] - 1.0))
    df_se = pd.DataFrame({"Period": timeline_full.astype(int), "SE": se_vec})

    df_base = pd.DataFrame({"Period": timeline_full, "ATT": att_full_pre, "count": counts_full})
    out = df_base.merge(df_se, on="Period", how="left")
    if "SE" in out.columns:
        try:
            from scipy.stats import norm as _scipy_norm
            Z97 = float(_scipy_norm.ppf(0.975))
        except Exception:
            Z97 = 1.959963984540054
        out["CI.lower"] = out["ATT"] - Z97 * out["SE"]
        out["CI.upper"] = out["ATT"] + Z97 * out["SE"]
    else:
        out["CI.lower"] = np.nan
        out["CI.upper"] = np.nan

    def _se_from_reps(values: List[float], method: str) -> float:
        if len(values) <= 1:
            return float("nan")
        arr = np.array(values, dtype=float)
        if method == "jackknife":
            n = float(arr.size)
            meanv = float(np.mean(arr))
            return float(np.sqrt((n - 1.0) / n * np.sum((arr - meanv) ** 2)))
        else:
            return float(np.std(arr, ddof=1))

    se_obs = _se_from_reps(overall_obs_vals, vartype.lower())
    se_unit = _se_from_reps(overall_unit_vals, vartype.lower())
    out.attrs["se_obs"] = se_obs
    out.attrs["se_unit"] = se_unit

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
                se_beta[mask_b] = _np.sqrt(((valid_counts_b[mask_b] - 1.0) / valid_counts_b[mask_b]) * sumsq[mask_b])
            else:
                se_beta[mask_b] = _np.sqrt(sumsq[mask_b] / (valid_counts_b[mask_b] - 1.0))
            out.attrs["beta_se_boot"] = se_beta.astype(float)
        except Exception:
            pass
    return out


def _get_effect(D: np.ndarray, I: np.ndarray, eff: np.ndarray, cumu: bool, period: Tuple[int, int]) -> np.ndarray:
    # Convert D to relative time per unit
    T, N = D.shape
    D_cum = np.cumsum(D, axis=0)
    rel = np.zeros_like(D_cum)
    for j in range(N):
        t0 = int(np.sum(D_cum[:, j] == 0))
        rel[:, j] = np.arange(1, T + 1) - t0
    # mask out excluded
    rel = rel.astype(float)
    rel[I == 0] = np.nan

    vd = rel.flatten()
    veff = eff.flatten()
    keep = ~np.isnan(vd)
    vd = vd[keep]
    veff = veff[keep]

    ts, te = period
    te = min(te, int(np.max(vd)) if len(vd) else te)
    times = np.arange(ts, te + 1, dtype=int)

    out = np.full(len(times), np.nan)
    if cumu:
        pos = []
        for i, t in enumerate(times):
            pos.extend(list(np.where(vd == t)[0]))
            if pos:
                out[i] = np.nanmean(veff[pos]) * (i + 1)
    else:
        for i, t in enumerate(times):
            idx = (vd == t)
            if np.any(idx):
                out[i] = np.nanmean(veff[idx])
    return out


def effect(
    x: FectResult,
    cumu: bool = True,
    id: Optional[Iterable] = None,
    period: Optional[Tuple[int, int]] = None,
    plot: bool = False,
    count: bool = True,
    xlab: Optional[str] = None,
    ylab: Optional[str] = None,
    main: Optional[str] = None,
) -> FectResult:
    # Compute point estimates even if no bootstrap/jackknife results are present
    # When treatments have reversals, cumulative effects are not well-defined.
    # Fallback: compute per-period (non-cumulative) estimates so callers can proceed.
    if x.hasRevs and cumu:
        cumu = False

    # Select units
    if id is None:
        mask_units = (x.D_dat.sum(axis=0) > 0)
    else:
        unit_names = np.array(x.id)
        mask_units = np.isin(unit_names, list(id))

    eff = x.eff[:, mask_units]
    D = x.D_dat[:, mask_units]
    I = x.I_dat[:, mask_units]

    # Determine default period
    T = eff.shape[0]
    D_cum = np.cumsum(D, axis=0)
    minT0 = int(np.min(np.sum(D_cum == 0, axis=0))) if D.size > 0 else 0
    default_period = (1, T - minT0)
    if period is None:
        period = default_period
    else:
        if period[1] > default_period[1]:
            raise ValueError(f"Ending period should not be greater than {default_period[1]}")

    catt = _get_effect(D, I, eff, cumu=cumu, period=period)

    # Attach point estimates for compatibility
    x.effect_est_avg = catt
    return x
