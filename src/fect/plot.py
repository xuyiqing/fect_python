from __future__ import annotations

from typing import Optional, Tuple, Sequence, Dict, Any

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


def _esplot_like(
    data: pd.DataFrame,
    Period: str,
    Estimate: str,
    SE: Optional[str],
    CI_lower: str,
    CI_upper: str,
    Count: Optional[str],
    xlim: Optional[Tuple[float, float]],
    ylim: Optional[Tuple[float, float]],
    show_points: bool = True,
    show_ref0: bool = True,
    # new optional styling/behavior flags (kept minimal for compatibility)
    start0: bool = False,
    plot_ci: Optional[str] = None,  # "0.9", "0.95", or "none"
    xbreaks: Optional[Sequence[float]] = None,
    ybreaks: Optional[Sequence[float]] = None,
    xangle: float = 0.0,
    yangle: float = 0.0,
    color: str = "#000000",
    est_lwidth: float = 1.6,
    xlab: Optional[str] = None,
    theme_bw: bool = True,
    count_height: float = 0.1,
):
    df = pd.DataFrame(data).copy()

    # Ensure Estimate present
    if Estimate not in df.columns:
        raise ValueError(f"Estimate column '{Estimate}' not found.")

    # Compute CI from SE if missing or NaN-only
    if CI_lower not in df.columns:
        if SE and SE in df.columns:
            try:
                from scipy.stats import norm as _scipy_norm  # optional
                z97 = float(_scipy_norm.ppf(0.975))
            except Exception:
                z97 = 1.959963984540054
            df[CI_lower] = df[Estimate] - z97 * df[SE]
        else:
            df[CI_lower] = np.nan
    if CI_upper not in df.columns:
        if SE and SE in df.columns:
            try:
                from scipy.stats import norm as _scipy_norm
                z97 = float(_scipy_norm.ppf(0.975))
            except Exception:
                z97 = 1.959963984540054
            df[CI_upper] = df[Estimate] + z97 * df[SE]
        else:
            df[CI_upper] = np.nan

    if SE and SE in df.columns:
        try:
            from scipy.stats import norm as _scipy_norm
            z97 = float(_scipy_norm.ppf(0.975))
        except Exception:
            z97 = 1.959963984540054
        if df[CI_lower].isna().all():
            df[CI_lower] = df[Estimate] - z97 * df[SE]
        if df[CI_upper].isna().all():
            df[CI_upper] = df[Estimate] + z97 * df[SE]

    x = pd.to_numeric(df[Period], errors="coerce")
    if bool(start0):
        try:
            x = x - 1
        except Exception:
            pass
    y = pd.to_numeric(df[Estimate], errors="coerce")
    cil = pd.to_numeric(df[CI_lower], errors="coerce")
    ciu = pd.to_numeric(df[CI_upper], errors="coerce")

    # style
    if theme_bw:
        try:
            plt.style.use("default")
        except Exception:
            pass
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.axhline(0, color="#AAAAAA70", linewidth=1.5)

    # Vertical reference: 0 if only pre or post exists; 0.5 if 0 and 1 both exist
    if show_ref0 and x.notna().any():
        try:
            if (x.min() <= 0 <= x.max()):
                vpos = 0.0
                if ((x == 0).any() and (x == 1).any()):
                    vpos = 0.5
                ax.axvline(vpos, color="#BBBBBB", linewidth=2.0, alpha=0.6)
        except Exception:
            pass

    # optionally recompute CI from SE with specified level
    if plot_ci and plot_ci.lower() != "none" and SE and (SE in df.columns):
        try:
            from scipy.stats import norm as _scipy_norm
            z = float(_scipy_norm.ppf(0.5 + float(plot_ci) / 2.0))
        except Exception:
            z = 1.959963984540054
        _se = pd.to_numeric(df[SE], errors="coerce")
        cil = y - z * _se
        ciu = y + z * _se

    mask_valid_ci = (~x.isna()) & (~cil.isna()) & (~ciu.isna()) & (ciu > cil)
    if mask_valid_ci.any():
        xv = x[mask_valid_ci].to_numpy(dtype=float)
        yv = y[mask_valid_ci].to_numpy(dtype=float)
        cilv = cil[mask_valid_ci].to_numpy(dtype=float)
        ciuv = ciu[mask_valid_ci].to_numpy(dtype=float)
        ax.vlines(xv, cilv, ciuv, color=color, linewidth=1.2, zorder=2)
        yerr_low = np.clip(yv - cilv, a_min=0.0, a_max=None)
        yerr_high = np.clip(ciuv - yv, a_min=0.0, a_max=None)
        yerr = np.vstack([yerr_low, yerr_high])
        ax.errorbar(xv, yv, yerr=yerr, fmt='none', ecolor=color, elinewidth=1.0, capsize=3, capthick=1, zorder=2)
    if show_points:
        ax.plot(x, y, marker="o", color=color, linewidth=est_lwidth, zorder=3)
    else:
        ax.scatter(x, y, s=25, color=color, zorder=3)

    if Count and Count in df.columns:
        counts = pd.to_numeric(df[Count], errors="coerce")
        if counts.notna().any() and np.nanmax(counts) > 0:
            ymin, ymax = ax.get_ylim()
            height = (ymax - ymin) * float(max(0.0, min(1.0, count_height)))
            maxc = np.nanmax(counts)
            for xi, ci in zip(x, counts):
                if pd.notna(xi) and pd.notna(ci) and ci > 0:
                    ax.add_patch(plt.Rectangle((xi - 0.2, ymin), 0.4, height * (ci / maxc), color="gray", alpha=0.4, linewidth=0.2))

    if xlim is not None:
        try:
            ax.set_xlim(float(xlim[0]), float(xlim[1]))
        except Exception:
            pass
    if ylim is not None:
        try:
            ax.set_ylim(float(ylim[0]), float(ylim[1]))
        except Exception:
            pass

    # optional ticks/angles
    if xbreaks is not None:
        try:
            ax.set_xticks([float(v) for v in xbreaks])
        except Exception:
            pass
    if ybreaks is not None:
        try:
            ax.set_yticks([float(v) for v in ybreaks])
        except Exception:
            pass
    if isinstance(xangle, (int, float)) and abs(float(xangle)) > 0:
        for lbl in ax.get_xticklabels():
            lbl.set_rotation(float(xangle))
    if isinstance(yangle, (int, float)) and abs(float(yangle)) > 0:
        for lbl in ax.get_yticklabels():
            lbl.set_rotation(float(yangle))

    ax.set_title("Estimated ATT (FEct)")
    ax.set_xlabel(str(xlab) if xlab is not None else "Time Since the Treatment's Onset")
    ax.set_ylabel("Effect of D on Y")
    fig.tight_layout()
    return ax


def plot_result(
    out,
    type: str = "gap",
    show_count: bool = True,
    proportion: float = 0.3,
    main: Optional[str] = None,
    ylab: Optional[str] = None,
    xlab: Optional[str] = None,
    cex_main: Optional[float] = None,
    cex_lab: Optional[float] = None,
    cex_axis: Optional[float] = None,
    cex_text: Optional[float] = None,
    xlim: Optional[Tuple[float, float]] = None,
    ylim: Optional[Tuple[float, float]] = None,
    # new options (subset of R's)
    start0: bool = False,
    plot_ci: Optional[str] = None,
    xbreaks: Optional[Sequence[float]] = None,
    ybreaks: Optional[Sequence[float]] = None,
    xangle: float = 0.0,
    yangle: float = 0.0,
    color: str = "#000000",
    est_lwidth: float = 1.6,
    theme_bw: bool = True,
    count_height: float = 0.1,
):
    import numpy as _np
    import pandas as pd

    if type not in ("gap", "calendar", "status", "counterfactual", "box", "equiv"):
        raise NotImplementedError("Supported types: 'gap', 'calendar', 'status', 'counterfactual', 'box', 'equiv'.")

    if type == "calendar":
        df = out.eff_calendar.copy()
        df = df.rename(columns={"ATT-calendar": "ATT", "count": "count"})
        df = df.reset_index().rename(columns={"index": "Period"})
        ax = _esplot_like(
            df,
            Period="Period",
            Estimate="ATT",
            SE=("SE" if "SE" in df.columns else None),
            CI_lower=("CI.lower" if "CI.lower" in df.columns else "CI.lower"),
            CI_upper=("CI.upper" if "CI.upper" in df.columns else "CI.upper"),
            Count=("count" if show_count else None),
            xlim=xlim,
            ylim=ylim,
            show_points=True,
            start0=start0,
            plot_ci=plot_ci,
            xbreaks=xbreaks,
            ybreaks=ybreaks,
            xangle=xangle,
            yangle=yangle,
            color=color,
            est_lwidth=est_lwidth,
            xlab=xlab,
            theme_bw=theme_bw,
            count_height=count_height,
        )
        try:
            if main is None:
                main_to_use = "Estimated ATT (FEct)"
            else:
                main_to_use = main
            ax.set_title(main_to_use, fontsize=int(16 * (cex_main if cex_main else 1.0)))
            if ylab is not None:
                ax.set_ylabel(ylab, fontsize=int(15 * (cex_lab if cex_lab else 1.0)))
            if cex_axis is not None:
                ax.tick_params(axis='both', labelsize=int(15 * cex_axis))
            if cex_text is not None:
                for t in ax.texts:
                    try:
                        t.set_fontsize(int(5 * cex_text))
                    except Exception:
                        pass
        except Exception:
            pass
        return ax

    if type == "status":
        return _status_plot(out)

    if type == "counterfactual":
        # observed vs counterfactual means among treated units by calendar time
        Y = out.Y_orig if out.Y_orig is not None else out.Y_dat
        D = out.D_orig if out.D_orig is not None else out.D_dat
        I = out.I_orig if out.I_orig is not None else out.I_dat
        if out.Y0_dat is not None:
            Y0 = out.Y0_dat
        else:
            from .fect import _predict_counterfactual as __predict_cf
            T, N = Y.shape
            Y0, _, _, _ = __predict_cf(Y, D, I, np.zeros((T, N, 0)), "two-way")
        treated_any = (D > 0).any(axis=0)
        treat_idx = np.where(treated_any)[0]
        if treat_idx.size == 0:
            raise ValueError("No treated units to plot.")
        Ym = np.nanmean(np.where(I[:, treat_idx] > 0, Y[:, treat_idx], np.nan), axis=1)
        Y0m = np.nanmean(np.where(I[:, treat_idx] > 0, Y0[:, treat_idx], np.nan), axis=1)
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.plot(Ym, label="Observed (treated avg)", color=color, linewidth=est_lwidth)
        ax.plot(Y0m, label="Counterfactual (treated avg)", color="#888888", linestyle="--", linewidth=est_lwidth)
        ax.axhline(0, color="#AAAAAA70", linewidth=1.2)
        ax.set_title("Observed vs Counterfactual (treated units)")
        ax.set_xlabel(str(xlab) if xlab is not None else "Calendar Time Index")
        if ylab is not None:
            ax.set_ylabel(str(ylab))
        ax.legend(loc="best")
        fig.tight_layout()
        return ax

    if type == "box":
        # distribution of individual effects by event time
        Y = out.Y_orig if out.Y_orig is not None else out.Y_dat
        D = out.D_orig if out.D_orig is not None else out.D_dat
        I = out.I_orig if out.I_orig is not None else out.I_dat
        T, N = Y.shape
        if out.Y0_dat is not None:
            Y0 = out.Y0_dat
        else:
            from .fect import _predict_counterfactual as __predict_cf
            Y0, _, _, _ = __predict_cf(Y, D, I, np.zeros((T, N, 0)), "two-way")
        # effect only defined where unit-time observed and treated (post-onset)
        eff = np.where((D > 0) & (I > 0), Y - Y0, np.nan)
        D_cum = np.cumsum(D, axis=0)
        rel = np.zeros_like(D_cum)
        for j in range(N):
            t0 = int(np.sum(D_cum[:, j] == 0))
            rel[:, j] = np.arange(1, T + 1) - t0
        rel = rel.astype(float)
        rel[I == 0] = np.nan
        # collect values by event time where any data exist
        unique_r = np.unique(rel[np.isfinite(rel)])
        unique_r = unique_r.astype(int)
        unique_r.sort()
        # counts per r and values
        counts = []
        values_by_r = []
        for r in unique_r.tolist():
            mask = (rel == r)
            vals = eff[mask]
            vals = vals[np.isfinite(vals)]
            values_by_r.append(vals)
            counts.append(int(vals.size))
        # optional filtering by proportion of max count (similar to gap)
        if len(counts) > 0:
            maxc = int(np.nanmax(np.asarray(counts)))
        else:
            maxc = 0
        if maxc > 0 and proportion is not None and 0.0 <= float(proportion) <= 1.0:
            thresh = float(proportion) * float(maxc)
            keep_idx = [i for i, c in enumerate(counts) if c >= thresh]
            if len(keep_idx) == 0 and len(counts) > 0:
                keep_idx = [int(np.argmax(np.asarray(counts)))]
            unique_r = unique_r[keep_idx]
            values_by_r = [values_by_r[i] for i in keep_idx]
            counts = [counts[i] for i in keep_idx]

        # plotting
        if theme_bw:
            try:
                plt.style.use("default")
            except Exception:
                pass
        fig, ax = plt.subplots(figsize=(max(7, len(unique_r) * 0.5), 4))
        if len(unique_r) == 0:
            ax.set_title("Effect Heterogeneity by Event Time (box plot)")
            fig.tight_layout()
            return ax
        ax.boxplot(values_by_r, positions=np.arange(len(unique_r)), widths=0.6, showfliers=False)
        ax.axhline(0, color="#AAAAAA70", linewidth=1.2)
        # counts histogram at bottom (scaled)
        if len(counts) > 0 and np.nanmax(np.asarray(counts)) > 0 and count_height is not None:
            ymin, ymax = ax.get_ylim()
            height = (ymax - ymin) * float(max(0.0, min(1.0, count_height)))
            maxc = float(np.nanmax(np.asarray(counts)))
            for xpos, ci in zip(np.arange(len(unique_r)), counts):
                if ci and ci > 0:
                    ax.add_patch(plt.Rectangle((float(xpos) - 0.3, ymin), 0.6, height * (float(ci) / maxc), color="gray", alpha=0.35, linewidth=0.2))
        # x tick labels are event times
        ax.set_xticks(np.arange(len(unique_r)))
        ax.set_xticklabels([int(r) for r in unique_r])
        ax.set_xlabel(str(xlab) if xlab is not None else "Event Time")
        if ylab is not None:
            ax.set_ylabel(str(ylab))
        ax.set_title("Effect Heterogeneity by Event Time (box plot)")
        fig.tight_layout()
        return ax

    # gap
    Y = out.Y_orig if out.Y_orig is not None else out.Y_dat
    D = out.D_orig if out.D_orig is not None else out.D_dat
    I = out.I_orig if out.I_orig is not None else out.I_dat
    T, N = Y.shape

    if out.Y0_dat is not None and (out.Y_orig is None):
        Y0 = out.Y0_dat
    else:
        # local import to avoid circular
        from .fect import _predict_counterfactual as __predict_cf
        Y0, _, _, _ = __predict_cf(Y, D, I, _np.zeros((T, N, 0)), "two-way")

    first_on = _np.full(N, _np.nan)
    for j in range(N):
        dcol = (D[:, j] > 0).astype(int)
        idx = _np.where(dcol == 1)[0]
        if idx.size > 0:
            first_on[j] = idx[0]
    treated = _np.where(_np.isfinite(first_on))[0]
    if treated.size == 0:
        raise ValueError("No treated units to plot.")

    T0_counts = _np.full(N, _np.nan)
    for j in treated:
        T0_counts[j] = int(first_on[j]) - 1
    T0_valid = T0_counts[_np.isfinite(T0_counts)].astype(int)
    if T0_valid.size == 0:
        raise ValueError("No valid treated onsets for plotting.")
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

    df = pd.DataFrame({
        "Period": timeline,
        "ATT": att,
        "CI.lower": _np.nan,
        "CI.upper": _np.nan,
        "count": counts,
    })

    # Merge SE/CI if available
    if getattr(out, "est_att_df", None) is not None:
        try:
            df = df.merge(out.est_att_df[["Period", "SE", "CI.lower", "CI.upper"]], on="Period", how="left")
        except Exception:
            pass

    # proportion filter unless xlim provided
    if xlim is None and proportion is not None:
        try:
            maxc = int(_np.nanmax(df["count"].to_numpy())) if len(df) > 0 else 0
        except Exception:
            maxc = 0
        if maxc > 0 and 0.0 <= float(proportion) <= 1.0:
            thresh = float(proportion) * float(maxc)
            filtered = df[df["count"] >= thresh]
            if filtered.empty:
                idx_max = int(_np.nanargmax(df["count"].to_numpy())) if len(df) > 0 else None
                if idx_max is not None:
                    df = df.iloc[[idx_max]].copy()
            else:
                df = filtered.sort_values("Period").copy()

    # xlim: honor if provided; else pad ±0.2
    xlim_to_use: Optional[Tuple[float, float]] = None
    try:
        if xlim is not None and len(xlim) == 2:
            xlim_to_use = (float(xlim[0]), float(xlim[1]))
        else:
            if len(df) > 0:
                xmin = float(_np.nanmin(df["Period"].to_numpy()))
                xmax = float(_np.nanmax(df["Period"].to_numpy()))
                xlim_to_use = (xmin - 0.2, xmax + 0.2)
    except Exception:
        xlim_to_use = None

    ax = _esplot_like(
        df,
        Period="Period",
        Estimate="ATT",
        SE=("SE" if "SE" in df.columns else None),
        CI_lower="CI.lower",
        CI_upper="CI.upper",
        Count=("count" if show_count else None),
        xlim=xlim_to_use,
        ylim=ylim,
        show_points=True,
        start0=start0,
        plot_ci=plot_ci,
        xbreaks=xbreaks,
        ybreaks=ybreaks,
        xangle=xangle,
        yangle=yangle,
        color=color,
        est_lwidth=est_lwidth,
        xlab=xlab,
        theme_bw=theme_bw,
        count_height=count_height,
    )

    try:
        if main is None:
            main_to_use = "Estimated ATT (FEct)"
        else:
            main_to_use = main
        ax.set_title(main_to_use, fontsize=int(16 * (cex_main if cex_main else 1.0)))
        if ylab is not None:
            ax.set_ylabel(ylab, fontsize=int(15 * (cex_lab if cex_lab else 1.0)))
        if cex_axis is not None:
            ax.tick_params(axis='both', labelsize=int(15 * cex_axis))
        if cex_text is not None:
            for t in ax.texts:
                try:
                    t.set_fontsize(int(5 * cex_text))
                except Exception:
                    pass
    except Exception:
        pass
    return ax


def _status_plot(out, *, colors: Optional[Dict[str, Any]] = None, main: Optional[str] = None):
    import numpy as _np
    Y = out.Y_orig if out.Y_orig is not None else out.Y_dat
    D = out.D_orig if out.D_orig is not None else out.D_dat
    I = out.I_orig if out.I_orig is not None else out.I_dat
    T, N = Y.shape
    status = _np.zeros((T, N))
    status[I <= 0] = -1  # missing
    status[(I > 0) & (D <= 0)] = 0  # control
    status[(I > 0) & (D > 0)] = 1   # treated
    cmap = {
        -1: (0.8, 0.95, 0.9),
        0: (0.2, 0.45, 0.7),
        1: (0.83, 0.37, 0.0),
    }
    if colors:
        import matplotlib.colors as mcolors
        cmap = {
            -1: mcolors.to_rgb(colors.get("missing", "#009E73")),
            0: mcolors.to_rgb(colors.get("control", "#0072B2")),
            1: mcolors.to_rgb(colors.get("treated", "#D55E00")),
        }
    arr = _np.zeros((T, N, 3))
    for key, col in cmap.items():
        mask = (status == key)
        for c in range(3):
            arr[:, :, c][mask] = col[c]
    fig, ax = plt.subplots(figsize=(max(6, N * 0.12), max(4, T * 0.12)))
    ax.imshow(arr, aspect="auto", origin="lower")
    ax.set_xlabel("Unit")
    ax.set_ylabel("Time")
    if main:
        ax.set_title(str(main))
    fig.tight_layout()
    return ax


