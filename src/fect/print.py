from __future__ import annotations

from typing import List, Tuple

import numpy as np


def format_summary(out) -> str:
    lines: List[str] = []

    call = out.call or {}
    Xdisp = (" + " + " + ".join(out.X)) if (out.X is not None and len(out.X) > 0) else ""
    formula = f"{out.Y} ~ {out.D}{Xdisp}"
    idx = f"index = ({out.index[0]!r}, {out.index[1]!r})"
    method = f"method = {call.get('method', 'fe')!r}"
    force = "force = 'two-way'"
    lines.append("Call:")
    lines.append(f"  fect(formula = {formula}, {idx}, {force}, {method})")
    lines.append("")

    eff = out.eff
    v = eff[~np.isnan(eff)]
    att_obs = float(np.nanmean(v)) if v.size else float('nan')
    unit_means: List[float] = []
    for j in range(eff.shape[1]):
        col = eff[:, j]
        if np.all(np.isnan(col)):
            continue
        m = np.nanmean(col)
        if np.isfinite(m):
            unit_means.append(float(m))
    att_unit = float(np.nanmean(unit_means)) if unit_means else float('nan')

    show_se: bool = bool(call.get('se', False))

    try:
        from scipy.stats import norm as _scipy_norm
        Z97 = float(_scipy_norm.ppf(0.975))
    except Exception:
        Z97 = 1.959963984540054

    def ci_and_p(att: float, se: float) -> Tuple[float, float, float]:
        if not (np.isfinite(att) and np.isfinite(se) and se > 0):
            return (float('nan'), float('nan'), float('nan'))
        z = att / se
        from math import erf, sqrt
        p = 2.0 * (1.0 - 0.5 * (1.0 + erf(abs(z) / sqrt(2.0))))
        lo = att - Z97 * se
        hi = att + Z97 * se
        return (lo, hi, p)

    se_obs_use = se_unit_use = None
    if show_se:
        se_obs_final = (out.est_summary.get("se_obs") if out.est_summary else None)
        se_unit_final = (out.est_summary.get("se_unit") if out.est_summary else None)
        if np.isfinite(se_obs_final) if se_obs_final is not None else False:
            se_obs_use = float(se_obs_final)
        if np.isfinite(se_unit_final) if se_unit_final is not None else False:
            se_unit_use = float(se_unit_final)

    lo_obs = hi_obs = p_obs = float('nan')
    lo_unit = hi_unit = p_unit = float('nan')
    if show_se:
        lo_obs, hi_obs, p_obs = ci_and_p(att_obs, se_obs_use if se_obs_use is not None else float('nan'))
        lo_unit, hi_unit, p_unit = ci_and_p(att_unit, se_unit_use if se_unit_use is not None else float('nan'))

    def fmt_num(x: float, digits: int = 3) -> str:
        try:
            xv = float(x)
        except Exception:
            return ""
        return (f"{xv:.{digits}f}" if np.isfinite(xv) else "")

    def fmt_p(p: float) -> str:
        if not np.isfinite(p):
            return ""
        return "0" if p < 1e-6 else f"{p:.3f}"

    # If effect() was called and effect summaries exist, show them first (R behavior)
    if getattr(out, "effect_est_avg", None) is not None:
        lines.append("Overall cumulative effect:")
        try:
            import pandas as _pd
            if isinstance(out.effect_est_avg, (list, tuple, np.ndarray)):
                arr = np.asarray(out.effect_est_avg, dtype=float)
                # Print as a simple vector-like row
                lines.append("  " + " ".join([f"{v:.4f}" if np.isfinite(v) else "" for v in arr]))
            else:
                lines.append(f"  {out.effect_est_avg}")
        except Exception:
            lines.append(f"  {out.effect_est_avg}")
        if getattr(out, "effect_est_att", None) is not None:
            lines.append("")
            lines.append("Period-by-period cumulative effect:")
            try:
                df = out.effect_est_att
                if hasattr(df, "to_string"):
                    s = df.to_string(index=False)
                    for ln in s.splitlines():
                        lines.append("  " + ln)
                else:
                    lines.append(str(df))
            except Exception:
                lines.append(str(out.effect_est_att))
        return "\n".join(["#> " + ln if ln else "#>" for ln in lines])

    lines.append("ATT:")
    lines.append("")
    label_w = 28
    if show_se:
        header = (
            "  "
            + f"{'':<{label_w}}"
            + f" {'ATT':>7} {'S.E.':>7} {'CI.lower':>8} {'CI.upper':>8} {'p.value':>7}"
        )
    else:
        header = "  " + f"{'':<{label_w}} {'ATT':>7}"
    lines.append(header)
    if show_se:
        row1 = (
            "  "
            + f"{'Tr obs equally weighted':<{label_w}}"
            + f" {fmt_num(att_obs):>7} {fmt_num(se_obs_use,4):>7} {fmt_num(lo_obs):>8} {fmt_num(hi_obs):>8} {fmt_p(p_obs):>7}"
        )
    else:
        row1 = "  " + f"{'Tr obs equally weighted':<{label_w}} {fmt_num(att_obs):>7}"
    lines.append(row1)
    if show_se:
        row2 = (
            "  "
            + f"{'Tr units equally weighted':<{label_w}}"
            + f" {fmt_num(att_unit):>7} {fmt_num(se_unit_use,4):>7} {fmt_num(lo_unit):>8} {fmt_num(hi_unit):>8} {fmt_p(p_unit):>7}"
        )
    else:
        row2 = "  " + f"{'Tr units equally weighted':<{label_w}} {fmt_num(att_unit):>7}"
    lines.append(row2)
    lines.append("")

    if out.beta is not None and out.beta_names:
        lines.append("Covariates:")
        if show_se:
            lines.append("  " + f"{'':<{label_w}} {'Coef':>7} {'S.E.':>7} {'CI.lower':>8} {'CI.upper':>8} {'p.value':>7}")
        else:
            lines.append("  " + f"{'':<{label_w}} {'Coef':>7}")
        se_attr = out.est_att_df.attrs.get("beta_se_boot") if (out.est_att_df is not None and "beta_se_boot" in out.est_att_df.attrs) else None
        for k, name in enumerate(out.beta_names):
            b = float(out.beta[k])
            if show_se:
                se = float('nan')
                if se_attr is not None and len(se_attr) > k and np.isfinite(se_attr[k]):
                    se = float(se_attr[k])
                lo, hi, p = ci_and_p(b, se)
                lines.append("  " + f"{name:<{label_w}} {fmt_num(b):>7} {fmt_num(se,5):>7} {fmt_num(lo,4):>8} {fmt_num(hi,4):>8} {fmt_p(p):>7}")
            else:
                lines.append("  " + f"{name:<{label_w}} {fmt_num(b):>7}")

    return "\n".join(["#> " + ln if ln else "#>" for ln in lines])


