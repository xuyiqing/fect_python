#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <limits>

namespace py = pybind11;

static inline size_t idx(size_t t, size_t n, size_t N) { return t * N + n; }

// Compute mu, alpha (N), xi (T) under mask II (1=use, 0=ignore)
// force: 0 none, 1 unit, 2 time, 3 two-way
static void recover_additive_fe(const double* Y, const double* II,
                                size_t T, size_t N, int force,
                                double& mu,
                                std::vector<double>& alpha,
                                std::vector<double>& xi) {
  alpha.assign(N, 0.0);
  xi.assign(T, 0.0);

  // weighted grand mean
  double wsum = 0.0, ysum = 0.0;
  for (size_t t = 0; t < T; ++t) {
    for (size_t n = 0; n < N; ++n) {
      const double w = II[idx(t, n, N)];
      if (w > 0.0) { wsum += 1.0; ysum += Y[idx(t, n, N)]; }
    }
  }
  mu = (wsum > 0.0) ? (ysum / wsum) : 0.0;

  if (force == 0) return;

  if (force == 1) {
    // unit FE only
    for (size_t n = 0; n < N; ++n) {
      double sw = 0.0, sy = 0.0;
      for (size_t t = 0; t < T; ++t) {
        const double w = II[idx(t, n, N)];
        if (w > 0.0) { sw += 1.0; sy += (Y[idx(t, n, N)] - mu); }
      }
      alpha[n] = (sw > 0.0) ? (sy / sw) : 0.0;
    }
    return;
  }

  if (force == 2) {
    // time FE only
    for (size_t t = 0; t < T; ++t) {
      double sw = 0.0, sy = 0.0;
      for (size_t n = 0; n < N; ++n) {
        const double w = II[idx(t, n, N)];
        if (w > 0.0) { sw += 1.0; sy += (Y[idx(t, n, N)] - mu); }
      }
      xi[t] = (sw > 0.0) ? (sy / sw) : 0.0;
    }
    return;
  }

  // two-way via alternating projections
  const int max_iter = 100;
  const double tol = 1e-12;
  for (int it = 0; it < max_iter; ++it) {
    std::vector<double> alpha_prev = alpha;
    std::vector<double> xi_prev = xi;

    // update alpha
    for (size_t n = 0; n < N; ++n) {
      double sw = 0.0, sy = 0.0;
      for (size_t t = 0; t < T; ++t) {
        const double w = II[idx(t, n, N)];
        if (w > 0.0) { sw += 1.0; sy += (Y[idx(t, n, N)] - mu - xi[t]); }
      }
      alpha[n] = (sw > 0.0) ? (sy / sw) : 0.0;
    }
    // center alpha
    double am = 0.0; for (double v : alpha) am += v; am /= (N > 0 ? double(N) : 1.0);
    for (double &v : alpha) v -= am;

    // update xi
    for (size_t t = 0; t < T; ++t) {
      double sw = 0.0, sy = 0.0;
      for (size_t n = 0; n < N; ++n) {
        const double w = II[idx(t, n, N)];
        if (w > 0.0) { sw += 1.0; sy += (Y[idx(t, n, N)] - mu - alpha[n]); }
      }
      xi[t] = (sw > 0.0) ? (sy / sw) : 0.0;
    }
    // center xi
    double xm = 0.0; for (double v : xi) xm += v; xm /= (T > 0 ? double(T) : 1.0);
    for (double &v : xi) v -= xm;

    // update mu
    double sw2 = 0.0, sy2 = 0.0;
    for (size_t t = 0; t < T; ++t) {
      for (size_t n = 0; n < N; ++n) {
        const double w = II[idx(t, n, N)];
        if (w > 0.0) { sw2 += 1.0; sy2 += (Y[idx(t, n, N)] - alpha[n] - xi[t]); }
      }
    }
    mu = (sw2 > 0.0) ? (sy2 / sw2) : 0.0;

    // convergence check
    double da = 0.0, dx = 0.0;
    for (size_t n = 0; n < N; ++n) da = std::max(da, std::abs(alpha[n] - alpha_prev[n]));
    for (size_t t = 0; t < T; ++t) dx = std::max(dx, std::abs(xi[t] - xi_prev[t]));
    if (da < tol && dx < tol) break;
  }
}

// Build within-transformed matrix (row-major) in out[T*N] using same demeaning as above
static void within_tilde(const double* M, const double* II,
                         size_t T, size_t N, int force,
                         std::vector<double>& out) {
  double mu; std::vector<double> a, x;
  recover_additive_fe(M, II, T, N, force, mu, a, x);
  out.assign(T * N, 0.0);
  for (size_t t = 0; t < T; ++t) {
    for (size_t n = 0; n < N; ++n) {
      out[idx(t, n, N)] = M[idx(t, n, N)] - (mu + a[n] + x[t]);
    }
  }
}

// Solve symmetric positive-definite system using simple Cholesky (no pivot)
static bool solve_spd(std::vector<double>& A, std::vector<double>& b, size_t p) {
  // A is p x p row-major; perform in-place Cholesky
  for (size_t i = 0; i < p; ++i) {
    for (size_t j = 0; j <= i; ++j) {
      double sum = A[i * p + j];
      for (size_t k = 0; k < j; ++k) sum -= A[i * p + k] * A[j * p + k];
      if (i == j) {
        if (sum <= 0.0) return false;
        A[i * p + j] = std::sqrt(sum);
      } else {
        A[i * p + j] = sum / A[j * p + j];
      }
    }
    for (size_t j = i + 1; j < p; ++j) A[i * p + j] = 0.0; // zero upper
  }
  // Solve L y = b
  for (size_t i = 0; i < p; ++i) {
    double sum = b[i];
    for (size_t k = 0; k < i; ++k) sum -= A[i * p + k] * b[k];
    b[i] = sum / A[i * p + i];
  }
  // Solve L^T x = y
  for (ptrdiff_t i = (ptrdiff_t)p - 1; i >= 0; --i) {
    double sum = b[(size_t)i];
    for (size_t k = (size_t)i + 1; k < p; ++k) sum -= A[k * p + (size_t)i] * b[k];
    b[(size_t)i] = sum / A[(size_t)i * p + (size_t)i];
  }
  return true;
}

// Main entry: fast FE counterfactual predictor
// Returns (Y0: T x N array, II: T x N mask, beta: p array or empty, beta_se: None)
py::tuple fe_predict_cf(py::array_t<double, py::array::c_style | py::array::forcecast> Y,
                        py::array_t<double, py::array::c_style | py::array::forcecast> D,
                        py::array_t<double, py::array::c_style | py::array::forcecast> I,
                        py::object X_obj,
                        const std::string& force_str) {
  // Shape checks
  if (Y.ndim() != 2 || D.ndim() != 2 || I.ndim() != 2)
    throw std::invalid_argument("Y, D, I must be 2D arrays");
  const size_t T = (size_t)Y.shape(0);
  const size_t N = (size_t)Y.shape(1);
  if ((size_t)D.shape(0) != T || (size_t)D.shape(1) != N || (size_t)I.shape(0) != T || (size_t)I.shape(1) != N)
    throw std::invalid_argument("Shapes of D/I must match Y");

  int force = 3;
  if (force_str == "none") force = 0;
  else if (force_str == "unit") force = 1;
  else if (force_str == "time") force = 2;
  else force = 3;

  // Build II = I with treated set to 0
  std::vector<double> II(T * N, 0.0);
  const double* Dp = D.data();
  const double* Ip = I.data();
  for (size_t t = 0; t < T; ++t) {
    for (size_t n = 0; n < N; ++n) {
      double mask = (Dp[idx(t, n, N)] > 0.0) ? 0.0 : Ip[idx(t, n, N)];
      II[idx(t, n, N)] = (mask > 0.0) ? 1.0 : 0.0;
    }
  }

  // Prepare outputs
  py::array_t<double> Y0({(py::ssize_t)T, (py::ssize_t)N});
  double* Y0p = Y0.mutable_data();
  std::fill(Y0p, Y0p + T * N, 0.0);

  const double* Yp = Y.data();

  size_t p = 0;
  bool hasX = false;
  py::array_t<double> X;
  if (!X_obj.is_none()) {
    X = X_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>();
    if (X.ndim() != 3 || (size_t)X.shape(0) != T || (size_t)X.shape(1) != N)
      throw std::invalid_argument("X must be a 3D array with shape (T, N, p)");
    p = (size_t)X.shape(2);
    hasX = (p > 0);
  }

  // No covariates: just FE parts
  if (!hasX) {
    double mu; std::vector<double> a, x;
    recover_additive_fe(Yp, II.data(), T, N, force, mu, a, x);
    for (size_t t = 0; t < T; ++t) {
      for (size_t n = 0; n < N; ++n) {
        Y0p[idx(t, n, N)] = mu + a[n] + x[t];
      }
    }
    // Return beta empty
    py::array_t<double> beta({(py::ssize_t)0});
    py::object beta_se = py::none();
    // Also return II mask as array
    py::array_t<double> II_arr({(py::ssize_t)T, (py::ssize_t)N});
    std::copy(II.begin(), II.end(), II_arr.mutable_data());
    return py::make_tuple(Y0, II_arr, beta, beta_se);
  }

  // With covariates: iterate as in Python implementation
  const double* Xp = X.data();
  std::vector<double> beta(p, 0.0);
  const int max_iter = 15;
  const double tol = 1e-8;

  for (int it = 0; it < max_iter; ++it) {
    // covar_fit = sum_k X_k * beta_k
    std::vector<double> covar_fit(T * N, 0.0);
    for (size_t k = 0; k < p; ++k) {
      const size_t offset = k; // in third dim
      for (size_t t = 0; t < T; ++t) {
        for (size_t n = 0; n < N; ++n) {
          const size_t linear = ((t * N + n) * p) + offset;
          covar_fit[idx(t, n, N)] += Xp[linear] * beta[k];
        }
      }
    }

    // resid = Y - covar_fit
    std::vector<double> resid(T * N, 0.0);
    for (size_t i = 0; i < T * N; ++i) resid[i] = Yp[i] - covar_fit[i];

    // Update FE by demeaning residuals on controls
    double mu; std::vector<double> a, x;
    recover_additive_fe(resid.data(), II.data(), T, N, force, mu, a, x);

    // y_tilde = resid - (mu + a + x)
    std::vector<double> y_tilde(T * N, 0.0);
    for (size_t t = 0; t < T; ++t) {
      for (size_t n = 0; n < N; ++n) {
        y_tilde[idx(t, n, N)] = resid[idx(t, n, N)] - (mu + a[n] + x[t]);
      }
    }

    // Build normal equations
    std::vector<double> XtX(p * p, 0.0);
    std::vector<double> Xty(p, 0.0);
    // Precompute demeaned X columns
    std::vector<std::vector<double>> Xtil_cols(p, std::vector<double>(T * N));
    for (size_t k = 0; k < p; ++k) {
      // Extract X_k
      std::vector<double> Xk(T * N, 0.0);
      for (size_t t = 0; t < T; ++t) {
        for (size_t n = 0; n < N; ++n) {
          const size_t linear = ((t * N + n) * p) + k;
          Xk[idx(t, n, N)] = Xp[linear];
        }
      }
      within_tilde(Xk.data(), II.data(), T, N, force, Xtil_cols[k]);
    }

    for (size_t k = 0; k < p; ++k) {
      // Masked dot for X'y
      double s = 0.0;
      for (size_t t = 0; t < T; ++t) {
        for (size_t n = 0; n < N; ++n) {
          if (II[idx(t, n, N)] > 0.0) s += Xtil_cols[k][idx(t, n, N)] * y_tilde[idx(t, n, N)];
        }
      }
      Xty[k] = s;
      for (size_t j = k; j < p; ++j) {
        double v = 0.0;
        for (size_t t = 0; t < T; ++t) {
          for (size_t n = 0; n < N; ++n) {
            if (II[idx(t, n, N)] > 0.0) v += Xtil_cols[k][idx(t, n, N)] * Xtil_cols[j][idx(t, n, N)];
          }
        }
        XtX[k * p + j] = v;
        XtX[j * p + k] = v;
      }
    }
    // tiny ridge for stability
    for (size_t d = 0; d < p; ++d) XtX[d * p + d] += 1e-12;

    // Solve
    std::vector<double> new_beta = Xty;
    bool ok = solve_spd(XtX, new_beta, p);
    if (!ok) break;

    // Convergence
    double maxdiff = 0.0;
    for (size_t k = 0; k < p; ++k) maxdiff = std::max(maxdiff, std::abs(new_beta[k] - beta[k]));
    beta.swap(new_beta);
    if (maxdiff < tol) break;
  }

  // Final Y0 = mu + a + x + covar_fit with final beta
  std::vector<double> covar_fit(T * N, 0.0);
  for (size_t k = 0; k < p; ++k) {
    for (size_t t = 0; t < T; ++t) {
      for (size_t n = 0; n < N; ++n) {
        const size_t linear = ((t * N + n) * p) + k;
        covar_fit[idx(t, n, N)] += Xp[linear] * beta[k];
      }
    }
  }
  double mu; std::vector<double> a, x;
  std::vector<double> resid(T * N, 0.0);
  for (size_t i = 0; i < T * N; ++i) resid[i] = Yp[i] - covar_fit[i];
  recover_additive_fe(resid.data(), II.data(), T, N, force, mu, a, x);
  for (size_t t = 0; t < T; ++t) {
    for (size_t n = 0; n < N; ++n) {
      Y0p[idx(t, n, N)] = mu + a[n] + x[t] + covar_fit[idx(t, n, N)];
    }
  }

  // Build outputs
  py::array_t<double> II_arr({(py::ssize_t)T, (py::ssize_t)N});
  std::copy(II.begin(), II.end(), II_arr.mutable_data());
  py::array_t<double> beta_arr({(py::ssize_t)p});
  std::copy(beta.begin(), beta.end(), beta_arr.mutable_data());
  py::object beta_se = py::none();
  return py::make_tuple(Y0, II_arr, beta_arr, beta_se);
}

PYBIND11_MODULE(_fe, m) {
  m.doc() = "Fast FE counterfactuals (two-way FE with covariates)";
  m.def("fe_predict_cf", &fe_predict_cf, py::arg("Y"), py::arg("D"), py::arg("I"), py::arg("X") = py::none(), py::arg("force") = std::string("two-way"));
  // Event-study like aggregator: returns (timeline:int[], att:double[], counts:int[])
  m.def("event_study", [](py::array_t<double, py::array::c_style | py::array::forcecast> Y,
                           py::array_t<double, py::array::c_style | py::array::forcecast> D,
                           py::array_t<double, py::array::c_style | py::array::forcecast> I,
                           py::array_t<double, py::array::c_style | py::array::forcecast> Y0) {
    if (Y.ndim() != 2 || D.ndim() != 2 || I.ndim() != 2 || Y0.ndim() != 2) throw std::invalid_argument("All inputs must be 2D arrays");
    size_t T = (size_t)Y.shape(0), N = (size_t)Y.shape(1);
    if ((size_t)D.shape(0) != T || (size_t)D.shape(1) != N || (size_t)I.shape(0) != T || (size_t)I.shape(1) != N || (size_t)Y0.shape(0) != T || (size_t)Y0.shape(1) != N)
      throw std::invalid_argument("Shapes must match");
    const double* Yp = Y.data();
    const double* Dp = D.data();
    const double* Ip = I.data();
    const double* Y0p = Y0.data();

    // first treated index per unit
    std::vector<int> first_on(N, -1);
    for (size_t n = 0; n < N; ++n) {
      for (size_t t = 0; t < T; ++t) {
        if (Dp[idx(t,n,N)] > 0.0) { first_on[n] = (int)t; break; }
      }
    }
    int have = 0;
    for (int v : first_on) if (v >= 0) { have = 1; break; }
    if (!have) {
      return py::make_tuple(py::array_t<int>({0}), py::array_t<double>({0}), py::array_t<int>({0}));
    }
    // rmin, rmax based on T0 counts
    int maxT0 = 0, minT0 = (int)T - 1;
    for (size_t n = 0; n < N; ++n) if (first_on[n] >= 0) {
      int T0c = first_on[n] - 1;
      if (T0c > maxT0) maxT0 = T0c;
      if (T0c < minT0) minT0 = T0c;
    }
    int rmin = 1 - maxT0;
    int rmax = (int)T - minT0;
    int R = rmax - rmin + 1;
    py::array_t<int> timeline({R});
    auto tl = timeline.mutable_unchecked<1>();
    for (int i = 0; i < R; ++i) tl(i) = rmin + i;

    std::vector<double> Y_tr(R * N, std::numeric_limits<double>::quiet_NaN());
    std::vector<double> Y_ct(R * N, std::numeric_limits<double>::quiet_NaN());
    auto ridx_of = [&](int r){ return (r - rmin); };
    for (size_t n = 0; n < N; ++n) {
      if (first_on[n] < 0) continue;
      int T0c = first_on[n] - 1;
      for (size_t t = 0; t < T; ++t) {
        if (Ip[idx(t,n,N)] != 1.0) continue;
        int r = (int)t - T0c;
        int ri = ridx_of(r);
        if (ri < 0 || ri >= R) continue;
        bool include = (r <= 0) || ((r > 0) && (Dp[idx(t,n,N)] == 1.0));
        if (include) {
          Y_tr[ri * N + (int)n] = Yp[idx(t,n,N)];
          Y_ct[ri * N + (int)n] = Y0p[idx(t,n,N)];
        }
      }
    }
    py::array_t<double> att({R});
    py::array_t<int> counts({R});
    auto attv = att.mutable_unchecked<1>();
    auto cntv = counts.mutable_unchecked<1>();
    for (int r = 0; r < R; ++r) {
      double sum_tr = 0.0, sum_ct = 0.0; int c_tr = 0, c_ct = 0, c = 0;
      for (size_t n = 0; n < N; ++n) {
        double ytr = Y_tr[r * N + (int)n];
        double yct = Y_ct[r * N + (int)n];
        if (!std::isnan(ytr)) { sum_tr += ytr; c_tr++; }
        if (!std::isnan(yct)) { sum_ct += yct; c_ct++; }
        if (!std::isnan(ytr)) c++;
      }
      double tr_bar = (c_tr > 0) ? (sum_tr / (double)c_tr) : std::numeric_limits<double>::quiet_NaN();
      double ct_bar = (c_ct > 0) ? (sum_ct / (double)c_ct) : std::numeric_limits<double>::quiet_NaN();
      attv(r) = (std::isnan(tr_bar) || std::isnan(ct_bar)) ? std::numeric_limits<double>::quiet_NaN() : (tr_bar - ct_bar);
      cntv(r) = c;
    }
    return py::make_tuple(timeline, att, counts);
  });
  // ATT-by-time and per-cell effects: returns (eff: T x N, att_t: T, counts_t: T)
  m.def("att_from_diff", [](py::array_t<double, py::array::c_style | py::array::forcecast> Y,
                             py::array_t<double, py::array::c_style | py::array::forcecast> Y0,
                             py::array_t<double, py::array::c_style | py::array::forcecast> D,
                             py::array_t<double, py::array::c_style | py::array::forcecast> I) {
    if (Y.ndim() != 2 || D.ndim() != 2 || I.ndim() != 2 || Y0.ndim() != 2) throw std::invalid_argument("All inputs must be 2D arrays");
    size_t T = (size_t)Y.shape(0), N = (size_t)Y.shape(1);
    if ((size_t)D.shape(0) != T || (size_t)D.shape(1) != N || (size_t)I.shape(0) != T || (size_t)I.shape(1) != N || (size_t)Y0.shape(0) != T || (size_t)Y0.shape(1) != N)
      throw std::invalid_argument("Shapes must match");
    const double* Yp = Y.data();
    const double* Y0p = Y0.data();
    const double* Dp = D.data();
    const double* Ip = I.data();
    py::array_t<double> eff({(py::ssize_t)T, (py::ssize_t)N});
    auto ef = eff.mutable_unchecked<2>();
    double nan = std::numeric_limits<double>::quiet_NaN();
    for (size_t t = 0; t < T; ++t) {
      for (size_t n = 0; n < N; ++n) {
        if (Dp[idx(t,n,N)] > 0.0 && Ip[idx(t,n,N)] > 0.0) {
          ef(t,n) = Yp[idx(t,n,N)] - Y0p[idx(t,n,N)];
        } else {
          ef(t,n) = nan;
        }
      }
    }
    py::array_t<double> att_t({(py::ssize_t)T});
    py::array_t<int> counts_t({(py::ssize_t)T});
    auto av = att_t.mutable_unchecked<1>();
    auto cv = counts_t.mutable_unchecked<1>();
    for (size_t t = 0; t < T; ++t) {
      double sum = 0.0; int c = 0;
      for (size_t n = 0; n < N; ++n) {
        double v = ef(t,n);
        if (!std::isnan(v)) { sum += v; c++; }
      }
      av((py::ssize_t)t) = (c > 0) ? (sum / (double)c) : nan;
      cv((py::ssize_t)t) = c;
    }
    return py::make_tuple(eff, att_t, counts_t);
  });
}


