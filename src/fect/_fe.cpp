#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <stdexcept>
#include <set>
#include <limits>
#include <algorithm>

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
  const double tol = 1e-5;  // Match R's C++ tolerance

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

// Use numpy's SVD for more accurate results
py::tuple numpy_svd(py::array_t<double> A, bool full_matrices = false) {
  py::object np = py::module::import("numpy");
  py::object linalg = np.attr("linalg");
  py::object svd_func = linalg.attr("svd");
  
  py::tuple result = svd_func(A, py::arg("full_matrices")=full_matrices);
  return result;
}

// Panel factor model using proper SVD (matching R implementation exactly)
py::tuple panel_factor(py::array_t<double, py::array::c_style | py::array::forcecast> Y,
                       int r) {
  if (Y.ndim() != 2) throw std::invalid_argument("Y must be 2D");
  const size_t T = (size_t)Y.shape(0);
  const size_t N = (size_t)Y.shape(1);
  if (r <= 0 || (size_t)r > std::min(T, N)) {
    throw std::invalid_argument("r must be positive and <= min(T, N)");
  }

  const double* Yp = Y.data();
  
  py::array_t<double> factors({(py::ssize_t)T, (py::ssize_t)r});
  py::array_t<double> loadings({(py::ssize_t)N, (py::ssize_t)r});
  py::array_t<double> VNT({(py::ssize_t)r, (py::ssize_t)r});
  
  double* F_data = factors.mutable_data();
  double* L_data = loadings.mutable_data();
  double* V_data = VNT.mutable_data();
  
  // Initialize to zero
  std::fill(F_data, F_data + T * r, 0.0);
  std::fill(L_data, L_data + N * r, 0.0);
  std::fill(V_data, V_data + r * r, 0.0);
  
  // Copy Y data and replace NaN/Inf with 0
  std::vector<double> E(T * N);
  for (size_t i = 0; i < T * N; ++i) {
    E[i] = std::isfinite(Yp[i]) ? Yp[i] : 0.0;
  }
  
  // Follow R's panel_factor implementation exactly
  if (T < N) {
    // Case: T < N - use SVD of E * E.t()
    py::array_t<double> EE({(py::ssize_t)T, (py::ssize_t)T});
    double* EE_data = EE.mutable_data();
    
    // Compute EE = E * E.t() / (N * T)
    for (size_t i = 0; i < T; ++i) {
      for (size_t j = 0; j < T; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < N; ++k) {
          sum += E[i * N + k] * E[j * N + k];
        }
        EE_data[i * T + j] = sum / (N * T);
      }
    }
    
    // Use numpy's SVD
    auto svd_result = numpy_svd(EE, false);
    auto U_svd = svd_result[0].cast<py::array_t<double>>();
    auto S_svd = svd_result[1].cast<py::array_t<double>>();
    
    const double* U_data = U_svd.data();
    const double* S_data = S_svd.data();
    
    // Set factors and loadings following R's convention
    for (int f = 0; f < r; ++f) {
      // factor = U.head_cols(r) * sqrt(T)
      for (size_t t = 0; t < T; ++t) {
        F_data[t * r + f] = U_data[t * T + f] * std::sqrt(double(T));
      }
      
      // lambda = E.t() * factor / T
      for (size_t n = 0; n < N; ++n) {
        double sum = 0.0;
        for (size_t t = 0; t < T; ++t) {
          sum += E[t * N + n] * F_data[t * r + f];
        }
        L_data[n * r + f] = sum / T;
      }
      
      V_data[f * r + f] = S_data[f];
    }
  } else {
    // Case: T >= N - use SVD of E.t() * E
    py::array_t<double> EE({(py::ssize_t)N, (py::ssize_t)N});
    double* EE_data = EE.mutable_data();
    
    // Compute EE = E.t() * E / (N * T)
    for (size_t i = 0; i < N; ++i) {
      for (size_t j = 0; j < N; ++j) {
        double sum = 0.0;
        for (size_t k = 0; k < T; ++k) {
          sum += E[k * N + i] * E[k * N + j];
        }
        EE_data[i * N + j] = sum / (N * T);
      }
    }
    
    // Use numpy's SVD
    auto svd_result = numpy_svd(EE, false);
    auto U_svd = svd_result[0].cast<py::array_t<double>>();
    auto S_svd = svd_result[1].cast<py::array_t<double>>();
    
    const double* U_data = U_svd.data();
    const double* S_data = S_svd.data();
    
    // Set loadings and factors following R's convention
    for (int f = 0; f < r; ++f) {
      // lambda = U.head_cols(r) * sqrt(N)
      for (size_t n = 0; n < N; ++n) {
        L_data[n * r + f] = U_data[n * N + f] * std::sqrt(double(N));
      }
      
      // factor = E * lambda / N
      for (size_t t = 0; t < T; ++t) {
        double sum = 0.0;
        for (size_t n = 0; n < N; ++n) {
          sum += E[t * N + n] * L_data[n * r + f];
        }
        F_data[t * r + f] = sum / N;
      }
      
      V_data[f * r + f] = S_data[f];
    }
  }
  
  return py::make_tuple(factors, loadings, VNT);
}

// CORRECT R-STYLE IFE IMPLEMENTATION 
// Based on R's fe_ad_inter_iter() and ife() functions
py::tuple ife_predict_cf(py::array_t<double, py::array::c_style | py::array::forcecast> Y,
                         py::array_t<double, py::array::c_style | py::array::forcecast> D,
                         py::array_t<double, py::array::c_style | py::array::forcecast> I,
                         py::object X_obj,
                         const std::string& force_str,
                         int r) {
  if (Y.ndim() != 2 || D.ndim() != 2 || I.ndim() != 2)
    throw std::invalid_argument("Y, D, I must be 2D arrays");
  const size_t T = (size_t)Y.shape(0);
  const size_t N = (size_t)Y.shape(1);
  if ((size_t)D.shape(0) != T || (size_t)D.shape(1) != N || (size_t)I.shape(0) != T || (size_t)I.shape(1) != N)
    throw std::invalid_argument("Shapes of D/I must match Y");
    
  if (r <= 0) {
    return fe_predict_cf(Y, D, I, X_obj, force_str);
  }
  
  int force = 3;
  if (force_str == "none") force = 0;
  else if (force_str == "unit") force = 1;
  else if (force_str == "time") force = 2;
  else force = 3;

  const double* Yp = Y.data();
  const double* Dp = D.data();
  const double* Ip = I.data();
  
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

  // ============================================================================
  // IMPLEMENT R's CORRECT EM ALGORITHM (fe_ad_inter_iter + ife functions)
  // This is the SIMPLE algorithm R actually uses, not the complex two-stage one
  // ============================================================================

  // Build II = I with treated set to 0 (R's approach)
  std::vector<double> II(T * N, 0.0);
  for (size_t t = 0; t < T; ++t) {
    for (size_t n = 0; n < N; ++n) {
      double mask = (Dp[idx(t, n, N)] > 0.0) ? 0.0 : Ip[idx(t, n, N)];
      II[idx(t, n, N)] = (mask > 0.0) ? 1.0 : 0.0;
    }
  }
  
  const int max_iter = 1000;
  const double tol = 1e-5;  // Match R's tolerance
  
  // Initialize fit with FE or simple mean
  std::vector<double> fit(T * N, 0.0);
  std::vector<double> fit_old(T * N, 0.0);
  
  // Get initial fit from FE method
  if (hasX) {
    auto fe_result = fe_predict_cf(Y, D, I, X_obj, force_str);
    py::array_t<double> Y0_fe = fe_result.cast<py::tuple>()[0].cast<py::array_t<double>>();
    const double* Y0_fe_data = Y0_fe.data();
    for (size_t i = 0; i < T * N; ++i) {
      fit[i] = Y0_fe_data[i];
      fit_old[i] = Y0_fe_data[i];
    }
  } else {
    // Simple mean initialization
    double mean_y = 0.0;
    double count = 0.0;
    for (size_t i = 0; i < T * N; ++i) {
      if (II[i] > 0.0) {  // Use II (observed control data)
        mean_y += Yp[i];
        count += 1.0;
      }
    }
    if (count > 0) mean_y /= count;
    
    for (size_t i = 0; i < T * N; ++i) {
      fit[i] = mean_y;
      fit_old[i] = mean_y;
    }
  }
  
  std::vector<double> beta(p, 0.0);
  
  // EM Algorithm Loop - Exact R implementation
  for (int iter = 0; iter < max_iter; ++iter) {
    
    // E-step: YY = E_adj(Y, fit, I) - Fill missing with current fit
    std::vector<double> YY(T * N);
    for (size_t i = 0; i < T * N; ++i) {
      if (II[i] > 0.0) {
        YY[i] = Yp[i];     // Observed control data
      } else {
        YY[i] = fit[i];    // Fill missing/treated with current fit
      }
    }
    
    // M-step: Apply ife() function logic
    
    // 1. Update beta if covariates exist
    if (hasX) {
      // Remove current FE to get residuals for beta estimation
      std::vector<double> Y_for_beta(T * N);
      for (size_t i = 0; i < T * N; ++i) {
        Y_for_beta[i] = YY[i];
        // Remove previous covariate effects
        const double* Xp = X.data();
        for (size_t k = 0; k < p; ++k) {
          size_t t = i / N;
          size_t n = i % N;
          const size_t linear = ((t * N + n) * p) + k;
          Y_for_beta[i] -= Xp[linear] * beta[k];
        }
      }
      
      // Get FE residuals (without covariates)
      double mu_temp; std::vector<double> alpha_temp, xi_temp;
      recover_additive_fe(Y_for_beta.data(), II.data(), T, N, force, mu_temp, alpha_temp, xi_temp);
      
      std::vector<double> FE_resid(T * N);
      for (size_t t = 0; t < T; ++t) {
        for (size_t n = 0; n < N; ++n) {
          FE_resid[t * N + n] = Y_for_beta[t * N + n] - mu_temp;
          if (force == 1 || force == 3) FE_resid[t * N + n] -= alpha_temp[n];
          if (force == 2 || force == 3) FE_resid[t * N + n] -= xi_temp[t];
        }
      }
      
      // Factor decomposition
      py::array_t<double> FE_resid_arr({(py::ssize_t)T, (py::ssize_t)N});
      std::copy(FE_resid.begin(), FE_resid.end(), FE_resid_arr.mutable_data());
      
      auto factor_result = panel_factor(FE_resid_arr, r);
      auto factors = factor_result.cast<py::tuple>()[0].cast<py::array_t<double>>();
      auto loadings = factor_result.cast<py::tuple>()[1].cast<py::array_t<double>>();
      
      // Reconstruct interactive FE
      std::vector<double> FE_inter(T * N, 0.0);
      const double* F_data = factors.data();
      const double* L_data = loadings.data();
      
      for (size_t t = 0; t < T; ++t) {
        for (size_t n = 0; n < N; ++n) {
          for (int f = 0; f < r; ++f) {
            FE_inter[t * N + n] += F_data[t * r + f] * L_data[n * r + f];
          }
        }
      }
      
      // Update beta: regress (Y - additive_FE - interactive_FE) on X
      std::vector<double> Y_clean(T * N);
      for (size_t t = 0; t < T; ++t) {
        for (size_t n = 0; n < N; ++n) {
          Y_clean[t * N + n] = YY[t * N + n] - mu_temp - FE_inter[t * N + n];
          if (force == 1 || force == 3) Y_clean[t * N + n] -= alpha_temp[n];
          if (force == 2 || force == 3) Y_clean[t * N + n] -= xi_temp[t];
        }
      }
      
      // Solve for beta using least squares on observed data
      std::vector<double> XtX(p * p, 0.0);
      std::vector<double> XtY(p, 0.0);
      
      const double* Xp = X.data();
      for (size_t t = 0; t < T; ++t) {
        for (size_t n = 0; n < N; ++n) {
          if (II[t * N + n] > 0.0) {  // Only use observed data
            for (size_t i = 0; i < p; ++i) {
              const size_t linear_i = ((t * N + n) * p) + i;
              XtY[i] += Xp[linear_i] * Y_clean[t * N + n];
              
              for (size_t j = 0; j < p; ++j) {
                const size_t linear_j = ((t * N + n) * p) + j;
                XtX[i * p + j] += Xp[linear_i] * Xp[linear_j];
              }
            }
          }
        }
      }
      
      // Regularization
      for (size_t d = 0; d < p; ++d) XtX[d * p + d] += 1e-12;
      
      std::vector<double> beta_new = XtY;
      if (solve_spd(XtX, beta_new, p)) {
        beta = beta_new;
      }
    }
    
    // 2. Remove covariate effects from YY
    std::vector<double> U(T * N);
    for (size_t t = 0; t < T; ++t) {
      for (size_t n = 0; n < N; ++n) {
        U[t * N + n] = YY[t * N + n];
        if (hasX) {
          const double* Xp = X.data();
          for (size_t k = 0; k < p; ++k) {
            const size_t linear = ((t * N + n) * p) + k;
            U[t * N + n] -= Xp[linear] * beta[k];
          }
        }
      }
    }
    
    // 3. Apply ife() function: Y_demean -> panel_factor -> reconstruct
    
    // 3a. Remove additive fixed effects (Y_demean)
    double mu; std::vector<double> alpha, xi;
    recover_additive_fe(U.data(), II.data(), T, N, force, mu, alpha, xi);
    
    std::vector<double> EE(T * N);
    for (size_t t = 0; t < T; ++t) {
      for (size_t n = 0; n < N; ++n) {
        EE[t * N + n] = U[t * N + n] - mu;
        if (force == 1 || force == 3) EE[t * N + n] -= alpha[n];
        if (force == 2 || force == 3) EE[t * N + n] -= xi[t];
      }
    }
    
    // 3b. Factor decomposition (panel_factor)
    py::array_t<double> EE_arr({(py::ssize_t)T, (py::ssize_t)N});
    std::copy(EE.begin(), EE.end(), EE_arr.mutable_data());
    
    auto factor_result = panel_factor(EE_arr, r);
    auto factors = factor_result.cast<py::tuple>()[0].cast<py::array_t<double>>();
    auto loadings = factor_result.cast<py::tuple>()[1].cast<py::array_t<double>>();
    
    // 3c. Reconstruct fit: covariate_effects + additive_FE + interactive_FE
    const double* F_data = factors.data();
    const double* L_data = loadings.data();
    
    for (size_t t = 0; t < T; ++t) {
      for (size_t n = 0; n < N; ++n) {
        double val = mu;
        
        // Add additive fixed effects
        if (force == 1 || force == 3) val += alpha[n];
        if (force == 2 || force == 3) val += xi[t];
        
        // Add interactive fixed effects: F * L'
        for (int f = 0; f < r; ++f) {
          val += F_data[t * r + f] * L_data[n * r + f];
        }
        
        // Add covariate effects
        if (hasX) {
          const double* Xp = X.data();
          for (size_t k = 0; k < p; ++k) {
            const size_t linear = ((t * N + n) * p) + k;
            val += Xp[linear] * beta[k];
          }
        }
        
        fit[t * N + n] = val;
      }
    }
    
    // 4. Check convergence
    double diff = 0.0, norm_old = 0.0;
    for (size_t i = 0; i < T * N; ++i) {
      double delta = fit[i] - fit_old[i];
      diff += delta * delta;
      norm_old += fit_old[i] * fit_old[i];
    }
    
    fit_old = fit;
    
    if (norm_old > 0 && sqrt(diff / norm_old) < tol) {
      break;
    }
  }
  
  // Build final Y0
  py::array_t<double> Y0({(py::ssize_t)T, (py::ssize_t)N});
  double* Y0p = Y0.mutable_data();
  
  for (size_t i = 0; i < T * N; ++i) {
    Y0p[i] = fit[i];
  }
  
  // Build outputs
  py::array_t<double> II_arr({(py::ssize_t)T, (py::ssize_t)N});
  std::copy(II.begin(), II.end(), II_arr.mutable_data());
  
  py::array_t<double> beta_arr({(py::ssize_t)p});
  if (p > 0) {
    std::copy(beta.begin(), beta.end(), beta_arr.mutable_data());
  }
  
  py::object beta_se = py::none();
  return py::make_tuple(Y0, II_arr, beta_arr, beta_se);
}

// ATT calculation function
py::tuple att_from_diff(
  py::array_t<double, py::array::c_style | py::array::forcecast> Y,
  py::array_t<double, py::array::c_style | py::array::forcecast> Y0,
  py::array_t<double, py::array::c_style | py::array::forcecast> D,
  py::array_t<double, py::array::c_style | py::array::forcecast> I
) {
  if (Y.ndim() != 2 || Y0.ndim() != 2 || D.ndim() != 2 || I.ndim() != 2)
    throw std::invalid_argument("All inputs must be 2D arrays");
    
  const size_t T = (size_t)Y.shape(0);
  const size_t N = (size_t)Y.shape(1);
  
  if ((size_t)Y0.shape(0) != T || (size_t)Y0.shape(1) != N ||
      (size_t)D.shape(0) != T || (size_t)D.shape(1) != N ||
      (size_t)I.shape(0) != T || (size_t)I.shape(1) != N)
    throw std::invalid_argument("All arrays must have the same shape");
    
  const double* Yp = Y.data();
  const double* Y0p = Y0.data();
  const double* Dp = D.data();
  const double* Ip = I.data();
  
  // Calculate effects: eff = Y - Y0 where D == 1
  py::array_t<double> eff({(py::ssize_t)T, (py::ssize_t)N});
  double* effp = eff.mutable_data();
  
  for (size_t t = 0; t < T; ++t) {
    for (size_t n = 0; n < N; ++n) {
      size_t idx = t * N + n;
      if (Dp[idx] > 0.0 && Ip[idx] > 0.0) {  // Treated and observed
        effp[idx] = Yp[idx] - Y0p[idx];
      } else {
        effp[idx] = std::numeric_limits<double>::quiet_NaN();
      }
    }
  }
  
  // Aggregate by time
  std::vector<double> att_t_vec(T, 0.0);
  std::vector<int> counts_t_vec(T, 0);
  
  for (size_t t = 0; t < T; ++t) {
    double sum = 0.0;
    int count = 0;
    for (size_t n = 0; n < N; ++n) {
      size_t idx = t * N + n;
      if (std::isfinite(effp[idx])) {
        sum += effp[idx];
        count++;
      }
    }
    if (count > 0) {
      att_t_vec[t] = sum / count;
      counts_t_vec[t] = count;
    } else {
      att_t_vec[t] = std::numeric_limits<double>::quiet_NaN();
      counts_t_vec[t] = 0;
    }
  }
  
  py::array_t<double> att_t({(py::ssize_t)T});
  py::array_t<int> counts_t({(py::ssize_t)T});
  
  std::copy(att_t_vec.begin(), att_t_vec.end(), att_t.mutable_data());
  std::copy(counts_t_vec.begin(), counts_t_vec.end(), counts_t.mutable_data());
  
  return py::make_tuple(eff, att_t, counts_t);
}

// Event study function implementation
py::tuple event_study(
  py::array_t<double, py::array::c_style | py::array::forcecast> Y,
  py::array_t<double, py::array::c_style | py::array::forcecast> D,
  py::array_t<double, py::array::c_style | py::array::forcecast> I,
  py::array_t<double, py::array::c_style | py::array::forcecast> Y0
) {
  if (Y.ndim() != 2 || D.ndim() != 2 || I.ndim() != 2 || Y0.ndim() != 2)
    throw std::invalid_argument("All inputs must be 2D arrays");
    
  const size_t T = (size_t)Y.shape(0);
  const size_t N = (size_t)Y.shape(1);
  
  if ((size_t)D.shape(0) != T || (size_t)D.shape(1) != N ||
      (size_t)I.shape(0) != T || (size_t)I.shape(1) != N ||
      (size_t)Y0.shape(0) != T || (size_t)Y0.shape(1) != N)
    throw std::invalid_argument("All arrays must have the same shape");
    
  const double* Yp = Y.data();
  const double* Dp = D.data();
  const double* Ip = I.data();
  const double* Y0p = Y0.data();
  
  // Find the range of relative treatment times
  std::vector<int> rel_times;
  std::set<int> unique_rel_times;
  
  for (size_t n = 0; n < N; ++n) {
    int first_treat = -1;
    for (size_t t = 0; t < T; ++t) {
      if (Dp[t * N + n] > 0.0) {
        if (first_treat < 0) first_treat = (int)t;
      }
    }
    
    if (first_treat >= 0) {
      for (size_t t = 0; t < T; ++t) {
        if (Dp[t * N + n] > 0.0 && Ip[t * N + n] > 0.0) {
          int rel_time = (int)t - first_treat;
          unique_rel_times.insert(rel_time);
        }
      }
    }
  }
  
  rel_times.assign(unique_rel_times.begin(), unique_rel_times.end());
  const size_t R = rel_times.size();
  
  if (R == 0) {
    // No treated observations
    return py::make_tuple(py::array_t<int>({0}), py::array_t<double>({0}), py::array_t<int>({0}));
  }
  
  py::array_t<int> timeline({(py::ssize_t)R});
  py::array_t<double> att({(py::ssize_t)R});
  py::array_t<int> counts({(py::ssize_t)R});
  
  int* timeline_ptr = timeline.mutable_data();
  double* att_ptr = att.mutable_data();
  int* counts_ptr = counts.mutable_data();
  
  for (size_t r = 0; r < R; ++r) {
    timeline_ptr[r] = rel_times[r];
    
    double sum = 0.0;
    int count = 0;
    
    for (size_t n = 0; n < N; ++n) {
      int first_treat = -1;
      for (size_t t = 0; t < T; ++t) {
        if (Dp[t * N + n] > 0.0) {
          if (first_treat < 0) first_treat = (int)t;
        }
      }
      
      if (first_treat >= 0) {
        for (size_t t = 0; t < T; ++t) {
          if (Dp[t * N + n] > 0.0 && Ip[t * N + n] > 0.0) {
            int rel_time = (int)t - first_treat;
            if (rel_time == rel_times[r]) {
              double effect = Yp[t * N + n] - Y0p[t * N + n];
              sum += effect;
              count++;
            }
          }
        }
      }
    }
    
    if (count > 0) {
      att_ptr[r] = sum / count;
      counts_ptr[r] = count;
    } else {
      att_ptr[r] = std::numeric_limits<double>::quiet_NaN();
      counts_ptr[r] = 0;
    }
  }
  
  return py::make_tuple(timeline, att, counts);
}

PYBIND11_MODULE(_fe, m) {
  m.doc() = "Fast FE counterfactuals (two-way FE with covariates and IFE)";
  m.def("fe_predict_cf", &fe_predict_cf, py::arg("Y"), py::arg("D"), py::arg("I"), py::arg("X") = py::none(), py::arg("force") = std::string("two-way"));
  m.def("ife_predict_cf", &ife_predict_cf, py::arg("Y"), py::arg("D"), py::arg("I"), py::arg("X") = py::none(), py::arg("force") = std::string("two-way"), py::arg("r") = 1);
  m.def("panel_factor", &panel_factor, py::arg("Y"), py::arg("r"));
  m.def("att_from_diff", &att_from_diff, py::arg("Y"), py::arg("Y0"), py::arg("D"), py::arg("I"));
  m.def("event_study", &event_study, py::arg("Y"), py::arg("D"), py::arg("I"), py::arg("Y0"));
}
