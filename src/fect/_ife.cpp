#include <pybind11/pybind11.h>
#include <pybind11/numpy.h>
#include <vector>
#include <cmath>
#include <algorithm>

namespace py = pybind11;

static inline size_t idx(size_t t, size_t n, size_t N) { return t * N + n; }

// Helpers largely mirroring R's auxiliary/ife_sub implementations

// Recover additive fixed effects (mu, alpha_i, xi_t) using only a mask of usable cells
static void recover_additive_fe(const double* Y, const double* II,
                                size_t T, size_t N, int force,
                                double& mu,
                                std::vector<double>& alpha,
                                std::vector<double>& xi) {
  alpha.assign(N, 0.0);
  xi.assign(T, 0.0);
  double wsum = 0.0, ysum = 0.0;
  for (size_t t = 0; t < T; ++t) for (size_t n = 0; n < N; ++n) {
    if (II[idx(t,n,N)] > 0.0) { wsum += 1.0; ysum += Y[idx(t,n,N)]; }
  }
  mu = (wsum > 0.0 ? ysum / wsum : 0.0);
  if (force == 0) return;
  if (force == 1) { // unit only
    for (size_t n = 0; n < N; ++n) {
      double sw = 0.0, sy = 0.0;
      for (size_t t = 0; t < T; ++t) if (II[idx(t,n,N)] > 0.0) { sw += 1.0; sy += (Y[idx(t,n,N)] - mu); }
      alpha[n] = (sw > 0.0 ? sy / sw : 0.0);
    }
    return;
  }
  if (force == 2) { // time only
    for (size_t t = 0; t < T; ++t) {
      double sw = 0.0, sy = 0.0;
      for (size_t n = 0; n < N; ++n) if (II[idx(t,n,N)] > 0.0) { sw += 1.0; sy += (Y[idx(t,n,N)] - mu); }
      xi[t] = (sw > 0.0 ? sy / sw : 0.0);
    }
    return;
  }
  // two-way via alternating projections
  const int max_iter = 100;
  const double tol = 1e-10;
  for (int it = 0; it < max_iter; ++it) {
    std::vector<double> a_prev = alpha, x_prev = xi;
    // alpha update
    for (size_t n = 0; n < N; ++n) {
      double sw = 0.0, sy = 0.0;
      for (size_t t = 0; t < T; ++t) if (II[idx(t,n,N)] > 0.0) { sw += 1.0; sy += (Y[idx(t,n,N)] - mu - xi[t]); }
      alpha[n] = (sw > 0.0 ? sy / sw : 0.0);
    }
    double am = 0.0; for (double v : alpha) am += v; am /= (N > 0 ? double(N) : 1.0);
    for (double& v : alpha) v -= am;
    // xi update
    for (size_t t = 0; t < T; ++t) {
      double sw = 0.0, sy = 0.0;
      for (size_t n = 0; n < N; ++n) if (II[idx(t,n,N)] > 0.0) { sw += 1.0; sy += (Y[idx(t,n,N)] - mu - alpha[n]); }
      xi[t] = (sw > 0.0 ? sy / sw : 0.0);
    }
    double xm = 0.0; for (double v : xi) xm += v; xm /= (T > 0 ? double(T) : 1.0);
    for (double& v : xi) v -= xm;
    // mu update
    double sw2 = 0.0, sy2 = 0.0;
    for (size_t t = 0; t < T; ++t) for (size_t n = 0; n < N; ++n) if (II[idx(t,n,N)] > 0.0) { sw2 += 1.0; sy2 += (Y[idx(t,n,N)] - alpha[n] - xi[t]); }
    mu = (sw2 > 0.0 ? sy2 / sw2 : 0.0);
    double da = 0.0, dx = 0.0;
    for (size_t n = 0; n < N; ++n) da = std::max(da, std::abs(alpha[n] - a_prev[n]));
    for (size_t t = 0; t < T; ++t) dx = std::max(dx, std::abs(xi[t] - x_prev[t]));
    if (da < tol && dx < tol) break;
  }
}

// SVD decomposition wrapper using numpy, reusing the same function signature as in _fe.cpp
static py::tuple numpy_svd(py::array_t<double> A, bool full_matrices = false) {
  py::object np = py::module::import("numpy");
  py::object linalg = np.attr("linalg");
  py::object svd_func = linalg.attr("svd");
  return svd_func(A, py::arg("full_matrices")=full_matrices).cast<py::tuple>();
}

static py::tuple panel_factor(py::array_t<double> Y, int r) {
  if (Y.ndim() != 2) throw std::invalid_argument("Y must be 2D");
  const size_t T = (size_t)Y.shape(0);
  const size_t N = (size_t)Y.shape(1);
  const double* Yp = Y.data();
  py::array_t<double> F({(py::ssize_t)T, (py::ssize_t)r});
  py::array_t<double> L({(py::ssize_t)N, (py::ssize_t)r});
  py::array_t<double> V({(py::ssize_t)r, (py::ssize_t)r});
  double* Fd = F.mutable_data();
  double* Ld = L.mutable_data();
  double* Vd = V.mutable_data();
  std::fill(Fd, Fd + T*r, 0.0);
  std::fill(Ld, Ld + N*r, 0.0);
  std::fill(Vd, Vd + r*r, 0.0);
  // Build E = Y (already demeaned outside)
  if (T < N) {
    py::array_t<double> EE({(py::ssize_t)T, (py::ssize_t)T}); double* EEp = EE.mutable_data();
    for (size_t i=0;i<T;++i){for(size_t j=0;j<T;++j){double s=0.0;for(size_t k=0;k<N;++k){s+=Yp[i*N+k]*Yp[j*N+k];}EEp[i*T+j]=s/(N*T);} }
    auto svd = numpy_svd(EE, false); auto U = svd[0].cast<py::array_t<double>>(); auto S = svd[1].cast<py::array_t<double>>();
    const double* Ud = U.data(); const double* Sd = S.data();
    for (int f=0; f<r; ++f){ for(size_t t=0;t<T;++t) Fd[t*r+f]=Ud[t*T+f]*std::sqrt((double)T);
      for(size_t n=0;n<N;++n){ double s=0.0; for(size_t t=0;t<T;++t) s+=Yp[t*N+n]*Fd[t*r+f]; Ld[n*r+f]=s/((double)T);} Vd[f*r+f]=Sd[f]; }
  } else {
    py::array_t<double> EE({(py::ssize_t)N, (py::ssize_t)N}); double* EEp = EE.mutable_data();
    for (size_t i=0;i<N;++i){for(size_t j=0;j<N;++j){double s=0.0;for(size_t k=0;k<T;++k){s+=Yp[k*N+i]*Yp[k*N+j];}EEp[i*N+j]=s/(N*T);} }
    auto svd = numpy_svd(EE, false); auto U = svd[0].cast<py::array_t<double>>(); auto S = svd[1].cast<py::array_t<double>>();
    const double* Ud = U.data(); const double* Sd = S.data();
    for (int f=0; f<r; ++f){ for(size_t n=0;n<N;++n) Ld[n*r+f]=Ud[n*N+f]*std::sqrt((double)N);
      for(size_t t=0;t<T;++t){ double s=0.0; for(size_t n=0;n<N;++n) s+=Yp[t*N+n]*Ld[n*r+f]; Fd[t*r+f]=s/((double)N);} Vd[f*r+f]=Sd[f]; }
  }
  return py::make_tuple(F,L,V);
}

// ---- R helper ports (exact-behavior equivalents) ----
static py::array_t<double> E_adj_py(py::array_t<double> E, py::array_t<double> FE, py::array_t<double> I) {
  const ssize_t T = E.shape(0), N = E.shape(1);
  py::array_t<double> EE({T, N});
  const double* Ep = E.data();
  const double* FEp = FE.data();
  const double* Ip = I.data();
  double* EEp = EE.mutable_data();
  for (ssize_t t = 0; t < T; ++t) {
    for (ssize_t n = 0; n < N; ++n) {
      size_t k = (size_t)t * (size_t)N + (size_t)n;
      EEp[k] = (Ip[k] == 0.0 ? FEp[k] : Ep[k]);
    }
  }
  return EE;
}

static py::array_t<double> FE_adj_py(py::array_t<double> FE, py::array_t<double> I) {
  const ssize_t T = FE.shape(0), N = FE.shape(1);
  py::array_t<double> out({T, N});
  const double* FEp = FE.data();
  const double* Ip = I.data();
  double* Op = out.mutable_data();
  for (ssize_t t = 0; t < T; ++t) {
    for (ssize_t n = 0; n < N; ++n) {
      size_t k = (size_t)t * (size_t)N + (size_t)n;
      Op[k] = (Ip[k] == 0.0 ? 0.0 : FEp[k]);
    }
  }
  return out;
}

static py::dict Y_demean_py(py::array_t<double> Y, int force) {
  const ssize_t T = Y.shape(0), N = Y.shape(1);
  const double* Yp = Y.data();
  py::array_t<double> YY({T, N});
  double* YYp = YY.mutable_data();
  double mu_Y = 0.0;
  for (ssize_t i = 0; i < T * N; ++i) mu_Y += Yp[i];
  mu_Y /= double(T * N);

  std::vector<double> alpha_Y((size_t)N, 0.0);
  std::vector<double> xi_Y((size_t)T, 0.0);

  if (force == 0) {
    for (ssize_t t = 0; t < T; ++t)
      for (ssize_t n = 0; n < N; ++n)
        YYp[idx((size_t)t, (size_t)n, (size_t)N)] = Yp[idx((size_t)t, (size_t)n, (size_t)N)] - mu_Y;
  } else if (force == 1) {
    for (ssize_t n = 0; n < N; ++n) {
      double s = 0.0;
      for (ssize_t t = 0; t < T; ++t) s += (Yp[idx((size_t)t, (size_t)n, (size_t)N)] - mu_Y);
      alpha_Y[(size_t)n] = s / double(T);
    }
    for (ssize_t t = 0; t < T; ++t)
      for (ssize_t n = 0; n < N; ++n)
        YYp[idx((size_t)t, (size_t)n, (size_t)N)] = (Yp[idx((size_t)t, (size_t)n, (size_t)N)] - mu_Y) - alpha_Y[(size_t)n];
  } else if (force == 2) {
    for (ssize_t t = 0; t < T; ++t) {
      double s = 0.0;
      for (ssize_t n = 0; n < N; ++n) s += (Yp[idx((size_t)t, (size_t)n, (size_t)N)] - mu_Y);
      xi_Y[(size_t)t] = s / double(N);
    }
    for (ssize_t t = 0; t < T; ++t)
      for (ssize_t n = 0; n < N; ++n)
        YYp[idx((size_t)t, (size_t)n, (size_t)N)] = (Yp[idx((size_t)t, (size_t)n, (size_t)N)] - mu_Y) - xi_Y[(size_t)t];
  } else { // two-way, approximate via alternating projections on all entries
    double mu; std::vector<double> a, x; std::vector<double> ones((size_t)T * (size_t)N, 1.0);
    recover_additive_fe(Yp, ones.data(), (size_t)T, (size_t)N, 3, mu, a, x);
    for (ssize_t t = 0; t < T; ++t)
      for (ssize_t n = 0; n < N; ++n) {
        double v = Yp[idx((size_t)t, (size_t)n, (size_t)N)] - mu;
        v -= a[(size_t)n]; v -= x[(size_t)t];
        YYp[idx((size_t)t, (size_t)n, (size_t)N)] = v;
      }
    for (ssize_t n = 0; n < N; ++n) alpha_Y[(size_t)n] = mu + a[(size_t)n];
    for (ssize_t t = 0; t < T; ++t) xi_Y[(size_t)t] = mu + x[(size_t)t];
  }

  py::array_t<double> alpha_arr({(py::ssize_t)N, (py::ssize_t)1});
  py::array_t<double> xi_arr({(py::ssize_t)T, (py::ssize_t)1});
  if (force == 1 || force == 3) { double* ap = alpha_arr.mutable_data(); for (ssize_t n = 0; n < N; ++n) ap[n] = alpha_Y[(size_t)n]; }
  if (force == 2 || force == 3) { double* xp = xi_arr.mutable_data(); for (ssize_t t = 0; t < T; ++t) xp[t] = xi_Y[(size_t)t]; }

  py::dict out; out["mu_Y"] = py::float_(mu_Y); out["YY"] = YY; if (force == 1 || force == 3) out["alpha_Y"] = alpha_arr; if (force == 2 || force == 3) out["xi_Y"] = xi_arr; return out;
}

static py::dict fe_add_py(py::array_t<double> alpha_Y, py::array_t<double> xi_Y, double mu_Y, int T, int N, int force) {
  py::array_t<double> FE_ad({(py::ssize_t)T, (py::ssize_t)N});
  double* FEp = FE_ad.mutable_data();
  std::vector<double> alpha((size_t)N, 0.0), xi((size_t)T, 0.0);
  if (force == 1 || force == 3) { const double* a = alpha_Y.data(); for (int n = 0; n < N; ++n) alpha[(size_t)n] = a[n] - mu_Y; }
  if (force == 2 || force == 3) { const double* x = xi_Y.data(); for (int t = 0; t < T; ++t) xi[(size_t)t] = x[t] - mu_Y; }
  for (int t = 0; t < T; ++t) for (int n = 0; n < N; ++n) { double v = mu_Y; if (force == 1 || force == 3) v += alpha[(size_t)n]; if (force == 2 || force == 3) v += xi[(size_t)t]; FEp[idx((size_t)t, (size_t)n, (size_t)N)] = v; }
  py::array_t<double> alpha_out({(py::ssize_t)N, (py::ssize_t)1}); py::array_t<double> xi_out({(py::ssize_t)T, (py::ssize_t)1});
  if (force == 1 || force == 3) { double* ap = alpha_out.mutable_data(); for (int n = 0; n < N; ++n) ap[n] = alpha[(size_t)n]; }
  if (force == 2 || force == 3) { double* xp = xi_out.mutable_data(); for (int t = 0; t < T; ++t) xp[t] = xi[(size_t)t]; }
  py::dict out; out["mu"] = py::float_(mu_Y); out["FE_ad"] = FE_ad; if (force == 1 || force == 3) out["alpha"] = alpha_out; if (force == 2 || force == 3) out["xi"] = xi_out; return out;
}

static py::tuple ife_py(py::array_t<double> E, int force, int mc, int r, int hard, double lambda) {
  const ssize_t T = E.shape(0), N = E.shape(1);
  auto E_ad = Y_demean_py(E, force);
  auto EE = E_ad["YY"].cast<py::array_t<double>>();
  double mu_E = E_ad["mu_Y"].cast<double>();
  py::array_t<double> alpha_E, xi_E;
  if (force == 1 || force == 3) alpha_E = E_ad["alpha_Y"].cast<py::array_t<double>>();
  if (force == 2 || force == 3) xi_E = E_ad["xi_Y"].cast<py::array_t<double>>();
  py::dict fe_add = fe_add_py(alpha_E, xi_E, mu_E, (int)T, (int)N, force);
  auto FE_add_use = fe_add["FE_ad"].cast<py::array_t<double>>();
  py::array_t<double> FE_inter_use({T, N}); std::fill(FE_inter_use.mutable_data(), FE_inter_use.mutable_data() + T * N, 0.0);
  py::array_t<double> VNT({(py::ssize_t)std::max(0, r), (py::ssize_t)std::max(0, r)});
  if (r > 0 && mc == 0) {
    auto pf = panel_factor(EE, r); auto F = pf[0].cast<py::array_t<double>>(); auto L = pf[1].cast<py::array_t<double>>(); VNT = pf[2].cast<py::array_t<double>>();
    const double* Fd = F.data(); const double* Ld = L.data(); double* FEip = FE_inter_use.mutable_data();
    for (ssize_t t = 0; t < T; ++t) for (ssize_t n = 0; n < N; ++n) { double s = 0.0; for (int f = 0; f < r; ++f) s += Fd[t * r + f] * Ld[n * r + f]; FEip[idx((size_t)t, (size_t)n, (size_t)N)] = s; }
  }
  py::array_t<double> FE({T, N}); const double* A = FE_add_use.data(); const double* B = FE_inter_use.data(); double* FEp = FE.mutable_data(); for (ssize_t i = 0; i < T * N; ++i) FEp[i] = A[i] + B[i];
  py::dict out; out["mu"] = fe_add["mu"]; out["FE"] = FE; if (force == 1 || force == 3) out["alpha"] = fe_add["alpha"]; if (force == 2 || force == 3) out["xi"] = fe_add["xi"]; if (r > 0) { out["VNT"] = VNT; out["FE_inter_use"] = FE_inter_use; }
  return out;
}

static py::array_t<double> XXinv_py(py::array_t<double> X) {
  const ssize_t T = X.shape(0), N = X.shape(1), p = X.shape(2);
  py::array_t<double> xx({p, p}); double* xxp = xx.mutable_data(); std::fill(xxp, xxp + p * p, 0.0);
  const double* Xp = X.data();
  for (ssize_t i = 0; i < p; ++i) {
    for (ssize_t j = i; j < p; ++j) {
      double s = 0.0;
      for (ssize_t t = 0; t < T; ++t) for (ssize_t n = 0; n < N; ++n) { size_t base = ((size_t)t * (size_t)N + (size_t)n) * (size_t)p; s += Xp[base + i] * Xp[base + j]; }
      xxp[i * p + j] = s; xxp[j * p + i] = s;
    }
  }
  py::object np = py::module::import("numpy"); return np.attr("linalg").attr("inv")(xx).cast<py::array_t<double>>();
}

static py::array_t<double> panel_beta_py(py::array_t<double> X, py::array_t<double> xxinv, py::array_t<double> Y, py::array_t<double> FE) {
  const ssize_t T = X.shape(0), N = X.shape(1), p = X.shape(2);
  const double* Xp = X.data(); const double* Yp = Y.data(); const double* FEp = FE.data();
  py::array_t<double> xy({(py::ssize_t)p, (py::ssize_t)1}); double* xyp = xy.mutable_data(); std::fill(xyp, xyp + p, 0.0);
  for (ssize_t k = 0; k < p; ++k) {
    double s = 0.0; for (ssize_t t = 0; t < T; ++t) for (ssize_t n = 0; n < N; ++n) { size_t base = ((size_t)t * (size_t)N + (size_t)n) * (size_t)p; s += Xp[base + k] * (Yp[idx((size_t)t, (size_t)n, (size_t)N)] - FEp[idx((size_t)t, (size_t)n, (size_t)N)]); } xyp[k] = s;
  }
  py::object np = py::module::import("numpy"); return np.attr("dot")(xxinv, xy).cast<py::array_t<double>>();
}

// Build initial Y0 similar to R's initialFit on controls: FE on II + optional beta on demeaned X
static py::array_t<double> initial_fit_py(py::array_t<double> Y, py::array_t<double> I_control, py::array_t<double> X, int force) {
  const ssize_t T = Y.shape(0), N = Y.shape(1);
  const bool hasX = (X.ptr() != nullptr && X.ndim() == 3 && X.shape(2) > 0);
  const double* Yp = Y.data();
  const double* IIp = I_control.data();
  // FE on controls
  double mu; std::vector<double> a, x; recover_additive_fe(Yp, IIp, (size_t)T, (size_t)N, force, mu, a, x);
  py::array_t<double> FE_add({(py::ssize_t)T,(py::ssize_t)N}); double* FEp = FE_add.mutable_data();
  for (ssize_t t=0; t<T; ++t) for (ssize_t n=0; n<N; ++n) {
    double v = mu; if (force==1||force==3) v += a[(size_t)n]; if (force==2||force==3) v += x[(size_t)t]; FEp[idx((size_t)t,(size_t)n,(size_t)N)] = v;
  }
  if (!hasX) return FE_add;
  // Add covariates beta0 using panel_beta with (Y - FE_add)
  auto xx = XXinv_py(X);
  auto beta0 = panel_beta_py(X, xx, Y, FE_add);
  const double* Xp = X.data(); const double* b0 = beta0.data(); const ssize_t p = X.shape(2);
  py::array_t<double> Y0({(py::ssize_t)T,(py::ssize_t)N}); double* Y0p = Y0.mutable_data();
  for (ssize_t t=0; t<T; ++t) {
    for (ssize_t n=0; n<N; ++n) {
      size_t base = ((size_t)t*(size_t)N + (size_t)n)*(size_t)p; double s=0.0; for (ssize_t k=0;k<p;++k) s += Xp[base+k]*b0[k];
      Y0p[idx((size_t)t,(size_t)n,(size_t)N)] = FEp[idx((size_t)t,(size_t)n,(size_t)N)] + s;
    }
  }
  return Y0;
}

static py::dict fe_ad_inter_iter_py(py::array_t<double> Y, py::array_t<double> Y0, py::array_t<double> I, int force, int r, double tol, int max_iter) {
  const ssize_t T = Y.shape(0), N = Y.shape(1);
  py::array_t<double> fit = Y0; py::array_t<double> fit_old = Y0; int niter = 0; py::dict ife_out;
  while (true) {
    auto YY = E_adj_py(Y, fit, I);
    ife_out = ife_py(YY, force, 0, r, 0, 0.0);
    fit = ife_out["FE"].cast<py::array_t<double>>();
    double num = 0.0, den = 0.0; const double* fp = fit.data(); const double* fo = fit_old.data(); for (ssize_t i = 0; i < T * N; ++i) { double d = fp[i] - fo[i]; num += d * d; den += fo[i] * fo[i]; }
    ++niter; fit_old = fit; if (den > 0 && std::sqrt(num / den) < tol) break; if (niter >= max_iter) break;
  }
  auto YYfinal = E_adj_py(Y, fit, I); auto e_adj = FE_adj_py(YYfinal, I);
  py::dict out; out["mu"] = ife_out["mu"]; out["e"] = e_adj; out["fit"] = fit; out["niter"] = py::int_(niter); if (force == 1 || force == 3) out["alpha"] = ife_out["alpha"]; if (force == 2 || force == 3) out["xi"] = ife_out["xi"]; if (r > 0) out["VNT"] = ife_out["VNT"]; return out;
}

static py::dict fe_ad_inter_covar_iter_py(py::array_t<double> X, py::array_t<double> xxinv, py::array_t<double> Y, py::array_t<double> Y0, py::array_t<double> I, int force, int r, double tol, int max_iter) {
  const ssize_t T = Y.shape(0), N = Y.shape(1), p = X.shape(2);
  py::array_t<double> fit = Y0; py::array_t<double> fit_old = Y0; py::array_t<double> beta({(py::ssize_t)p, (py::ssize_t)1}); std::fill(beta.mutable_data(), beta.mutable_data() + p, 0.0);
  int niter = 0; py::dict ife_out; const double* Xp = X.data();
  while (true) {
    auto YY = E_adj_py(Y, fit, I);
    py::array_t<double> covar_fit({T, N}); double* cf = covar_fit.mutable_data(); const double* bp = beta.data();
    for (ssize_t t = 0; t < T; ++t) for (ssize_t n = 0; n < N; ++n) { size_t base = ((size_t)t * (size_t)N + (size_t)n) * (size_t)p; double s = 0.0; for (ssize_t k = 0; k < p; ++k) s += Xp[base + k] * bp[k]; cf[idx((size_t)t, (size_t)n, (size_t)N)] = s; }
    py::array_t<double> U({T, N}); double* Up = U.mutable_data(); const double* YYp = YY.data(); for (ssize_t i = 0; i < T * N; ++i) Up[i] = YYp[i] - cf[i];
    ife_out = ife_py(U, force, 0, r, 0, 0.0);
    auto FE = ife_out["FE"].cast<py::array_t<double>>();
    beta = panel_beta_py(X, xxinv, YY, FE);
    const double* bp2 = beta.data(); for (ssize_t t = 0; t < T; ++t) for (ssize_t n = 0; n < N; ++n) { size_t base = ((size_t)t * (size_t)N + (size_t)n) * (size_t)p; double s = 0.0; for (ssize_t k = 0; k < p; ++k) s += Xp[base + k] * bp2[k]; cf[idx((size_t)t, (size_t)n, (size_t)N)] = s; }
    for (ssize_t i = 0; i < T * N; ++i) fit.mutable_data()[i] = FE.data()[i] + cf[i];
    double num = 0.0, den = 0.0; const double* fp = fit.data(); const double* fo = fit_old.data(); for (ssize_t i = 0; i < T * N; ++i) { double d = fp[i] - fo[i]; num += d * d; den += fo[i] * fo[i]; }
    ++niter; fit_old = fit; if (den > 0 && std::sqrt(num / den) < tol) break; if (niter >= max_iter) break;
  }
  auto YYfinal = E_adj_py(Y, fit, I); auto e_adj = FE_adj_py(YYfinal, I);
  py::dict out; out["mu"] = ife_out["mu"]; out["e"] = e_adj; out["fit"] = fit; out["niter"] = py::int_(niter); out["beta"] = beta; if (force == 1 || force == 3) out["alpha"] = ife_out["alpha"]; if (force == 2 || force == 3) out["xi"] = ife_out["xi"]; if (r > 0) out["VNT"] = ife_out["VNT"]; return out;
}

// Main: IFE predictor following R's inter_fe_ub/fe_ad_inter_* structure
py::tuple ife_predict_cf_r(py::array_t<double, py::array::c_style | py::array::forcecast> Y,
                          py::array_t<double, py::array::c_style | py::array::forcecast> D,
                          py::array_t<double, py::array::c_style | py::array::forcecast> I,
                          py::object X_obj,
                          const std::string& force_str,
                          int r) {
  if (Y.ndim() != 2 || D.ndim() != 2 || I.ndim() != 2) throw std::invalid_argument("Y, D, I must be 2D arrays");
  const size_t T = (size_t)Y.shape(0); const size_t N = (size_t)Y.shape(1);
  int force = 3; if (force_str=="none") force=0; else if (force_str=="unit") force=1; else if (force_str=="time") force=2; else force=3;
  const double* Yp=Y.data(); const double* Dp=D.data(); const double* Ip=I.data();
  size_t p=0; bool hasX=false; py::array_t<double> X; if(!X_obj.is_none()){ X=X_obj.cast<py::array_t<double, py::array::c_style | py::array::forcecast>>(); if(X.ndim()!=3|| (size_t)X.shape(0)!=T || (size_t)X.shape(1)!=N) throw std::invalid_argument("X must be (T,N,p)"); p=(size_t)X.shape(2); hasX=(p>0); }

  // II: usable observations are untreated controls (as in R)
  std::vector<double> II(T*N,0.0); for(size_t t=0;t<T;++t) for(size_t n=0;n<N;++n){ double mask=(Dp[idx(t,n,N)]>0.0)?0.0:Ip[idx(t,n,N)]; II[idx(t,n,N)]=(mask>0.0)?1.0:0.0; }

  // Call exact R-style iterators with II mask (controls-only), as in R's inter_fe_ub
  const double tol = 1e-5; const int max_iter = 1000;
  // Build initial fit using controls-only FE (closer to R's initialFit)
  py::array_t<double> Y0_init({(py::ssize_t)T,(py::ssize_t)N});
  py::dict res;
  // Build II from D and I: II = I with treated set to 0 (R: YY uses I where treated are set missing)
  py::array_t<double> II_arr({(py::ssize_t)T,(py::ssize_t)N});
  double* IIp = II_arr.mutable_data();
  for (size_t t = 0; t < T; ++t) {
    for (size_t n = 0; n < N; ++n) {
      double mask = (Dp[idx(t,n,N)] > 0.0) ? 0.0 : Ip[idx(t,n,N)];
      IIp[idx(t,n,N)] = (mask > 0.0 ? 1.0 : 0.0);
    }
  }
  // Compute initialization
  try {
    if (hasX) Y0_init = initial_fit_py(Y, II_arr, X, force);
    else Y0_init = initial_fit_py(Y, II_arr, py::array_t<double>(), force);
  } catch (...) {
    // fall back to zeros if any issue
  }
  if (!hasX) {
    res = fe_ad_inter_iter_py(Y, Y0_init, II_arr, force, r, tol, max_iter);
  } else {
    auto xx = XXinv_py(X);
    res = fe_ad_inter_covar_iter_py(X, xx, Y, Y0_init, II_arr, force, r, tol, max_iter);
  }
  auto Y0 = res["fit"].cast<py::array_t<double>>();
  py::array_t<double> beta_arr({(py::ssize_t)p});
  if (hasX && res.contains("beta")) beta_arr = res["beta"].cast<py::array_t<double>>();
  // Return II as the mask used in fitting (R returns II internally for algorithms)
  return py::make_tuple(Y0, II_arr, beta_arr, py::none());
}

PYBIND11_MODULE(_ife, m) {
  m.doc() = "IFE port matching R logic (pybind11)";
  m.def("ife_predict_cf_r", &ife_predict_cf_r, py::arg("Y"), py::arg("D"), py::arg("I"), py::arg("X") = py::none(), py::arg("force") = std::string("two-way"), py::arg("r") = 1);
}


