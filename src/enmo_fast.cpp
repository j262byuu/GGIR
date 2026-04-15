// enmo_fast.cpp
// Fused C++ implementation of ENMO-family epoch metrics for GGIR
//
// Replaces the R implementations of: EuclideanNorm() + averagePerEpoch() for ENMO, EN, ENMOa, MAD
//
// Performance (7-day 100Hz simulated data): ~8x speedup, ~80% memory reduction vs ENMO epoch aggregation in g.applymetrics
//
// This patch primarily benefits HPC batch processing where multiple workers compete for memory
//
// Author: Xiaoyu Zong j262byuu@gmail.com
// License: Same as GGIR (Apache-2.0)

#include <Rcpp.h>
#include <cmath>
using namespace Rcpp;

// ---------------------------------------------------------------------------
// Standalone epoch aggregation functions
// Drop-in replacements for the R averagePerEpoch / sumPerEpoch
// Zero extra memory allocation (no cumsum vector)
// Properly propagates NA: if any value in an epoch is NA, output is NA
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
NumericVector averagePerEpochCpp(const NumericVector& x,
                                 int sf,
                                 int epochsize) {
  int n = x.size();
  int block = sf * epochsize;
  int n_epochs = n / block;
  if (n_epochs <= 0) return NumericVector(0);

  NumericVector out(n_epochs);
  int pos = 0;
  for (int i = 0; i < n_epochs; i++) {
    double s = 0.0;
    bool has_na = false;
    int end = pos + block;
    for (int j = pos; j < end; j++) {
      if (ISNAN(x[j])) {
        has_na = true;
        break;
      }
      s += x[j];
    }
    out[i] = has_na ? NA_REAL : s / block;
    pos = end;
  }
  return out;
}

// [[Rcpp::export]]
NumericVector sumPerEpochCpp(const NumericVector& x,
                              int sf,
                              int epochsize) {
  int n = x.size();
  int block = sf * epochsize;
  int n_epochs = n / block;
  if (n_epochs <= 0) return NumericVector(0);

  NumericVector out(n_epochs);
  int pos = 0;
  for (int i = 0; i < n_epochs; i++) {
    double s = 0.0;
    bool has_na = false;
    int end = pos + block;
    for (int j = pos; j < end; j++) {
      if (ISNAN(x[j])) {
        has_na = true;
        break;
      }
      s += x[j];
    }
    out[i] = has_na ? NA_REAL : s;
    pos = end;
  }
  return out;
}

// ---------------------------------------------------------------------------
// Fused ENMO-family metric computation
//
// Computes all of the following in a single pass over raw x/y/z data:
//   - ENMO:  mean(max(sqrt(x^2+y^2+z^2) - 1, 0)) per epoch
//   - EN:    mean(sqrt(x^2+y^2+z^2)) per epoch
//   - ENMOa: mean(|sqrt(x^2+y^2+z^2) - 1|) per epoch
//   - MAD:   mean(|EN - epoch_mean_EN|) per epoch (requires 2nd pass)
//
// NA handling: if ANY sample in an epoch contains NA (in any axis),
// ALL metrics for that epoch are set to NA. This matches R's behavior
// where NA propagates through arithmetic operations.
//
// Memory: only allocates output vectors (n_epochs length, not n_samples)
// vs original R: allocates ~5 intermediate vectors of n_samples length
//
// Interface: accepts NumericMatrix to avoid column-copy overhead.
// Rcpp's NumericMatrix is a pointer to the original R memory, no copy.
// ---------------------------------------------------------------------------

// [[Rcpp::export]]
List enmoFusedCpp(const NumericMatrix& data,
                  int sf,
                  int epochsize,
                  bool do_enmo,
                  bool do_en,
                  bool do_enmoa,
                  bool do_mad) {
  int n = data.nrow();
  int block = sf * epochsize;
  int n_epochs = n / block;

  // Pre-allocate only what's requested
  NumericVector out_enmo, out_en, out_enmoa, out_mad;
  if (do_enmo)  out_enmo  = NumericVector(n_epochs);
  if (do_en)    out_en    = NumericVector(n_epochs);
  if (do_enmoa) out_enmoa = NumericVector(n_epochs);

  // For MAD we need epoch means of EN from pass 1
  NumericVector en_epoch_means;
  if (do_mad) en_epoch_means = NumericVector(n_epochs);

  // --- Pass 1: EN + ENMO + ENMOa ---
  int pos = 0;
  for (int i = 0; i < n_epochs; i++) {
    double sum_enmo = 0.0;
    double sum_en = 0.0;
    double sum_enmoa = 0.0;
    bool has_na = false;
    int end = pos + block;

    for (int j = pos; j < end; j++) {
      double xj = data(j, 0);
      double yj = data(j, 1);
      double zj = data(j, 2);

      if (ISNAN(xj) || ISNAN(yj) || ISNAN(zj)) {
        has_na = true;
        break;
      }

      double en = std::sqrt(xj*xj + yj*yj + zj*zj);
      sum_en += en;
      if (do_enmo) {
        double v = en - 1.0;
        sum_enmo += (v > 0.0) ? v : 0.0;
      }
      if (do_enmoa) {
        sum_enmoa += std::fabs(en - 1.0);
      }
    }

    if (has_na) {
      if (do_en)    out_en[i]    = NA_REAL;
      if (do_enmo)  out_enmo[i]  = NA_REAL;
      if (do_enmoa) out_enmoa[i] = NA_REAL;
      if (do_mad)   en_epoch_means[i] = NA_REAL;
    } else {
      double mean_en = sum_en / block;
      if (do_en)    out_en[i]    = mean_en;
      if (do_enmo)  out_enmo[i]  = sum_enmo / block;
      if (do_enmoa) out_enmoa[i] = sum_enmoa / block;
      if (do_mad)   en_epoch_means[i] = mean_en;
    }
    pos = end;
  }

  // --- Pass 2 (only if MAD): recompute EN, accumulate |EN - mean| ---
  if (do_mad) {
    out_mad = NumericVector(n_epochs);
    pos = 0;
    for (int i = 0; i < n_epochs; i++) {
      if (ISNAN(en_epoch_means[i])) {
        out_mad[i] = NA_REAL;
        pos += block;
        continue;
      }
      double sum_mad = 0.0;
      double mu = en_epoch_means[i];
      int end = pos + block;
      for (int j = pos; j < end; j++) {
        double xj = data(j, 0);
        double yj = data(j, 1);
        double zj = data(j, 2);
        double en = std::sqrt(xj*xj + yj*yj + zj*zj);
        sum_mad += std::fabs(en - mu);
      }
      out_mad[i] = sum_mad / block;
      pos = end;
    }
  }

  // Build return list
  List result;
  if (do_enmo)  result["ENMO"]  = out_enmo;
  if (do_en)    result["EN"]    = out_en;
  if (do_enmoa) result["ENMOa"] = out_enmoa;
  if (do_mad)   result["MAD"]   = out_mad;
  return result;
}
