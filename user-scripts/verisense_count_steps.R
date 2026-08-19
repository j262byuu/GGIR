# This script was originally copied from
# https://github.com/ShimmerEngineering/Verisense-Toolbox/tree/master/Verisense_step_algorithm
# where it included the following software license:

# Copyright (c) 2020 Shimmer
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
#   The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.


# ---------------------------------------------------------------------------
# Vectorised 2026-08. Same signature, same coefficient order, same hardcoded fs.
# identical() on all 15 chunks of ten UK Biobank .cwa recordings, 11.9x faster
# (56.8 s -> 4.8 s, mean per captured chunk). Two R-level loops were replaced;
# nothing else moved. Two divergences on degenerate input are listed at the end.
#
# 1. The segment loop ran floor(n/k) times -- 489,375 for a 36 h chunk at 15 Hz
#    -- slicing acc twice and calling which.max twice per iteration. It becomes
#    a comparison of the k phases of acc, plus the same centred test written
#    without the slice:
#        which.max(acc[loc-half_k .. loc+half_k]) == half_k + 1
#    is exactly
#        acc[loc] >  max(acc[loc-half_k .. loc-1])   and
#        acc[loc] >= max(acc[loc+1 .. loc+half_k])
#    because which.max returns the FIRST maximum: an equal value to the left
#    wins the tie, an equal value to the right does not. That asymmetry IS the
#    test on a plateau; reversing it changes which peaks survive. Segments whose
#    window would clamp at either end of acc go through the original scalar
#    code, since with a clamped window half_k + 1 is no longer the centre.
#
# 2. The continuity loop called var() on the same peak-to-peak interval
#    cont_thres times. Each interval is computed once and the inner count
#    becomes a rolling sum over a cumsum.
#
#    var() itself is deliberately untouched. A hand-written two-pass variance is
#    3.8x faster and disagrees with var() on 102,402 of 386,949 real intervals
#    -- 26% -- by up to one ulp. No verdict against var_thres flipped on the
#    data tested, but that value is consumed by a threshold comparison, so the
#    faster form was dropped rather than argued for.
#
# Also changed: table(factor(x, levels = seq_along(v))) built a 130,500-level
# factor to count integers already in 1..length(v); tabulate() counts them
# directly and returns the same vector.
#
# Behaviour deliberately NOT changed, so that this stays a performance patch:
#   - the periodicity filters use bare logical indexing on a column whose last
#     row is NA, so they keep an all-NA row. Later cleaned up by the col-1 NA
#     filter. Dropping it early would change nothing observable and would be a
#     different function.
#   - coeffs[[5]] is read as cont_win_size and coeffs[[6]] as cont_thres, while
#     the loop bound and inner count use cont_thres and the final comparison
#     uses cont_win_size. This reads backwards against the names, and it is
#     exactly what the upstream Shimmer implementation does. With the published
#     coefficients (Rowlands et al 2022) both are 4, so it is inert. Changing it
#     here would silently diverge from the published algorithm for anyone using
#     asymmetric values.
#
# Two divergences from the original, both on degenerate input:
#
#   a. Fewer samples than one segment (n < k). The original raises "argument is
#      of length zero"; this returns numeric(0).
#   b. Exactly one segment peak surviving the centred test. The original's row
#      filter has no drop = FALSE, so peak_info collapses to a length-5 vector
#      and the next two-dimensional subscript raises "incorrect number of
#      dimensions"; this builds a 1 x 5 matrix, falls through to the no-steps
#      path and returns zeros. Reachable whenever k <= n < 2k, though on a real
#      chunk of ~1M samples "sd >= 0.025 and exactly one centred peak" is
#      essentially unreachable.
#
# Everything else agreed, including a flat signal and a 30-sample input that
# raises the same error in both.
# ---------------------------------------------------------------------------

verisense_count_steps <- function(input_data = runif(500, min = -1.5, max = 1.5),
                                  coeffs = c(0, 0, 0)) {
  # # TO BE USED WITH GGIR AS FOLLOWS:
  # 
  # source("your_file_path/verisense_count_steps.R")
  # myfun = list(FUN = verisense_count_steps,
  #              parameters = c(4, 4, 20, -1.0, 4, 4, 0.01, 1.25), # updated based on Rowlands et al Stepping up with GGIR 2022
  #              expected_sample_rate = 15,
  #              expected_unit = "g",
  #              colnames = c("step_count"),
  #              outputres = 1,
  #              minlength = 1,
  #              outputtype = "numeric",
  #              aggfunction = sum,
  #              timestamp = F,
  #              reporttype = "event")
  # 
  # GGIR(myfun = myfun, ...)
  # 
  # See also https://wadpac.github.io/GGIR/articles/ExternalFunction.html

  fs = 15 # temporary for now, this is manually set
  acc <- sqrt(input_data[, 1]^2 + input_data[, 2]^2 + input_data[, 3]^2)

  if (sd(acc) < 0.025) {
    num_seconds = round(length(acc) / fs)
    steps_per_sec = rep(0, num_seconds)
  } else {
    k <- coeffs[[1]]
    period_min <- coeffs[[2]]
    period_max <- coeffs[[3]]
    sim_thres <- coeffs[[4]]
    cont_win_size <- coeffs[[5]]
    cont_thres <- coeffs[[6]]
    var_thres <- coeffs[[7]]
    mag_thres <- coeffs[[8]]

    half_k <- round(k / 2)
    segments <- floor(length(acc) / k)

    # ---- (a) per-segment peak detection, vectorised -----------------------
    n_acc <- length(acc)
    loc_in_seg <- rep(1L, segments)
    seg_max <- acc[seq(1, by = k, length.out = segments)]
    if (k >= 2) {
      for (r in 2:k) {
        v <- acc[seq(r, by = k, length.out = segments)]
        upd <- v > seg_max          # strict: the FIRST maximum wins, as which.max does
        loc_in_seg[upd] <- r
        seg_max[upd] <- v[upd]
      }
    }
    loc <- (seq_len(segments) - 1L) * k + loc_in_seg

    interior <- (loc - half_k) >= 1L & (loc + half_k) <= n_acc
    keep <- logical(segments)
    if (any(interior)) {
      li <- loc[interior]
      leftmax <- rep(-Inf, length(li)); rightmax <- rep(-Inf, length(li))
      for (s in seq_len(half_k)) {
        leftmax  <- pmax(leftmax,  acc[li - s])
        rightmax <- pmax(rightmax, acc[li + s])
      }
      keep[interior] <- acc[li] > leftmax & acc[li] >= rightmax
    }
    for (i in which(!interior)) {                    # original code, verbatim
      tmp_loc_b <- loc[i]
      s_ctr <- tmp_loc_b - half_k; if (s_ctr < 1) s_ctr <- 1
      e_ctr <- tmp_loc_b + half_k; if (e_ctr > n_acc) e_ctr <- n_acc
      keep[i] <- which.max(acc[s_ctr:e_ctr]) == (half_k + 1)
    }

    peak_info <- matrix(NA, nrow = sum(keep), ncol = 5)
    peak_info[, 1] <- loc[keep]
    peak_info[, 2] <- seg_max[keep]

    # ---- unchanged from here, including the NA-keeping filters ------------
    peak_info <- peak_info[peak_info[, 2] > mag_thres, ]
    if (length(peak_info) > 10) {
      num_peaks <- length(peak_info[, 1])
      no_steps = FALSE
      if (num_peaks > 2) {
        peak_info[1:(num_peaks - 1), 3] <- diff(peak_info[, 1])
        peak_info <- peak_info[peak_info[, 3] > period_min, ]
        peak_info <- peak_info[peak_info[, 3] < period_max, ]
      } else {
        no_steps = TRUE
      }
    } else {
      no_steps = TRUE
    }

    if (length(peak_info) == 0 ||
        length(peak_info) == sum(is.na(peak_info)) || no_steps == TRUE) {
      num_seconds = round(length(acc) / fs)
      steps_per_sec = rep(0, num_seconds)
    } else {
      num_peaks <- length(peak_info[, 1])
      peak_info[1:(num_peaks - 2), 4] <- -abs(diff(peak_info[, 2], 2))
      peak_info <- peak_info[peak_info[, 4] > sim_thres, , drop = FALSE]
      peak_info <- peak_info[is.na(peak_info[, 1]) != TRUE, , drop = FALSE]

      # ---- (b) continuity, each interval variance computed once -----------
      np <- length(peak_info[, 3])
      if (np > 5) {
        end_for <- np - 1
        idx <- cont_thres:end_for
        if (min(idx) - cont_thres + 1L >= 1L && all(idx >= 1L)) {
          p <- peak_info[, 1]
          v <- vapply(seq_len(np - 1L),
                      function(j) var(acc[p[j]:p[j + 1L]]), numeric(1))
          cs <- c(0L, cumsum(as.integer(v > var_thres)))
          v_count <- cs[idx + 1L] - cs[idx - cont_thres + 1L]
          peak_info[idx, 5] <- as.numeric(v_count >= cont_win_size)
        } else {
          # Pathological coefficients make cont_thres:end_for descend or index
          # out of range. Run the original loop so the behaviour -- including
          # its error -- is whatever it always was.
          for (i in idx) {
            v_count <- 0
            for (x in 1:cont_thres) {
              if (var(acc[peak_info[i - x + 1, 1]:peak_info[i - x + 2, 1]]) > var_thres) {
                v_count = v_count + 1
              }
            }
            peak_info[i, 5] <- if (v_count >= cont_win_size) 1 else 0
          }
        }
      }
      peak_info <- peak_info[peak_info[, 5] == 1, 1]
      peak_info <- peak_info[is.na(peak_info) != TRUE]

      if (length(peak_info) == 0) {
        num_seconds = round(length(acc) / fs)
        steps_per_sec = rep(0, num_seconds)
      } else {
        start_idx_vec <- seq(from = 1, to = length(acc), by = fs)
        steps_per_sec <- as.numeric(tabulate(findInterval(peak_info, start_idx_vec),
                                             nbins = length(start_idx_vec)))
      }
    }
  }
  return(steps_per_sec)
}
