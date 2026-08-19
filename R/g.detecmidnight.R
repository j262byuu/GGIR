g.detecmidnight = function(time, desiredtz, dayborder) {
  # code in this function is able to deal with two types of timestamp format
  convert2clock = function(x) { #ISO format
    return(format(as.POSIXlt(x, format = "%Y-%m-%dT%H:%M:%S%z", tz = desiredtz), "%H:%M:%S"))
  }
  convert2clock_space = function(x) { #POSIX format
    return(unlist(strsplit(x, " "))[2])
  }
  checkmidnight = function(x){
    temp1 = as.numeric(unlist(strsplit(x, ":")))
    return(sum(temp1 / c(1, 60, 3600)))
  }
  # Vectorised form of checkmidnight over a whole vector of "HH:MM:SS" strings.
  # checkmidnight is otherwise called once per timestamp -- about 121k times for
  # a week at 5 s epochs -- and each call does its own strsplit and coercion.
  #
  # rowSums, not h + m/60 + s/3600: sum() accumulates in long double and so does
  # rowSums(), which makes them agree exactly, whereas the plain expression
  # differs on 20510 of the 86400 seconds in a day (by up to 3.6e-15).
  #
  # Only used when every element is exactly "HH:MM:SS". Fixed-position substr
  # would silently drop a fractional second, which strsplit(":") would have kept.
  # names() is carried over because sapply() names its result in the branch
  # below, and those names travel through which() into the returned midnightsi.
  checkmidnight_vec = function(x) {
    hh = as.numeric(substr(x, 1, 2))
    mm = as.numeric(substr(x, 4, 5))
    ss = as.numeric(substr(x, 7, 8))
    if (anyNA(hh) || anyNA(mm) || anyNA(ss)) return(NULL) # not really HMS, caller falls back
    out = rowSums(cbind(hh, mm / 60, ss / 3600))
    names(out) = names(x)
    return(out)
  }
  # Length alone is not enough: "00-00-00" is also 8 characters, and the colon
  # parser would have produced NA for it where fixed-position substr yields a
  # valid-looking 0. Require the separators too, and checkmidnight_vec returns
  # NULL if the fields still fail to convert.
  fixed_hms = function(x) {
    !anyNA(x) && all(nchar(x) == 8L) &&
      all(substr(x, 3, 3) == ":") && all(substr(x, 6, 6) == ":")
  }
  space = ifelse(length(unlist(strsplit(time[1], " "))) > 1,TRUE,FALSE)
  if (space == TRUE) {
    # sapply(time, convert2clock_space) runs a strsplit per timestamp. When every
    # timestamp has its space in the same position -- "YYYY-MM-DD HH:MM:SS" does --
    # the clock part is a fixed-position substring. sapply() names its result
    # after `time`, and those names reach the returned midnightsi, so they are
    # reproduced explicitly here.
    sp1 = regexpr(" ", time, fixed = TRUE)
    tail_after_first = if (anyNA(time) || any(sp1 < 1)) NULL else substring(time, sp1 + 1L)
    # convert2clock_space takes the SECOND space-delimited token, so the fast path
    # is only equivalent when nothing follows it -- "2024-01-01 00:00:00 UTC"
    # would otherwise keep the " UTC".
    if (!is.null(tail_after_first) && all(regexpr(" ", tail_after_first, fixed = TRUE) == -1L)) {
      time_clock = tail_after_first
      # sapply() names its result from names(time) when present, and from the
      # values otherwise; reproduce both cases rather than assuming unnamed input.
      names(time_clock) = if (!is.null(names(time))) names(time) else time
    } else {
      time_clock = sapply(time, FUN = convert2clock_space)
    }
    checkmidnight_out = if (fixed_hms(time_clock)) checkmidnight_vec(time_clock) else NULL
    if (is.null(checkmidnight_out)) {
      checkmidnight_out = lapply(time_clock, FUN = checkmidnight)
    }
  } else {
    time_clock = convert2clock(time)
    checkmidnight_out = if (fixed_hms(time_clock)) checkmidnight_vec(time_clock) else NULL
    if (is.null(checkmidnight_out)) {
      checkmidnight_out = unlist(lapply(time_clock, FUN = checkmidnight))
    }
  }
  midn = which(checkmidnight_out == dayborder) #replaced "== 0" for "== dayborder" to work in any scenario
  if (length(midn) == 0) { # measurement with no midnights, use last timestamp as dummy midnight for g.analyse() to work
    midnights = time[length(time)]
    midnightsi = length(time)
  } else {
    midnights = format(time[midn])
    midnightsi = midn
  }
  if (length(midn) == 0) { # measurement with no midnights, use last timestamp as dummy midnight for g.analyse() to work
    lastmidnight = time[length(time)]
    lastmidnighti = length(time)
    firstmidnight = time[1]
    firstmidnighti = 1
  } else {
    lastmidnight = midnights[length(midnights)]
    lastmidnighti = midnightsi[length(midnights)]
    firstmidnight = midnights[1]
    firstmidnighti = midnightsi[1]
  }
  invisible(list(firstmidnight = firstmidnight, firstmidnighti = firstmidnighti,
                 lastmidnight = lastmidnight, lastmidnighti = lastmidnighti,
                 midnights = midnights, midnightsi = midnightsi))
}
