makeSingleThreadedCluster = function(Ncores2use, ...) {
  # GGIR already parallelises over recordings. When each worker then starts a
  # thread pool of its own the machine is asked for Nworkers x Nthreads threads,
  # which oversubscribes the cores the process was actually granted on a shared
  # or scheduler-managed node (wadpac/GGIR#1442).
  #
  # The variables have to be set in this process, before the workers exist. A
  # PSOCK worker inherits the parent environment when it starts, and a threaded
  # BLAS reads its thread-count variable while initialising, which is earlier
  # than any clusterEvalQ() could reach it. Setting them inside a running worker
  # is therefore too late for BLAS, even though it works for data.table.
  #
  # Which variable is honoured depends on the build, so set all of them rather
  # than assume one: OpenBLAS consults OPENBLAS_NUM_THREADS, then
  # GOTO_NUM_THREADS, then OMP_NUM_THREADS; MKL consults MKL_NUM_THREADS; a
  # plain OpenMP runtime consults OMP_NUM_THREADS.
  threadvars = c("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS",
                 "GOTO_NUM_THREADS", "MKL_NUM_THREADS")
  previous = Sys.getenv(threadvars, names = TRUE, unset = NA)
  on.exit({
    # Put this process back the way it was. The workers keep the capped values
    # they inherited at startup; only the parent is restored, so a user calling
    # GGIR from an interactive session does not silently lose their own setting.
    keep = previous[!is.na(previous)]
    if (length(keep) > 0) do.call(Sys.setenv, as.list(keep))
    drop = names(previous)[is.na(previous)]
    if (length(drop) > 0) Sys.unsetenv(drop)
  }, add = TRUE)
  capped = as.list(rep("1", length(threadvars)))
  names(capped) = threadvars
  do.call(Sys.setenv, capped)
  cl = parallel::makeCluster(Ncores2use, ...)
  # data.table caches its thread count when the package loads, so the inherited
  # environment variable is not enough if it was already loaded in the worker.
  # It exposes a setter, so cap it explicitly as well.
  parallel::clusterEvalQ(cl, data.table::setDTthreads(1L))
  return(cl)
}
