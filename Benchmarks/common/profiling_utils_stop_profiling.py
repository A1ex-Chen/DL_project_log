def stop_profiling(do_prof):
    if do_prof:
        numba.cuda.profile_stop()
