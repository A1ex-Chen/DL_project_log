def measure_run_time_ms_catch_oom(self, batch_size, initial_repetitions=None):
    release_memory()
    try:
        return None, self.measure_run_time_ms(batch_size, initial_repetitions)
    except AnalysisError as ex:
        message = str(ex)
        if 'CUDA out of memory' in message:
            return ex, None
        else:
            raise
