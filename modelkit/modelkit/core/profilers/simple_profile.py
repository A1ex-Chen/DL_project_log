@contextmanager
def profile(self, model_name: str) ->Generator:
    if model_name == self.main_model_name:
        self.start_time = time.perf_counter()
    previous_calls = self._get_current_sub_calls(model_name)
    try:
        self.start(model_name)
        yield model_name
    finally:
        sub_calls = self._compute_sub_calls_and_update_graph_calls(model_name,
            previous_calls)
        self.end(model_name, sub_calls)
        if model_name == self.main_model_name:
            self.total_duration = time.perf_counter() - self.start_time
