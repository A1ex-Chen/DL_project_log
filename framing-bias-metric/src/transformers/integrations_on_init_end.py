def on_init_end(self, args, state, control, **kwargs):
    if self.azureml_run is None and state.is_world_process_zero:
        self.azureml_run = Run.get_context()
