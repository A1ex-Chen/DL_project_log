def serialize_config(self, model_dir: str):
    """Serializes and saves the experiment config."""
    params_save_path = os.path.join(model_dir, 'params.yaml')
    with open(params_save_path, 'w') as outfile:
        yaml.dump(vars(self.params), outfile, default_flow_style=False)
