def save_pretrained(self, folder):
    folder_abs = os.path.abspath(folder)
    if not os.path.exists(folder_abs):
        os.mkdir(folder_abs)
    with open(os.path.join(folder_abs, f'{self._config.model_type}.flax', 'wb')
        ) as f:
        model_bytes = to_bytes(self.params)
        f.write(model_bytes)
