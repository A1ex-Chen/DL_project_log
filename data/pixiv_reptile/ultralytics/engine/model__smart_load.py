def _smart_load(self, key: str):
    """Load model/trainer/validator/predictor."""
    try:
        return self.task_map[self.task][key]
    except Exception as e:
        name = self.__class__.__name__
        mode = inspect.stack()[1][3]
        raise NotImplementedError(emojis(
            f"WARNING ⚠️ '{name}' model does not support '{mode}' mode for '{self.task}' task yet."
            )) from e
