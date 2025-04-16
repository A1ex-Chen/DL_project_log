def info(self, detailed=False, verbose=True):
    """
        Logs model info.

        Args:
            detailed (bool): Show detailed information about model.
            verbose (bool): Controls verbosity.
        """
    return model_info(self.model, detailed=detailed, verbose=verbose, imgsz=640
        )
