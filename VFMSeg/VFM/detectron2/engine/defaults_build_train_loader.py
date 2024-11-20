@classmethod
def build_train_loader(cls, cfg):
    """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
    return build_detection_train_loader(cfg)
