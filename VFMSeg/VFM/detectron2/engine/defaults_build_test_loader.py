@classmethod
def build_test_loader(cls, cfg, dataset_name):
    """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
    return build_detection_test_loader(cfg, dataset_name)
