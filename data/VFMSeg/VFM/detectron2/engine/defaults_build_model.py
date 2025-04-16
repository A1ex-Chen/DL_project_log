@classmethod
def build_model(cls, cfg):
    """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
    model = build_model(cfg)
    logger = logging.getLogger(__name__)
    logger.info('Model:\n{}'.format(model))
    return model
