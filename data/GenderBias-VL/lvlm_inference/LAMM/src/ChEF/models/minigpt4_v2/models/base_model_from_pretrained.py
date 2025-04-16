@classmethod
def from_pretrained(cls, model_type):
    """
        Build a pretrained model from default configuration file, specified by model_type.

        Args:
            - model_type (str): model type, specifying architecture and checkpoints.

        Returns:
            - model (nn.Module): pretrained or finetuned model, depending on the configuration.
        """
    model_cfg = OmegaConf.load(cls.default_config_path(model_type)).model
    model = cls.from_config(model_cfg)
    return model
