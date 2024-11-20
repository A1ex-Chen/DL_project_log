@staticmethod
def _validate_config_for_finetuning(config: Config, checkpoint: Checkpoint):
    assert config.pretrained == checkpoint.model.algorithm.pretrained, 'config.pretrained is inconsistent with fine-tuning one: {} vs {}'.format(
        config.pretrained, checkpoint.model.algorithm.pretrained)
