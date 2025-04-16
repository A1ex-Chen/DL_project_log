@staticmethod
def _validate_config_for_finetuning(config: Config, checkpoint: Checkpoint):
    assert config.anchor_sizes == checkpoint.model.algorithm.anchor_sizes, 'config.anchor_sizes is inconsistent with fine-tuning one: {} vs {}'.format(
        config.anchor_sizes, checkpoint.model.algorithm.anchor_sizes)
    assert config.anchor_ratios == checkpoint.model.algorithm.anchor_ratios, 'config.anchor_ratios is inconsistent with fine-tuning one: {} vs {}'.format(
        config.anchor_ratios, checkpoint.model.algorithm.anchor_ratios)
    assert config.backbone_pretrained == checkpoint.model.algorithm.backbone.pretrained, 'config.backbone_pretrained is inconsistent with fine-tuning one: {} vs {}'.format(
        config.backbone_pretrained, checkpoint.model.algorithm.backbone.
        pretrained)
