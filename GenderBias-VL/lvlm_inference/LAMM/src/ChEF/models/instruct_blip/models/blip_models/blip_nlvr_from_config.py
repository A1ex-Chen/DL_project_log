@classmethod
def from_config(cls, cfg=None):
    image_encoder = VisionTransformerEncoder.from_config(cfg)
    bert_config = BertConfig.from_json_file(get_abs_path(cfg[
        'med_config_path']))
    text_encoder = BertModel(config=bert_config, add_pooling_layer=False)
    num_classes = cfg.get('num_classes', 3)
    assert num_classes > 1, 'Invalid number of classes provided, found {}'.format(
        num_classes)
    model = cls(image_encoder=image_encoder, text_encoder=text_encoder,
        num_classes=num_classes)
    model.load_checkpoint_from_config(cfg)
    return model
