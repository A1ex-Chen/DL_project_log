@classmethod
def from_config(cls, cfg, from_pretrained=False):
    med_config_path = get_abs_path(cfg.get('med_config_path'))
    med_config = BertConfig.from_json_file(med_config_path)
    if from_pretrained:
        return cls.from_pretrained('bert-base-uncased', config=med_config,
            add_pooling_layer=False)
    else:
        return cls(config=med_config, add_pooling_layer=False)
