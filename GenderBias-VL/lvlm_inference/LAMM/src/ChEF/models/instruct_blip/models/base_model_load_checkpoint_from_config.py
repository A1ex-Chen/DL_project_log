def load_checkpoint_from_config(self, cfg, **kwargs):
    """
        Load checkpoint as specified in the config file.

        If load_finetuned is True, load the finetuned model; otherwise, load the pretrained model.
        When loading the pretrained model, each task-specific architecture may define their
        own load_from_pretrained() method.
        """
    load_finetuned = cfg.get('load_finetuned', True)
    if load_finetuned:
        finetune_path = cfg.get('finetuned', None)
        assert finetune_path is not None, 'Found load_finetuned is True, but finetune_path is None.'
        self.load_checkpoint(url_or_filename=finetune_path)
    else:
        load_pretrained = cfg.get('load_pretrained', True)
        if load_pretrained:
            pretrain_path = cfg.get('pretrained', None)
            assert 'Found load_finetuned is False, but pretrain_path is None.'
            self.load_from_pretrained(url_or_filename=pretrain_path, **kwargs)
