def run(self):
    if self._model_type == 'albert':
        try:
            from transformers.models.albert.convert_albert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
        except ImportError:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        convert_tf_checkpoint_to_pytorch(self._tf_checkpoint, self._config,
            self._pytorch_dump_output)
    elif self._model_type == 'bert':
        try:
            from transformers.models.bert.convert_bert_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
        except ImportError:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        convert_tf_checkpoint_to_pytorch(self._tf_checkpoint, self._config,
            self._pytorch_dump_output)
    elif self._model_type == 'funnel':
        try:
            from transformers.models.funnel.convert_funnel_original_tf_checkpoint_to_pytorch import convert_tf_checkpoint_to_pytorch
        except ImportError:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        convert_tf_checkpoint_to_pytorch(self._tf_checkpoint, self._config,
            self._pytorch_dump_output)
    elif self._model_type == 'gpt':
        from transformers.models.openai.convert_openai_original_tf_checkpoint_to_pytorch import convert_openai_checkpoint_to_pytorch
        convert_openai_checkpoint_to_pytorch(self._tf_checkpoint, self.
            _config, self._pytorch_dump_output)
    elif self._model_type == 'transfo_xl':
        try:
            from transformers.models.transfo_xl.convert_transfo_xl_original_tf_checkpoint_to_pytorch import convert_transfo_xl_checkpoint_to_pytorch
        except ImportError:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        if 'ckpt' in self._tf_checkpoint.lower():
            TF_CHECKPOINT = self._tf_checkpoint
            TF_DATASET_FILE = ''
        else:
            TF_DATASET_FILE = self._tf_checkpoint
            TF_CHECKPOINT = ''
        convert_transfo_xl_checkpoint_to_pytorch(TF_CHECKPOINT, self.
            _config, self._pytorch_dump_output, TF_DATASET_FILE)
    elif self._model_type == 'gpt2':
        try:
            from transformers.models.gpt2.convert_gpt2_original_tf_checkpoint_to_pytorch import convert_gpt2_checkpoint_to_pytorch
        except ImportError:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        convert_gpt2_checkpoint_to_pytorch(self._tf_checkpoint, self.
            _config, self._pytorch_dump_output)
    elif self._model_type == 'xlnet':
        try:
            from transformers.models.xlnet.convert_xlnet_original_tf_checkpoint_to_pytorch import convert_xlnet_checkpoint_to_pytorch
        except ImportError:
            raise ImportError(IMPORT_ERROR_MESSAGE)
        convert_xlnet_checkpoint_to_pytorch(self._tf_checkpoint, self.
            _config, self._pytorch_dump_output, self._finetuning_task_name)
    elif self._model_type == 'xlm':
        from transformers.models.xlm.convert_xlm_original_pytorch_checkpoint_to_pytorch import convert_xlm_checkpoint_to_pytorch
        convert_xlm_checkpoint_to_pytorch(self._tf_checkpoint, self.
            _pytorch_dump_output)
    elif self._model_type == 'lxmert':
        from transformers.models.lxmert.convert_lxmert_original_pytorch_checkpoint_to_pytorch import convert_lxmert_checkpoint_to_pytorch
        convert_lxmert_checkpoint_to_pytorch(self._tf_checkpoint, self.
            _pytorch_dump_output)
    else:
        raise ValueError(
            '--model_type should be selected in the list [bert, gpt, gpt2, transfo_xl, xlnet, xlm, lxmert]'
            )
