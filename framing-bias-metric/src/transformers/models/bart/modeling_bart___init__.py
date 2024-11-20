def __init__(self, config: BartConfig):
    super().__init__(config)
    base_model = BartModel(config)
    self.model = base_model
    self.extra_task_num_labels = 2
    self.classification_head = BartClassificationHead(config.d_model,
        config.d_model, self.extra_task_num_labels, config.classifier_dropout)
    self.model._init_weights(self.classification_head.dense)
    self.model._init_weights(self.classification_head.out_proj)
    self.register_buffer('final_logits_bias', torch.zeros((1, self.model.
        shared.num_embeddings)))
