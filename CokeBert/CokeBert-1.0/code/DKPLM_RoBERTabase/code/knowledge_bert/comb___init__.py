def __init__(self, config):
    super(Model, self).__init__(config)
    self.model_att, _ = BertForSequenceClassification_att.from_pretrained(args
        .ernie_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /
        'distributed_{}'.format(args.local_rank), num_labels=num_labels,
        args=args)
    self.model_ernie, _ = BertForSequenceClassification_ernie.from_pretrained(
        '../' + args.ernie_model, cache_dir=PYTORCH_PRETRAINED_BERT_CACHE /
        'distributed_{}'.format(args.local_rank), num_labels=num_labels)
