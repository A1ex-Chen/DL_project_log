def build_optimizer(opt, model):
    no_decay = ['bias', 'LayerNorm.weight']
    optimizer_grouped_parameters = [{'params': [p for n, p in model.
        named_parameters() if not any(nd in n for nd in no_decay)],
        'weight_decay': opt.weight_decay}, {'params': [p for n, p in model.
        named_parameters() if any(nd in n for nd in no_decay)],
        'weight_decay': 0.0}]
    assert opt.num_training_steps > 0
    optimizer = AdamW(optimizer_grouped_parameters, lr=opt.learning_rate,
        eps=opt.adam_epsilon)
    scheduler = get_constant_schedule(optimizer)
    return AdamWOpt(optimizer, scheduler)
