def deepview_iteration_provider(model):
    model.parameters()
    optimizer = optim.AdamW(params=model.parameters(), betas=(0.9, 0.999),
        eps=1e-06, weight_decay=0.01, lr=0.0001)
    scheduler = get_linear_schedule_with_warmup(optimizer, 10000, 500000)
    trainer = Trainer(model=model, optimizers=(optimizer, scheduler))

    def iteration(source, label):
        trainer.training_step(model, {'input_ids': source, 'labels': label})
    return iteration
