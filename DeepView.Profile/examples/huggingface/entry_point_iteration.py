def iteration(source, label):
    trainer.training_step(model, {'input_ids': source, 'labels': label})
