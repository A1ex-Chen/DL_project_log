def on_pretrain_routine_end(trainer):
    """Callback."""
    if RANK in {-1, 0}:
        names = [name.split('/')[0] for name in list(trainer.test_loader.
            dataset.data['names'].values())]
        de_parallel(trainer.ema.ema).set_classes(names, cache_clip_model=False)
    device = next(trainer.model.parameters()).device
    trainer.text_model, _ = trainer.clip.load('ViT-B/32', device=device)
    for p in trainer.text_model.parameters():
        p.requires_grad_(False)
