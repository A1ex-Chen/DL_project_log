def train():
    export_root = setup_train(args)
    train_loader, val_loader, test_loader = dataloader_factory(args)
    model = model_factory(args)
    trainer = trainer_factory(args, model, train_loader, val_loader,
        test_loader, export_root)
    trainer.train()
    test_model = input('Test model with test dataset? y/[n]: ') == 'y'
    if test_model:
        trainer.test()
