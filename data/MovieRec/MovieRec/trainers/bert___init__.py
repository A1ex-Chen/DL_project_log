def __init__(self, args, model, train_loader, val_loader, test_loader,
    export_root):
    super().__init__(args, model, train_loader, val_loader, test_loader,
        export_root)
    self.ce = nn.CrossEntropyLoss(ignore_index=0)
