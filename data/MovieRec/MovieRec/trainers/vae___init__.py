def __init__(self, args, model, train_loader, val_loader, test_loader,
    export_root):
    super().__init__(args, model, train_loader, val_loader, test_loader,
        export_root)
    self.__beta = 0
    self.finding_best_beta = args.find_best_beta
    self.anneal_amount = 1.0 / args.total_anneal_steps
    if self.finding_best_beta:
        self.current_best_metric = 0.0
        self.anneal_cap = 1.0
    else:
        self.anneal_cap = args.anneal_cap
