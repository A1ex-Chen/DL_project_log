def get_optimizer(self, args, cfg, model):
    accumulate = max(1, round(64 / args.batch_size))
    cfg.solver.weight_decay *= args.batch_size * accumulate / 64
    cfg.solver.lr0 *= args.batch_size / (self.world_size * args.bs_per_gpu)
    optimizer = build_optimizer(cfg, model)
    return optimizer
