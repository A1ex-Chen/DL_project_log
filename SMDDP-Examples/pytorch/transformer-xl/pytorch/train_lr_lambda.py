def lr_lambda(step):
    if step == 0 and args.warmup_step == 0:
        return 1.0
    else:
        return (1.0 / step ** 0.5 if step > args.warmup_step else step / 
            args.warmup_step ** 1.5)
