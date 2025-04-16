def save_model_hook(models, weights, output_dir):
    if args.use_ema:
        ema_prior.save_pretrained(os.path.join(output_dir, 'prior_ema'))
    for i, model in enumerate(models):
        model.save_pretrained(os.path.join(output_dir, 'prior'))
        weights.pop()
