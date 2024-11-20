def _forward():
    with amp_autocast():
        output = model(input)
        loss = loss_fn(output, target)
        if model_kd is not None:
            if not args.use_kd_loss_only:
                loss += args.alpha_kd * compute_kd_loss(input, output,
                    model, model_kd)
            else:
                loss = compute_kd_loss(input, output, model, model_kd)
    if accum_steps > 1:
        loss /= accum_steps
    return loss
