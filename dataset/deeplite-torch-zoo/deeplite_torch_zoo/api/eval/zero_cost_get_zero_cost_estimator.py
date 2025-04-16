def get_zero_cost_estimator(metric_name: str):
    compute_zc_score_fn = ZERO_COST_SCORES.get(metric_name)

    def compute_zc_score_wrapper(model, dataloader=None,
        model_output_generator=None, loss_fn=None, inplace=False,
        do_gaussian_init=False, eval_mode=False, fuse=False, **kwargs):
        if dataloader is not None and model_output_generator is not None:
            raise ValueError(
                'Zero-cost estimator computation requires either a `dataloader` or a `model_output_generator` argument not equal to None, not both at the same time. In case when `dataloader` is passed, a standard interface to compute model output is assumed.'
                )
        model_ = model
        if not inplace:
            model_ = deepcopy(model)
        if model_output_generator is None:
            device = next(model_.parameters()).device
            model_output_generator = partial(default_model_output_generator,
                dataloader=dataloader, device=device)
        model_.train(not eval_mode)
        model_.zero_grad()
        if fuse and hasattr(model, 'fuse'):
            model_ = model_.fuse()
        if do_gaussian_init:
            weight_gaussian_init(model_)
        return compute_zc_score_fn(model_, model_output_generator, loss_fn,
            **kwargs)
    return compute_zc_score_wrapper
