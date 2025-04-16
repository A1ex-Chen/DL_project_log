def create_model_card(args, model_name):
    if not is_jinja_available():
        raise ValueError(
            'Modelcard rendering is based on Jinja templates. Please make sure to have `jinja` installed before using `create_model_card`. To install it, please run `pip install Jinja2`.'
            )
    if hasattr(args, 'local_rank') and args.local_rank not in [-1, 0]:
        return
    hub_token = args.hub_token if hasattr(args, 'hub_token') else None
    repo_name = get_full_repo_name(model_name, token=hub_token)
    model_card = ModelCard.from_template(card_data=ModelCardData(language=
        'en', license='apache-2.0', library_name='diffusers', tags=[],
        datasets=args.dataset_name, metrics=[]), template_path=
        MODEL_CARD_TEMPLATE_PATH, model_name=model_name, repo_name=
        repo_name, dataset_name=args.dataset_name if hasattr(args,
        'dataset_name') else None, learning_rate=args.learning_rate,
        train_batch_size=args.train_batch_size, eval_batch_size=args.
        eval_batch_size, gradient_accumulation_steps=args.
        gradient_accumulation_steps if hasattr(args,
        'gradient_accumulation_steps') else None, adam_beta1=args.
        adam_beta1 if hasattr(args, 'adam_beta1') else None, adam_beta2=
        args.adam_beta2 if hasattr(args, 'adam_beta2') else None,
        adam_weight_decay=args.adam_weight_decay if hasattr(args,
        'adam_weight_decay') else None, adam_epsilon=args.adam_epsilon if
        hasattr(args, 'adam_epsilon') else None, lr_scheduler=args.
        lr_scheduler if hasattr(args, 'lr_scheduler') else None,
        lr_warmup_steps=args.lr_warmup_steps if hasattr(args,
        'lr_warmup_steps') else None, ema_inv_gamma=args.ema_inv_gamma if
        hasattr(args, 'ema_inv_gamma') else None, ema_power=args.ema_power if
        hasattr(args, 'ema_power') else None, ema_max_decay=args.
        ema_max_decay if hasattr(args, 'ema_max_decay') else None,
        mixed_precision=args.mixed_precision)
    card_path = os.path.join(args.output_dir, 'README.md')
    model_card.save(card_path)
