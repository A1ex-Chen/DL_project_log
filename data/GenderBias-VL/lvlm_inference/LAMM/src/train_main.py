def main(**args):
    start_time = time.time()
    config_env(args)
    build_directory(args['save_path'])
    build_directory(args['log_path'])
    with open(os.path.join(args['log_path'], 'training_args.json'), 'w') as fw:
        json.dump(args, fw, indent=4)
    dschf = HfDeepSpeedConfig(args['deepspeed'])
    args['dschf'] = dschf
    if args['log_path']:
        logging.basicConfig(format=
            '%(asctime)s - %(pathname)s[line:%(lineno)d] - %(levelname)s: %(message)s'
            , level=logging.DEBUG, filename=
            f"{args['log_path']}/train_{time.asctime()}.log", filemode='w')
    if args['use_flash_attn']:
        from model.LAMM.flash_attn_patch import replace_llama_attn_with_flash_attn
        logging.info('⚡⚡⚡ enable flash attention.')
        replace_llama_attn_with_flash_attn()
    if args['use_xformers']:
        from model.LAMM.xformers_patch import replace_llama_attn_with_xformers_attn
        logging.info('xxx enable xformers attention.')
        replace_llama_attn_with_xformers_attn()
    train_data, train_iter, sampler = load_dataset(args)
    length = args['epochs'] * len(train_data) // args['world_size'
        ] // dschf.config['train_micro_batch_size_per_gpu']
    total_steps = args['epochs'] * len(train_data) // dschf.config[
        'train_batch_size']
    args['total_steps'] = total_steps
    agent = load_model(args)
    torch.distributed.barrier()
    with open(os.path.join(args['log_path'], 'training_args.yaml'), 'w') as fw:
        yaml.dump(args, fw)
    pbar = tqdm(total=length)
    current_step = 0
    for epoch_i in tqdm(range(args['epochs'])):
        for batch in train_iter:
            agent.train_model(batch, current_step=current_step, pbar=pbar)
            current_step += 1
        if epoch_i % max(args['epochs'] // 5, 1) == 0:
            agent.save_model(args['save_path'], epoch_i + 1)
    torch.distributed.barrier()
    agent.save_model(args['save_path'], 0)
    print(f'Done! Total Training time: {time.time() - start_time}')
