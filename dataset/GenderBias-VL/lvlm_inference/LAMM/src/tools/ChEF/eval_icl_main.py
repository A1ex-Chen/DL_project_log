def main(dist_args):
    model_cfg, recipe_cfg, save_dir, sample_len, time = load_config()
    devices = get_useable_cuda()
    if model_cfg['model_name'] in ['GPT', 'Gemini']:
        model = get_model(model_cfg, device='cpu')
    else:
        model = get_model(model_cfg, device=devices[dist_args['global_rank']])
    scenario_cfg = recipe_cfg['scenario_cfg']
    dataset_name = scenario_cfg['dataset_name']
    dataset = dataset_dict[dataset_name](**scenario_cfg)
    dataset = sample_dataset(dataset, sample_len=sample_len, sample_seed=0,
        dist_args=dist_args)
    save_base_dir = os.path.join(save_dir, model_cfg['model_name'],
        dataset_name, 'ICL')
    os.makedirs(save_base_dir, exist_ok=True)
    with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8'
        ) as f:
        yaml.dump(data=dict(model_cfg=model_cfg, recipe_cfg=recipe_cfg),
            stream=f, allow_unicode=True)
    print(f'Save results in {save_base_dir}!')
    ice_strategy = ['random', 'topk_text', 'fixed', 'topk_img']
    ice_nums = [1, 2, 3]
    results = []
    for strategy in ice_strategy:
        for ice_num in ice_nums:
            eval_cfg = recipe_cfg['eval_cfg']
            eval_cfg['instruction_cfg']['incontext_cfg']['ice_num'] = ice_num
            eval_cfg['instruction_cfg']['incontext_cfg']['retriever_type'
                ] = strategy
            evaluater = Evaluator(dataset, save_base_dir, eval_cfg,
                dist_args=dist_args)
            result_path, result = evaluater.evaluate(model)
            if result_path is None:
                print('Rank!=0')
                continue
            if 'vanilla_acc' in result:
                metric_result = result['vanilla_acc']
            else:
                metric_result = result['ACC']
            results.append(dict(ice_num=ice_num, strategy=strategy,
                result_path=result_path, metric_result=metric_result))
    if len(results) > 0:
        with open(os.path.join(save_base_dir, 'results.json'), 'w',
            encoding='utf-8') as f:
            json.dump(results, f, indent=4)
    print('finish')
