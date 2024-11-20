def main():
    model_cfg, recipe_cfg, save_dir, sample_len = load_config()
    model = get_model(model_cfg)
    scenario_cfg = recipe_cfg['scenario_cfg']
    dataset_name = scenario_cfg['dataset_name']
    dataset = dataset_dict[dataset_name](**scenario_cfg)
    dataset = sample_dataset(dataset, sample_len=sample_len, sample_seed=0)
    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    save_base_dir = os.path.join(save_dir, model_cfg['model_name'],
        'Calibration', dataset_name, time)
    os.makedirs(save_base_dir, exist_ok=True)
    with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8'
        ) as f:
        yaml.dump(data=dict(model_cfg=model_cfg, recipe_cfg=recipe_cfg),
            stream=f, allow_unicode=True)
    print(f'Save results in {save_base_dir}!')
    eval_cfg = recipe_cfg['eval_cfg']
    evaluater = Evaluator(dataset, save_base_dir, eval_cfg)
    evaluater.evaluate(model)
