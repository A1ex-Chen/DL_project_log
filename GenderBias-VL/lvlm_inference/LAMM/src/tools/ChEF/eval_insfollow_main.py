def main():
    model_cfg, recipe_cfg, save_dir, sample_len = load_config()
    model = get_model(model_cfg)
    scenario_cfg = recipe_cfg['scenario_cfg']
    settings = [('natural', 0), ('natural', 1), ('natural', 2), ('neutral',
        0), ('neutral', 1), ('unnatural', 0)]
    dataset_name = scenario_cfg['dataset_name']
    time = datetime.datetime.now().strftime('%Y%m%d%H%M%S')
    base_save_dir = os.path.join(save_dir, model_cfg['model_name'],
        'Instruct_follow', dataset_name, time)
    scenario_cfg['option_map'] = None
    dataset = dataset_dict[dataset_name](**scenario_cfg)
    dataset = sample_dataset(dataset, sample_len=sample_len, sample_seed=0)
    save_base_dir = os.path.join(base_save_dir, 'origin')
    os.makedirs(save_base_dir, exist_ok=True)
    with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding='utf-8'
        ) as f:
        yaml.dump(data=dict(model_cfg=model_cfg, recipe_cfg=recipe_cfg),
            stream=f, allow_unicode=True)
    print(f'Save origin results in {save_base_dir}!')
    eval_cfg = recipe_cfg['eval_cfg']
    evaluater = Evaluator(dataset, save_base_dir, eval_cfg)
    evaluater.evaluate(model)
    origin_save_base_dir = save_base_dir
    for setting in settings:
        ins_dict = {'type': setting[0], 'ids': setting[1]}
        scenario_cfg['option_map'] = ins_dict
        dataset = dataset_dict[dataset_name](**scenario_cfg)
        dataset = sample_dataset(dataset, sample_len=sample_len, sample_seed=0)
        save_base_dir = os.path.join(base_save_dir,
            f'{setting[0]}_{setting[1]}')
        os.makedirs(save_base_dir, exist_ok=True)
        with open(os.path.join(save_base_dir, 'config.yaml'), 'w', encoding
            ='utf-8') as f:
            yaml.dump(data=dict(model_cfg=model_cfg, recipe_cfg=recipe_cfg),
                stream=f, allow_unicode=True)
        print(f'Save {setting[0]}_{setting[1]} results in {save_base_dir}!')
        eval_cfg = recipe_cfg['eval_cfg']
        evaluater = Evaluator(dataset, save_base_dir, eval_cfg)
        evaluater.evaluate(model)
    with open(find_res_json(origin_save_base_dir, dataset_name), 'r') as f:
        origin_res = json.load(f)
    types_dirs = {'natural': [], 'neutral': [], 'unnatural': []}
    types_accs = {'natural': [], 'neutral': [], 'unnatural': []}
    types_mrs = {'natural': [], 'neutral': [], 'unnatural': []}
    for setting in settings:
        dir = os.path.join(base_save_dir, f'{setting[0]}_{setting[1]}')
        types_dirs[setting[0]].append(dir)
        acc_json_path = os.path.join(dir, 'results.json')
        with open(acc_json_path, 'r') as f:
            acc_data = json.load(f)
        types_accs[setting[0]].append(acc_data['result'])
        result_json_path = find_res_json(dir, dataset_name)
        with open(result_json_path, 'r') as f:
            result_data = json.load(f)
        mr = compute_MR(origin_res, result_data)
        types_mrs[setting[0]].append(mr)
        print(
            f"{setting[0]}_{setting[1]}: Acc: {acc_data['result']}, follow_MR: {mr}"
            )
    avg_acc, avg_mr = 0, 0
    for type, accs in types_accs.items():
        for acc in accs:
            avg_acc += acc
    avg_acc /= len(settings)
    for type, mrs in types_mrs.items():
        for mr in mrs:
            avg_mr += mr
    avg_mr /= len(settings)
    print(f'weighted_avg_MR: {avg_mr}, weighted_avg_Acc: {avg_acc}')
    final_res = {'res_dirs': types_dirs, 'Accs': types_accs, 'MRs':
        types_mrs, 'weighted_avg_MR': avg_mr, 'weighted_avg_Acc': avg_acc}
    with open(os.path.join(base_save_dir, 'Instruction_Follow_Results.json'
        ), 'w', encoding='utf-8') as f:
        f.write(json.dumps(final_res, indent=4))
