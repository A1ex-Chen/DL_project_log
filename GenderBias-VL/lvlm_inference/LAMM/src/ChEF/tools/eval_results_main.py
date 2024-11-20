def main():
    args = parse_args()
    base_path = args.result_path
    if base_path.endswith('.json'):
        base_path = os.path.dirname(base_path)
    cfg_path = os.path.join(base_path, 'config.yaml')
    yaml_dict = load_yaml(cfg_path)
    recipe_cfg = yaml_dict['recipe_cfg']
    scenario_cfg = recipe_cfg['scenario_cfg']
    dataset_name = scenario_cfg['dataset_name']
    metric_func = build_metric(dataset_name=dataset_name, **recipe_cfg[
        'eval_cfg']['metric_cfg'])
    result_path = None
    for filename in os.listdir(base_path):
        if filename.endswith('.json') and not filename.startswith('result'):
            result_path = os.path.join(base_path, filename)
            break
    assert result_path is not None, 'No result file!'
    result = metric_func.metric(result_path)
    with open(os.path.join(base_path, 'results.json'), 'w', encoding='utf-8'
        ) as f:
        f.write(json.dumps(dict(answer_path=result_path, result=result),
            indent=4))
