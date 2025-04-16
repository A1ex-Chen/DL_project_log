def generate_zoo_checkpoint_name(model, test_dataloader, pth_filename,
    model_name, dataset_name, metric_key='acc', ndigits=4):
    ckpt_hash = get_file_hash(pth_filename)
    model.load_state_dict(torch.load(pth_filename), strict=True)
    eval_fn = deeplite_torch_zoo.get_eval_function(model_name=model_name,
        dataset_name=dataset_name)
    metric_val = eval_fn(model, test_dataloader, progressbar=True)[metric_key]
    if isinstance(metric_val, torch.Tensor):
        metric_val = metric_val.item()
    metric_str = str(metric_val).lstrip('0').replace('.', '')[:ndigits]
    checkpoint_name = (
        f'{model_name}_{dataset_name}_{metric_str}_{ckpt_hash}.pt')
    return checkpoint_name
