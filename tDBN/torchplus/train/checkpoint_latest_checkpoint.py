def latest_checkpoint(model_dir, model_name):
    """return path of latest checkpoint in a model_dir
    Args:
        model_dir: string, indicate your model dir(save ckpts, summarys,
            logs, etc).
        model_name: name of your model. we find ckpts by name
    Returns:
        path: None if isn't exist or latest checkpoint path.
    """
    ckpt_info_path = Path(model_dir) / 'checkpoints.json'
    if not ckpt_info_path.is_file():
        return None
    with open(ckpt_info_path, 'r') as f:
        ckpt_dict = json.loads(f.read())
    if model_name not in ckpt_dict['latest_ckpt']:
        return None
    latest_ckpt = ckpt_dict['latest_ckpt'][model_name]
    ckpt_file_name = Path(model_dir) / latest_ckpt
    if not ckpt_file_name.is_file():
        return None
    return str(ckpt_file_name)
