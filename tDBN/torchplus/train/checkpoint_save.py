def save(model_dir, model, model_name, global_step, max_to_keep=8,
    keep_latest=True):
    """save a model into model_dir.
    Args:
        model_dir: string, indicate your model dir(save ckpts, summarys,
            logs, etc).
        model: torch.nn.Module instance.
        model_name: name of your model. we find ckpts by name
        global_step: int, indicate current global step.
        max_to_keep: int, maximum checkpoints to keep.
        keep_latest: bool, if True and there are too much ckpts,
            will delete oldest ckpt. else will delete ckpt which has
            smallest global step.
    Returns:
        path: None if isn't exist or latest checkpoint path.
    """
    with DelayedKeyboardInterrupt():
        ckpt_info_path = Path(model_dir) / 'checkpoints.json'
        ckpt_filename = '{}-{}.tckpt'.format(model_name, global_step)
        ckpt_path = Path(model_dir) / ckpt_filename
        if not ckpt_info_path.is_file():
            ckpt_info_dict = {'latest_ckpt': {}, 'all_ckpts': {}}
        else:
            with open(ckpt_info_path, 'r') as f:
                ckpt_info_dict = json.loads(f.read())
        ckpt_info_dict['latest_ckpt'][model_name] = ckpt_filename
        if model_name in ckpt_info_dict['all_ckpts']:
            ckpt_info_dict['all_ckpts'][model_name].append(ckpt_filename)
        else:
            ckpt_info_dict['all_ckpts'][model_name] = [ckpt_filename]
        all_ckpts = ckpt_info_dict['all_ckpts'][model_name]
        torch.save(model.state_dict(), ckpt_path)
        all_ckpts_checked = []
        for ckpt in all_ckpts:
            ckpt_path_uncheck = Path(model_dir) / ckpt
            if ckpt_path_uncheck.is_file():
                all_ckpts_checked.append(str(ckpt_path_uncheck))
        all_ckpts = all_ckpts_checked
        if len(all_ckpts) > max_to_keep:
            if keep_latest:
                ckpt_to_delete = all_ckpts.pop(0)
            else:
                get_step = lambda name: int(name.split('.')[0].split('-')[1])
                min_step = min([get_step(name) for name in all_ckpts])
                ckpt_to_delete = '{}-{}.tckpt'.format(model_name, min_step)
                all_ckpts.remove(ckpt_to_delete)
            os.remove(str(Path(model_dir) / ckpt_to_delete))
        all_ckpts_filename = _ordered_unique([Path(f).name for f in all_ckpts])
        ckpt_info_dict['all_ckpts'][model_name] = all_ckpts_filename
        with open(ckpt_info_path, 'w') as f:
            f.write(json.dumps(ckpt_info_dict, indent=2))
