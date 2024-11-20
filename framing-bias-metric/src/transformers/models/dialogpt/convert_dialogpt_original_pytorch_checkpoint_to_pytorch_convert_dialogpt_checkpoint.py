def convert_dialogpt_checkpoint(checkpoint_path: str,
    pytorch_dump_folder_path: str):
    d = torch.load(checkpoint_path)
    d[NEW_KEY] = d.pop(OLD_KEY)
    os.makedirs(pytorch_dump_folder_path, exist_ok=True)
    torch.save(d, os.path.join(pytorch_dump_folder_path, WEIGHTS_NAME))
