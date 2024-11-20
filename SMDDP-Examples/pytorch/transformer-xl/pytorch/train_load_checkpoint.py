def load_checkpoint(path):
    if os.path.isdir(path):
        path = os.path.join(path, 'checkpoint_last.pt')
    dst = f'cuda:{torch.cuda.current_device()}'
    logging.info(f'Loading checkpoint from {path}')
    checkpoint = torch.load(path, map_location=dst)
    return checkpoint
