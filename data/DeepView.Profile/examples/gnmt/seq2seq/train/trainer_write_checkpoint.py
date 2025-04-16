def write_checkpoint(state, filename):
    filename = os.path.join(self.save_path, filename)
    logging.info(f'Saving model to {filename}')
    torch.save(state, filename)
