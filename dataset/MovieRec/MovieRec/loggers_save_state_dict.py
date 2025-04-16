def save_state_dict(state_dict, path, filename):
    torch.save(state_dict, os.path.join(path, filename))
