def load_checkpoint_to_model(checkpoint, model):
    with tempfile.NamedTemporaryFile(delete=False) as file:
        torch.save(checkpoint, file.name)
        del checkpoint
        model.load_state_dict(torch.load(file.name), strict=False)
    os.remove(file.name)
