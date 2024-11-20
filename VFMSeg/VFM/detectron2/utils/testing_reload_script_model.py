def reload_script_model(module):
    """
    Save a jit module and load it back.
    Similar to the `getExportImportCopy` function in torch/testing/
    """
    buffer = io.BytesIO()
    torch.jit.save(module, buffer)
    buffer.seek(0)
    return torch.jit.load(buffer)
