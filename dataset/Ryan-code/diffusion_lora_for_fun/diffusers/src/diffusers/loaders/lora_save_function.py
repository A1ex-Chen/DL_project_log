def save_function(weights, filename):
    return safetensors.torch.save_file(weights, filename, metadata={
        'format': 'pt'})
