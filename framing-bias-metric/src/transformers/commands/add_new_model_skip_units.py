def skip_units(line):
    return ('generating PyTorch' in line and not output_pytorch or 
        'generating TensorFlow' in line and not output_tensorflow)
