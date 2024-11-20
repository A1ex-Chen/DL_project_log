def postprocess(self, sample: torch.Tensor, output_type: str='pil'):
    if output_type not in ['pt', 'np', 'pil']:
        raise ValueError(
            f"output_type={output_type} is not supported. Make sure to choose one of ['pt', 'np', or 'pil']"
            )
    sample = (sample / 2 + 0.5).clamp(0, 1)
    if output_type == 'pt':
        return sample
    sample = sample.cpu().permute(0, 2, 3, 1).numpy()
    if output_type == 'np':
        return sample
    sample = numpy_to_pil(sample)
    return sample
