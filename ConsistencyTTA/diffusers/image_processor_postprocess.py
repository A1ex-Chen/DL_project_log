def postprocess(self, image, output_type: str='pil'):
    if isinstance(image, torch.Tensor) and output_type == 'pt':
        return image
    image = self.pt_to_numpy(image)
    if output_type == 'np':
        return image
    elif output_type == 'pil':
        return self.numpy_to_pil(image)
    else:
        raise ValueError(f'Unsupported output_type {output_type}.')
