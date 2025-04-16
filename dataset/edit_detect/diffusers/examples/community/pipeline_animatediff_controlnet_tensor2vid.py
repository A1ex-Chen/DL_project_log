def tensor2vid(video: torch.Tensor, processor, output_type='np'):
    batch_size, channels, num_frames, height, width = video.shape
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = processor.postprocess(batch_vid, output_type)
        outputs.append(batch_output)
    if output_type == 'np':
        outputs = np.stack(outputs)
    elif output_type == 'pt':
        outputs = torch.stack(outputs)
    elif not output_type == 'pil':
        raise ValueError(
            f"{output_type} does not exist. Please choose one of ['np', 'pt', 'pil']"
            )
    return outputs
