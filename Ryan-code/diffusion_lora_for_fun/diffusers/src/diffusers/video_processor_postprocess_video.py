def postprocess_video(self, video: torch.Tensor, output_type: str='np'
    ) ->Union[np.ndarray, torch.Tensor, List[PIL.Image.Image]]:
    """
        Converts a video tensor to a list of frames for export.

        Args:
            video (`torch.Tensor`): The video as a tensor.
            output_type (`str`, defaults to `"np"`): Output type of the postprocessed `video` tensor.
        """
    batch_size = video.shape[0]
    outputs = []
    for batch_idx in range(batch_size):
        batch_vid = video[batch_idx].permute(1, 0, 2, 3)
        batch_output = self.postprocess(batch_vid, output_type)
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
