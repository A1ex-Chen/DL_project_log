def forward(model, generator, return_input=False, return_target=False):
    """Forward data to a model.

    Args:
      model: object
      generator: object
      return_input: bool
      return_target: bool

    Returns:
      audio_name: (audios_num,)
      clipwise_output: (audios_num, classes_num)
      (ifexist) segmentwise_output: (audios_num, segments_num, classes_num)
      (ifexist) framewise_output: (audios_num, frames_num, classes_num)
      (optional) return_input: (audios_num, segment_samples)
      (optional) return_target: (audios_num, classes_num)
    """
    output_dict = {}
    device = next(model.parameters()).device
    time1 = time.time()
    for n, batch_data_dict in enumerate(generator):
        print(n)
        batch_waveform = move_data_to_device(batch_data_dict['waveform'],
            device)
        with torch.no_grad():
            model.eval()
            batch_output = model(batch_waveform)
        append_to_dict(output_dict, 'audio_name', batch_data_dict['audio_name']
            )
        append_to_dict(output_dict, 'clipwise_output', batch_output[
            'clipwise_output'].data.cpu().numpy())
        if 'segmentwise_output' in batch_output.keys():
            append_to_dict(output_dict, 'segmentwise_output', batch_output[
                'segmentwise_output'].data.cpu().numpy())
        if 'framewise_output' in batch_output.keys():
            append_to_dict(output_dict, 'framewise_output', batch_output[
                'framewise_output'].data.cpu().numpy())
        if return_input:
            append_to_dict(output_dict, 'waveform', batch_data_dict['waveform']
                )
        if return_target:
            if 'target' in batch_data_dict.keys():
                append_to_dict(output_dict, 'target', batch_data_dict['target']
                    )
        if n % 10 == 0:
            print(' --- Inference time: {:.3f} s / 10 iterations ---'.
                format(time.time() - time1))
            time1 = time.time()
    for key in output_dict.keys():
        output_dict[key] = np.concatenate(output_dict[key], axis=0)
    return output_dict
