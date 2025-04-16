def pad_framewise_output(framewise_output, frames_num):
    """Pad framewise_output to the same length as input frames. The pad value
    is the same as the value of the last frame.

    Args:
      framewise_output: (batch_size, frames_num, classes_num)
      frames_num: int, number of frames to pad

    Outputs:
      output: (batch_size, frames_num, classes_num)
    """
    pad = framewise_output[:, -1:, :].repeat(1, frames_num -
        framewise_output.shape[1], 1)
    """tensor for padding"""
    output = torch.cat((framewise_output, pad), dim=1)
    """(batch_size, frames_num, classes_num)"""
    return output
