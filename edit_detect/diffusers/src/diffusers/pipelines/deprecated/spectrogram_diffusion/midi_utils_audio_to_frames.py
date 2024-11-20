def audio_to_frames(samples, hop_size: int, frame_rate: int) ->Tuple[
    Sequence[Sequence[int]], torch.Tensor]:
    """Convert audio samples to non-overlapping frames and frame times."""
    frame_size = hop_size
    samples = np.pad(samples, [0, frame_size - len(samples) % frame_size],
        mode='constant')
    frames = frame(torch.Tensor(samples).unsqueeze(0), frame_length=
        frame_size, frame_step=frame_size, pad_end=False)
    num_frames = len(samples) // frame_size
    times = np.arange(num_frames) / frame_rate
    return frames, times
