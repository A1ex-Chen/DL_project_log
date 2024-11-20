def smooth_sequence(sequence, window_size=21):
    sequence = np.array(sequence)
    smoothed_sequence = np.zeros_like(sequence)
    for i in range(sequence.shape[0]):
        start = max(0, i - window_size + 1)
        end = i + 1
        smoothed_sequence[i] = np.mean(sequence[start:end], axis=0)
    return smoothed_sequence
