def audio_from_file(file_path, offset=0, duration=0, trim=False, target_sr=
    16000):
    audio = AudioSegment(file_path, target_sr=target_sr, int_values=False,
        offset=offset, duration=duration, trim=trim)
    samples = torch.tensor(audio.samples, dtype=torch.float).cuda()
    num_samples = torch.tensor(samples.shape[0]).int().cuda()
    return samples.unsqueeze(0), num_samples.unsqueeze(0)
