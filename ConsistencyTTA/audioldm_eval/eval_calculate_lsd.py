def calculate_lsd(self, pairedloader, same_name=True, time_offset=160 * 7):
    if same_name == False:
        return {'lsd': -1, 'ssim_stft': -1}
    lsd_avg = []
    ssim_stft_avg = []
    for _, _, _, (audio1, audio2) in tqdm(pairedloader):
        audio1, audio2 = audio1.numpy()[0, :], audio2.numpy()[0, :]
        audio1 = audio1[time_offset:]
        audio1 = (audio1 - audio1.mean()) / np.abs(audio1).max()
        audio2 = (audio2 - audio2.mean()) / np.abs(audio2).max()
        min_len = min(audio1.shape[0], audio2.shape[0])
        audio1, audio2 = audio1[:min_len], audio2[:min_len]
        result = self.lsd(audio1, audio2)
        lsd_avg.append(result['lsd'])
        ssim_stft_avg.append(result['ssim'])
    return {'lsd': np.mean(lsd_avg), 'ssim_stft': np.mean(ssim_stft_avg)}
