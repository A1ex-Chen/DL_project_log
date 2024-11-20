def calculate_psnr_ssim(self, pairedloader, same_name=True):
    if same_name == False:
        return {'psnr': -1, 'ssim': -1}
    psnr_avg, ssim_avg = [], []
    for mel_gen, mel_target, filename, _ in tqdm(pairedloader):
        mel_gen = mel_gen.cpu().numpy()[0]
        mel_target = mel_target.cpu().numpy()[0]
        psnrval = psnr(mel_gen, mel_target)
        if np.isinf(psnrval):
            print('Infinite value encountered in psnr %s ' % filename)
            continue
        psnr_avg.append(psnrval)
        ssim_avg.append(ssim(mel_gen, mel_target, data_range=1.0))
    return {'psnr': np.mean(psnr_avg), 'ssim': np.mean(ssim_avg)}
