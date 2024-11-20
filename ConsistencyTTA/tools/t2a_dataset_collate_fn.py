def collate_fn(self, data):
    """ Return:
        a list of captions,
        a tensor containing the groundtruth waveforms,
        a tensor containing the generated waveforms.
        """
    df = pd.DataFrame(data)
    captions, gt_waveforms, gen_waveforms, gen_mel = [df[i].tolist() for i in
        df]
    gt_waveforms = torch.cat(gt_waveforms, dim=0)
    gen_waveforms = torch.cat(gen_waveforms, dim=0)
    if gen_mel is None or None in gen_mel:
        gen_mel = None
    else:
        gen_mel = torch.cat(gen_mel, dim=0)
    return captions, gt_waveforms, gen_waveforms, gen_mel
