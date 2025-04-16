def get_featuresdict(self, dataloader):
    out, out_meta = None, None
    for waveform, filename in tqdm(dataloader):
        metadict = {'file_path_': filename}
        waveform = waveform.squeeze(1).float().to(self.device)
        with torch.no_grad():
            featuresdict = self.mel_model(waveform)
            featuresdict = {k: [v.cpu()] for k, v in featuresdict.items()}
        out = featuresdict if out is None else {k: (out[k] + featuresdict[k
            ]) for k in out.keys()}
        out_meta = metadict if out_meta is None else {k: (out_meta[k] +
            metadict[k]) for k in out_meta.keys()}
    out = {k: torch.cat(v, dim=0) for k, v in out.items()}
    return {**out, **out_meta}
