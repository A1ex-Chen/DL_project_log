def get_embeddings(self, audio_paths, sr=16000, target_length=1000):
    """
        Get embeddings using VGGish model.
        Params:
        -- audio_paths  :   A list of np.ndarray audio samples
        -- sr           :   Sampling rate. Default value is 16000.
        -- target_length:   Target audio length in centiseconds.
        """
    embd_lst = []
    for _, fname in enumerate(tqdm(os.listdir(audio_paths))):
        if fname.endswith('.wav'):
            audio = load_audio_task(os.path.join(audio_paths, fname),
                target_sr=sr, target_length=target_length)
            embd = self.model.forward(audio, sr).cpu().detach().numpy()
            embd_lst.append(embd)
    return np.concatenate(embd_lst, axis=0)
