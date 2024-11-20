def cos_similarity(self, waveform, text):
    with torch.no_grad():
        self.embed_mode = 'audio'
        audio_emb = self(waveform.cuda())
        self.embed_mode = 'text'
        text_emb = self(text)
        similarity = F.cosine_similarity(audio_emb, text_emb, dim=2
            ), audio_emb, text_emb
        return similarity.squeeze()
