def score(self, generated_dir, groundtruth_dir, target_length=1000,
    store_embds=False):
    generated_embds = self.get_embeddings(generated_dir, target_length=
        target_length)
    groundtruth_embds = self.get_embeddings(groundtruth_dir, target_length=1000
        )
    if store_embds:
        np.save('generated_embds.npy', generated_embds)
        np.save('groundtruth_embds.npy', groundtruth_embds)
    if len(generated_embds) == 0:
        print('[Frechet Audio Distance] generated dir is empty, exitting...')
        return -1
    if len(groundtruth_embds) == 0:
        print('[Frechet Audio Distance] ground truth dir is empty, exitting...'
            )
        return -1
    groundtruth_mu, groundtruth_sigma = self.calculate_embd_statistics(
        groundtruth_embds)
    generated_mu, generated_sigma = self.calculate_embd_statistics(
        generated_embds)
    fad_score = self.calculate_frechet_distance(generated_mu,
        generated_sigma, groundtruth_mu, groundtruth_sigma)
    return {'frechet_audio_distance': fad_score}
